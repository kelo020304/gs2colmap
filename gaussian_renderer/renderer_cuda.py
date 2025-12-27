'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''

from gs2colmap.gaussian_renderer import util_gau
import numpy as np
import torch
from dataclasses import dataclass
from diff_gaussian_rasterization import GaussianRasterizer

@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-2]

@dataclass
class GaussianRasterizationSettingsStorage:
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

def gaus_cuda_from_cpu(gau: util_gau) -> GaussianDataCUDA:
    gaus =  GaussianDataCUDA(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus

class CUDARenderer:
    def __init__(self, w, h):
        super().__init__()
        raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": 1,
            "tanfovy": 1,
            "bg": torch.Tensor([0., 0., 0]).float().cuda(),
            "scale_modifier": 1.,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 3,
            "campos": None,
            "prefiltered": False,
            "debug": False
        }
        self.raster_settings = GaussianRasterizationSettingsStorage(**raster_settings)

        self.depth_render = False
        self.need_rerender = True
        self.render_rgb_img = None
        self.render_depth_img = None

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.need_rerender = True
        if type(gaus) is dict:
            gau_xyz = []
            gau_rot = []
            gau_s = []
            gau_a = []
            gau_c = []
            for gaus_item in gaus.values():
                gau_xyz.append(gaus_item.xyz)
                gau_rot.append(gaus_item.rot)
                gau_s.append(gaus_item.scale)
                gau_a.append(gaus_item.opacity)
                gau_c.append(gaus_item.sh)
            self.gau_env_idx = gau_xyz[0].shape[0]
            gau_xyz = np.concatenate(gau_xyz, axis=0)
            gau_rot = np.concatenate(gau_rot, axis=0)
            gau_s = np.concatenate(gau_s, axis=0)
            gau_a = np.concatenate(gau_a, axis=0)
            gau_c = np.concatenate(gau_c, axis=0)
            gaus_all = util_gau.GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c)
            self.gaussians = gaus_cuda_from_cpu(gaus_all)
        else:
            self.gaussians = gaus_cuda_from_cpu(gaus)
        self.raster_settings.sh_degree = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1

        num_points = self.gaussians.xyz.shape[0]

        self.gau_ori_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_ori_xyz_all_cu[..., :] = torch.from_numpy(gau_xyz).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu[..., :] = torch.from_numpy(gau_rot).cuda().requires_grad_(False)

        self.gau_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)

    def set_scale_modifier(self, modifier):
        self.need_rerender = True
        self.raster_settings.scale_modifier = float(modifier)

    def set_render_reso(self, w, h):
        self.need_rerender = True
        self.raster_settings.image_height = int(h)
        self.raster_settings.image_width = int(w)

    def update_camera_pose(self, camera: util_gau.Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0,2], :] *= -1
        
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings.viewmatrix = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings.campos = torch.tensor(camera.position).float().cuda()
        self.raster_settings.projmatrix = torch.tensor(proj.T).float().cuda()

    def update_camera_pose_from_topic(self, camera: util_gau.Camera, rmat, trans):
        self.need_rerender = True

        camera.position = np.array(trans).astype(np.float32)
        camera.target = camera.position - (1. * rmat[:3,2]).astype(np.float32)

        Tmat = np.eye(4)
        Tmat[:3,:3] = rmat
        Tmat[:3,3] = trans
        Tmat[0:3, [1,2]] *= -1
        transpose = np.array([[-1.0,  0.0,  0.0,  0.0],
                              [ 0.0, -1.0,  0.0,  0.0],
                              [ 0.0,  0.0,  1.0,  0.0],
                              [ 0.0,  0.0,  0.0,  1.0]])
        view_matrix = transpose @ np.linalg.inv(Tmat)

        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings.projmatrix = torch.tensor(proj.T).float().cuda()
        self.raster_settings.viewmatrix = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings.campos = torch.tensor(camera.position).float().cuda()

    def update_camera_intrin(self, camera: util_gau.Camera):
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings.tanfovx = hfovx
        self.raster_settings.tanfovy = hfovy

    def draw(self, render_depth=False, render_normal=False, use_surface_depth=True):
        if not self.need_rerender:
            results = [self.render_rgb_img]
            if render_depth:
                # return self.render_depth_img
                results.append(self.render_depth_img)
            # else:
            #     return self.render_rgb_img
            if render_normal:
                results.append(self.render_normal_img)
            return tuple(results) if len(results) > 1 else results[0]

        self.need_rerender = False
        # run cuda rasterizer now is just a placeholder
        rasterizer = GaussianRasterizer(raster_settings=self.raster_settings)

        with torch.no_grad():
            color_img, radii, depth_img, _alpha = rasterizer(
                means3D = self.gaussians.xyz,
                means2D = None,
                shs = self.gaussians.sh,
                colors_precomp = None,
                opacities = self.gaussians.opacity,
                scales = self.gaussians.scale,
                rotations = self.gaussians.rot,
                cov3D_precomp = None
            )

            self.render_depth_img = depth_img.permute(1, 2, 0).contiguous().cpu().numpy()
            self.render_rgb_img = (255. * torch.clamp(color_img, 0.0, 1.0)).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()

            if render_normal:
                self.render_normal_img = self._compute_normal_from_depth_with_alpha(
                    depth_img.squeeze(),  # (H, W)
                    _alpha.squeeze(),  # (H, W)
                    self.raster_settings.image_width,
                    self.raster_settings.image_height
                )

        # if render_depth:
        #     return self.render_rgb_img, self.render_depth_img
        
        # else:
        #     return self.render_rgb_img
        results = [self.render_rgb_img]
        if render_depth:
            results.append(self.render_depth_img)
        if render_normal:
            results.append(self.render_normal_img)
        
        return tuple(results) if len(results) > 1 else results[0]

    def _compute_normal_from_depth(self, depth_tensor, width, height):
        """
        从深度 tensor 计算法向
        
        Args:
            depth_tensor: (H, W) torch tensor
            width, height: 图像尺寸
        
        Returns:
            normal: (H, W, 3) numpy array，范围 [-1, 1]
        """
        # 获取相机内参
        fx = fy = width / (2 * np.tan(np.arctan(self.raster_settings.tanfovx)))
        cx = width / 2
        cy = height / 2
        
        H, W = depth_tensor.shape
        device = depth_tensor.device
        
        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        depth_4d = depth_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        depth_dx = torch.nn.functional.conv2d(depth_4d, sobel_x, padding=1).squeeze()
        depth_dy = torch.nn.functional.conv2d(depth_4d, sobel_y, padding=1).squeeze()
        
        # 构建坐标网格
        y, x = torch.meshgrid(torch.arange(H, device=device), 
                            torch.arange(W, device=device), indexing='ij')
        
        x_norm = (x.float() - cx) / fx
        y_norm = (y.float() - cy) / fy
        
        # 3D 坐标
        z = depth_tensor
        
        # 偏导数
        du_x = x_norm * depth_dx + z / fx
        du_y = y_norm * depth_dx
        du_z = depth_dx
        
        dv_x = x_norm * depth_dy
        dv_y = y_norm * depth_dy + z / fy
        dv_z = depth_dy
        
        # 叉乘
        normal_x = du_y * dv_z - du_z * dv_y
        normal_y = du_z * dv_x - du_x * dv_z
        normal_z = du_x * dv_y - du_y * dv_x
        
        normal = torch.stack([normal_x, normal_y, normal_z], dim=-1)  # (H, W, 3)
        
        # 归一化
        norm = torch.norm(normal, dim=-1, keepdim=True)
        norm = torch.where(norm > 1e-6, norm, torch.ones_like(norm))
        normal = normal / norm
        
        # 转换为 numpy
        normal_np = normal.cpu().numpy()
        
        return normal_np    
    
    def _compute_normal_from_depth_with_alpha(self, depth_tensor, alpha_tensor, width, height):
        """基于置信度的法向计算"""
        
        fx = fy = width / (2 * np.tan(np.arctan(self.raster_settings.tanfovx)))
        cx = width / 2
        cy = height / 2
        
        H, W = depth_tensor.shape
        device = depth_tensor.device
        
        # ===== 关键：置信度 mask =====
        alpha_threshold = 0.9  # 可调：0.85-0.95
        high_confidence = alpha_tensor > alpha_threshold
        
        # ===== 轻微平滑（只在高置信度区域）=====
        smooth_sigma = 1.0  # 可调：0.5-2.0
        
        if smooth_sigma > 0:
            kernel_size = int(smooth_sigma * 3) * 2 + 1
            kernel = torch.exp(-torch.arange(kernel_size, dtype=torch.float32, device=device)
                            .sub_(kernel_size // 2).pow(2) / (2 * smooth_sigma**2))
            kernel_2d = (kernel.unsqueeze(0) * kernel.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
            kernel_2d = kernel_2d / kernel_2d.sum()
            
            # 只平滑高置信度
            depth_masked = depth_tensor.clone()
            depth_masked[~high_confidence] = 0
            
            depth_smooth = torch.nn.functional.conv2d(
                depth_masked.unsqueeze(0).unsqueeze(0), kernel_2d, padding=kernel_size//2
            ).squeeze()
            
            # 归一化
            mask_smooth = torch.nn.functional.conv2d(
                high_confidence.float().unsqueeze(0).unsqueeze(0), kernel_2d, padding=kernel_size//2
            ).squeeze()
            depth_smooth = depth_smooth / (mask_smooth + 1e-6)
            depth_smooth[~high_confidence] = depth_tensor[~high_confidence]
        else:
            depth_smooth = depth_tensor
        
        # ===== 计算梯度（Scharr）=====
        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], 
                                dtype=torch.float32, device=device).view(1, 1, 3, 3) / 32.0
        scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], 
                                dtype=torch.float32, device=device).view(1, 1, 3, 3) / 32.0
        
        depth_4d = depth_smooth.unsqueeze(0).unsqueeze(0)
        depth_dx = torch.nn.functional.conv2d(depth_4d, scharr_x, padding=1).squeeze()
        depth_dy = torch.nn.functional.conv2d(depth_4d, scharr_y, padding=1).squeeze()
        
        # ===== 计算法向 =====
        y, x = torch.meshgrid(torch.arange(H, device=device), 
                            torch.arange(W, device=device), indexing='ij')
        
        x_norm = (x.float() - cx) / fx
        y_norm = (y.float() - cy) / fy
        z = depth_smooth
        
        du_x = z / fx + x_norm * depth_dx
        du_y = y_norm * depth_dx
        du_z = depth_dx
        
        dv_x = x_norm * depth_dy
        dv_y = z / fy + y_norm * depth_dy
        dv_z = depth_dy
        
        normal_x = du_y * dv_z - du_z * dv_y
        normal_y = du_z * dv_x - du_x * dv_z
        normal_z = du_x * dv_y - du_y * dv_x
        
        normal = torch.stack([normal_x, normal_y, normal_z], dim=-1)
        
        norm = torch.norm(normal, dim=-1, keepdim=True)
        normal = normal / torch.where(norm > 1e-6, norm, torch.ones_like(norm))
        
        # ===== 低置信度区域设为朝向相机 =====
        normal[~high_confidence] = torch.tensor([0., 0., 1.], device=device)
        
        return normal.cpu().numpy()