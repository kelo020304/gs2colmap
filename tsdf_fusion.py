#!/usr/bin/env python3
"""
使用 Open3D TSDF Integration 从深度图融合 SDF
"""

import open3d as o3d
import numpy as np
import json
from pathlib import Path
import cv2
from tqdm import tqdm


class TSDFFusion:
    """TSDF 融合器"""
    
    def __init__(self, voxel_length=0.02, sdf_trunc=0.08):
        """
        Args:
            voxel_length: 体素大小（米）
            sdf_trunc: TSDF 截断距离（米）
        """
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        
        # 创建 TSDF Volume
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
    def integrate_frame(self, rgb, depth, intrinsic, extrinsic):
        """
        融合一帧
        
        Args:
            rgb: (H, W, 3) uint8
            depth: (H, W) float32, 单位：米
            intrinsic: Open3D PinholeCameraIntrinsic
            extrinsic: (4, 4) 世界到相机的变换矩阵
        """
        # 创建 RGBD 图像
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1.0,  # 深度已经是米
            depth_trunc=10.0,  # 截断远处深度
            convert_rgb_to_intensity=False
        )
        
        # 融合
        self.volume.integrate(rgbd, intrinsic, extrinsic)
    
    def extract_mesh(self):
        """提取 Mesh"""
        print("提取 Triangle Mesh...")
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh
    
    def extract_point_cloud(self):
        """提取点云"""
        print("提取点云...")
        pcd = self.volume.extract_point_cloud()
        return pcd
    
    def save_mesh(self, output_path):
        """保存 Mesh"""
        mesh = self.extract_mesh()
        o3d.io.write_triangle_mesh(str(output_path), mesh)
        print(f"✓ 保存 Mesh: {output_path}")
        return mesh


def fusion_from_renders(render_dir, output_mesh):
    """
    从渲染结果融合 TSDF
    
    Args:
        render_dir: 渲染输出目录（包含 transforms.json）
        output_mesh: 输出 Mesh 路径
    """
    render_dir = Path(render_dir)
    
    # 加载 transforms.json
    transforms_path = render_dir / "transforms.json"
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    # 相机内参
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(data['w']),
        height=int(data['h']),
        fx=data['fl_x'],
        fy=data['fl_y'],
        cx=data['cx'],
        cy=data['cy']
    )
    
    # 创建融合器
    fusion = TSDFFusion(voxel_length=0.01, sdf_trunc=0.04)
    
    # ⚠️ OpenGL -> OpenCV 坐标系转换矩阵
    # OpenGL: X右, Y上, Z后 -> OpenCV: X右, Y下, Z前
    gl_to_cv = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],  # Y 翻转
        [0,  0, -1, 0],  # Z 翻转
        [0,  0,  0, 1]
    ], dtype=np.float64)
    
    print(f"开始融合 {len(data['frames'])} 帧...")
    print(f"体素大小: {fusion.voxel_length}m")
    print(f"TSDF 截断: {fusion.sdf_trunc}m")
    
    for idx, frame in enumerate(tqdm(data['frames'])):
        # 加载 RGB
        rgb_path = render_dir / frame['file_path']
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            print(f"警告: 无法加载 {rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 加载深度（NPY 格式,米）
        depth_path = render_dir / frame['depth_npy_path']
        depth = np.load(depth_path)
        
        # 过滤无效深度
        depth[depth <= 0] = 0
        depth[depth > 10.0] = 0
        
        # C2W（OpenGL 约定）
        c2w_gl = np.array(frame['transform_matrix'], dtype=np.float64)
        
        # 转换到 OpenCV 约定
        c2w_cv = c2w_gl @ gl_to_cv
        # c2w_cv = c2w_gl
        
        # 转换为 W2C（Open3D 需要的外参）
        w2c_cv = np.linalg.inv(c2w_cv)
        
        # 调试第一帧
        if idx == 0:
            print(f"\n第一帧调试信息:")
            print(f"  RGB shape: {rgb.shape}")
            print(f"  Depth shape: {depth.shape}")
            print(f"  Depth range: [{depth[depth>0].min():.3f}, {depth[depth>0].max():.3f}] 米")
            print(f"  C2W (OpenGL) 相机位置: {c2w_gl[:3, 3]}")
            print(f"  C2W (OpenCV) 相机位置: {c2w_cv[:3, 3]}")
            print(f"  W2C (OpenCV):\n{w2c_cv}")
        
        # 融合（使用转换后的 W2C）
        fusion.integrate_frame(rgb, depth, intrinsic, w2c_cv)
    
    # 保存 Mesh
    mesh = fusion.save_mesh(output_mesh)
    
    # 可视化
    print("\n可视化 Mesh...")
    o3d.visualization.draw_geometries([mesh])



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", required=True, help="渲染输出目录")
    parser.add_argument("--output", default="output.ply", help="输出 Mesh")
    args = parser.parse_args()
    
    fusion_from_renders(args.render_dir, args.output)