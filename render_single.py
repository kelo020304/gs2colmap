#!/usr/bin/env python3
"""
测试单个位姿的渲染 - MuJoCo 坐标系
世界: X前, Y左, Z上
相机: X右, Y上, Z后
"""
import pdb
import torch
import numpy as np
from pathlib import Path
import cv2
import sys
import os
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gs2colmap.gaussian_renderer.gsRenderer import GSRenderer


def create_mujoco_c2w(pos, look_at, up=np.array([0, 0, 1])):
    """
    创建 MuJoCo 坐标系的 C2W 矩阵
    
    世界坐标系: X前, Y左, Z上
    MuJoCo 相机: X右, Y上, Z后 (看向 -Z)
    
    Args:
        pos: 相机位置 (世界坐标)
        look_at: 看向的点 (世界坐标)
        up: 世界上方向 (默认 Z 轴)
    """
    # 计算相机坐标系的 Z 轴（后方）
    # 相机看向 -Z，所以 Z_cam = -(look_at - pos) 的方向
    forward_world = look_at - pos
    forward_world = forward_world / np.linalg.norm(forward_world)
    z_cam = -forward_world  # 相机 Z 轴指向后方
    
    # 计算相机坐标系的 X 轴（右方）
    # X_cam = forward × up（叉乘）
    x_cam = np.cross(forward_world, up)
    x_cam = x_cam / np.linalg.norm(x_cam)
    
    # 计算相机坐标系的 Y 轴（上方）
    # Y_cam = Z_cam × X_cam
    y_cam = np.cross(z_cam, x_cam)
    
    # 构建旋转矩阵（列向量是相机坐标系的基）
    R = np.column_stack([x_cam, y_cam, z_cam])
    
    # 构建 C2W 矩阵
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = pos
    
    return c2w


def test_render_single_pose():
    print("=" * 70)
    print("测试 MuJoCo 坐标系渲染")
    print("世界: X前, Y左, Z上")
    print("相机: X右, Y上, Z后")
    print("=" * 70)
    
    ply_path = "gs2colmap/gs_ply/point_cloud_n.ply"
    width = 1280
    height = 720
    fovy_deg = 65.0
    
    # 测试位姿 1: 从后方看（沿 -X 看向原点）
    print("\n位姿 1: 从后方看（沿 -X）")
    pos1 = np.array([-3.0, 0.0, 1.5])  # 后方
    look_at1 = np.array([0.0, 0.0, 1.0])
    c2w1 = create_mujoco_c2w(pos1, look_at1)
    print(f"相机位置: {pos1}")
    print(f"看向: {look_at1}")
    print(f"C2W:\n{c2w1}")
    
    rendered1, depth1, normal1 = render_single(ply_path, c2w1, width, height, fovy_deg)
    import pdb; pdb.set_trace()
    save_image(rendered1, "gs2colmap/test_mj_pose1.png")
    save_depth(depth1, "gs2colmap/pose1_depth_vis.png", colormap=True)
    save_normal(normal1, "gs2colmap/pose1_normal.png")
    
    print("\n" + "=" * 70)



def render_single(ply_path, c2w, width, height, fovy_deg):
    ply_path_abs = str(Path(ply_path).resolve())
    models_dict = {"background": ply_path_abs}
    renderer = GSRenderer(
        models_dict=models_dict,
        render_width=width,
        render_height=height
    )
    
    trans = c2w[:3, 3].copy()
    rmat = c2w[:3, :3].copy()
    rot = Rotation.from_matrix(rmat)
    quat_xyzw = rot.as_quat()
    
    fovy_rad = np.radians(fovy_deg)
    renderer.set_camera_pose(trans, quat_xyzw)
    renderer.set_camera_fovy(fovy_rad)
    
    with torch.no_grad():
        out_rgb, out_depth, out_normal = renderer.render()
    # pdb.set_trace()
    # rendering = out["render"].clamp(0.0, 1.0)
    # rgb = rendering.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
    # _, H, W = rendering.shape
    
    # depth = out["plane_depth"].squeeze()
    # depth_tsdf = depth.clone()
    # depth = depth.detach().cpu().numpy()
    # depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
    # depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
    # depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

    # normal = out["rendered_normal"].permute(1,2,0)  
    # normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
    # normal = normal.detach().cpu().numpy()
    # normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)


    return out_rgb, out_depth, out_normal


def save_image(rgb, path):
    # rgb = np.clip(rgb, 0, 1)
    # rgb_uint8 = (rgb * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

def save_depth(depth, output_path, colormap=True):
    """保存深度图"""
    if depth is None:
        print(f"  [警告] 深度为空")
        return
    
    if depth.ndim == 3:
        depth = depth.squeeze()
    
    if colormap:
        # 彩色可视化
        valid_mask = depth > 0
        if valid_mask.any():
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            depth_norm = np.zeros_like(depth)
            depth_norm[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min + 1e-6)
        else:
            depth_norm = depth
        
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(output_path), depth_colored)
    else:
        # 原始深度（16位，单位：毫米）
        depth_uint16 = (depth * 1000).astype(np.uint16)
        cv2.imwrite(str(output_path), depth_uint16)
    
    print(f"  保存深度: {output_path}")


def save_normal(normal, output_path):
    """保存法向图"""
    if normal is None:
        print(f"  [警告] 法向为空")
        return
    
    print(f"  保存法向: {output_path}")
    print(f"    shape: {normal.shape}")
    print(f"    range: [{normal.min():.3f}, {normal.max():.3f}]")
    
    # 从 [-1, 1] 转换到 [0, 255]
    normal_vis = ((normal + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # BGR 转换
    normal_bgr = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), normal_bgr)
    
    print(f"    ✓ 保存成功")


if __name__ == "__main__":
    test_render_single_pose()