#!/usr/bin/env python3
"""
使用 GSRenderer 批量渲染轨迹（RGB + 深度 + 法向）
输出格式适配 SDF Studio，深度为绝对尺度
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import cv2
from argparse import ArgumentParser
import sys
import os
from scipy.spatial.transform import Rotation

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gs2colmap.gaussian_renderer.gsRenderer import GSRenderer


class SDFStudioRenderer:
    """渲染器 - 输出适配 SDF Studio"""
    
    def __init__(self, ply_path, width=1280, height=720, fovy_deg=65.0, max_depth=10.0):
        """
        初始化渲染器
        
        Args:
            ply_path: PLY 文件路径（LiDAR 点云训练的 GS）
            width: 渲染宽度
            height: 渲染高度
            fovy_deg: 垂直 FOV（角度）
        """
        print(f"\n{'='*70}")
        print(f"初始化渲染器")
        print(f"{'='*70}")
        print(f"PLY 路径: {ply_path}")
        print(f"分辨率: {width} x {height}")
        print(f"FOV: {fovy_deg}°")
        
        self.render_width = width
        self.render_height = height
        self.fovy_deg = fovy_deg
        self.fovy_rad = np.radians(fovy_deg)
        self.max_depth = max_depth
        
        # 计算内参
        self.fy = height / (2 * np.tan(self.fovy_rad / 2))
        self.fx = self.fy  # 假设正方形像素
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        print(f"\n相机内参:")
        print(f"  fx = {self.fx:.2f}")
        print(f"  fy = {self.fy:.2f}")
        print(f"  cx = {self.cx:.2f}")
        print(f"  cy = {self.cy:.2f}")
        
        # 创建渲染器
        ply_path_abs = str(Path(ply_path).resolve())
        models_dict = {"background": ply_path_abs}
        
        print(f"\n创建 GSRenderer...")
        self.renderer = GSRenderer(
            models_dict=models_dict,
            render_width=width,
            render_height=height
        )
        
        print(f"✓ 渲染器初始化完成\n")
    
    def set_camera_from_c2w(self, c2w):
        """从 C2W 矩阵设置相机位姿"""
        trans = c2w[:3, 3].copy()
        rmat = c2w[:3, :3].copy()
        
        rot = Rotation.from_matrix(rmat)
        quat_xyzw = rot.as_quat()
        
        self.renderer.set_camera_pose(trans, quat_xyzw)
        self.renderer.set_camera_fovy(self.fovy_rad)
    
    def save_rgb(self, rgb, output_path):
        """保存 RGB 图像"""
        # rgb 已经是 uint8 格式 (H, W, 3)
        cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    def save_depth_raw(self, depth, output_path):
        """
        保存原始深度（绝对尺度）
        
        Args:
            depth: (H, W) 深度图，单位：米
            output_path: 输出路径
        
        说明：
            - 保存为 16 位 PNG，单位：毫米
            - 适配 SDF Studio 输入格式
            - 保留绝对尺度（LiDAR 训练的 GS 尺度准确）
        """
        if depth.ndim == 3:
            depth = depth.squeeze()
        depth_cleaned = depth.copy()
        depth_cleaned[depth == 0] = self.max_depth
        depth_cleaned[depth > self.max_depth] = self.max_depth
        # 转换为毫米，保存为 16 位
        depth_mm = (depth_cleaned * 1000.0).astype(np.uint16)
        cv2.imwrite(str(output_path), depth_mm)
    
    def save_depth_npy(self, depth, output_path):
        """
        保存 NumPy 格式深度（浮点，单位：米）
        
        用于需要精确浮点深度的场景
        """
        if depth.ndim == 3:
            depth = depth.squeeze()
        depth_cleaned = depth.copy()
        # depth_cleaned[depth == 0] = self.max_depth
        # depth_cleaned[depth > self.max_depth] = self.max_depth
        np.save(output_path, depth_cleaned.astype(np.float32))
    
    def save_depth_vis(self, depth, output_path):
        """保存深度可视化（彩色 colormap）"""
        if depth.ndim == 3:
            depth = depth.squeeze()
        
        # 归一化
        valid_mask = (depth > 0) & (depth < self.max_depth)
        if valid_mask.any():
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            # depth_max = self.max_depth
            depth_norm = np.zeros_like(depth)
            depth_norm[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min + 1e-6 )
            depth_norm[~valid_mask] = (depth_max - depth_min) / (depth_max - depth_min + 1e-6)
        else:
            depth_norm = np.zeros_like(depth)
        
        # 应用 colormap
        depth_colored = cv2.applyColorMap(
            (depth_norm * 255).astype(np.uint8), 
            cv2.COLORMAP_TURBO
        )
        cv2.imwrite(str(output_path), depth_colored)
     
    
    def save_normal(self, normal, output_path):
        """
        保存法向图
        
        Args:
            normal: (H, W, 3) 法向，范围 [-1, 1]
            output_path: 输出路径
        """
        # 转换到 [0, 255]
        normal_vis = ((normal + 1) * 127.5).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(output_path), cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))
    
    def save_normal_npy(self, normal, output_path):
        """保存 NumPy 格式法向（浮点，范围 [-1, 1]）"""
        np.save(output_path, normal.astype(np.float32))
    
    def render_trajectory(self, trajectory_json, output_dir):
        """
        渲染整个轨迹
        
        Args:
            trajectory_json: transforms.json 格式的轨迹文件
            output_dir: 输出目录
        
        输出结构（适配 SDF Studio）:
            output_dir/
                images/          # RGB 图像
                    0000.png
                    0001.png
                    ...
                depth/           # 原始深度（16位 PNG，毫米）
                    0000.png
                    0001.png
                    ...
                depth_npy/       # 浮点深度（NPY，米）
                    0000.npy
                    0001.npy
                    ...
                depth_vis/       # 深度可视化
                    0000.png
                    0001.png
                    ...
                normal/          # 法向图
                    0000.png
                    0001.png
                    ...
                normal_npy/      # 浮点法向（NPY）
                    0000.npy
                    0001.npy
                    ...
                transforms.json  # 相机参数
        """
        # 加载轨迹
        print(f"{'='*70}")
        print(f"加载轨迹")
        print(f"{'='*70}")
        print(f"文件: {trajectory_json}")
        
        with open(trajectory_json, 'r') as f:
            data = json.load(f)
        
        num_frames = len(data['frames'])
        print(f"帧数: {num_frames}\n")
        
        # 创建输出目录
        output_dir = Path(output_dir)
        
        rgb_dir = output_dir / "images"
        depth_dir = output_dir / "depth_img"
        depth_npy_dir = output_dir / "depth"
        depth_vis_dir = output_dir / "depth_vis"
        normal_dir = output_dir / "normal"
        normal_npy_dir = output_dir / "normal_npy"
        
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_npy_dir.mkdir(parents=True, exist_ok=True)
        depth_vis_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)
        normal_npy_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{'='*70}")
        print(f"开始渲染")
        print(f"{'='*70}\n")
        
        # 保存所有帧的数据
        frames_output = []
        # 渲染每一帧
        for idx, frame in enumerate(tqdm(data['frames'], desc="渲染进度")):
            # 获取 C2W 位姿
            c2w = np.array(frame['transform_matrix'], dtype=np.float32)
            # c2w_mujoco = np.array(frame['transform_matrix'], dtype=np.float32)
            
            # 设置相机
            self.set_camera_from_c2w(c2w)
            
            # 渲染 - 返回 (rgb, depth, normal)
            with torch.no_grad():
                rgb, depth, normal = self.renderer.render()
            
            # DEBUG: 第一帧打印详细信息
            # if idx == 0:
                # print(f"\n[第一帧渲染结果]")
                # print(f"  RGB:")
                # print(f"    shape: {rgb.shape}")
                # print(f"    dtype: {rgb.dtype}")
                # print(f"    range: [{rgb.min()}, {rgb.max()}]")
                # print(f"  Depth:")
                # print(f"    shape: {depth.shape}")
                # print(f"    dtype: {depth.dtype}")
                # print(f"    range: [{depth.min():.3f}, {depth.max():.3f}] 米")
                # print(f"  Normal:")
                # print(f"    shape: {normal.shape}")
                # print(f"    dtype: {normal.dtype}")
                # print(f"    range: [{normal.min():.3f}, {normal.max():.3f}]")
                # print()
            
            # 文件名
            frame_name = f"{idx:04d}"
            
            # 保存 RGB
            self.save_rgb(rgb, rgb_dir / f"{frame_name}.png")
            
            # 保存深度（多种格式）
            self.save_depth_raw(depth, depth_dir / f"{frame_name}.png")      # 16位 PNG（毫米）
            self.save_depth_npy(depth, depth_npy_dir / f"{frame_name}.npy")  # 浮点 NPY（米）
            self.save_depth_vis(depth, depth_vis_dir / f"{frame_name}.png")  # 可视化
            
            # 保存法向（多种格式）
            self.save_normal(normal, normal_dir / f"{frame_name}.png")        # PNG 可视化
            self.save_normal_npy(normal, normal_npy_dir / f"{frame_name}.npy")  # 浮点 NPY


            
            # 记录帧信息
            frame_info = {
                "file_path": f"images/{frame_name}.png",
                "depth_path": f"depth_img/{frame_name}.png",
                "depth_npy_path": f"depth/{frame_name}.npy",
                "normal_path": f"normal/{frame_name}.png",
                "normal_npy_path": f"normal_npy/{frame_name}.npy",
                "transform_matrix": c2w.tolist()
            }
            frames_output.append(frame_info)
        
        # 保存 transforms.json
        output_transforms = {
            "camera_model": "OPENCV",
            "fl_x": self.fx,
            "fl_y": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.render_width,
            "h": self.render_height,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "frames": frames_output
        }
        
        transforms_path = output_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(output_transforms, f, indent=2)
        
        # 输出统计信息
        # print(f"\n{'='*70}")
        # print(f"渲染完成！")
        # print(f"{'='*70}")
        # print(f"输出目录: {output_dir}")
        # print(f"  RGB:            {rgb_dir} ({num_frames} 张)")
        # print(f"  深度（PNG）:     {depth_dir} ({num_frames} 张，16位，单位：毫米)")
        # print(f"  深度（NPY）:     {depth_npy_dir} ({num_frames} 个，浮点，单位：米)")
        # print(f"  深度可视化:      {depth_vis_dir} ({num_frames} 张)")
        # print(f"  法向（PNG）:     {normal_dir} ({num_frames} 张)")
        # print(f"  法向（NPY）:     {normal_npy_dir} ({num_frames} 个)")
        # print(f"  相机参数:       {transforms_path}")
        # print(f"\n深度格式说明:")
        # print(f"  - PNG (depth/):     16位无符号整数，单位：毫米")
        # print(f"  - NPY (depth_npy/): 32位浮点，单位：米")
        # print(f"  - 深度为绝对尺度（LiDAR 训练的 GS）")
        # print(f"\n适配 SDF Studio:")
        # print(f"  使用 depth/ 或 depth_npy/ 作为深度输入")
        # print(f"  transforms.json 包含相机内外参")
        # print(f"{'='*70}\n")


def main():
    parser = ArgumentParser(description="批量渲染轨迹 - 输出适配 SDF Studio")
    parser.add_argument("--ply", type=str, required=True,
                       help="PLY 文件路径（LiDAR 点云训练的 GS）")
    parser.add_argument("--trajectory", type=str, required=True,
                       help="轨迹 JSON 文件（transforms.json 格式）")
    parser.add_argument("--output", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--width", type=int, default=1280,
                       help="渲染宽度（默认 1280）")
    parser.add_argument("--height", type=int, default=720,
                       help="渲染高度（默认 720）")
    parser.add_argument("--fovy", type=float, default=65.0,
                       help="垂直 FOV 角度（默认 65°）")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"GS 批量渲染 - 适配 SDF Studio")
    print(f"{'='*70}")
    print(f"输入:")
    print(f"  PLY:       {args.ply}")
    print(f"  轨迹:      {args.trajectory}")
    print(f"输出:")
    print(f"  目录:      {args.output}")
    print(f"参数:")
    print(f"  分辨率:    {args.width} x {args.height}")
    print(f"  FOV:       {args.fovy}°")
    
    # 创建渲染器
    renderer = SDFStudioRenderer(
        ply_path=args.ply,
        width=args.width,
        height=args.height,
        fovy_deg=args.fovy
    )
    
    # 渲染轨迹
    renderer.render_trajectory(
        trajectory_json=args.trajectory,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()