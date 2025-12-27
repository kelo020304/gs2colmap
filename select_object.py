#!/usr/bin/env python3
"""
交互式相机轨迹生成器 - MuJoCo 坐标系
世界坐标系: X前, Y左, Z上
相机坐标系: X右, Y上, Z后 (看向 -Z)

操作说明:
1. 使用 Shift+左键 点击点云选择中心点
2. 自动生成轨迹并可视化（显示相机位姿）
3. 查看满意后按 Q 保存
"""

import open3d as o3d
import numpy as np
import json
from pathlib import Path
import argparse


class InteractiveTrajectoryGenerator:
    """交互式轨迹生成器"""
    
    def __init__(self, pcd_path, output_path="trajectory.json",
                 radius=1.0, height_offset=0.3, num_views=50,
                 clockwise=True, start_angle_deg=0.0):
        self.pcd_path = pcd_path
        self.output_path = output_path
        
        # 加载点云
        print(f"加载点云: {pcd_path}")
        self.pcd = o3d.io.read_point_cloud(str(pcd_path))
        print(f"点数: {len(self.pcd.points)}")


        # 轨迹参数
        self.center = None
        self.radius = radius
        self.height_offset = height_offset
        self.num_views = num_views
        self.elevation_deg = 15.0
        self.start_angle_deg = start_angle_deg
        self.clockwise = clockwise
        
        # 渲染参数
        self.width = 1280
        self.height = 720
        self.fovy_deg = 50.0
        
        # 可视化参数
        self.show_every = max(1, num_views // 20)  # 显示约20个相机
        self.frustum_scale = 0.15
        
    def create_mujoco_c2w(self, pos, look_at, up=np.array([0, 0, 1])):
        """创建 MuJoCo C2W 矩阵"""
        forward_world = look_at - pos
        forward_world = forward_world / np.linalg.norm(forward_world)
        z_cam = -forward_world
        
        x_cam = np.cross(forward_world, up)
        x_cam = x_cam / np.linalg.norm(x_cam)
        
        y_cam = np.cross(z_cam, x_cam)
        
        R = np.column_stack([x_cam, y_cam, z_cam])
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        
        return c2w
    
    def generate_trajectory(self):
        """生成轨迹"""
        if self.center is None:
            return None
        
        poses = []
        angle_range = 2 * np.pi
        start_angle = np.radians(self.start_angle_deg)
        direction = -1 if self.clockwise else 1
        
        for i in range(self.num_views):
            theta = start_angle + direction * angle_range * i / self.num_views
            
            x = self.center[0] + self.radius * np.cos(theta)
            y = self.center[1] + self.radius * np.sin(theta)
            z = self.center[2] + self.height_offset + self.radius * np.sin(np.radians(self.elevation_deg))
            
            camera_pos = np.array([x, y, z])
            look_at = self.center.copy()
            
            c2w = self.create_mujoco_c2w(camera_pos, look_at)
            poses.append(c2w)
        
        return np.array(poses)
    
    def create_camera_frustum(self, c2w, scale=0.2, color=[1, 0, 0]):
        """创建相机视锥体"""
        frustum_points = np.array([
            [0, 0, 0],
            [-scale, -scale, -scale],
            [scale, -scale, -scale],
            [scale, scale, -scale],
            [-scale, scale, -scale],
        ])
        
        frustum_world = []
        for p in frustum_points:
            p_homo = np.append(p, 1)
            p_world = c2w @ p_homo
            frustum_world.append(p_world[:3])
        
        frustum_world = np.array(frustum_world)
        
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1],
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(frustum_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
        
        return line_set
    
    def create_camera_axis(self, c2w, scale=0.2):
        """创建相机坐标轴"""
        origin = c2w[:3, 3]
        x_axis = c2w[:3, 0] * scale  # 红色：X 右
        y_axis = c2w[:3, 1] * scale  # 绿色：Y 上
        z_axis = c2w[:3, 2] * scale  # 蓝色：Z 后
        
        axes = []
        
        # X 轴（红色）
        points_x = np.array([origin, origin + x_axis])
        lines_x = [[0, 1]]
        line_set_x = o3d.geometry.LineSet()
        line_set_x.points = o3d.utility.Vector3dVector(points_x)
        line_set_x.lines = o3d.utility.Vector2iVector(lines_x)
        line_set_x.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        axes.append(line_set_x)
        
        # Y 轴（绿色）
        points_y = np.array([origin, origin + y_axis])
        lines_y = [[0, 1]]
        line_set_y = o3d.geometry.LineSet()
        line_set_y.points = o3d.utility.Vector3dVector(points_y)
        line_set_y.lines = o3d.utility.Vector2iVector(lines_y)
        line_set_y.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        axes.append(line_set_y)
        
        # Z 轴（蓝色）
        points_z = np.array([origin, origin + z_axis])
        lines_z = [[0, 1]]
        line_set_z = o3d.geometry.LineSet()
        line_set_z.points = o3d.utility.Vector3dVector(points_z)
        line_set_z.lines = o3d.utility.Vector2iVector(lines_z)
        line_set_z.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
        axes.append(line_set_z)
        
        return axes
    
    def create_trajectory_path(self, poses, color=[0, 0.8, 0]):
        """创建轨迹路径"""
        positions = poses[:, :3, 3]
        lines = [[i, i+1] for i in range(len(positions)-1)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
        
        return line_set
    
    def visualize_trajectory(self, poses):
        """可视化轨迹和相机位姿"""
        print("\n" + "=" * 70)
        print("可视化轨迹")
        print("=" * 70)
        print(f"总视角数: {len(poses)}")
        print(f"显示相机: {len(poses) // self.show_every + 1} 个")
        print("\n可视化说明:")
        print("  🔴 红球: 中心点")
        print("  🟢 绿线: 轨迹路径")
        print("  🎥 相机视锥体: 红->蓝 (时间顺序)")
        print("  📍 相机坐标轴: 红=X右, 绿=Y上, 蓝=Z后")
        print("  🟢 绿球: 起点")
        print("  🔴 红球: 终点")
        print("\n按 Q 关闭窗口并保存轨迹")
        print("=" * 70)
        
        geometries = []
        
        # 点云
        self.pcd.paint_uniform_color([1, 0, 0])
        geometries.append(self.pcd)
        
        # 世界坐标系
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        geometries.append(world_frame)
        
        # 中心点（红色球）
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        center_sphere.translate(self.center)
        center_sphere.paint_uniform_color([1, 0, 0])
        geometries.append(center_sphere)
        
        # 轨迹路径（绿色线）
        trajectory_path = self.create_trajectory_path(poses, color=[0, 0.8, 0])
        geometries.append(trajectory_path)
        
        # 相机视锥体和坐标轴
        for i in range(0, len(poses), self.show_every):
            c2w = poses[i]
            t = i / len(poses)
            color = [1-t, 0, t]  # 红 -> 蓝
            
            # 视锥体
            frustum = self.create_camera_frustum(c2w, scale=self.frustum_scale, color=color)
            geometries.append(frustum)
            
            # 每隔更多帧显示坐标轴
            if i % (self.show_every * 2) == 0:
                axes = self.create_camera_axis(c2w, scale=0.1)
                geometries.extend(axes)
        
        # 起点（绿色球）
        start_pos = poses[0, :3, 3]
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
        start_sphere.translate(start_pos)
        start_sphere.paint_uniform_color([0, 1, 0])
        geometries.append(start_sphere)
        
        # 终点（红色球）
        end_pos = poses[-1, :3, 3]
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
        end_sphere.translate(end_pos)
        end_sphere.paint_uniform_color([1, 0, 0])
        geometries.append(end_sphere)
        
        # 显示
        o3d.visualization.draw_geometries(
            geometries,
            window_name="相机轨迹可视化 (按 Q 关闭并保存)",
            width=1600,
            height=1000,
            left=50,
            top=50
        )
    
    def save_trajectory(self, poses):
        """保存轨迹"""
        # 计算内参
        fovy_rad = np.radians(self.fovy_deg)
        fy = self.height / (2 * np.tan(fovy_rad / 2))
        
        aspect = self.width / self.height
        fovx_rad = 2 * np.arctan(np.tan(fovy_rad / 2) * aspect)
        fx = self.width / (2 * np.tan(fovx_rad / 2))
        
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        # 构建输出数据
        output_data = {
            "camera_model": "OPENCV",
            "w": self.width,
            "h": self.height,
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "camera_angle_x": fovx_rad,
            "camera_angle_y": fovy_rad,
            "object_info": {
                "center": self.center.tolist(),
                "radius": float(self.radius),
                "height_offset": float(self.height_offset),
            },
            "frames": []
        }
        
        for i, pose in enumerate(poses):
            frame = {
                "file_path": f"./images/{i:04d}.png",
                "transform_matrix": pose.tolist()
            }
            output_data["frames"].append(frame)
        
        # 保存
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ 保存轨迹: {output_path}")
        print(f"   视角数: {len(poses)}")
        print(f"   分辨率: {self.width}x{self.height}")
        print(f"   中心点: [{self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f}]")
        print(f"   半径: {self.radius:.3f}m")
        print(f"   高度偏移: {self.height_offset:.3f}m")
    
    def run(self):
        """运行交互式可视化"""
        print("\n" + "=" * 70)
        print("交互式相机轨迹生成器")
        print("=" * 70)
        print("轨迹参数:")
        print(f"  半径: {self.radius:.3f}m")
        print(f"  高度偏移: {self.height_offset:.3f}m")
        print(f"  视角数: {self.num_views}")
        print(f"  旋转方向: {'顺时针' if self.clockwise else '逆时针'}")
        print(f"  起始角度: {self.start_angle_deg}°")
        print("=" * 70)
        print("\n步骤 1: 选择中心点")
        print("  - 使用 Shift+左键 点击点云选择中心点")
        print("  - 关闭窗口继续")
        print("=" * 70)
        
        # 第一步：选择中心点
        vis_pick = o3d.visualization.VisualizerWithEditing()
        vis_pick.create_window(
            window_name="步骤 1: Shift+左键选择中心点",
            width=1600,
            height=1000
        )
        self.pcd.paint_uniform_color([1, 0, 0])
        vis_pick.add_geometry(self.pcd)
        
        # 添加世界坐标系
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        vis_pick.add_geometry(world_frame)
        
        vis_pick.run()
        
        # 获取选择的点
        picked_points = vis_pick.get_picked_points()
        vis_pick.destroy_window()
        
        if not picked_points:
            print("没有选择点，退出")
            return
        
        # 获取中心点
        points = np.asarray(self.pcd.points)
        idx = picked_points[0]
        self.center = points[idx].copy()
        
        print(f"\n✅ 选择中心点: [{self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f}]")
        
        # 第二步：生成轨迹
        print("\n" + "=" * 70)
        print("步骤 2: 生成轨迹")
        print("=" * 70)
        
        poses = self.generate_trajectory()
        if poses is None:
            print("❌ 生成轨迹失败")
            return
        
        print(f"✅ 生成 {len(poses)} 个相机位姿")
        
        # 第三步：可视化
        print("\n" + "=" * 70)
        print("步骤 3: 可视化轨迹和相机位姿")
        print("=" * 70)
        
        self.visualize_trajectory(poses)
        
        # 第四步：保存
        self.save_trajectory(poses)
        
        print("\n现在可以渲染:")
        print(f"python gs2colmap/render.py \\")
        print(f"    --ply {self.pcd_path} \\")
        print(f"    --trajectory {self.output_path} \\")
        print(f"    --output gs2colmap/renders/custom \\")
        print(f"    --fovy {self.fovy_deg}")
        
        print("\n👋 完成！")


def main():
    parser = argparse.ArgumentParser(description="交互式相机轨迹生成器")
    parser.add_argument("--ply", type=str, required=True,
                       help="点云 PLY 文件")
    parser.add_argument("--output", type=str, default="gs2colmap/trajectory.json",
                       help="输出轨迹文件")
    parser.add_argument("--num-views", type=int, default=1000,
                       help="视角数量")
    parser.add_argument("--radius", type=float, default=2.0,
                       help="环绕半径（米）")
    parser.add_argument("--height", type=float, default=0.3,
                       help="相机高度偏移（米）")
    parser.add_argument("--start-angle", type=float, default=0.0,
                       help="起始角度（度），0=前方, 90=左侧, 180=后方, 270=右侧")
    parser.add_argument("--counterclockwise", action="store_true",
                       help="逆时针旋转（默认顺时针）")
    parser.add_argument("--width", type=int, default=1280,
                       help="渲染宽度")
    parser.add_argument("--img-height", type=int, default=800,
                       help="渲染高度")
    parser.add_argument("--fovy", type=float, default=65.0,
                       help="垂直 FOV（角度）")
    
    args = parser.parse_args()
    
    generator = InteractiveTrajectoryGenerator(
        pcd_path=args.ply,
        output_path=args.output,
        radius=args.radius,
        height_offset=args.height,
        num_views=args.num_views,
        clockwise=not args.counterclockwise,
        start_angle_deg=args.start_angle
    )
    
    generator.width = args.width
    generator.height = args.img_height
    generator.fovy_deg = args.fovy
    
    generator.run()


if __name__ == "__main__":
    main()