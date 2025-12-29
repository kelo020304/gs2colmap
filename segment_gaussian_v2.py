#!/usr/bin/env python3
"""
用2D mask分割3D Gaussian点云
投票阈值（严格过滤）+ 连通性清理 + 保留最大连通域
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import cv2
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


class GaussianSegmenter:
    """用2D mask分割3D Gaussian"""
    
    def __init__(self, ply_path):
        print(f"\n{'='*70}")
        print(f"加载Gaussian点云")
        print(f"{'='*70}")
        print(f"文件: {ply_path}")
        
        # 加载PLY
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        
        # 提取位置
        self.positions = np.stack([
            vertices['x'],
            vertices['y'],
            vertices['z']
        ], axis=1).astype(np.float32)
        
        self.num_points = len(self.positions)
        
        print(f"点数: {self.num_points:,}")
        print(f"位置范围:")
        print(f"  X: [{self.positions[:, 0].min():.3f}, {self.positions[:, 0].max():.3f}]")
        print(f"  Y: [{self.positions[:, 1].min():.3f}, {self.positions[:, 1].max():.3f}]")
        print(f"  Z: [{self.positions[:, 2].min():.3f}, {self.positions[:, 2].max():.3f}]")
        
        # 保存完整的vertex数据
        self.vertices = vertices
        
        # 投票系统
        self.vote_count = np.zeros(self.num_points, dtype=np.int32)
        self.total_views = 0
    
    def project_points(self, c2w, fx, fy, cx, cy, width, height):
        """将3D点投影到2D图像（模拟GSRenderer）"""
        # GSRenderer的坐标转换
        Tmat = c2w.copy()
        Tmat[0:3, [1,2]] *= -1
        
        transpose = np.array([[1.0,  0.0,  0.0,  0.0],
                              [ 0.0, 1.0,  0.0,  0.0],
                              [ 0.0,  0.0,  1.0,  0.0],
                              [ 0.0,  0.0,  0.0,  1.0]], dtype=np.float32)
        
        w2c = transpose @ np.linalg.inv(Tmat)
        
        # 转换到相机坐标系
        positions_homo = np.concatenate([
            self.positions,
            np.ones((self.num_points, 1))
        ], axis=1)
        
        points_cam = (w2c @ positions_homo.T).T[:, :3]
        
        # 过滤掉相机后面的点
        valid_depth = points_cam[:, 2] > 0.01
        
        # 投影到像素
        pixel_x = (points_cam[:, 0] * fx / points_cam[:, 2]) + cx
        pixel_y = (points_cam[:, 1] * fy / points_cam[:, 2]) + cy
        
        pixel_coords = np.stack([pixel_x, pixel_y], axis=1)
        
        # 检查是否在图像范围内
        in_image = (
            (pixel_coords[:, 0] >= 0) &
            (pixel_coords[:, 0] < width) &
            (pixel_coords[:, 1] >= 0) &
            (pixel_coords[:, 1] < height)
        )
        
        valid_mask = valid_depth & in_image
        
        return pixel_coords, valid_mask
    
    def mark_with_mask(self, mask, c2w, fx, fy, cx, cy):
        """用一个mask对3D点投票"""
        H, W = mask.shape
        
        # 投影3D点到2D
        pixel_coords, valid_mask = self.project_points(c2w, fx, fy, cx, cy, W, H)
        
        valid_indices = np.where(valid_mask)[0]
        
        # 投票
        for idx in valid_indices:
            x, y = pixel_coords[idx]
            x_int = int(round(x))
            y_int = int(round(y))
            
            if 0 <= x_int < W and 0 <= y_int < H:
                if mask[y_int, x_int]:
                    self.vote_count[idx] += 1
        
        self.total_views += 1
    
    def get_segmented_points(self, 
                            vote_threshold=0.5,
                            connectivity_radius=0.02,
                            keep_largest_only=True,
                            cluster_eps=0.02,
                            cluster_min_samples=10):
        """
        投票阈值 + 连通性清理 + 保留最大连通域
        
        Args:
            vote_threshold: 投票阈值（必须在至少这么多比例的视角中可见）
            connectivity_radius: 孤立点判定半径（米）
            keep_largest_only: 是否只保留最大连通域
            cluster_eps: DBSCAN聚类半径（米）
            cluster_min_samples: DBSCAN最小样本数
        """
        print(f"\n{'='*70}")
        print(f"提取分割结果")
        print(f"{'='*70}")
        print(f"处理视角数: {self.total_views}")
        
        # 计算投票率
        vote_ratio = self.vote_count / max(self.total_views, 1)
        
        # Step 1: 投票阈值严格过滤（保证都在mask内）
        candidate_mask = vote_ratio >= vote_threshold
        candidate_indices = np.where(candidate_mask)[0]
        
        print(f"\nStep 1: 投票阈值过滤")
        print(f"  投票阈值: {vote_threshold * 100:.0f}%")
        print(f"  候选点数: {len(candidate_indices):,} / {self.num_points:,} "
              f"({len(candidate_indices) / self.num_points * 100:.2f}%)")
        
        if len(candidate_indices) == 0:
            print("❌ 没有满足阈值的点！尝试降低 --vote-threshold")
            return np.array([], dtype=np.int64)
        
        # Step 2: 移除孤立点
        print(f"\nStep 2: 连通性清理")
        print(f"  移除孤立点（半径={connectivity_radius*100:.1f}cm内无邻居）...")
        
        candidate_positions = self.positions[candidate_indices]
        candidate_tree = cKDTree(candidate_positions)
        
        # 统计每个点的近邻数量
        neighbor_counts = candidate_tree.query_ball_point(
            candidate_positions, 
            r=connectivity_radius,
            return_length=True
        )
        
        # 保留至少有2个点（自己+1个邻居）的位置
        min_neighbors = 2
        connected_mask = neighbor_counts >= min_neighbors
        connected_indices = candidate_indices[connected_mask]
        
        isolated_count = len(candidate_indices) - len(connected_indices)
        print(f"  移除孤立点: {isolated_count:,}")
        print(f"  剩余点数: {len(connected_indices):,}")
        
        if len(connected_indices) == 0:
            print("❌ 连通性过滤后无点剩余！")
            return np.array([], dtype=np.int64)
        
        # Step 3: 保留最大连通域
        selected_indices = connected_indices
        
        if keep_largest_only and len(connected_indices) > cluster_min_samples:
            print(f"\nStep 3: 保留最大连通域")
            
            connected_positions = self.positions[connected_indices]
            
            # DBSCAN聚类
            print(f"  DBSCAN (eps={cluster_eps*100:.1f}cm, min_samples={cluster_min_samples})...")
            clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(connected_positions)
            labels = clustering.labels_
            
            # 统计聚类
            unique_labels = np.unique(labels[labels >= 0])
            noise_count = (labels == -1).sum()
            
            if len(unique_labels) > 0:
                label_counts = [(label, (labels == label).sum()) for label in unique_labels]
                label_counts.sort(key=lambda x: x[1], reverse=True)
                
                print(f"  发现 {len(unique_labels)} 个连通域 + {noise_count:,} 噪声点")
                for i, (label, count) in enumerate(label_counts[:3]):
                    print(f"    域{i+1}: {count:,} 点")
                
                # 只保留最大的
                largest_label = label_counts[0][0]
                largest_mask = (labels == largest_label)
                selected_indices = connected_indices[largest_mask]
                
                removed_count = len(connected_indices) - len(selected_indices)
                print(f"  移除其他域: {removed_count:,} 点")
                print(f"  ✅ 最大连通域: {len(selected_indices):,} 点")
                
                # 警告：如果有多个大聚类
                if len(label_counts) > 1:
                    second_count = label_counts[1][1]
                    if second_count > len(selected_indices) * 0.2:
                        print(f"  ⚠️ 第二大连通域较大 ({second_count:,} 点)")
                        print(f"     如果结果不对，可能mask标注有问题")
            else:
                print(f"  ⚠️ 全是噪声点，保留所有连通点")
        
        print(f"\n{'='*70}")
        print(f"最终选中: {len(selected_indices):,} / {self.num_points:,} "
              f"({len(selected_indices) / self.num_points * 100:.2f}%)")
        print(f"{'='*70}")
        
        # 投票分布统计
        print(f"\n投票率分布:")
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i+1]
            count = ((vote_ratio >= low) & (vote_ratio < high)).sum()
            print(f"  [{low*100:>3.0f}%-{high*100:>3.0f}%): {count:,} points")
        
        return selected_indices
    
    def visualize_result(self, selected_indices, title="Segmentation Result"):
        """可视化分割结果"""
        print(f"\n{'='*70}")
        print(f"可视化: {title}")
        print(f"{'='*70}")
        print(f"被选中点数: {len(selected_indices):,}")
        print(f"按 Q 关闭...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.positions)
        
        colors = np.ones((self.num_points, 3)) * 0.5  # 灰色
        selected_mask = np.zeros(self.num_points, dtype=bool)
        selected_mask[selected_indices] = True
        colors[selected_mask] = [1.0, 0.0, 0.0]  # 红色
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=1280,
            height=720
        )
    
    def save_segmented_ply(self, selected_indices, output_path):
        """保存分割后的PLY文件"""
        selected_vertices = self.vertices[selected_indices]
        
        new_ply = PlyData([
            PlyElement.describe(selected_vertices, 'vertex')
        ], text=False)
        
        new_ply.write(output_path)
        
        print(f"\n✓ 已保存: {output_path}")
        print(f"  点数: {len(selected_indices):,}")


def main():
    parser = ArgumentParser(description="用2D mask分割3D Gaussian点云")
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--transforms", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    # 核心参数
    parser.add_argument("--vote-threshold", type=float, default=0.5,
                       help="投票阈值(0-1)，必须在这么多比例的视角可见，默认0.5")
    
    # 连通性参数
    parser.add_argument("--connectivity-radius", type=float, default=0.02,
                       help="孤立点判定半径(米)，默认2cm")
    parser.add_argument("--cluster-eps", type=float, default=0.02,
                       help="DBSCAN聚类半径(米)，默认2cm")
    parser.add_argument("--cluster-min-samples", type=int, default=10,
                       help="DBSCAN最小样本数，默认10")
    parser.add_argument("--keep-largest-only", dest="keep_largest_only", 
                       action="store_true", default=False,
                       help="只保留最大连通域（默认关闭）")
    parser.add_argument("--no-keep-largest", dest="keep_largest_only", 
                       action="store_false",
                       help="不过滤连通域，保留所有点（默认）")
    
    # 输出选项
    parser.add_argument("--save-background", action="store_true",
                       help="同时保存背景（mask外的点）")
    parser.add_argument("--visualize", action="store_true",
                       help="可视化结果")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"3D Gaussian 点云分割")
    print(f"{'='*70}")
    print(f"输入:")
    print(f"  PLY:        {args.ply}")
    print(f"  Masks:      {args.masks}")
    print(f"  Transforms: {args.transforms}")
    print(f"输出:")
    print(f"  PLY:        {args.output}")
    print(f"参数:")
    print(f"  投票阈值:   {args.vote_threshold}")
    print(f"  连通半径:   {args.connectivity_radius}m")
    print(f"  聚类参数:   eps={args.cluster_eps}m, min_samples={args.cluster_min_samples}")
    print(f"  最大连通域: {'是' if args.keep_largest_only else '否'}")
    
    # 加载Gaussian
    segmenter = GaussianSegmenter(args.ply)
    
    # 加载transforms.json
    print(f"\n{'='*70}")
    print(f"加载相机参数")
    print(f"{'='*70}")
    
    with open(args.transforms, 'r') as f:
        transforms = json.load(f)
    
    fx = transforms['fl_x']
    fy = transforms['fl_y']
    cx = transforms['cx']
    cy = transforms['cy']
    width = transforms['w']
    height = transforms['h']
    
    print(f"内参: fx={fx:.2f}, fy={fy:.2f}")
    print(f"分辨率: {width} x {height}")
    
    # 处理Masks
    print(f"\n{'='*70}")
    print(f"处理Masks")
    print(f"{'='*70}")
    
    masks_dir = Path(args.masks)
    frames = transforms['frames']
    
    mask_files = sorted(masks_dir.glob("*.png"))
    print(f"找到 {len(mask_files)} 个mask文件")
    print(f"Transforms总帧数: {len(frames)}")
    
    if len(mask_files) == 0:
        print("❌ 没有找到mask文件！")
        return
    
    processed = 0
    
    for mask_file in tqdm(mask_files, desc="投票中"):
        mask_name = mask_file.stem
        
        try:
            mask_idx = int(mask_name)
        except ValueError:
            continue
        
        if mask_idx >= len(frames):
            continue
        
        # 加载mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        mask = mask > 127
        
        # 获取pose
        frame = frames[mask_idx]
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        
        # 投票
        segmenter.mark_with_mask(mask, c2w, fx, fy, cx, cy)
        processed += 1
    
    print(f"\n实际处理: {processed} 帧")
    
    if processed == 0:
        print("❌ 没有处理任何帧！")
        return
    
    # 获取分割结果
    selected_indices = segmenter.get_segmented_points(
        vote_threshold=args.vote_threshold,
        connectivity_radius=args.connectivity_radius,
        keep_largest_only=args.keep_largest_only,
        cluster_eps=args.cluster_eps,
        cluster_min_samples=args.cluster_min_samples
    )
    
    if len(selected_indices) == 0:
        print("❌ 没有选中任何点！")
        print("   建议: 降低 --vote-threshold")
        return
    
    # 可视化
    if args.visualize:
        segmenter.visualize_result(selected_indices, "Mask内的点")
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segmenter.save_segmented_ply(selected_indices, output_path)
    
    # 保存背景
    if args.save_background:
        all_indices = np.arange(segmenter.num_points)
        background_indices = np.setdiff1d(all_indices, selected_indices)
        
        bg_path = output_path.parent / f"{output_path.stem}_background{output_path.suffix}"
        
        if args.visualize:
            segmenter.visualize_result(background_indices, "背景点")
        
        segmenter.save_segmented_ply(background_indices, bg_path)
    
    print(f"\n{'='*70}")
    print(f"完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()