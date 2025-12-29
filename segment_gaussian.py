#!/usr/bin/env python3
"""
用2D mask分割3D Gaussian点云
分区域自适应阈值 + 连通性约束 + 只保留最大连通域
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
from gaussian_restore import GaussianAttributeRestorer

class GaussianSegmenter:
    """用2D mask分割3D Gaussian"""
    
    def __init__(self, ply_path, mode='vote'):
        print(f"\n{'='*70}")
        print(f"加载Gaussian点云")
        print(f"{'='*70}")
        print(f"文件: {ply_path}")
        print(f"分割模式: {mode}")
        
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
        self.mode = mode
        
        print(f"点数: {self.num_points:,}")
        print(f"位置范围:")
        print(f"  X: [{self.positions[:, 0].min():.3f}, {self.positions[:, 0].max():.3f}]")
        print(f"  Y: [{self.positions[:, 1].min():.3f}, {self.positions[:, 1].max():.3f}]")
        print(f"  Z: [{self.positions[:, 2].min():.3f}, {self.positions[:, 2].max():.3f}]")
        
        # 保存完整的vertex数据
        self.vertices = vertices
        
        # 投票系统
        self.vote_count = np.zeros(self.num_points, dtype=np.int32)
        self.weighted_vote = np.zeros(self.num_points, dtype=np.float32)
        self.intersection_mask = np.ones(self.num_points, dtype=bool)
        self.total_views = 0
        
        # 统计信息
        self.mask_areas = []
        
        # 构建KD-Tree（用于连通性检查）
        print("构建KD-Tree...")
        self.kdtree = cKDTree(self.positions)
    
    def project_points(self, c2w, fx, fy, cx, cy, width, height):
        """将3D点投影到2D图像"""
        Tmat = c2w.copy()
        Tmat[0:3, [1,2]] *= -1
        
        transpose = np.array([[1.0,  0.0,  0.0,  0.0],
                              [ 0.0, 1.0,  0.0,  0.0],
                              [ 0.0,  0.0,  1.0,  0.0],
                              [ 0.0,  0.0,  0.0,  1.0]], dtype=np.float32)
        
        w2c = transpose @ np.linalg.inv(Tmat)
        
        positions_homo = np.concatenate([
            self.positions,
            np.ones((self.num_points, 1))
        ], axis=1)
        
        points_cam = (w2c @ positions_homo.T).T[:, :3]
        
        valid_depth = points_cam[:, 2] > 0.01
        
        pixel_x = (points_cam[:, 0] * fx / points_cam[:, 2]) + cx
        pixel_y = (points_cam[:, 1] * fy / points_cam[:, 2]) + cy
        
        pixel_coords = np.stack([pixel_x, pixel_y], axis=1)
        
        in_image = (
            (pixel_coords[:, 0] >= 0) &
            (pixel_coords[:, 0] < width) &
            (pixel_coords[:, 1] >= 0) &
            (pixel_coords[:, 1] < height)
        )
        
        valid_mask = valid_depth & in_image
        
        return pixel_coords, valid_mask
    
    def mark_with_mask(self, mask, c2w, fx, fy, cx, cy):
        """用一个mask标记3D点"""
        H, W = mask.shape
        
        # 计算mask面积
        mask_area = mask.sum()
        total_pixels = H * W
        area_ratio = mask_area / total_pixels
        
        self.mask_areas.append(area_ratio)
        
        # 投影3D点到2D
        pixel_coords, valid_mask = self.project_points(c2w, fx, fy, cx, cy, W, H)
        
        valid_indices = np.where(valid_mask)[0]
        
        # 当前帧mask内的点
        frame_mask_points = np.zeros(self.num_points, dtype=bool)
        
        for idx in valid_indices:
            x, y = pixel_coords[idx]
            x_int = int(round(x))
            y_int = int(round(y))
            
            if 0 <= x_int < W and 0 <= y_int < H:
                if mask[y_int, x_int]:
                    frame_mask_points[idx] = True
        
        # 根据模式更新
        if self.mode == 'vote':
            self.vote_count[frame_mask_points] += 1
        elif self.mode == 'area_weighted':
            self.vote_count[frame_mask_points] += 1
            self.weighted_vote[frame_mask_points] += area_ratio
        elif self.mode == 'intersection':
            self.intersection_mask &= frame_mask_points
        
        self.total_views += 1
        return frame_mask_points.sum()
    
    def visualize_result(self, selected_indices, title="Segmentation Result"):
        """可视化最终分割结果"""
        print(f"\n{'='*70}")
        print(f"可视化: {title}")
        print(f"{'='*70}")
        print(f"被选中点数: {len(selected_indices):,}")
        print(f"按 Q 关闭窗口...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.positions)
        
        colors = np.ones((self.num_points, 3)) * 0.5
        selected_mask = np.zeros(self.num_points, dtype=bool)
        selected_mask[selected_indices] = True
        colors[selected_mask] = [1.0, 0.0, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=1280,
            height=720,
            point_show_normal=False
        )
    
    def get_segmented_points(self, 
                            core_threshold=0.7,
                            edge_threshold=0.3,
                            connectivity_radius=0.05,
                            keep_largest_only=True,
                            cluster_eps=0.1,
                            cluster_min_samples=10):
        """
        分区域自适应阈值 + 只保留最大连通域
        
        Args:
            core_threshold: 核心区域阈值（高）
            edge_threshold: 边缘区域阈值（低）
            connectivity_radius: 连通性半径（米）
            keep_largest_only: 是否只保留最大连通域
            cluster_eps: DBSCAN聚类半径
            cluster_min_samples: DBSCAN最小样本数
        """
        print(f"\n{'='*70}")
        print(f"提取分割结果 - 分区域自适应")
        print(f"{'='*70}")
        print(f"处理视角数: {self.total_views}")
        
        if len(self.mask_areas) > 0:
            avg_area = np.mean(self.mask_areas)
            print(f"平均Mask面积: {avg_area*100:.1f}%")
        
        # 计算综合分数
        vote_ratio = self.vote_count / max(self.total_views, 1)
        
        if self.mode == 'area_weighted':
            max_weighted = self.weighted_vote.max()
            if max_weighted > 0:
                weight_ratio = self.weighted_vote / max_weighted
            else:
                weight_ratio = np.zeros_like(self.weighted_vote)
            
            combined_score = vote_ratio * 0.6 + weight_ratio * 0.4
        else:
            combined_score = vote_ratio
        
        # Step 1: 选择核心区域（高阈值，高置信度）
        core_mask = combined_score >= core_threshold
        core_indices = np.where(core_mask)[0]
        
        print(f"\n核心阈值: {core_threshold * 100:.0f}%")
        print(f"核心点数: {len(core_indices):,}")
        
        if len(core_indices) == 0:
            print("❌ 警告: 没有核心点！尝试降低core_threshold")
            return np.array([], dtype=np.int64)
        
        # Step 2: 选择边缘候选点（低阈值）
        edge_candidate_mask = (combined_score >= edge_threshold) & (combined_score < core_threshold)
        edge_candidate_indices = np.where(edge_candidate_mask)[0]
        
        print(f"\n边缘阈值: {edge_threshold * 100:.0f}%")
        print(f"边缘候选点数: {len(edge_candidate_indices):,}")
        
        # Step 3: 边缘点必须邻近核心点（连通性约束）
        if len(edge_candidate_indices) > 0:
            print(f"\n检查边缘点连通性 (半径={connectivity_radius*100:.1f}cm)...")
            
            # 查询每个边缘候选点到核心点的最近距离
            core_positions = self.positions[core_indices]
            core_tree = cKDTree(core_positions)
            
            distances, _ = core_tree.query(self.positions[edge_candidate_indices])
            
            # 保留距离核心点足够近的边缘点
            valid_edge_mask = distances < connectivity_radius
            valid_edge_indices = edge_candidate_indices[valid_edge_mask]
            
            print(f"有效边缘点数: {len(valid_edge_indices):,}")
        else:
            valid_edge_indices = np.array([], dtype=np.int64)
        
        # Step 4: 合并核心点和有效边缘点
        selected_indices = np.concatenate([core_indices, valid_edge_indices])
        selected_indices = np.unique(selected_indices)
        
        print(f"\n初步选中点数: {len(selected_indices):,} / {self.num_points:,} "
              f"({len(selected_indices) / self.num_points * 100:.2f}%)")
        
        # Step 5: 只保留最大连通域（过滤掉其他独立的聚类）
        if keep_largest_only and len(selected_indices) > 0:
            print(f"\n{'='*70}")
            print(f"清理：只保留最大连通域")
            print(f"{'='*70}")
            
            selected_positions = self.positions[selected_indices]
            
            # 使用DBSCAN聚类
            print(f"运行DBSCAN聚类 (eps={cluster_eps}m, min_samples={cluster_min_samples})...")
            clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(selected_positions)
            labels = clustering.labels_
            
            # 统计每个聚类的大小
            unique_labels = np.unique(labels[labels >= 0])
            
            if len(unique_labels) > 0:
                # 统计每个聚类的大小
                label_counts = []
                for label in unique_labels:
                    count = (labels == label).sum()
                    label_counts.append((label, count))
                
                # 按大小排序
                label_counts.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\n发现 {len(unique_labels)} 个连通域:")
                for i, (label, count) in enumerate(label_counts[:5]):  # 显示前5个
                    print(f"  域 {i+1} (label={label}): {count:,} 点")
                
                # 只保留最大的那个
                largest_label = label_counts[0][0]
                largest_mask = (labels == largest_label)
                selected_indices = selected_indices[largest_mask]
                
                print(f"\n✅ 保留最大连通域: {len(selected_indices):,} 点")
                
                # 如果有多个较大的聚类，警告用户
                if len(label_counts) > 1:
                    second_largest_count = label_counts[1][1]
                    if second_largest_count > len(selected_indices) * 0.1:  # 如果第二大的超过10%
                        print(f"\n⚠️  注意: 发现第二大连通域 ({second_largest_count:,} 点)")
                        print(f"   如果结果不对，可能需要调整参数或重新标注mask")
            else:
                print(f"⚠️ 没有找到有效聚类（全是噪声点）")
        
        print(f"\n{'='*70}")
        print(f"最终选中点数: {len(selected_indices):,} / {self.num_points:,} "
              f"({len(selected_indices) / self.num_points * 100:.2f}%)")
        print(f"{'='*70}")
        
        # 投票分布
        print(f"\n投票率分布:")
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i+1]
            count = ((vote_ratio >= low) & (vote_ratio < high)).sum()
            print(f"  [{low*100:>3.0f}%-{high*100:>3.0f}%): {count:,} points")
        
        return selected_indices
    
    def save_segmented_ply(self, selected_indices, output_path):
        """保存分割后的PLY文件"""
        selected_vertices = self.vertices[selected_indices]
        
        new_ply = PlyData([
            PlyElement.describe(selected_vertices, 'vertex')
        ], text=False)
        
        new_ply.write(output_path)
        
        print(f"\n✓ 已保存PLY: {output_path}")
        print(f"  点数: {len(selected_indices):,}")


def main():
    parser = ArgumentParser(description="用2D mask分割3D Gaussian点云")
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--transforms", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    # 分割模式
    parser.add_argument("--mode", type=str, default="area_weighted", 
                       choices=["vote", "area_weighted", "intersection"],
                       help="分割模式: vote | area_weighted(推荐) | intersection")
    
    # 分区域阈值参数
    parser.add_argument("--core-threshold", type=float, default=0.5,
                       help="核心区域阈值(高置信度), 默认0.5")
    parser.add_argument("--edge-threshold", type=float, default=0.2,
                       help="边缘区域阈值(低置信度), 默认0.2")
    parser.add_argument("--connectivity-radius", type=float, default=0.02,
                       help="连通性半径(米), 边缘点必须在此距离内, 默认2cm")
    
    # 连通域过滤参数
    parser.add_argument("--keep-largest-only", action="store_true", default=True,
                       help="只保留最大连通域（默认开启）")
    parser.add_argument("--no-keep-largest", dest="keep_largest_only", action="store_false",
                       help="不过滤连通域，保留所有点")
    parser.add_argument("--cluster-eps", type=float, default=0.03,
                       help="DBSCAN聚类半径(米), 默认3cm")
    parser.add_argument("--cluster-min-samples", type=int, default=10,
                       help="DBSCAN最小样本数, 默认10")
    
    # 双向输出
    parser.add_argument("--save-inverse", action="store_true",
                       help="同时保存mask外的点")
    parser.add_argument("--restore-attributes", action="store_true",
                       help="恢复完整的Gaussian Splatting属性")
    parser.add_argument("--restore-max-distance", type=float, default=0.001,
                       help="属性恢复最大匹配距离(米), 默认1mm")
    
    # 可视化
    parser.add_argument("--visualize", action="store_true")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"3D Gaussian点云分割")
    print(f"{'='*70}")
    print(f"输入:")
    print(f"  PLY:        {args.ply}")
    print(f"  Masks:      {args.masks}")
    print(f"  Transforms: {args.transforms}")
    print(f"输出:")
    print(f"  PLY:        {args.output}")
    print(f"模式:         {args.mode}")
    print(f"核心阈值:     {args.core_threshold}")
    print(f"边缘阈值:     {args.edge_threshold}")
    print(f"连通半径:     {args.connectivity_radius}m")
    print(f"最大连通域:   {'是' if args.keep_largest_only else '否'}")
    print(f"保存背景:     {'是' if args.save_inverse else '否'}")
    
    # 加载Gaussian
    segmenter = GaussianSegmenter(args.ply, mode=args.mode)
    
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
    
    print(f"内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"分辨率: {width} x {height}")
    
    # 处理Masks
    print(f"\n{'='*70}")
    print(f"处理Masks")
    print(f"{'='*70}")
    
    masks_dir = Path(args.masks)
    frames = transforms['frames']
    
    mask_files = sorted(masks_dir.glob("*.png"))
    print(f"找到 {len(mask_files)} 个mask文件")
    
    if len(mask_files) == 0:
        print("❌ 错误: 没有找到mask文件！")
        return
    
    processed = 0
    
    for mask_file in tqdm(mask_files, desc="处理进度"):
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
        
        # 标记
        segmenter.mark_with_mask(mask, c2w, fx, fy, cx, cy)
        processed += 1
    
    print(f"\n实际处理帧数: {processed}")
    
    if processed == 0:
        print("❌ 错误: 没有处理任何帧！")
        return
    
    # 打印投票统计
    print(f"\n🔍 投票率统计（详细）:")
    vote_ratio = segmenter.vote_count / max(segmenter.total_views, 1)

    if segmenter.mode == 'area_weighted':
        max_weighted = segmenter.weighted_vote.max()
        if max_weighted > 0:
            weight_ratio = segmenter.weighted_vote / max_weighted
        else:
            weight_ratio = np.zeros_like(segmenter.weighted_vote)
        combined_score = vote_ratio * 0.6 + weight_ratio * 0.4
    else:
        combined_score = vote_ratio

    # 统计不同分数区间的点数
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"综合分数分布:")
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        count = ((combined_score >= low) & (combined_score < high)).sum()
        print(f"  [{low:.1f}-{high:.1f}): {count:,} points")

    # 获取分割结果
    selected_indices = segmenter.get_segmented_points(
        core_threshold=args.core_threshold,
        edge_threshold=args.edge_threshold,
        connectivity_radius=args.connectivity_radius,
        keep_largest_only=args.keep_largest_only,
        cluster_eps=args.cluster_eps,
        cluster_min_samples=args.cluster_min_samples
    )
    
    if len(selected_indices) == 0:
        print("❌ 警告: 没有选中任何点！")
        return
    
    # 可视化
    if args.visualize:
        segmenter.visualize_result(selected_indices, "Mask内的点")
    
    # 保存mask内的点
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segmenter.save_segmented_ply(selected_indices, output_path)
    
    # 保存mask外的点
    if args.save_inverse:
        all_indices = np.arange(segmenter.num_points)
        inverse_indices = np.setdiff1d(all_indices, selected_indices)
        
        inverse_output = output_path.parent / f"{output_path.stem}_background{output_path.suffix}"
        
        if args.visualize:
            segmenter.visualize_result(inverse_indices, "Mask外的点（背景）")
        
        segmenter.save_segmented_ply(inverse_indices, inverse_output)
    
    # ========== 恢复Gaussian属性 ==========
    if args.restore_attributes:
        print(f"\n{'='*70}")
        print(f"恢复 Gaussian Splatting 属性")
        print(f"{'='*70}")
        
        # 创建属性恢复器
        restorer = GaussianAttributeRestorer(args.ply, verbose=True)
        
        # 需要恢复的文件列表
        files_to_restore = [output_path]
        if args.save_inverse:
            files_to_restore.append(inverse_output)
        
        # 批量恢复
        restored_paths = restorer.batch_restore(
            files_to_restore,
            suffix="_gs",  # 恢复后的文件加 _gs 后缀
            max_distance=args.restore_max_distance,
            overwrite=True
        )
        
        print(f"\n✓ 属性恢复完成！")
        print(f"恢复后的文件:")
        for path in restored_paths:
            print(f"  - {path}")
    
    print(f"\n{'='*70}")
    print(f"完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()