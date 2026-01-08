#!/usr/bin/env python3
"""
用2D mask分割3D Gaussian点云
分区域自适应阈值 + 连通性约束 + 只保留最大连通域
"""

import numpy as np
import json
import re
from pathlib import Path
from tqdm import tqdm
import cv2
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement
import open3d as o3d
from scipy.spatial import cKDTree
from gaussian_restore import GaussianAttributeRestorer
from segment_mesh_by_ply import segment_mesh

class GaussianSegmenter:
    """用2D mask分割3D Gaussian"""
    
    def __init__(self, ply_path, mode='vote'):
        # print(f"\n{'='*70}")
        # print(f"加载Gaussian点云")
        # print(f"{'='*70}")
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
        
        # print(f"点数: {self.num_points:,}")
        # print(f"位置范围:")
        # print(f"  X: [{self.positions[:, 0].min():.3f}, {self.positions[:, 0].max():.3f}]")
        # print(f"  Y: [{self.positions[:, 1].min():.3f}, {self.positions[:, 1].max():.3f}]")
        # print(f"  Z: [{self.positions[:, 2].min():.3f}, {self.positions[:, 2].max():.3f}]")
        
        # 保存完整的vertex数据
        self.vertices = vertices
        
        # 投票系统
        self.vote_count = np.zeros(self.num_points, dtype=np.int32)
        self.weighted_vote = np.zeros(self.num_points, dtype=np.float32)
        self.intersection_mask = np.ones(self.num_points, dtype=bool)
        self.total_views = 0
        
        # 统计信息
        self.mask_areas = []
        
    
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
    
    def get_segmented_points(self, core_threshold=0.3):
        """简单阈值分割"""
        # print(f"\n{'='*70}")
        # print(f"提取分割结果")
        # print(f"{'='*70}")
        # print(f"处理视角数: {self.total_views}")
        
        if self.mode == 'intersection':
            selected_indices = np.where(self.intersection_mask)[0]
        else:
            combined_score = _compute_combined_score(self)
            selected_indices = np.where(combined_score >= core_threshold)[0]
        
        print(f"\n阈值: {core_threshold * 100:.0f}%")
        print(f"选中点数: {len(selected_indices):,} / {self.num_points:,} "
              f"({len(selected_indices) / self.num_points * 100:.2f}%)")
        
        return selected_indices
    
    def save_segmented_ply(self, selected_indices, output_path):
        """保存分割后的PLY文件"""
        selected_vertices = self.vertices[selected_indices]
        
        new_ply = PlyData([
            PlyElement.describe(selected_vertices, 'vertex')
        ], text=False)
        
        new_ply.write(output_path)
        
        print(f"\n 已保存PLY: {output_path}")
        # print(f"  点数: {len(selected_indices):,}")


def _mask_dir_sort_key(path):
    name = path.name
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return (0, int(digits))
    return (1, name)


def _resolve_mask_dirs(masks_path):
    """解析mask目录，支持 mask0/mask1 子目录"""
    masks_path = Path(masks_path)
    candidate_dirs = [
        p for p in masks_path.glob("mask*")
        if p.is_dir() and p.name != "masks" and any(p.glob("*.png"))
    ]
    if not candidate_dirs and masks_path.name == "masks":
        parent = masks_path.parent
        candidate_dirs = [
            p for p in parent.glob("mask*")
            if p.is_dir() and p.name != "masks" and any(p.glob("*.png"))
        ]
        if candidate_dirs:
            masks_path = parent
    candidate_dirs = sorted(candidate_dirs, key=_mask_dir_sort_key)
    if candidate_dirs:
        return candidate_dirs
    return [masks_path]




def _compute_combined_score(segmenter):
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
    return combined_score


def _axis_quat_from_dir(axis):
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v = np.cross(z, axis)
    c = np.dot(z, axis)
    if np.linalg.norm(v) < 1e-8:
        return [1.0, 0.0, 0.0, 0.0] if c > 0 else [0.0, 1.0, 0.0, 0.0]
    s = np.sqrt((1.0 + c) * 2.0)
    v = v / np.linalg.norm(v)
    w = s * 0.5
    x, y, zq = v * (1.0 / s)
    return [float(w), float(x), float(y), float(zq)]


def _load_ply_points(path):
    ply = PlyData.read(path)
    verts = ply["vertex"].data
    return np.stack([verts["x"], verts["y"], verts["z"]], axis=-1).astype(np.float32)


def _compute_axis_from_aabb(selected_pts, background_pts):
    if len(selected_pts) == 0 or len(background_pts) == 0:
        return None
    s_min = selected_pts.min(axis=0)
    s_max = selected_pts.max(axis=0)
    b_min = background_pts.min(axis=0)
    b_max = background_pts.max(axis=0)

    overlap_min = np.maximum(s_min, b_min)
    overlap_max = np.minimum(s_max, b_max)
    overlap = overlap_max - overlap_min
    if np.any(overlap <= 0):
        return None

    axis_idx = int(np.argmax(overlap))
    other_axes = [ax for ax in [0, 1, 2] if ax != axis_idx]

    pos = np.zeros(3, dtype=np.float32)
    pos[axis_idx] = 0.5 * (overlap_min[axis_idx] + overlap_max[axis_idx])
    pos[other_axes[0]] = 0.5 * (overlap_min[other_axes[0]] + overlap_max[other_axes[0]])
    pos[other_axes[1]] = 0.5 * (overlap_min[other_axes[1]] + overlap_max[other_axes[1]])

    axis = np.zeros(3, dtype=np.float32)
    axis[axis_idx] = 1.0

    p0 = pos.copy()
    p1 = pos.copy()
    p0[axis_idx] = overlap_min[axis_idx]
    p1[axis_idx] = overlap_max[axis_idx]

    quat = _axis_quat_from_dir(axis)
    return {
        "pos": pos.tolist(),
        "axis": axis.tolist(),
        "quat_wxyz": quat,
        "endpoints": [p0.tolist(), p1.tolist()],
        "contact_points": 0,
    }


def _compute_axis_from_contact(selected_pts, background_pts, max_dist=0.005, min_points=50):
    if len(selected_pts) == 0 or len(background_pts) == 0:
        return None
    tree = cKDTree(background_pts)
    dists, nn = tree.query(selected_pts, k=1, distance_upper_bound=max_dist)
    finite = np.isfinite(dists)
    if finite.sum() < min_points:
        return None
    finite_dists = dists[finite]
    thresh = np.percentile(finite_dists, 10) * 1.5
    thresh = min(thresh, max_dist)
    keep = finite & (dists <= thresh)
    if keep.sum() < min_points:
        keep = finite
    contact = selected_pts[keep]
    nearest_bg = background_pts[nn[keep]]
    midpoints = 0.5 * (contact + nearest_bg)
    center = midpoints.mean(axis=0)
    centered = midpoints - center
    cov = centered.T @ centered / max(len(contact) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    if axis[2] < 0:
        axis = -axis
    proj = centered @ axis
    lo = np.percentile(proj, 5)
    hi = np.percentile(proj, 95)
    p0 = center + axis * lo
    p1 = center + axis * hi
    quat = _axis_quat_from_dir(axis)
    return {
        "pos": center.tolist(),
        "axis": axis.tolist(),
        "quat_wxyz": quat,
        "endpoints": [p0.tolist(), p1.tolist()],
        "contact_points": int(len(contact)),
    }




def _process_mask_dir(segmenter, masks_dir, frames, fx, fy, cx, cy, width, height, args):
    """处理单个mask目录，返回分数"""
    masks_dir = Path(masks_dir)
    mask_files = sorted(masks_dir.glob("*.png"))
    # print(f"找到 {len(mask_files)} 个mask文件")
    
    if len(mask_files) == 0:
        print("❌ 错误: 没有找到mask文件！")
        return None, 0
    
    processed = 0
    
    for mask_file in tqdm(mask_files, desc="processing"):
        mask_name = mask_file.stem
        
        try:
            mask_idx = int(mask_name)
        except ValueError:
            continue
        
        if mask_idx >= len(frames):
            continue
        
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        mask = mask > 127
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        
        frame = frames[mask_idx]
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        segmenter.mark_with_mask(mask, c2w, fx, fy, cx, cy)
        processed += 1

    
    # print(f"\n实际处理帧数: {processed}")
    
    if processed == 0:
        print("❌ 错误: 没有处理任何帧！")
        return None, 0
    
    # print(f"\n🔍 投票率统计（详细）:")
    combined_score = _compute_combined_score(segmenter)

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # print(f"综合分数分布:")
    # for i in range(len(bins) - 1):
    #     low, high = bins[i], bins[i+1]
    #     count = ((combined_score >= low) & (combined_score < high)).sum()
    #     print(f"  [{low:.1f}-{high:.1f}): {count:,} points")
    
    return combined_score, processed


def main():
    parser = ArgumentParser(description="用2D mask分割3D Gaussian点云")
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--transforms", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="", help="原始prompt(可选)")
    
    # 分割模式
    parser.add_argument("--mode", type=str, default="area_weighted", 
                       choices=["vote", "area_weighted", "intersection"],
                       help="分割模式: vote | area_weighted(推荐) | intersection")
    
    # 分割参数
    parser.add_argument("--core-threshold", type=float, default=0.3,
                       help="阈值(高置信度), 默认0.3")
    
    # 双向输出
    parser.add_argument("--save-inverse", action="store_true",
                       help="同时保存mask外的点")
    parser.add_argument("--restore-attributes", action="store_true",
                       help="恢复完整的Gaussian Splatting属性")
    parser.add_argument("--restore-max-distance", type=float, default=0.001,
                       help="属性恢复最大匹配距离(米), 默认1mm")
    parser.add_argument("--segment-mesh", action="store_true", default=True,
                       help="分割OBJ网格（默认开启）")
    parser.add_argument("--no-segment-mesh", dest="segment_mesh", action="store_false",
                       help="不分割OBJ网格")
    
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
    print(f"保存背景:     {'是' if args.save_inverse else '否(多mask会自动保存)'}")
    
    # 加载transforms.json
    # print(f"\n{'='*70}")
    # print(f"加载相机参数")
    # print(f"{'='*70}")
    
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
    # print(f"\n{'='*70}")
    # print(f"处理Masks")
    # print(f"{'='*70}")
    
    frames = transforms['frames']
    mask_dirs = _resolve_mask_dirs(args.masks)
    multiple_masks = len(mask_dirs) > 1
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.stem
    output_suffix = output_path.suffix if output_path.suffix else ".ply"
    
    union_selected = None
    union_selected_points = None
    files_to_restore = []
    segmenter_for_save = None
    
    for mask_dir in mask_dirs:
        label = mask_dir.name if multiple_masks else "masks"
        # print(f"\n{'-'*60}")
        # print(f"处理Mask目录: {mask_dir}")
        # print(f"{'-'*60}")
        
        segmenter = GaussianSegmenter(args.ply, mode=args.mode)
        segmenter_for_save = segmenter
        
        combined_score, processed = _process_mask_dir(
            segmenter, mask_dir, frames, fx, fy, cx, cy, width, height, args
        )
        if processed == 0 or combined_score is None:
            continue
        
        selected_indices = segmenter.get_segmented_points(
            core_threshold=args.core_threshold
        )
        
        if len(selected_indices) == 0:
            print("❌ 警告: 没有选中任何点！")
            continue
        
        if args.visualize:
            segmenter.visualize_result(selected_indices, f"{label} 内的点")
        
        if multiple_masks:
            output_mask_path = output_path.parent / f"{output_stem}_{label}{output_suffix}"
        else:
            output_mask_path = output_path
        
        segmenter.save_segmented_ply(selected_indices, output_mask_path)
        
        if union_selected_points is None:
            union_selected_points = np.zeros(segmenter.num_points, dtype=bool)
        union_selected_points[selected_indices] = True
        files_to_restore.append(output_mask_path)
        
        if args.segment_mesh:
            mesh_path = Path(args.ply).with_suffix(".obj")
            if mesh_path.is_file():
                mesh_out = output_mask_path.with_suffix(".obj")
                segment_mesh(
                    str(mesh_path),
                    str(output_mask_path),
                    str(mesh_out),
                    dist=0.006,
                    min_keep=1,
                    smooth_iter=10,
                    fill_holes=200,
                )
        
        if union_selected is None:
            union_selected = np.zeros(segmenter.num_points, dtype=bool)
        union_selected[selected_indices] = True

    
    if union_selected is None or segmenter_for_save is None:
        print("❌ 错误: 没有成功处理任何mask目录！")
        return
    
    # 保存剩余主体（背景）
    save_background = args.save_inverse or multiple_masks
    background_indices = None
    if save_background:
        all_indices = np.arange(segmenter_for_save.num_points)
        inverse_indices = np.setdiff1d(all_indices, np.where(union_selected)[0])
        background_indices = inverse_indices
        inverse_output = output_path.parent / f"{output_stem}_background{output_suffix}"
        
        if args.visualize:
            segmenter_for_save.visualize_result(inverse_indices, "Mask外的点（背景）")
        
        segmenter_for_save.save_segmented_ply(inverse_indices, inverse_output)
        files_to_restore.append(inverse_output)
        
        if args.segment_mesh:
            mesh_path = Path(args.ply).with_suffix(".obj")
            if mesh_path.is_file():
                mesh_out = inverse_output.with_suffix(".obj")
                segment_mesh(
                    str(mesh_path),
                    str(inverse_output),
                    str(mesh_out),
                    dist=0.006,
                    min_keep=1,
                    smooth_iter=10,
                    fill_holes=200,
                )
    
    # ========== 轴参数输出(仅sphere prompt) ==========
    if args.prompt:
        has_sphere = re.search(r"\bspheres?\b", args.prompt, re.IGNORECASE) is not None
        has_drawer = re.search(r"\bdrawers?\b", args.prompt, re.IGNORECASE) is not None
        if (has_sphere or not has_drawer) and union_selected_points is not None and background_indices is not None:
            pts = segmenter_for_save.positions
            selected_pts = pts[union_selected_points]
            clean_bg_path = output_path.parent / f"{output_stem}_background_clean{output_suffix}"
            background_pts = pts[background_indices]
            if clean_bg_path.is_file():
                background_pts = _load_ply_points(clean_bg_path)
            if has_sphere:
                axis_info = _compute_axis_from_contact(selected_pts, background_pts)
            else:
                axis_info = _compute_axis_from_aabb(selected_pts, background_pts)
                if axis_info is None:
                    axis_info = _compute_axis_from_contact(selected_pts, background_pts)
            if axis_info is not None:
                axis_path = output_path.parent / f"{output_stem}_axis.json"
                with open(axis_path, "w") as f:
                    json.dump(axis_info, f, indent=2)
                print(f"✓ 已保存转轴参数: {axis_path}")

    # ========== 恢复Gaussian属性 ==========
    if args.restore_attributes:
        # print(f"\n{'='*70}")
        # print(f"恢复 Gaussian Splatting 属性")
        # print(f"{'='*70}")
        
        # 创建属性恢复器
        restorer = GaussianAttributeRestorer(args.ply, verbose=True)
        
        # 批量恢复
        restored_paths = restorer.batch_restore(
            files_to_restore,
            suffix="_gs",  # 恢复后的文件加 _gs 后缀
            max_distance=args.restore_max_distance,
            overwrite=True
        )
        
        # print(f"\n✓ 属性恢复完成！")
        # print(f"恢复后的文件:")
        for path in restored_paths:
            print(f"  - {path}")
        
        # 删除未恢复的原始ply
        for path in files_to_restore:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as e:
                print(f"⚠️ 删除失败: {path} ({e})")
    
    print(f"\n{'='*70}")
    print(f"完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
