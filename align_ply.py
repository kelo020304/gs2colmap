#!/usr/bin/env python3
"""
对齐两个PLY：估计尺度+位姿并输出变换后的源PLY
"""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from gaussian_restore import GaussianAttributeRestorer


def _pca_basis(points):
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    return center, basis


def _robust_bounds(points, low=5.0, high=95.0):
    return np.percentile(points, low, axis=0), np.percentile(points, high, axis=0)


def _aabb_lengths(points):
    return points.max(axis=0) - points.min(axis=0)


def _best_face_from_aabb(src_pts, tgt_pts):
    src_min, src_max = _robust_bounds(src_pts)
    tgt_min, tgt_max = _robust_bounds(tgt_pts)
    src = src_max - src_min
    tgt = tgt_max - tgt_min
    face_pairs = [(0, 1), (0, 2), (1, 2)]
    best_ratio = None
    best = None
    for i, j in face_pairs:
        if src[i] <= 1e-8 or src[j] <= 1e-8 or tgt[i] <= 1e-8 or tgt[j] <= 1e-8:
            continue
        ratio_src = src[i] / src[j]
        ratio_tgt = tgt[i] / tgt[j]
        ratio = min(ratio_src, ratio_tgt) / max(ratio_src, ratio_tgt)
        scale = 0.5 * ((tgt[i] / src[i]) + (tgt[j] / src[j]))
        if best_ratio is None or ratio > best_ratio:
            best_ratio = ratio
            best = (i, j, scale)
    if best is None:
        return (0, 1, 1.0)
    return best


def _fit_plane_offset(points, axis, face_value, band=0.01):
    """拟合靠近指定AABB面的平面位置（无旋转，仅返回轴向偏移与点数）"""
    coords = points[:, axis]
    close = np.abs(coords - face_value) <= band
    if not np.any(close):
        return face_value, 0
    return np.median(coords[close]), int(close.sum())


def compute_align_transform(src, tgt):
    src_pts = np.asarray(src.points)
    tgt_pts = np.asarray(tgt.points)
    
    i, j, _ = _best_face_from_aabb(src_pts, tgt_pts)
    k = 3 - i - j
    src_min, src_max = _robust_bounds(src_pts)
    tgt_min, tgt_max = _robust_bounds(tgt_pts)
    
    extent = np.linalg.norm(tgt_max - tgt_min)
    band = max(extent * 0.01, 1e-4)

    src_face_val = src_min[k]
    tgt_face_val = tgt_min[k]
    src_plane, _ = _fit_plane_offset(src_pts, k, src_face_val, band=band)
    tgt_plane, _ = _fit_plane_offset(tgt_pts, k, tgt_face_val, band=band)

    scale_i = (tgt_max[i] - tgt_min[i]) / max(src_max[i] - src_min[i], 1e-8)
    scale_j = (tgt_max[j] - tgt_min[j]) / max(src_max[j] - src_min[j], 1e-8)
    scale_k = 0.5 * (scale_i + scale_j)
    
    S = np.eye(3)
    S[i, i] = scale_i
    S[j, j] = scale_j
    S[k, k] = scale_k
    
    T = np.eye(4)
    T[:3, :3] = S
    # 让i/j两轴的面边界完全重合（min对min, max对max）
    T[i, 3] = tgt_min[i] - (S[i, i] * src_min[i])
    T[j, 3] = tgt_min[j] - (S[j, j] * src_min[j])
    # k轴仅对齐选中面（最小面）
    T[k, 3] = tgt_plane - (S[k, k] * src_plane)
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="待对齐PLY(蓝色)")
    parser.add_argument("--target", required=True, help="目标PLY(红色)")
    parser.add_argument("--output", required=True, help="输出PLY")
    parser.add_argument("--voxel", type=float, default=0.002, help="下采样体素大小")
    parser.add_argument("--restore-attributes", action="store_true",
                        help="恢复Gaussian属性")
    parser.add_argument("--restore-max-distance", type=float, default=0.001,
                        help="属性恢复最大匹配距离(米), 默认1mm")
    args = parser.parse_args()
    
    src = o3d.io.read_point_cloud(str(args.source))
    tgt = o3d.io.read_point_cloud(str(args.target))
    
    if src.is_empty() or tgt.is_empty():
        raise RuntimeError("输入点云为空")
    
    src_down = src.voxel_down_sample(args.voxel)
    tgt_down = tgt.voxel_down_sample(args.voxel)
    
    src_clean, _ = src_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    tgt_clean, _ = tgt_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    T = compute_align_transform(src_clean, tgt_clean)
    src.transform(T)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), src)
    print(f"✅ Saved aligned PLY: {out_path}")
    
    if args.restore_attributes:
        print("\n恢复 Gaussian Splatting 属性...")
        restorer = GaussianAttributeRestorer(args.source, verbose=True)
        restored_paths = restorer.batch_restore(
            [out_path],
            suffix="_gs",
            max_distance=args.restore_max_distance,
            overwrite=True
        )
        for path in restored_paths:
            print(f"✅ 已保存带属性PLY: {path}")


if __name__ == "__main__":
    main()
