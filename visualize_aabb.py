#!/usr/bin/env python3
"""
可视化两个PLY的AABB及其面比例
"""
import argparse
import numpy as np
import open3d as o3d


def face_ratios(lengths):
    x, y, z = lengths
    pairs = {
        "xy": x / y if y > 1e-9 else 0.0,
        "xz": x / z if z > 1e-9 else 0.0,
        "yz": y / z if z > 1e-9 else 0.0,
    }
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="源PLY(蓝)")
    parser.add_argument("--target", required=True, help="目标PLY(白)")
    args = parser.parse_args()
    
    src = o3d.io.read_point_cloud(args.source)
    tgt = o3d.io.read_point_cloud(args.target)
    
    if src.is_empty() or tgt.is_empty():
        raise RuntimeError("输入点云为空")
    
    src.paint_uniform_color([0.1, 0.3, 1.0])
    tgt.paint_uniform_color([0.9, 0.9, 0.9])
    
    src_aabb = src.get_axis_aligned_bounding_box()
    tgt_aabb = tgt.get_axis_aligned_bounding_box()
    src_aabb.color = (0.1, 0.3, 1.0)
    tgt_aabb.color = (0.9, 0.9, 0.9)
    
    src_len = src_aabb.get_extent()
    tgt_len = tgt_aabb.get_extent()
    
    print("Source AABB extents (x,y,z):", src_len)
    print("Target AABB extents (x,y,z):", tgt_len)
    print("Source face ratios:", face_ratios(src_len))
    print("Target face ratios:", face_ratios(tgt_len))
    
    def face_lines(aabb):
        min_pt = aabb.get_min_bound()
        max_pt = aabb.get_max_bound()
        x0, y0, z0 = min_pt
        x1, y1, z1 = max_pt
        # XY face at z0
        xy = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
        ])
        # XZ face at y0
        xz = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z1],
            [x0, y0, z1],
        ])
        # YZ face at x0
        yz = np.array([
            [x0, y0, z0],
            [x0, y1, z0],
            [x0, y1, z1],
            [x0, y0, z1],
        ])
        faces = {"xy": xy, "xz": xz, "yz": yz}
        colors = {
            "xy": [1.0, 0.2, 0.2],
            "xz": [0.2, 1.0, 0.2],
            "yz": [0.2, 0.6, 1.0],
        }
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        geoms = []
        for key, pts in faces.items():
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector([colors[key] for _ in lines])
            geoms.append(ls)
        return geoms
    
    print("面颜色标注: xy=红, xz=绿, yz=蓝")
    
    o3d.visualization.draw_geometries(
        [src, tgt, src_aabb, tgt_aabb] + face_lines(tgt_aabb),
        window_name="AABB Visualizer",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
