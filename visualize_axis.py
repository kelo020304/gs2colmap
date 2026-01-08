#!/usr/bin/env python3
import argparse
import json
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="可视化点云主轴")
    parser.add_argument("--ply", required=True, help="点云PLY路径")
    parser.add_argument("--axis-json", required=True, help="轴参数JSON路径")
    parser.add_argument("--scale", type=float, default=0.6, help="轴线长度缩放")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    if pcd.is_empty():
        raise RuntimeError("PLY为空")

    with open(args.axis_json, "r") as f:
        axis_info = json.load(f)

    pos = np.array(axis_info["pos"], dtype=np.float32)
    axis = np.array(axis_info["axis"], dtype=np.float32)
    axis = axis / max(np.linalg.norm(axis), 1e-8)

    pts = np.asarray(pcd.points)
    diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    length = diag * args.scale

    p0 = pos - axis * (0.5 * length)
    p1 = pos + axis * (0.5 * length)

    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p0, p1]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=diag * 0.003,
        height=length,
        resolution=20,
        split=4,
    )
    cylinder.paint_uniform_color([1.0, 0.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v = np.cross(z_axis, axis)
    c = float(np.dot(z_axis, axis))
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3, dtype=np.float32)
    else:
        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * (1.0 / (1.0 + c))
    cylinder.rotate(R, center=np.zeros(3))
    cylinder.translate((p0 + p1) * 0.5)

    o3d.visualization.draw_geometries([pcd, line, cylinder])


if __name__ == "__main__":
    main()
