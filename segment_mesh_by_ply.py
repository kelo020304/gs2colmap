#!/usr/bin/env python3
"""
用分割后的PLY点云切割同名OBJ网格，并做简单平滑
"""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def segment_mesh(mesh_path, mask_ply, output_path,
                 dist=0.006, min_keep=1, smooth_iter=10, fill_holes=200):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError("Mesh为空")

    mask_pc = o3d.io.read_point_cloud(mask_ply)
    if mask_pc.is_empty():
        raise RuntimeError("Mask点云为空")

    verts = np.asarray(mesh.vertices)

    tree = o3d.geometry.KDTreeFlann(mask_pc)
    keep_vertex = np.zeros(len(verts), dtype=bool)
    dist2_thresh = dist * dist
    for i, v in enumerate(verts):
        _, _, dist2 = tree.search_knn_vector_3d(v, 1)
        if len(dist2) > 0 and dist2[0] <= dist2_thresh:
            keep_vertex[i] = True

    triangles = np.asarray(mesh.triangles)
    tri_keep = np.sum(keep_vertex[triangles], axis=1) >= min_keep
    kept_tris = triangles[tri_keep]

    if len(kept_tris) == 0:
        raise RuntimeError("没有保留下任何三角形，请调大dist或降低min_keep")

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(verts)
    new_mesh.triangles = o3d.utility.Vector3iVector(kept_tris)
    if len(mesh.vertex_colors) == len(verts):
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
    if len(mesh.vertex_normals) == len(verts):
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    new_mesh.remove_unreferenced_vertices()
    new_mesh.remove_duplicated_vertices()
    new_mesh.remove_degenerate_triangles()
    new_mesh.remove_duplicated_triangles()

    if fill_holes > 0 and hasattr(new_mesh, "fill_holes"):
        try:
            new_mesh = new_mesh.fill_holes(max_hole_size=fill_holes)
        except Exception:
            pass

    if smooth_iter > 0:
        new_mesh = new_mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
        new_mesh.compute_vertex_normals()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), new_mesh)
    print(f"✅ Saved mesh: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="输入OBJ路径")
    parser.add_argument("--mask-ply", required=True, help="分割后的PLY点云路径")
    parser.add_argument("--output", required=True, help="输出OBJ路径")
    parser.add_argument("--dist", type=float, default=0.006, help="顶点到mask点的最大距离(米)")
    parser.add_argument("--min-keep", type=int, default=1, help="三角形保留的最少顶点数")
    parser.add_argument("--smooth-iter", type=int, default=10, help="Taubin平滑迭代次数")
    parser.add_argument("--fill-holes", type=int, default=200, help="填洞最大尺寸")
    args = parser.parse_args()

    segment_mesh(
        args.mesh,
        args.mask_ply,
        args.output,
        dist=args.dist,
        min_keep=args.min_keep,
        smooth_iter=args.smooth_iter,
        fill_holes=args.fill_holes,
    )


if __name__ == "__main__":
    main()
