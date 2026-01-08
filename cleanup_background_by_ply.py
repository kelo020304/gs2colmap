#!/usr/bin/env python3
"""
用对齐后的抽屉PLY去清理background PLY/OBJ
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def _preserve_obj_materials(src_obj: Path, dst_obj: Path) -> None:
    if not src_obj.is_file() or not dst_obj.is_file():
        return
    mtllib_line = ""
    mtl_name = ""
    with open(src_obj, "r") as f:
        for line in f:
            if line.startswith("mtllib "):
                mtllib_line = line.rstrip("\n")
                mtl_name = line.strip().split(maxsplit=1)[-1]
                break

    if mtllib_line:
        with open(dst_obj, "r") as f:
            out_lines = f.read().splitlines()
        if not any(l.startswith("mtllib ") for l in out_lines):
            out_lines.insert(0, mtllib_line)
            with open(dst_obj, "w") as f:
                f.write("\n".join(out_lines) + "\n")

        src_dir = src_obj.parent
        dst_dir = dst_obj.parent
        src_mtl = src_dir / mtl_name
        dst_mtl = dst_dir / mtl_name
        if src_mtl.is_file() and src_mtl.resolve() != dst_mtl.resolve():
            shutil.copy2(src_mtl, dst_mtl)
        if src_mtl.is_file():
            with open(src_mtl, "r") as f:
                for line in f:
                    if line.strip().startswith("map_Kd "):
                        tex_name = line.strip().split(maxsplit=1)[-1]
                        src_tex = src_dir / tex_name
                        dst_tex = dst_dir / tex_name
                        if src_tex.is_file() and src_tex.resolve() != dst_tex.resolve():
                            shutil.copy2(src_tex, dst_tex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-ply", required=True)
    parser.add_argument("--drawer-ply", required=True)
    parser.add_argument("--output-ply", required=True)
    parser.add_argument("--dist", type=float, default=0.006)
    parser.add_argument("--background-obj", default="")
    parser.add_argument("--output-obj", default="")
    args = parser.parse_args()

    bg_ply = PlyData.read(args.background_ply)
    bg_verts = bg_ply["vertex"].data
    bg_pts = np.stack([bg_verts["x"], bg_verts["y"], bg_verts["z"]], axis=-1)

    drawer_pts_list = []
    drawer_paths = [Path(args.drawer_ply)]
    drawer_dir = Path(args.drawer_ply).parent
    aligned_paths = sorted(drawer_dir.glob("*_aligned.ply"))
    for p in aligned_paths:
        if p not in drawer_paths:
            drawer_paths.append(p)
    for p in drawer_paths:
        if not p.is_file():
            continue
        ply = PlyData.read(str(p))
        verts = ply["vertex"].data
        pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=-1)
        if len(pts) > 0:
            drawer_pts_list.append(pts)

    if len(bg_pts) == 0 or len(drawer_pts_list) == 0:
        raise RuntimeError("点云为空")

    drawer_pts = np.vstack(drawer_pts_list)
    use_aabb = len(aligned_paths) > 0
    if use_aabb:
        drawer_min = drawer_pts.min(axis=0) - args.dist
        drawer_max = drawer_pts.max(axis=0) + args.dist
        inside = np.all((bg_pts >= drawer_min) & (bg_pts <= drawer_max), axis=1)
        keep = ~inside
    else:
        tree = cKDTree(drawer_pts)
        dists, _ = tree.query(bg_pts, k=1, distance_upper_bound=args.dist)
        keep = np.isinf(dists) | (dists > args.dist)

    cleaned = np.empty(int(keep.sum()), dtype=bg_verts.dtype)
    for name in bg_verts.dtype.names:
        cleaned[name] = bg_verts[name][keep]
    cleaned_pts = np.stack([cleaned["x"], cleaned["y"], cleaned["z"]], axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cleaned_pts)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(ind) > 0 and len(ind) < len(cleaned):
        cleaned = cleaned[ind]
    out_ply = Path(args.output_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    el = PlyElement.describe(cleaned, "vertex")
    PlyData([el], text=False).write(out_ply)
    print(f"✅ Saved cleaned background PLY: {out_ply}")

    if args.background_obj and args.output_obj:
        mesh = o3d.io.read_triangle_mesh(args.background_obj)
        if not mesh.is_empty():
            verts = np.asarray(mesh.vertices)
            tri = np.asarray(mesh.triangles)
            if use_aabb:
                inside_v = np.all((verts >= drawer_min) & (verts <= drawer_max), axis=1)
                keep_v = ~inside_v
            else:
                tree = cKDTree(drawer_pts)
                mesh_dist = args.dist * 1.5
                dists_v, _ = tree.query(verts, k=1, distance_upper_bound=mesh_dist)
                keep_v = np.isinf(dists_v) | (dists_v > mesh_dist)
            tri_keep = np.sum(keep_v[tri], axis=1) == 3
            mesh.triangles = o3d.utility.Vector3iVector(tri[tri_keep])
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            out_obj = Path(args.output_obj)
            out_obj.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(str(out_obj), mesh)
            _preserve_obj_materials(Path(args.background_obj), out_obj)
            print(f"✅ Saved cleaned background OBJ: {out_obj}")


if __name__ == "__main__":
    main()
