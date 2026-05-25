# -*- coding: utf-8 -*-
import argparse
import json
import os
import random

import numpy as np
import open3d as o3d

from grasp_sampler import sample_grasps_from_mesh, pack_for_json


def set_global_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="6-DoF grasp generator using Open3D & grasp_sampler.py"
    )
    parser.add_argument("--obj",      required=True,
                        help="mesh file path (.ply/.obj/.stl, etc.)")
    parser.add_argument("--out",      required=True,
                        help="output json file for sampled grasps")
    parser.add_argument("--n",        type=int, default=2048,
                        help="number of raw grasp samples")
    parser.add_argument("--topk",     type=int, default=512,
                        help="keep top-K grasps by score")
    parser.add_argument("--table_z",  type=float, default=0.0,
                        help="table top height (used in scoring)")
    parser.add_argument("--voxel",    type=float, default=0.002,
                        help="voxel size for mesh down-sampling")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--world-pos", type=str, default=None,
                        help="x,y,z of object centre in robot world frame. "
                             "Grasps are translated from mesh-local to world frame. "
                             "Example: --world-pos 0.38,0.0,0.027")

    args = parser.parse_args()
    set_global_seed(args.seed)

    mesh_path = args.obj
    print(f"[INFO] Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"[ERROR] Empty/no-triangle mesh: {mesh_path}")

    # ── auto-compute workspace from mesh bounding box ──────────
    verts = np.asarray(mesh.vertices)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ctr    = 0.5 * (lo + hi)
    ext    = hi - lo
    pad    = 0.10   # 10 cm margin around the mesh
    workspace = (
        (float(lo[0] - pad), float(hi[0] + pad)),
        (float(lo[1] - pad), float(hi[1] + pad)),
        (float(max(args.table_z + 0.003, lo[2] - pad)), float(hi[2] + pad)),
    )
    print(f"[INFO] Mesh centre={np.round(ctr,3)}  extents={np.round(ext,3)}")
    print(f"[INFO] Auto workspace x={workspace[0]} y={workspace[1]} z={workspace[2]}")

    print(f"[INFO] Sampling: n={args.n}  topk={args.topk}  "
          f"voxel={args.voxel:.4f}  seed={args.seed}")

    grasps = sample_grasps_from_mesh(
        mesh_path=mesh_path,
        n_samples=args.n,
        down_sample_voxel=args.voxel,
        table_z=args.table_z,
        workspace=workspace,
        seed=args.seed,
    )
    print(f"[INFO] {len(grasps)} grasps passed scoring filter")

    data = pack_for_json(grasps, topk=args.topk)

    # ── optional world-frame translation ──────────────────────
    if args.world_pos:
        wx, wy, wz = [float(v) for v in args.world_pos.split(",")]
        world_target = np.array([wx, wy, wz])
        # compute centroid of all grasp positions
        if data:
            gpos = np.array([g["position"] for g in data])
            mesh_grasp_centre = gpos.mean(axis=0)
            # offset = world_target - mesh_grasp_centre,  but keep Z correction:
            # mesh grasps may be at mesh surface; we want them near the placed object
            offset = world_target - ctr   # translate mesh centroid → world position
            for g in data:
                g["position"] = [float(g["position"][i] + offset[i]) for i in range(3)]
            print(f"[INFO] Translated grasps: mesh_ctr={np.round(ctr,3)} → world={world_target}  "
                  f"offset={np.round(offset,3)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved {len(data)} grasps → {args.out}")


if __name__ == "__main__":
    main()
