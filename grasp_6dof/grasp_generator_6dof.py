# -*- coding: utf-8 -*-
import argparse, json, os
from grasp_6dof.grasp_sampler import sample_grasps_from_mesh, pack_for_json
import open3d as o3d
import random

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import pybullet as p
        # 让接触对计算使用确定性路径（尽量）
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
    except Exception:
        pass

mesh = o3d.io.read_triangle_mesh(args.obj)
if mesh.is_empty():
    print("[WARN] load mesh failed; fallback to unit sphere.")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    mesh.compute_vertex_normals()

def main():
    set_global_seed(args.seed)
    ap = argparse.ArgumentParser("Generate 6-DoF grasp candidates from mesh/pointcloud")
    ap.add_argument("--obj", type=str, required=True, help="Path to mesh (.obj/.stl/.ply)")
    ap.add_argument("--n", type=int, default=500, help="number of surface samples")
    ap.add_argument("--topk", type=int, default=200, help="save top-K by score")
    ap.add_argument("--out", type=str, default="grasp_6dof/dataset/sample_grasps.json")
    ap.add_argument("--table_z", type=float, default=0.0, help="table top z in mesh frame")
    ap.add_argument("--seed", type=int, default=19)
    args = ap.parse_args()
    parser.add_argument("--seed", type=int, default=42, help="global random seed")

    grasps = sample_grasps_from_mesh(
        mesh_path=args.obj,
        n_samples=args.n,
        table_z=args.table_z,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = pack_for_json(grasps, topk=args.topk)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved {len(data)} grasps → {args.out}")

if __name__ == "__main__":
    main()

