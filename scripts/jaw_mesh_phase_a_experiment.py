#!/usr/bin/env python3
"""
Phase A experiment (2026-08-05): systematic test of H1 -- is the sphere-proxy
jaw collision simplification actually necessary, or was the original
"explosive penetration" finding measured under a badly-posed grasp attempt?

Uses EnvironmentSoArm's enable_close_window_diagnostics=True (formal,
reviewed, opt-in interface -- see its docstring in env_soarm.py) instead of
an external throwaway probe. Grasp candidates come from the real
grasp_6dof/grasp_generator_6dof.py candidate generator, not hand-picked
poses -- exactly the fix that made the 2026-08-05 n=1 pilot work.

Scope (a practical first pass, not the full pre-registered 200-trial design):
  2 objects spanning the concavity spectrum measured in the CoACD-vs-VHACD
  pilot (PowerDrill = high concavity, MustardBottle = near-convex)
  x top-3 real candidates per object
  x 3 repeats per (object, candidate) pair (contact solver is not perfectly
    bit-reproducible on marginal contacts, per this project's own documented
    ~0.6-1% flip rate -- paper_tro.tex sec:selfcorrections)
  x 2 jaw collision geometries (sphere-proxy production default, real
    un-simplified mesh)
  = 36 trials total.

Usage:
  conda run -n tango python3 scripts/jaw_mesh_phase_a_experiment.py
"""
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from tango_robot.env_soarm import EnvironmentSoArm, GRASP_MODE_PHYSICS_WELD, TABLE_TOP_Z

_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)

OBJECTS = {
    "PowerDrill": {
        "mesh": "tango_robot/assets/ycb_objects/YcbPowerDrill/collision_vhacd.obj",
        "spawn_xy": (0.32, -0.12),
    },
    "MustardBottle": {
        "mesh": "tango_robot/assets/ycb_objects/YcbMustardBottle/collision_vhacd.obj",
        "spawn_xy": (0.32, -0.12),
    },
}
N_CANDIDATES_PER_OBJECT = 3
N_REPEATS = 3
GRIPPER_OPENING = 0.09
OBJ_HEIGHT = 0.06


def generate_candidates(object_name: str, spawn_xy) -> list:
    """Call the real candidate generator directly (same math as the CLI
    wrapper grasp_6dof/grasp_generator_6dof.py), no subprocess needed."""
    import trimesh
    from grasp_6dof.grasp_sampler import sample_grasps_from_mesh, pack_for_json

    mesh_path = OBJECTS[object_name]["mesh"]
    mesh = trimesh.load(mesh_path, force="mesh")
    ctr = mesh.bounds.mean(axis=0)
    ext = mesh.extents
    lo, hi = mesh.bounds
    pad = 0.02
    workspace = (
        (float(lo[0] - pad), float(hi[0] + pad)),
        (float(lo[1] - pad), float(hi[1] + pad)),
        (float(max(0.003, lo[2] - pad)), float(hi[2] + pad)),
    )
    grasps = sample_grasps_from_mesh(mesh_path=mesh_path, n_samples=300,
                                       down_sample_voxel=0.002, table_z=0.0,
                                       workspace=workspace, seed=0)
    data = pack_for_json(grasps, topk=N_CANDIDATES_PER_OBJECT)

    world_target = np.array([spawn_xy[0], spawn_xy[1], TABLE_TOP_Z])
    offset = world_target - ctr
    for g in data:
        g["position"] = [float(g["position"][i] + offset[i]) for i in range(3)]
    return data


def run_one(object_name: str, spawn_xy, candidate: dict, use_real_jaw_mesh: bool, seed: int):
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD,
                            enable_close_window_diagnostics=True)
    spawn_pos = [spawn_xy[0], spawn_xy[1], TABLE_TOP_Z + 0.02]
    obj_id = env.load_obj(object_name, pos=spawn_pos, yaw=0.0)

    if use_real_jaw_mesh:
        env.model.geom_type[env._jaw_fixed_geom_id] = _MESH
        env.model.geom_type[env._jaw_mv_geom_id] = _MESH

    pos = tuple(candidate["position"])
    roll = float(candidate["world_yaw"])
    try:
        success, grasped = env.grasp(pos, roll, GRIPPER_OPENING, OBJ_HEIGHT)
        m = env.last_grasp_metrics or {}
        # last_grasp_metrics stores some fields as numpy scalar types
        # (np.bool_/np.float32), not native Python -- json.dumps can't
        # serialize those directly, so cast explicitly rather than relying
        # on the source dict's types.
        bilateral = m.get("bilateral_contact")
        lifted_v = m.get("lifted")
        max_speed = m.get("close_window_max_speed_mps")
        min_dist = m.get("close_window_min_contact_dist_m")
        return {
            "object": object_name, "use_real_jaw_mesh": use_real_jaw_mesh, "seed": seed,
            "candidate_score": candidate["score"],
            "success": bool(success),
            "bilateral_contact": (bool(bilateral) if bilateral is not None else None),
            "lifted": (bool(lifted_v) if lifted_v is not None else None),
            "max_speed_mps": (float(max_speed) if max_speed is not None else None),
            "min_contact_dist_m": (float(min_dist) if min_dist is not None else None),
            "error": None,
        }
    except Exception as e:
        return {
            "object": object_name, "use_real_jaw_mesh": use_real_jaw_mesh, "seed": seed,
            "candidate_score": candidate["score"], "error": str(e),
        }
    finally:
        # Each trial constructs a fresh EnvironmentSoArm (own MjRenderer/EGL
        # context) -- without explicitly closing it, 36 trials in one process
        # accumulate live EGL contexts and eventually crash the interpreter
        # (found the hard way: all 36 trials completed with sane per-trial
        # output, but the process died before the final save/summary step).
        env.close()


def main():
    all_results = []
    for object_name, info in OBJECTS.items():
        print(f"\n{'='*100}\nGenerating candidates for {object_name}\n{'='*100}")
        candidates = generate_candidates(object_name, info["spawn_xy"])
        print(f"Top {len(candidates)} candidates (scores): "
              f"{[round(c['score'], 4) for c in candidates]}")

        for cand_idx, cand in enumerate(candidates):
            for use_real_jaw_mesh in (False, True):
                for repeat in range(N_REPEATS):
                    r = run_one(object_name, info["spawn_xy"], cand,
                                use_real_jaw_mesh, seed=repeat)
                    r["candidate_idx"] = cand_idx
                    all_results.append(r)
                    label = "real_mesh" if use_real_jaw_mesh else "sphere_proxy"
                    if r.get("error"):
                        print(f"  [{object_name} c{cand_idx} {label} r{repeat}] ERROR: {r['error']}")
                    else:
                        print(f"  [{object_name} c{cand_idx} {label} r{repeat}] "
                              f"success={r['success']!s:<5} "
                              f"max_speed={r['max_speed_mps']:.3f}m/s "
                              f"min_dist={r['min_contact_dist_m']*1000:.2f}mm")

    out_path = Path("results/jaw_mesh_phase_a_pilot.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {len(all_results)} trial records -> {out_path}")

    # quick aggregate summary
    print(f"\n{'='*100}\nSummary (pooled across objects/candidates/repeats)\n{'='*100}")
    for use_real_jaw_mesh in (False, True):
        label = "real_mesh" if use_real_jaw_mesh else "sphere_proxy"
        rows = [r for r in all_results if r.get("use_real_jaw_mesh") == use_real_jaw_mesh
                and not r.get("error")]
        n_success = sum(1 for r in rows if r["success"])
        speeds = [r["max_speed_mps"] for r in rows]
        dists = [r["min_contact_dist_m"] for r in rows]
        print(f"{label:<15} n={len(rows):3d}  success={n_success}/{len(rows)}  "
              f"median_max_speed={np.median(speeds):.3f}m/s  "
              f"median_min_dist={np.median(dists)*1000:.2f}mm  "
              f"worst_max_speed={max(speeds):.3f}m/s  worst_min_dist={min(dists)*1000:.2f}mm")


if __name__ == "__main__":
    main()
