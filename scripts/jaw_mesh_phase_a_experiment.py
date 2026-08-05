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

Scope: the full pre-registered design (scaled up 2026-08-05 from an initial
36-trial pilot -- results/jaw_mesh_phase_a_pilot.json -- whose n=3 discordant
pairs were too few to draw a conclusion from):
  4 objects spanning the concavity spectrum measured in the CoACD-vs-VHACD
  pilot (PowerDrill = high concavity, Banana = curved, CrackerBox = box-like,
  MustardBottle = near-convex)
  x top-5 real candidates per object
  x 5 repeats per (object, candidate) pair (contact solver is not perfectly
    bit-reproducible on marginal contacts, per this project's own documented
    ~0.6-1% flip rate -- paper_tro.tex sec:selfcorrections)
  x 2 jaw collision geometries (sphere-proxy production default, real
    un-simplified mesh)
  = 200 trials total.

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
    "Banana": {
        "mesh": "tango_robot/assets/ycb_objects/YcbBanana/collision_vhacd.obj",
        "spawn_xy": (0.32, -0.12),
    },
    # CrackerBox was originally chosen for the box-shaped/near-convex end of
    # the concavity spectrum, but its collision_vhacd.obj has 0 triangles
    # readable by Open3D (crashed sample_grasps_from_mesh mid-run 2026-08-05,
    # after PowerDrill+Banana had already completed 100 trials -- lost since
    # nothing was saved incrementally). TomatoSoupCan and Pear have the same
    # 0-triangle problem; verified MediumClamp (590 triangles) and Scissors
    # (654 triangles) both load fine before choosing this replacement. Using
    # "MediumClamp" (not CLAUDE.md's registry name "Cylinder") because
    # load_obj resolves path_or_name against the real YCB directory name
    # ("Ycb"+name) directly -- it does not go through OBJECT_REGISTRY's
    # logical-name translation, which only benchmark/runner.py and demo.py use.
    "MediumClamp": {
        "mesh": "tango_robot/assets/ycb_objects/YcbMediumClamp/collision_vhacd.obj",
        "spawn_xy": (0.32, -0.12),
    },
    "MustardBottle": {
        "mesh": "tango_robot/assets/ycb_objects/YcbMustardBottle/collision_vhacd.obj",
        "spawn_xy": (0.32, -0.12),
    },
}
N_CANDIDATES_PER_OBJECT = 5
N_REPEATS = 5
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
    out_path = Path("results/jaw_mesh_phase_a_confirm200.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = []

    for object_name, info in OBJECTS.items():
        print(f"\n{'='*100}\nGenerating candidates for {object_name}\n{'='*100}")
        try:
            candidates = generate_candidates(object_name, info["spawn_xy"])
        except Exception as e:
            print(f"  Candidate generation FAILED for {object_name}: {e}. Skipping this object.")
            continue
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

        # Incremental save after each object -- a crash partway through (this
        # exact thing happened once already, on 2026-08-05: 100 completed
        # trials were lost because nothing was written to disk until the very
        # end) now only costs the current object's in-flight trials, not
        # everything completed so far.
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"  [checkpoint] saved {len(all_results)} trial records so far -> {out_path}")

    print(f"\nSaved {len(all_results)} trial records -> {out_path}")

    # quick aggregate summary (descriptive only, not the statistical test)
    print(f"\n{'='*100}\nDescriptive summary (pooled across objects/candidates/repeats)\n{'='*100}")
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

    # ── paired McNemar per object (this project's canonical implementation),
    # Holm-Bonferroni corrected across the 4 objects, plus a pooled comparison ──
    from scripts.paired_stats import mcnemar_test

    print(f"\n{'='*100}\nPaired McNemar (sphere-proxy vs real-mesh), per object, "
          f"Holm-corrected across {len(OBJECTS)} objects\n{'='*100}")

    pairs_by_key = {}
    for r in all_results:
        if r.get("error"):
            continue
        key = (r["object"], r["candidate_idx"], r["seed"])
        pairs_by_key.setdefault(key, {})[r["use_real_jaw_mesh"]] = int(r["success"])

    per_object_results = []
    for object_name in OBJECTS:
        pairs = [(v[False], v[True]) for k, v in pairs_by_key.items()
                 if k[0] == object_name and False in v and True in v]
        # pairs = (sphere_success, mesh_success). mcnemar_test's convention:
        # n01 = a==0,b==1 = sphere FAILED, mesh SUCCEEDED = mesh-only.
        # n10 = a==1,b==0 = sphere SUCCEEDED, mesh FAILED = sphere-only.
        # (An earlier version of this script had these two labels swapped --
        # caught because the per-object mesh/sphere-only counts didn't add up
        # to the raw pooled success counts printed just above; fixed before
        # trusting any conclusion drawn from this table.)
        n01, n10, p, stat = mcnemar_test(pairs)
        per_object_results.append({"object": object_name, "n_pairs": len(pairs),
                                    "n_mesh_only_success": n01, "n_sphere_only_success": n10,
                                    "p_raw": p})

    # Holm-Bonferroni step-down
    sorted_by_p = sorted(range(len(per_object_results)),
                          key=lambda i: per_object_results[i]["p_raw"])
    m = len(per_object_results)
    running_max = 0.0
    for rank, idx in enumerate(sorted_by_p):
        adj = min(1.0, per_object_results[idx]["p_raw"] * (m - rank))
        running_max = max(running_max, adj)
        per_object_results[idx]["p_holm"] = running_max

    for row in per_object_results:
        print(f"  {row['object']:<15} n_pairs={row['n_pairs']:2d}  "
              f"sphere_only_success={row['n_sphere_only_success']}  "
              f"mesh_only_success={row['n_mesh_only_success']}  "
              f"p_raw={row['p_raw']:.4f}  p_holm={row['p_holm']:.4f}")

    all_pairs = [(v[False], v[True]) for v in pairs_by_key.values()
                 if False in v and True in v]
    n_mesh_only, n_sphere_only, p, stat = mcnemar_test(all_pairs)
    print(f"\n  POOLED (all objects) n_pairs={len(all_pairs)}  "
          f"sphere_only_success={n_sphere_only}  mesh_only_success={n_mesh_only}  p={p:.4f}")
    print(f"\nInterpretation: sphere_only_success = trials where sphere-proxy "
          f"succeeded and real mesh failed on the SAME (object,candidate,seed);\n"
          f"mesh_only_success is the reverse. If sphere_only >> mesh_only with "
          f"p_holm<0.05 for a given object, real mesh has a real cost there, not just noise.")


if __name__ == "__main__":
    main()
