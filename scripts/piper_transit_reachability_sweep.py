"""SAFE_TRANSIT_Z reachability sweep -- read-only, zero diff on
tango_robot/piper_robosuite/ and tango_robot/piper_assets/.

Follow-up to docs/PIPER_TRANSIT_ORIENTATION_ABC_AND_TILT_HYPOTHESIS_
20260807.md's finding that orientation relaxation barely moved transit_high
convergence (0/13 -> 1/13) -- meaning the constant SAFE_TRANSIT_Z=1.05m
itself, not the orientation forced onto it, is the more likely dominant
constraint for most failing seeds. This is the cheap "minimal version"
diagnostic of a reachability-map approach (no learned map, no RM4D/RichMap
needed for a first pass): sweep transit height at each failing/succeeding
candidate's own (x, y) and score every z by IK convergence + minimum
joint-limit margin, using DOWN_ORIENTATION throughout (Part 1 already
showed orientation isn't the dominant lever, so this isolates the pure
position-reachability question).

This does NOT touch run_pick_and_place at all -- solves IK directly against
a bare, unmodified ppp.ArmIK(env) instance for each swept z, without
running the full 8-phase pipeline. Much cheaper per data point than the
previous full-trial scripts, and answers a narrower, more precise question
than any of them: at this (x, y), which z values are actually reachable
with real joint-limit margin, and is 1.05m one of them?

Run:  conda run -n tango python scripts/piper_transit_reachability_sweep.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import scene_objects_for

OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_transit_reachability_sweep.jsonl"

# Same 13 pairs as the orientation A/B/C -- 7 known transit_high failures +
# 6 known successes, so the sweep can be read against both populations.
PAIRS = [
    ("cracker", 1045), ("cracker", 1048),
    ("pear", 1042), ("pear", 1044), ("pear", 1045), ("pear", 1046), ("pear", 1049),
    ("cracker", 1041), ("cracker", 1043), ("cracker", 1047),
    ("pear", 1041), ("pear", 1043), ("pear", 1047),
]

Z_SWEEP = np.round(np.arange(0.85, 1.16, 0.05), 2)  # 0.85 .. 1.15m, matches
                                                     # the range SAFE_TRANSIT_Z=1.05
                                                     # sits inside


def candidate_xy_for(env, obj_name):
    """Same reference candidate generation flow run_pick_and_place uses:
    true_centroid_xy(body_origin, quat, obj_name) -- not the raw body
    origin (see the geometry-decomposition script's own caught bug for why
    that distinction matters)."""
    body_id = env.object_body_ids[obj_name]
    body_pos = env.sim.data.xpos[body_id].copy()
    quat = env.sim.data.xquat[body_id].copy()
    return ppp.true_centroid_xy(body_pos, quat, obj_name)


def sweep_one(obj_name, seed):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for(obj_name),
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20,
    )
    try:
        env.reset()
        xy = candidate_xy_for(env, obj_name)
        ik = ppp.ArmIK(env)

        rows = []
        for z in Z_SWEEP:
            target = np.array([xy[0], xy[1], float(z)])
            result, converged, err, source = ik.solve_multi_seed(
                target, primary_seed=ppp.READY_QPOS, target_mat=ppp.DOWN_ORIENTATION)
            margins = [min(result[i] - lo, hi - result[i])
                      for i, (lo, hi) in enumerate(ppp.REAL_JOINT_LIMITS)]
            tight_j = int(np.argmin(margins))
            rows.append({
                "z": float(z), "converged": bool(converged), "pos_err_cm": float(err * 100),
                "source": source, "min_joint_margin_rad": float(margins[tight_j]),
                "tightest_joint": tight_j + 1,
            })
        return {"object": obj_name, "seed": seed, "candidate_xy": xy[:2].tolist(), "sweep": rows}
    finally:
        env.close()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as fh:
        for obj, seed in PAIRS:
            rec = sweep_one(obj, seed)
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

            print(f"\n=== {obj} seed={seed}  xy={np.round(rec['candidate_xy'],3)} ===")
            for r in rec["sweep"]:
                marker = " <-- SAFE_TRANSIT_Z" if abs(r["z"] - 1.05) < 1e-6 else ""
                print(f"  z={r['z']:.2f}  converged={r['converged']!s:5s} "
                      f"pos_err={r['pos_err_cm']:5.2f}cm  "
                      f"min_margin=joint{r['tightest_joint']}:{r['min_joint_margin_rad']:+.3f}rad{marker}")

    summarize(records)
    print(f"\nwrote {len(records)} records to {OUT}")


def summarize(records):
    print("\n" + "=" * 90)
    print("convergence rate by z (across all 13 candidates)")
    print("=" * 90)
    for zi in range(len(Z_SWEEP)):
        z = Z_SWEEP[zi]
        vs = [r["sweep"][zi]["converged"] for r in records]
        marker = " <-- SAFE_TRANSIT_Z=1.05" if abs(z - 1.05) < 1e-6 else ""
        print(f"  z={z:.2f}  {sum(vs)}/{len(vs)} converged{marker}")

    print("\n" + "=" * 90)
    print("per-candidate: best (highest-margin, converged if possible) z in range")
    print("=" * 90)
    for r in records:
        converged_rows = [s for s in r["sweep"] if s["converged"]]
        pool = converged_rows if converged_rows else r["sweep"]
        best = max(pool, key=lambda s: s["min_joint_margin_rad"])
        at_105 = next(s for s in r["sweep"] if abs(s["z"] - 1.05) < 1e-6)
        print(f"  {r['object']:8s} seed={r['seed']}  "
              f"best_z={best['z']:.2f} (converged={best['converged']!s:5s} margin={best['min_joint_margin_rad']:+.3f}rad)  "
              f"vs z=1.05 (converged={at_105['converged']!s:5s} margin={at_105['min_joint_margin_rad']:+.3f}rad)")


if __name__ == "__main__":
    main()
