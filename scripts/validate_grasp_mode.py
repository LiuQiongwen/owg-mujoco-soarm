#!/usr/bin/env python3
"""Headless validation for the physics_weld_after_bilateral grasp mode.

Runs N trials per object with random yaw sampling (geometry baseline) and
reports bilateral-contact rate, weld-trigger rate, table-contact rate,
and grasp success rate.

World-model (LGGSN) evaluation is not included here because it requires
a full VLM + LGGSN inference stack; run quick_eval.sh with --stage 4 for that.

Usage
-----
    MUJOCO_GL=egl python scripts/validate_grasp_mode.py
    MUJOCO_GL=egl python scripts/validate_grasp_mode.py \\
        --objects Banana MustardBottle TomatoSoupCan \\
        --n-trials 20 --seed 0
"""
import argparse
import os
import sys
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from owg_robot.env_soarm import (
    EnvironmentSoArm,
    TABLE_TOP_Z,
    GRASP_Z_TABLE_MARGIN,
    GRASP_MODE_PHYSICS_WELD,
)

_YAW_RANGE  = (-np.pi / 2, np.pi / 2)
_OPEN_RANGE = (0.05, 0.09)


def _run_trial(env: EnvironmentSoArm,
               obj_name: str,
               rng: np.random.Generator) -> Dict:
    """Run one grasp trial and return result dict."""
    env.remove_all_obj()   # clear pool before each trial to avoid accumulation
    env.reset()

    # Spawn object on the table
    obj_id = env.load_obj(obj_name, pos=[0.0, -0.20, TABLE_TOP_Z + 0.15])
    env.wait_until_still(obj_id)
    env._steps(300)

    com = env.get_obj_com_pos(obj_id)
    x, y   = float(com[0]), float(com[1])
    grasp_z = float(com[2]) + GRASP_Z_TABLE_MARGIN
    yaw     = float(rng.uniform(*_YAW_RANGE))
    opening = float(rng.uniform(*_OPEN_RANGE))

    t0 = time.time()
    success, grasped_id = env._execute_grasp_physics_topdown(
        pos=(x, y, grasp_z),
        yaw=yaw,
        gripper_opening_length=opening,
        obj_height=0.04,
    )
    elapsed = time.time() - t0

    m = env.last_grasp_metrics or {}
    return {
        "object":           obj_name,
        "yaw":              round(yaw, 4),
        "opening":          round(opening, 4),
        "grasp_z":          round(grasp_z, 4),
        "com_z":            round(float(com[2]), 4),
        "success":          success,
        "bilateral_contact": m.get("bilateral_contact", False),
        "weld_triggered":   m.get("weld_triggered", False),
        "table_contact":    m.get("table_contact", False),
        "final_z":          round(m.get("final_z", 0.0), 4),
        "lifted":           m.get("lifted", False),
        "jaw_obj_xy_gap_cm": round((m.get("jaw_obj_xy_gap") or 0) * 100, 2),
        "left_contacts":    m.get("left_contacts", 0),
        "right_contacts":   m.get("right_contacts", 0),
        "elapsed_s":        round(elapsed, 2),
    }


def _aggregate(results: List[Dict]) -> Dict:
    n = len(results)
    if n == 0:
        return {}

    def rate(key: str) -> float:
        return sum(1 for r in results if r.get(key)) / n

    final_zs = [r["final_z"] for r in results if r.get("weld_triggered")]

    return {
        "n_trials":            n,
        "success_rate":        round(rate("success"), 3),
        "bilateral_rate":      round(rate("bilateral_contact"), 3),
        "weld_trigger_rate":   round(rate("weld_triggered"), 3),
        "table_contact_rate":  round(rate("table_contact"), 3),
        "mean_final_z_weld":   round(float(np.mean(final_zs)), 4) if final_zs else None,
        "failure_breakdown": {
            "no_bilateral":    sum(1 for r in results if not r["bilateral_contact"]),
            "bilateral_no_lift": sum(1 for r in results
                                    if r["bilateral_contact"] and not r["lifted"]),
        },
    }


def _print_summary(obj: str, agg: Dict, trial_results: List[Dict]) -> None:
    n = agg["n_trials"]
    print(f"\n{'='*60}")
    print(f"  Object: {obj}   (n={n}  mode=physics_weld_after_bilateral)")
    print(f"{'='*60}")
    print(f"  Success rate         : {agg['success_rate']*100:.1f}%")
    print(f"  Bilateral rate       : {agg['bilateral_rate']*100:.1f}%")
    print(f"  Weld trigger rate    : {agg['weld_trigger_rate']*100:.1f}%")
    print(f"  Table contact rate   : {agg['table_contact_rate']*100:.1f}%  (should be 0%)")
    if agg.get("mean_final_z_weld") is not None:
        print(f"  Mean final_z (weld)  : {agg['mean_final_z_weld']:.4f} m")

    fb = agg["failure_breakdown"]
    print(f"\n  Failure breakdown:")
    print(f"    no_bilateral       : {fb['no_bilateral']}")
    print(f"    bilateral_no_lift  : {fb['bilateral_no_lift']}")

    # show any table-contact trials
    tc_trials = [r for r in trial_results if r.get("table_contact")]
    if tc_trials:
        print(f"\n  [WARN] Table contacts ({len(tc_trials)} trials):")
        for r in tc_trials[:5]:
            print(f"    grasp_z={r['grasp_z']:.4f}  com_z={r['com_z']:.4f}  "
                  f"gap_cm={r['jaw_obj_xy_gap_cm']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects",  nargs="+",
                    default=["Banana", "MustardBottle", "TomatoSoupCan"])
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--out",      default=None,
                    help="Write results JSON to this path")
    args = ap.parse_args()

    print(f"[validate] mode=physics_weld_after_bilateral  "
          f"objects={args.objects}  n_trials={args.n_trials}  seed={args.seed}")

    env = EnvironmentSoArm(
        grasp_mode=GRASP_MODE_PHYSICS_WELD,
        n_grasp_attempts=1,
    )

    all_results: Dict[str, List[Dict]] = {}
    global_rng = np.random.default_rng(args.seed)

    for obj_name in args.objects:
        print(f"\n[validate] ── {obj_name} ─────────────────────────────")
        trials: List[Dict] = []
        obj_rng = np.random.default_rng(int(global_rng.integers(0, 2**31)))

        for trial_idx in range(args.n_trials):
            print(f"  trial {trial_idx+1}/{args.n_trials} ...", end="  ", flush=True)
            try:
                r = _run_trial(env, obj_name, obj_rng)
                trials.append(r)
                status = ("SUCCESS" if r["success"]
                          else ("BILATERAL_NO_LIFT" if r["bilateral_contact"]
                                else "NO_CONTACT"))
                print(f"{status}  bilateral={r['bilateral_contact']}"
                      f"  weld={r['weld_triggered']}"
                      f"  table_contact={r['table_contact']}"
                      f"  final_z={r['final_z']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                trials.append({"object": obj_name, "success": False,
                                "bilateral_contact": False, "weld_triggered": False,
                                "table_contact": False, "error": str(e)})

        agg = _aggregate(trials)
        _print_summary(obj_name, agg, trials)
        all_results[obj_name] = {"trials": trials, "aggregate": agg}

    # ── overall summary ───────────────────────────────────────────────────────
    all_trials = [r for v in all_results.values() for r in v["trials"]]
    overall    = _aggregate(all_trials)
    print(f"\n{'='*60}")
    print(f"  OVERALL  (n={overall['n_trials']})")
    print(f"{'='*60}")
    print(f"  Success rate       : {overall['success_rate']*100:.1f}%")
    print(f"  Bilateral rate     : {overall['bilateral_rate']*100:.1f}%")
    print(f"  Weld trigger rate  : {overall['weld_trigger_rate']*100:.1f}%")
    print(f"  Table contact rate : {overall['table_contact_rate']*100:.1f}%")
    print()

    output = {
        "meta": {
            "mode":      "physics_weld_after_bilateral",
            "method":    "geometry",
            "objects":   args.objects,
            "n_trials":  args.n_trials,
            "seed":      args.seed,
            "grasp_z_table_margin": GRASP_Z_TABLE_MARGIN,
        },
        "per_object": all_results,
        "overall":    overall,
    }

    out_path = args.out or str(ROOT / "results" / "grasp_mode_validation.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[validate] Results written to {out_path}")


if __name__ == "__main__":
    main()
