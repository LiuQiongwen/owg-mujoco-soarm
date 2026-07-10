#!/usr/bin/env python3
"""
Phase 2, Part B3 (see /home/lina/.claude/plans/floating-crunching-yeti.md):
does _settle_at_pose's "revert" behave as a true no-op, or does object state
drift across repeated calls?

Hypothesis: _execute_grasp_physics_topdown's trust-but-verify correction
path calls _settle_at_pose THREE times without an intervening reset_robot()
-- (1) original target, (2) corrected target, (3) revert back to original
target -- exactly matching production behaviour. Each call restores the
object to wherever it CURRENTLY is at that call's start (not the pristine
spawn position), so if call (2) perturbs the object even slightly, call (3)
may not exactly reproduce call (1)'s result.

Test A: back-to-back calls at the SAME target, no intervening different
         settle -- pure repeatability check.
Test B: original target -> different target -> back to original target --
         tests whether an intervening correction attempt introduces drift.

Usage:
    conda run -n tango python paperA_data/scripts/check_settle_idempotency.py
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from tango_robot.env_soarm import EnvironmentSoArm, TABLE_TOP_Z

OBJECT_REGISTRY = {"pear": "YcbPear", "can": "YcbTomatoSoupCan", "cracker": "YcbCrackerBox"}
_SPAWN_X_LO, _SPAWN_X_HI = -0.15, 0.15
_SPAWN_Y_LO, _SPAWN_Y_HI = -0.35, -0.10
_DROP_Z = TABLE_TOP_Z + 0.12
_Z_OFFSET = 0.025
_OPEN = 0.06


def run_one(obj_key: str, seed: int) -> dict:
    ycb_name = OBJECT_REGISTRY[obj_key]
    rng = np.random.default_rng(seed)
    env = EnvironmentSoArm(vis=False, grasp_mode="physics_weld_after_bilateral")

    cx = float(rng.uniform(_SPAWN_X_LO, _SPAWN_X_HI))
    cy = float(rng.uniform(_SPAWN_Y_LO, _SPAWN_Y_HI))
    obj_id = env.load_obj(ycb_name, name=obj_key, pos=[cx, cy, _DROP_Z])
    env._steps(300)
    obj_pos = env.get_obj_pos(obj_id)

    x1 = float(obj_pos[0] + rng.uniform(-0.02, 0.02))
    y1 = float(obj_pos[1] + rng.uniform(-0.02, 0.02))
    z1 = float(obj_pos[2] + _Z_OFFSET)
    yaw1 = float(rng.uniform(-math.pi / 4, math.pi / 4))
    opening = _OPEN * env.GRIP_REDUCTION

    # different target for the intervening settle (matches the correction
    # search's delta magnitude: up to 0.03m / 0.2rad)
    x2 = x1 + 0.02
    y2 = y1 - 0.02
    yaw2 = yaw1 + 0.15

    env.reset_robot()
    m1a, _ = env._settle_at_pose(x1, y1, z1, yaw1, opening)   # call 1: original

    # Test A: immediate repeat, same target, no intervening settle
    m1b, _ = env._settle_at_pose(x1, y1, z1, yaw1, opening)   # call 2: same target again

    # Test B: intervening different target, then back to original
    m2, _ = env._settle_at_pose(x2, y2, z1, yaw2, opening)    # call 3: different target
    m1c, _ = env._settle_at_pose(x1, y1, z1, yaw1, opening)   # call 4: revert to original

    env.close()

    def gap(m):
        return m.get("jaw_obj_xy_gap")

    return {
        "object": obj_key, "seed": seed,
        "gap_call1_original":        gap(m1a),
        "gap_callA_immediate_repeat": gap(m1b),
        "gap_callB2_different":       gap(m2),
        "gap_callB3_reverted":        gap(m1c),
        "drift_A_vs_1":  None if gap(m1a) is None or gap(m1b) is None else round(gap(m1b) - gap(m1a), 5),
        "drift_B3_vs_1": None if gap(m1a) is None or gap(m1c) is None else round(gap(m1c) - gap(m1a), 5),
    }


def main():
    results = []
    for obj_key in ["pear", "can", "cracker"]:
        for seed in [1, 2, 3, 4, 5]:
            r = run_one(obj_key, seed)
            results.append(r)
            print(f"[idempotency] {obj_key} seed={seed}  "
                  f"gap1={r['gap_call1_original']:.4f}  "
                  f"repeatA={r['gap_callA_immediate_repeat']:.4f} (drift={r['drift_A_vs_1']})  "
                  f"revertB={r['gap_callB3_reverted']:.4f} (drift={r['drift_B3_vs_1']})")

    out_path = Path("paperA_data/worldmodel_trajs/settle_idempotency_check.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    drift_a = [abs(r["drift_A_vs_1"]) for r in results if r["drift_A_vs_1"] is not None]
    drift_b = [abs(r["drift_B3_vs_1"]) for r in results if r["drift_B3_vs_1"] is not None]
    print(f"\n[summary] Test A (immediate repeat, no intervening settle): "
          f"mean|drift|={np.mean(drift_a):.5f}  max|drift|={np.max(drift_a):.5f}")
    print(f"[summary] Test B (intervening different-target settle then revert): "
          f"mean|drift|={np.mean(drift_b):.5f}  max|drift|={np.max(drift_b):.5f}")
    print(f"[summary] saved -> {out_path}")


if __name__ == "__main__":
    main()
