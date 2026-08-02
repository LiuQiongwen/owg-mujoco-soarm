#!/usr/bin/env python3
"""
Paired A/B test: does wrist_friendly_orientation=True actually improve
tray-placement success, or was the seed=777/mustard flip (success=False ->
True) a one-off numerical nudge rather than a real effect?

For each seed, rebuilds the IDENTICAL scene (np.random.seed(seed) before
env construction, this project's established reproducibility convention --
see piper_multi_object_scene.py's 2026-07-16 fix) twice: once with the flag
off, once with it on, and runs run_pick_and_place on every object in that
scene both times. Uses this project's canonical McNemar implementation
(scripts/paired_stats.py::mcnemar_test) for the significance test, matching
the methodology used throughout results/risk_gated_vla/.

No VLM/grounding involved -- this is a pure physical-execution question, so
object names are read directly rather than through Claude/GPT grounding.

Usage:
    conda run -n tango python tango_robot/piper_robosuite/eval_wrist_friendly_tray_placement.py --seeds 10
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa: registers Piper/PiperGripper
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite.piper_pick_and_place import run_pick_and_place
from paired_stats import mcnemar_test  # noqa: E402

OBJECTS = ["pear", "can", "mustard"]


def run_one(seed, wrist_friendly):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=OBJECTS,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    results = {}
    for obj in OBJECTS:
        r = run_pick_and_place(env, obj, use_oriented_grasp=True,
                                wrist_friendly_orientation=wrist_friendly, verbose=False)
        results[obj] = int(r["success"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds to test (base_seed..base_seed+n-1)")
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    pairs_by_obj = {obj: [] for obj in OBJECTS}
    pairs_pooled = []

    for i in range(args.seeds):
        seed = args.base_seed + i
        off = run_one(seed, wrist_friendly=False)
        on = run_one(seed, wrist_friendly=True)
        for obj in OBJECTS:
            pairs_by_obj[obj].append((off[obj], on[obj]))
            pairs_pooled.append((off[obj], on[obj]))
        print(f"seed={seed}  off={off}  on={on}")

    print()
    print("| Object | off success | on success | n01 (on wins) | n10 (off wins) | McNemar p |")
    print("|---|---:|---:|---:|---:|---:|")
    for obj in OBJECTS:
        p = pairs_by_obj[obj]
        off_rate = sum(a for a, b in p) / len(p)
        on_rate = sum(b for a, b in p) / len(p)
        n01, n10, pval, _ = mcnemar_test(p)
        print(f"| {obj} | {off_rate:.0%} | {on_rate:.0%} | {n01} | {n10} | {pval:.4f} |")

    off_rate = sum(a for a, b in pairs_pooled) / len(pairs_pooled)
    on_rate = sum(b for a, b in pairs_pooled) / len(pairs_pooled)
    n01, n10, pval, _ = mcnemar_test(pairs_pooled)
    print(f"| **pooled (n={len(pairs_pooled)})** | {off_rate:.0%} | {on_rate:.0%} | {n01} | {n10} | {pval:.4f} |")


if __name__ == "__main__":
    main()
