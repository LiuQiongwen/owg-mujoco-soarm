"""
Pilot experiment: does orientation-aware grasping (compute_grasp_orientation)
actually beat the fixed-world-X approach it replaced?

Design (per /research conventions used elsewhere in this project -- smoke
test first, per-trial seeding, statistical comparison rather than eyeballing
raw counts):
  - Objects: pear, mustard, can. Can is kept in deliberately -- it is
    expected to fail regardless of strategy (its ~8.6cm diameter exceeds
    the gripper's ~7.6cm max opening, a hardware limit, not an orientation
    problem) and that is itself the reportable result for that object.
  - Strategies: oriented (compute_grasp_orientation) vs fixed (DOWN_ORIENTATION,
    the original world-X-locked approach) -- the ablation this experiment
    exists to test.
  - Success metric: final object position within PLACEMENT_TRAY_HALF_EXTENT
    of the tray centre (same threshold run_pick_and_place already uses).
  - n=20/cell for the pilot (3 objects x 2 strategies x 20 = 120 trials).
    Scale to n=50 only if the pilot direction looks stable, per this
    project's established smoke-test-then-scale convention.
  - Each trial reseeds numpy (np.random.seed(trial_id)) before env.reset()
    so the object spawn/orientation draw is reproducible per trial id and
    IDENTICAL across the two strategies at the same trial id -- i.e. this
    is a paired design (both strategies see the same spawn draws), not two
    independent samples, which is a strictly more powerful comparison for
    the same n.

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_experiment_runner [n_trials] [oriented|fixed|both] [object]

The strategy/object args let a single ~38min (3 objects x 2 strategies x 20
trials, at ~19s/trial with the multi-seed IK solver) pilot be split into 6
~6-7min chunks that each fit safely under the background command timeout --
run each (object, strategy) pair as its own background call (each writing
its own results file, tagged by both), then merge before analysis.
"""
import json
import sys
import time

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite.piper_pick_and_place import run_pick_and_place

OBJECTS = ["pear", "mustard", "can", "banana", "cracker", "drill", "clamp"]
RESULTS_PATH_TEMPLATE = "/lena/projects/OWG-main/tango_robot/piper_robosuite/experiment_results_{tag}.json"

# NOTE (2026-07-14): pear/mustard/can trials above were all run with the
# scene spawning all 3 together (ycb_objects=["pear","can","mustard"]) --
# the original pilot's fixed 3-object scene. New objects (banana, cracker,
# drill, clamp) are spawned ALONE instead: PiperMultiObjectScene's own
# placement sampler could only reliably fit small-radius objects together
# (see piper_multi_object_scene.py's DEFAULT_SCENE_OBJECTS comment -- banana/
# cracker/drill/clamp all have large horizontal_radius and were excluded
# from the default multi-object set for exactly this reason), and
# run_pick_and_place only ever looks at the one named target object anyway,
# so a solo scene is both necessary (avoids spawn-placement failures
# unrelated to grasp performance) and doesn't change what's being measured.
# This is a real condition difference from the pear/mustard/can trials
# (no other objects/clutter in the scene) -- noted so it isn't silently
# treated as the same experiment.


def run_one_trial(obj_name, use_oriented, trial_id):
    np.random.seed(trial_id)
    scene_objects = ["pear", "can", "mustard"] if obj_name in ("pear", "can", "mustard") else [obj_name]
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    result = run_pick_and_place(env, obj_name, use_oriented_grasp=use_oriented, verbose=False)
    result["trial_id"] = trial_id
    result["scene_objects"] = scene_objects
    return result


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    strategy_arg = sys.argv[2] if len(sys.argv) > 2 else "both"
    obj_filter = sys.argv[3] if len(sys.argv) > 3 else "all"
    start_trial = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    strategies = {"oriented": [True], "fixed": [False], "both": [True, False]}[strategy_arg]
    objects = OBJECTS if obj_filter == "all" else [obj_filter]
    trial_ids = range(start_trial, start_trial + n_trials)

    results = []
    t0 = time.time()
    total = len(objects) * len(strategies) * n_trials
    done = 0
    for obj_name in objects:
        for use_oriented in strategies:
            successes = 0
            for trial_id in trial_ids:
                r = run_one_trial(obj_name, use_oriented, trial_id)
                results.append(r)
                successes += r["success"]
                done += 1
            strat_name = "oriented" if use_oriented else "fixed"
            print(f"{obj_name:10s} {strat_name:10s} {successes}/{n_trials} success  "
                  f"({done}/{total} trials, {time.time()-t0:.0f}s elapsed)")

    tag = f"{strategy_arg}_{obj_filter}_{start_trial}-{start_trial + n_trials - 1}"
    out_path = RESULTS_PATH_TEMPLATE.format(tag=tag)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
