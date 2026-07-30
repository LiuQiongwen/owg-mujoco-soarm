"""
Consensus vs best-IK candidate selection pilot -- replicates the SO-ARM101
finding (median-of-pool beats lowest-IK-error) on the Piper embodiment. See
/home/lina/.claude/plans/floating-crunching-yeti.md for the full design
rationale (why the noise pool is injected explicitly, why these 3 objects,
why this is the right next Piper experiment for the T-RO paper).

Design, mirroring piper_experiment_runner.py's already-validated conventions:
  - Objects: pear (known-reliable baseline), mustard + cracker (known
    precision-limited failures -- the case consensus selection should help
    most, if it helps at all). Not can/drill/clamp: their failures are
    unrelated to candidate pose noise (width or unverified shape issues),
    so this experiment can't show a signal there either way.
  - use_oriented_grasp=True throughout (already established as not
    significantly different from fixed, so held constant here rather than
    adding a third crossed dimension).
  - Selection: "best" vs "consensus", paired by trial_id via a per-trial
    seeded candidate-pool RNG (np.random.default_rng(trial_id)) -- SAME
    pool drawn for both strategies at a given trial_id, so the comparison
    isolates the selection RULE, not which poses happened to be sampled.
    (Separately, np.random.seed(trial_id) still controls the object's
    spawn/orientation draw, exactly as in piper_experiment_runner.py.)

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_consensus_experiment_runner [n_trials] [best|consensus|both] [object] [start_trial] [kinematic|perception]

noise_model (last arg, default "kinematic"): "perception" perturbs the
TRUE object pose (position + full 3D orientation) and recomputes the
grasp target per noisy estimate -- a more faithful reproduction of
SO-ARM101's actual candidate-diversity source than "kinematic"'s direct
jitter on an already-computed grasp target. See
piper_candidate_selection.sample_perception_noisy_candidates and the
README entry explaining why "kinematic" didn't replicate the SO-ARM101
finding.
"""
import json
import sys
import time

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite.piper_pick_and_place import run_pick_and_place

OBJECTS = ["pear", "mustard", "cracker"]
RESULTS_PATH_TEMPLATE = "/lena/projects/OWG-main/tango_robot/piper_robosuite/consensus_results_{tag}.json"


def run_one_trial(obj_name, selection, trial_id, noise_model="kinematic"):
    np.random.seed(trial_id)
    scene_objects = ["pear", "can", "mustard"] if obj_name in ("pear", "can", "mustard") else [obj_name]
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    candidate_rng = np.random.default_rng(trial_id)  # same pool for best and consensus at this trial_id
    result = run_pick_and_place(env, obj_name, use_oriented_grasp=True, verbose=False,
                                 candidate_selection=selection, noise_model=noise_model, rng=candidate_rng)
    result["trial_id"] = trial_id
    result["scene_objects"] = scene_objects
    return result


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    selection_arg = sys.argv[2] if len(sys.argv) > 2 else "both"
    obj_filter = sys.argv[3] if len(sys.argv) > 3 else "all"
    start_trial = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    noise_model = sys.argv[5] if len(sys.argv) > 5 else "kinematic"
    selections = {"best": ["best"], "consensus": ["consensus"], "both": ["best", "consensus"]}[selection_arg]
    objects = OBJECTS if obj_filter == "all" else [obj_filter]
    trial_ids = range(start_trial, start_trial + n_trials)

    results = []
    t0 = time.time()
    total = len(objects) * len(selections) * n_trials
    done = 0
    for obj_name in objects:
        for selection in selections:
            successes = 0
            for trial_id in trial_ids:
                r = run_one_trial(obj_name, selection, trial_id, noise_model=noise_model)
                results.append(r)
                successes += r["success"]
                done += 1
            print(f"{obj_name:10s} {selection:10s} {successes}/{n_trials} success  "
                  f"({done}/{total} trials, {time.time()-t0:.0f}s elapsed)")

    tag = f"{noise_model}_{selection_arg}_{obj_filter}_{start_trial}-{start_trial + n_trials - 1}"
    out_path = RESULTS_PATH_TEMPLATE.format(tag=tag)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
