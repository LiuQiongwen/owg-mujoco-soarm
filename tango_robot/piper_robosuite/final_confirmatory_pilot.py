"""
Final confirmatory pilot for the T-RO paper's headline results table.

Frozen final baseline config: oriented grasp on, gripper-controller
double-scaling fix in place (piper_controller_config.py), interpolated
descend, pre-close refresh, multi-seed IK -- no compliant_descend,
force_compliant_descend, or pre_narrow_descend flags (all four tested and
closed negative/null, see piper_robosuite/README.md and IDEA_REPORT.md).

Uses a clean, previously-unused trial_id range (5000+) so this table is not
entangled with any of this session's earlier diagnostic/pilot batches
(400-419, 400-849, 900-926, 1000-1039).

Usage (chunked to fit under background command timeouts, ~19s/trial):
  conda run -n tango python3 -m tango_robot.piper_robosuite.final_confirmatory_pilot <object> <n_trials> <start_trial>
"""
import json
import sys
import time

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite.piper_pick_and_place import run_pick_and_place

RESULTS_PATH_TEMPLATE = "/lena/projects/OWG-main/tango_robot/piper_robosuite/final_pilot_{obj}_{start}-{end}.json"


def run_one_trial(obj_name, trial_id):
    np.random.seed(trial_id)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=[obj_name],
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    result = run_pick_and_place(env, obj_name, use_oriented_grasp=True, verbose=False)
    result["trial_id"] = trial_id
    env.close()
    return result


def main():
    obj_name = sys.argv[1]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    start_trial = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    trial_ids = range(start_trial, start_trial + n_trials)

    results = []
    t0 = time.time()
    successes = 0
    for trial_id in trial_ids:
        r = run_one_trial(obj_name, trial_id)
        results.append(r)
        successes += r["success"]
        print(f"trial {trial_id}: success={r['success']} dist_to_tray={r['dist_to_tray']:.3f}")

    print(f"\n{obj_name}: {successes}/{n_trials} success ({100*successes/n_trials:.0f}%), "
          f"{time.time()-t0:.0f}s elapsed")

    out_path = RESULTS_PATH_TEMPLATE.format(obj=obj_name, start=start_trial, end=start_trial + n_trials - 1)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
