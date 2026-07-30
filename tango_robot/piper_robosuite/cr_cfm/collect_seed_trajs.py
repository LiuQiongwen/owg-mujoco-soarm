"""CR-CFM Stage A: collect seed trajectories for flow-matching training.

Records full pick-and-place trajectories via the existing
PiperTrajectoryRecorder (fresh env per trial -- matches
piper_experiment_runner.py's established convention, NOT the reused-env
pattern that caused an invalid-pairing bug earlier this session, see
RULED_OUT_METHODS.md rows 12-13). Keeps only SUCCESSFUL trajectories as the
flow-matching target (x_1) distribution -- synthetic drift is injected at
training time (data.py) rather than relying on naturally-occurring failure
trajectories to supply "bad" examples.
"""
import sys

import numpy as np

from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite.piper_pick_and_place import run_pick_and_place
from tango_robot.piper_robosuite.piper_trajectory import PiperTrajectoryRecorder

OUT_DIR = "/lena/projects/OWG-main/tango_robot/piper_robosuite/piper_trajs"


def collect(obj_name, trial_ids):
    n_saved = 0
    for trial_id in trial_ids:
        np.random.seed(trial_id)
        env = PiperMultiObjectScene(
            robots="Piper", ycb_objects=[obj_name],
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
            control_freq=20,
        )
        env.reset()
        recorder = PiperTrajectoryRecorder()
        recorder.begin(metadata={"obj_name": obj_name, "trial_id": trial_id})
        result = run_pick_and_place(env, obj_name, use_oriented_grasp=True, verbose=False,
                                     wrist_friendly_orientation=True,
                                     step_hook=recorder.snap)
        traj = recorder.end(success=result["success"], dist_to_tray=result["dist_to_tray"])
        status = "OK" if result["success"] else "fail"
        print(f"{obj_name} trial={trial_id} {status} n_points={traj.n_points} dist={result['dist_to_tray']:.3f}")
        if result["success"]:
            traj.save(f"{OUT_DIR}/{obj_name}_{trial_id}.json")
            n_saved += 1
    return n_saved


def main():
    obj_name = sys.argv[1] if len(sys.argv) > 1 else "cracker"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    n_saved = collect(obj_name, range(start, start + n))
    print(f"saved {n_saved}/{n} successful trajectories for {obj_name}")


if __name__ == "__main__":
    main()
