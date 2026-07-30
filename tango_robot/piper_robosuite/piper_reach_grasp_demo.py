"""
Piper + RoboSuite: replace the placeholder zero/random test actions with a
real IK-driven reach, following the same "close the loop on end-effector
position error" idea as tango_robot/piper_ik_check.py -- except here the
low-level IK is RoboSuite's own OSC_POSE controller (Jacobian-based
operational-space control on `robot0_eef_site`), so we only need to feed it
proportional position-error deltas instead of re-solving DLS IK by hand
against a standalone MjModel.

Sequence: approach above the object -> descend to grasp height -> close
gripper -> lift. Prints eef/object position error each phase so convergence
can be checked the same way piper_ik_check.py reported pos_err_cm.

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_reach_grasp_demo [pear|can]
"""
import sys

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa: registers Piper/PiperGripper
from tango_robot.piper_robosuite.piper_lift_ycb import PiperLiftYCB

APPROACH_HEIGHT = 0.10   # metres above object CoM before descending
GRASP_HEIGHT_OFFSET = 0.015
KP = 4.0                 # proportional gain, position error (m) -> normalized OSC delta
MAX_STEP_NORM = 1.0
POS_TOL = 0.01


def goto(env, target_pos, gripper_action, max_steps=150, tol=POS_TOL):
    eef_site_id = env.sim.model.site("robot0_eef_site").id
    for step in range(max_steps):
        eef_pos = env.sim.data.site_xpos[eef_site_id].copy()
        err = target_pos - eef_pos
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            return True, step, err_norm
        delta = np.clip(KP * err, -MAX_STEP_NORM, MAX_STEP_NORM)
        action = np.concatenate([delta, np.zeros(3), [gripper_action]])
        env.step(action)
    eef_pos = env.sim.data.site_xpos[eef_site_id].copy()
    return False, max_steps, float(np.linalg.norm(target_pos - eef_pos))


def main():
    ycb_object = sys.argv[1] if len(sys.argv) > 1 else "pear"
    env = PiperLiftYCB(
        robots="Piper",
        ycb_object=ycb_object,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names="frontview",
        control_freq=20,
    )
    env.reset()
    obj_pos = env.sim.data.body_xpos[env.cube_body_id].copy()
    print(f"object ({ycb_object}) spawned at: {obj_pos.round(4)}")

    above = obj_pos + np.array([0, 0, APPROACH_HEIGHT])
    ok, steps, err = goto(env, above, gripper_action=-1.0)
    print(f"[approach above] converged={ok} steps={steps} pos_err_cm={err*100:.2f}")

    grasp_pos = obj_pos + np.array([0, 0, GRASP_HEIGHT_OFFSET])
    ok, steps, err = goto(env, grasp_pos, gripper_action=-1.0)
    print(f"[descend to grasp height] converged={ok} steps={steps} pos_err_cm={err*100:.2f}")

    for _ in range(30):
        env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    print("[close gripper] done")

    lift_pos = grasp_pos + np.array([0, 0, 0.15])
    ok, steps, err = goto(env, lift_pos, gripper_action=1.0, max_steps=100)
    final_obj_pos = env.sim.data.body_xpos[env.cube_body_id].copy()
    lifted = final_obj_pos[2] > obj_pos[2] + 0.04
    print(f"[lift] arm converged={ok} pos_err_cm={err*100:.2f}  "
          f"object_z: {obj_pos[2]:.3f} -> {final_obj_pos[2]:.3f}  lifted={lifted}")

    img = env.sim.render(camera_name="frontview", width=640, height=480)[::-1]
    import imageio
    out_path = f"/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/piper_{ycb_object}_reach_grasp.png"
    imageio.imwrite(out_path, img)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
