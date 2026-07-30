"""
Integrate tango_robot/piper_ik_check.py's DLS position-only IK into the
RoboSuite + Piper + real-YCB-object scene, replacing the placeholder
zero/random test actions with actual IK-driven reach verification.

Note on approach: RoboSuite's default OSC_POSE controller (impedance control
via operational-space dynamics) was tried first for closed-loop reaching, but
diverged on this custom, very lightweight Piper arm port (eef moved AWAY from
the commanded target and grew unstable regardless of gain -- likely an
operational-space mass-matrix conditioning issue with the ported link
inertials, not something to debug within this "minimal viable scene"
milestone). Instead, this script does the same thing piper_ik_check.py
already did successfully on the standalone scene: solve joint-space IK
directly against the compiled model (teleporting qpos + mj_forward, the way
piper_ik_check.py verifies reachability), just retargeted at RoboSuite's own
compiled sim/model instead of a separate standalone MjModel. This is the
faithful "integrate the IK check script" ask -- verifying reachability of the
new table+object RoboSuite scene -- without conflating it with a full
dynamic/torque-control grasp (a separate, larger task).
"""
import sys

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite.piper_lift_ycb import PiperLiftYCB

JOINTS = [f"robot0_joint{i}" for i in range(1, 7)]
EEF_SITE = "robot0_eef_site"
IK_DAMPING = 1e-3
IK_TOL = 5e-3


class RoboSuiteIKCheck:
    """Same DLS position-only IK as piper_ik_check.PiperIKCheck, retargeted
    at a live RoboSuite sim's model/data instead of a standalone MjModel."""

    def __init__(self, env):
        self.env = env
        self.model = env.sim.model
        self.data = env.sim.data
        self.qpos_adr = [self.model.joint(n).qposadr[0] for n in JOINTS]
        self.dof_adr = [self.model.joint(n).dofadr[0] for n in JOINTS]
        self.jnt_ids = [self.model.joint(n).id for n in JOINTS]
        self.eef_site_id = self.model.site(EEF_SITE).id

    def get_eef_pos(self):
        return self.data.site_xpos[self.eef_site_id].copy()

    def _ik_step(self, target_pos):
        jacp = np.zeros((3, self.model.nv))
        self.env.sim.forward()
        import mujoco
        mujoco.mj_jacSite(self.model._model, self.data._data, jacp, None, self.eef_site_id)
        J = jacp[:, self.dof_adr]
        err = target_pos - self.get_eef_pos()
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(3), err)
        for i, adr in enumerate(self.qpos_adr):
            lo, hi = self.model.jnt_range[self.jnt_ids[i]]
            self.data.qpos[adr] = np.clip(self.data.qpos[adr] + dq[i], lo, hi)
        self.env.sim.forward()
        return np.linalg.norm(err) < IK_TOL

    def solve_ik(self, target_pos, iters=1500, seed_qpos=None):
        if seed_qpos is not None:
            for adr, q in zip(self.qpos_adr, seed_qpos):
                self.data.qpos[adr] = q
        self.env.sim.forward()
        converged = False
        for _ in range(iters):
            if self._ik_step(target_pos):
                converged = True
                break
        pe = float(np.linalg.norm(self.get_eef_pos() - target_pos))
        return converged, pe

    def get_qpos(self):
        return np.array([self.data.qpos[a] for a in self.qpos_adr])


def main():
    ycb_object = sys.argv[1] if len(sys.argv) > 1 else "pear"
    env = PiperLiftYCB(
        robots="Piper", ycb_object=ycb_object,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
        camera_names="frontview", control_freq=20,
    )
    env.reset()
    obj_pos = env.sim.data.body_xpos[env.cube_body_id].copy()
    print(f"object ({ycb_object}) spawned at: {obj_pos.round(4)}")

    checker = RoboSuiteIKCheck(env)
    targets = {
        "above_object": obj_pos + np.array([0, 0, 0.10]),
        "grasp_height": obj_pos + np.array([0, 0, 0.015]),
        "lift_height":  obj_pos + np.array([0, 0, 0.20]),
    }

    # Solve each target from a fresh all-zero joint seed (same convention as
    # piper_ik_check.PiperIKCheck.solve_ik), not warm-started from the
    # previous target -- warm-starting was masking genuine convergence
    # failures by inheriting a lucky prior pose.
    print(f"{'target':<15} {'requested_xyz':<28} {'converged':<10} {'pos_err_cm'}")
    for name, pos in targets.items():
        converged, pe = checker.solve_ik(pos, seed_qpos=np.zeros(6))
        print(f"{name:<15} {str(pos.round(3)):<28} {str(converged):<10} {pe*100:.2f}")

    # Report the converged grasp-height pose's clearance to the object.
    # NOTE: this is a kinematic-only check (teleport qpos + mj_forward, no
    # mj_step dynamics) -- deliberately mirrors piper_ik_check.py's own
    # methodology. Calling env.sim.step() directly here (bypassing
    # env.step()/the OSC controller) was tried and is NOT valid: with no
    # actuator torque commanded, this light-mass arm has no gravity
    # compensation and free-falls into a QACC blowup within a few ms. Full
    # closed-loop dynamic holds must go through env.step(action).
    checker.solve_ik(targets["grasp_height"], seed_qpos=np.zeros(6))
    obj_pos_now = env.sim.data.body_xpos[env.cube_body_id].copy()
    eef_now = checker.get_eef_pos()
    print(f"grasp_height IK solution: eef={eef_now.round(4)} "
          f"obj={obj_pos_now.round(4)} gap_cm={np.linalg.norm(eef_now-obj_pos_now)*100:.2f}")

    img = env.sim.render(camera_name="frontview", width=640, height=480)[::-1]
    import imageio
    out_path = f"/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/piper_{ycb_object}_ik_check.png"
    imageio.imwrite(out_path, img)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
