# -*- coding: utf-8 -*-
"""
SO-ARM101 adapter: joint reset, IK, grasp execution inside a MuJoCo scene.
Works with MujocoGraspEnv — call attach(env) before use.
"""
import numpy as np
import mujoco
from typing import List, Optional

ARM_JOINTS = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]
GRIP_JOINT = "gripper"
EEF_SITE   = "gripperframe"

# Safe home position for SO-101 over the table
HOME_QPOS = np.array([0.0, -0.5, 1.0, -0.5, 0.0])
GRIP_OPEN  = 1.0
GRIP_CLOSE = 0.05

IK_ITERS = 200
IK_DAMP  = 0.05
IK_TOL   = 5e-4


class SoArm101Adapter:
    """
    Thin adapter around MujocoGraspEnv that exposes robot-level primitives.

    Usage:
        env = MujocoGraspEnv(...)
        robot = SoArm101Adapter()
        robot.attach(env)
        robot.joint_reset()
        robot.execute_grasp({"position": ..., "rpy": ..., "width": ...})
    """

    def __init__(self):
        self._model: Optional[mujoco.MjModel] = None
        self._data:  Optional[mujoco.MjData]  = None

        self._arm_qpos_adr: List[int] = []
        self._arm_jdof_adr: List[int] = []
        self._arm_act_ids:  List[int] = []
        self._grip_act:     int = -1
        self._grip_adr:     int = -1
        self._eef_site:     int = -1
        self._jnt_ranges: np.ndarray = np.zeros((5, 2))

    # ── attachment ────────────────────────────────────────────────────────────

    def attach(self, env) -> "SoArm101Adapter":
        """Bind this adapter to a MujocoGraspEnv (or any env with .model/.data)."""
        self._model = env.model
        self._data  = env.data

        self._arm_qpos_adr = [self._model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self._arm_jdof_adr = [self._model.joint(n).dofadr[0]  for n in ARM_JOINTS]
        self._arm_act_ids  = [self._model.actuator(n).id       for n in ARM_JOINTS]
        self._grip_act     = self._model.actuator(GRIP_JOINT).id
        self._grip_adr     = self._model.joint(GRIP_JOINT).qposadr[0]
        self._eef_site     = self._model.site(EEF_SITE).id
        self._jnt_ranges   = np.array([
            self._model.jnt_range[self._model.joint(n).id] for n in ARM_JOINTS
        ])
        return self

    def _check(self):
        if self._model is None:
            raise RuntimeError("SoArm101Adapter.attach(env) must be called first.")

    # ── joint control ─────────────────────────────────────────────────────────

    def joint_reset(self, qpos: Optional[np.ndarray] = None):
        """Reset arm to home (or given) joint positions and settle."""
        self._check()
        q = HOME_QPOS if qpos is None else np.asarray(qpos)
        for adr, qi in zip(self._arm_qpos_adr, q):
            self._data.qpos[adr] = qi
        self._data.qpos[self._grip_adr] = GRIP_OPEN
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)
        for act, qi in zip(self._arm_act_ids, q):
            self._data.ctrl[act] = qi
        self._data.ctrl[self._grip_act] = GRIP_OPEN
        self._step(80)

    def set_gripper(self, angle: float, steps: int = 60):
        """Set gripper joint angle (GRIP_CLOSE ≈ 0.05, GRIP_OPEN ≈ 1.0)."""
        self._check()
        self._data.ctrl[self._grip_act] = float(angle)
        self._step(steps)

    def get_joint_positions(self) -> np.ndarray:
        return np.array([self._data.qpos[a] for a in self._arm_qpos_adr])

    def get_eef_position(self) -> np.ndarray:
        self._check()
        return self._data.site_xpos[self._eef_site].copy()

    # ── IK ───────────────────────────────────────────────────────────────────

    def ik(self, target_pos: np.ndarray, max_iters: int = IK_ITERS,
           damping: float = IK_DAMP) -> bool:
        """
        Solve IK via damped least-squares Jacobian.
        Returns True if converged within tolerance.
        """
        self._check()
        target = np.asarray(target_pos, dtype=float)
        for _ in range(max_iters):
            jacp = np.zeros((3, self._model.nv))
            mujoco.mj_jacSite(self._model, self._data, jacp, None, self._eef_site)
            J   = jacp[:, self._arm_jdof_adr]
            err = target - self._data.site_xpos[self._eef_site]
            if np.linalg.norm(err) < IK_TOL:
                return True
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
            for i, adr in enumerate(self._arm_qpos_adr):
                lo, hi = self._jnt_ranges[i]
                self._data.qpos[adr] = np.clip(self._data.qpos[adr] + dq[i], lo, hi)
            mujoco.mj_forward(self._model, self._data)
        return np.linalg.norm(target - self._data.site_xpos[self._eef_site]) < 5e-3

    def move_to(self, target_pos: np.ndarray, sim_steps: int = 80) -> bool:
        """IK + actuator update + simulate."""
        ok = self.ik(target_pos)
        for act, adr in zip(self._arm_act_ids, self._arm_qpos_adr):
            self._data.ctrl[act] = self._data.qpos[adr]
        self._step(sim_steps)
        return ok

    # ── grasp execution ───────────────────────────────────────────────────────

    def execute_grasp(self, grasp: dict,
                      table_z: float = 0.0,
                      pre_height: float = 0.18,
                      lift_height: float = 0.25) -> bool:
        """
        Execute a single grasp.

        grasp: {"position": [x,y,z], "rpy": [r,p,y], "width": w, "score": s}
        Returns True if the object was lifted (caller checks contact).
        """
        self._check()
        pos   = np.asarray(grasp["position"], dtype=float)
        width = float(grasp.get("width", 0.04))
        # rpy ignored for now (IK is position-only; orientation via config)

        # 1. open gripper
        self.set_gripper(GRIP_OPEN, steps=40)

        # 2. pre-grasp above target
        pre = pos.copy(); pre[2] = table_z + pre_height
        self.move_to(pre, sim_steps=80)

        # 3. descend
        target_z = max(pos[2], table_z + 0.005)
        self.move_to(np.array([pos[0], pos[1], target_z]), sim_steps=80)

        # 4. close gripper
        self.set_gripper(GRIP_CLOSE, steps=80)

        # 5. lift
        lift = pos.copy(); lift[2] = table_z + lift_height
        self.move_to(lift, sim_steps=80)

        # success = EEF is above lift threshold
        eef_z = self.get_eef_position()[2]
        return eef_z > table_z + lift_height * 0.7

    def _step(self, n: int):
        for _ in range(n):
            mujoco.mj_step(self._model, self._data)
