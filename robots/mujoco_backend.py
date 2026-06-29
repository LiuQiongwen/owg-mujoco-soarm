"""MujocoBackend — RobotBackend implementation wrapping EnvironmentSoArm.

The planner (tango_robot/ui.py) and benchmark (benchmark/runner.py) continue to
use EnvironmentSoArm directly and are entirely unaffected by this module.

Trajectory recording
--------------------
    env    = EnvironmentSoArm(vis=False)
    env.load_obj("YcbBanana", ...)
    backend  = MujocoBackend(env)
    recorder = TrajectoryRecorder()
    backend.attach_recorder(recorder)

    action = GraspAction(eef_pos=np.array([0.0, -0.42, 0.82]),
                         yaw=0.0, opening_m=0.08, obj_height=0.05)
    result = backend.execute_grasp(action)   # recorder auto-managed
    result.trajectory.save("trajs/banana_42.json")

Step hook
---------
    The trajectory recorder is driven by a single hook in
    EnvironmentSoArm.step_simulation() (the leaf physics step).  That one-line
    addition in env_soarm.py is the only change required to the existing codebase.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cameras.base import RGBDFrame
from datasets.episode import GraspAction
from robots.base import GraspResult, RobotBackend
from robots.trajectory import TrajectoryRecorder

from tango_robot.env_soarm import (
    EnvironmentSoArm,
    GRIP_CLOSED,
    GRIP_OPEN,
)


class MujocoBackend(RobotBackend):
    """RobotBackend adapter wrapping EnvironmentSoArm.

    Exposes the standardised RobotBackend interface so trajectory recording
    and replay work identically across simulation and hardware backends.

    Parameters
    ----------
    env : EnvironmentSoArm
        Already-constructed MuJoCo environment (objects may be loaded before
        or after construction; execute_grasp always uses env.obj_ids at call
        time).
    recorder : TrajectoryRecorder | None
        Optional recorder.  Attach / detach at any time via attach_recorder()
        and detach_recorder().  When a recorder is active every physics step
        during execute_grasp() is captured into the trajectory buffer.
    """

    # env.move_gripper normalises opening_m ∈ [0, _GRIP_TRAVEL_M] to an
    # internal angle ∈ [GRIP_CLOSED, GRIP_OPEN].  We invert that mapping in
    # get_gripper_opening() so the public API always uses metres.
    _GRIP_TRAVEL_M: float = 0.10

    def __init__(
        self,
        env:      EnvironmentSoArm,
        recorder: Optional[TrajectoryRecorder] = None,
    ):
        self._env      = env
        self._recorder = recorder

    # ── recorder management ───────────────────────────────────────────────────

    def attach_recorder(self, recorder: TrajectoryRecorder) -> None:
        """Attach a recorder.  Replaces any previously attached one."""
        self._recorder = recorder

    def detach_recorder(self) -> None:
        """Remove the recorder and clear any residual step hook on the env."""
        self._recorder = None
        self._env._step_hook = None

    # ── state reads ───────────────────────────────────────────────────────────

    def get_joint_positions(self) -> np.ndarray:
        """Current arm joint positions. Returns (5,) float32 in radians."""
        env = self._env
        return np.array(
            [env.data.qpos[adr] for adr in env._arm_qpos_adr],
            dtype=np.float32,
        )

    def get_gripper_opening(self) -> float:
        """Gripper opening in metres, in [0.0, _GRIP_TRAVEL_M=0.10].

        env.move_gripper(m) maps via:
            t     = clip(m / 0.10, 0, 1)
            angle = GRIP_CLOSED + t * (GRIP_OPEN - GRIP_CLOSED)

        We invert this to recover metres from the raw qpos angle.
        """
        env   = self._env
        angle = float(env.data.qpos[env._grip_qpos_adr])
        t     = np.clip(
            (angle - GRIP_CLOSED) / (GRIP_OPEN - GRIP_CLOSED),
            0.0, 1.0,
        )
        return float(t * self._GRIP_TRAVEL_M)

    def get_eef_pos(self) -> np.ndarray:
        """EEF position in robot-base frame. Returns (3,) float32."""
        return self._env._get_eef_pos().astype(np.float32)

    # ── motion commands ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Move arm to home configuration and open gripper."""
        self._env.reset_robot()

    def move_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        """Command arm joints to q (radians). q.shape must be (5,).

        blocking=True  runs 40 simulation steps so the arm settles before
                       returning (typical use during trajectory recording).
        blocking=False sets actuator ctrl targets and returns immediately
                       (used by TrajectoryReplayer which manages its own timing).
        """
        env = self._env
        for act_id, qi in zip(env._arm_act_ids, np.asarray(q, dtype=float)):
            env.data.ctrl[act_id] = float(qi)
        if blocking:
            env._steps(40)

    def set_gripper(self, opening: float, blocking: bool = True) -> None:
        """Command gripper to opening (metres), clipped to [0, 0.10].

        blocking=True  runs 40 simulation steps.
        blocking=False sets the actuator target only (env.move_gripper step=0
                       is a safe no-op for the physics loop).
        """
        opening = float(np.clip(opening, 0.0, self._GRIP_TRAVEL_M))
        # move_gripper(_, step=0) sets ctrl and calls _steps(0) → no physics
        self._env.move_gripper(opening, step=40 if blocking else 0)

    # ── high-level execution ──────────────────────────────────────────────────

    def execute_grasp(self, action: GraspAction) -> GraspResult:
        """Execute a full grasp cycle: pre-grasp → descend → close → lift.

        The active recorder (if any) is started just before the grasp and
        ended just after; the completed Trajectory is attached to the result.

        dz is the vertical displacement of the grasped object (positive = up).
        When env._execute_grasp returns a grasped_id that id is used; otherwise
        the object closest to the action target XY is used as a fallback.
        """
        env     = self._env
        obj_ids = list(env.obj_ids) if hasattr(env, 'obj_ids') else []

        # snapshot pre-grasp Z for every loaded object
        z_before: dict[int, float] = {}
        for oid in obj_ids:
            try:
                z_before[oid] = float(env.get_obj_pos(oid)[2])
            except Exception:
                pass

        # install step hook and start recorder
        if self._recorder is not None:
            self._recorder.begin(metadata={
                "backend":      "mujoco",
                "grasp_mode":   env.grasp_mode,
                "grasp_action": action.as_vector().tolist(),
            })
            self._env._step_hook = lambda: self._recorder.snap(self)

        try:
            success, grasped_id = env._execute_grasp(
                pos=(float(action.eef_pos[0]),
                     float(action.eef_pos[1]),
                     float(action.eef_pos[2])),
                roll=float(action.yaw),
                gripper_opening_length=float(action.opening_m),
                obj_height=float(action.obj_height),
            )
        finally:
            # always clear hook even if _execute_grasp raises
            self._env._step_hook = None

        eef_pos = self.get_eef_pos()

        # compute dz from the reported grasped object, falling back to nearest
        dz     = 0.0
        ref_id = grasped_id
        if ref_id is None and z_before:
            tx, ty = float(action.eef_pos[0]), float(action.eef_pos[1])
            try:
                ref_id = min(
                    z_before,
                    key=lambda oid: float(np.linalg.norm(
                        env.get_obj_pos(oid)[:2] - np.array([tx, ty])
                    )),
                )
            except Exception:
                pass
        if ref_id is not None and ref_id in z_before:
            try:
                dz = float(env.get_obj_pos(ref_id)[2]) - z_before[ref_id]
            except Exception:
                pass

        traj = None
        if self._recorder is not None:
            traj = self._recorder.end(success=bool(success), dz=dz)

        return GraspResult(
            success    = bool(success),
            dz         = dz,
            eef_pos    = eef_pos,
            trajectory = traj,
        )

    # ── perception ────────────────────────────────────────────────────────────

    def capture_frame(self) -> RGBDFrame:
        """Capture one overhead RGB-D frame from the MuJoCo renderer."""
        rgb, depth, _ = self._env._render_rgb_depth_seg()
        return RGBDFrame(
            rgb       = rgb.astype(np.uint8),
            depth     = depth.astype(np.float32),
            timestamp = time.time(),
            camera_id = "overhead",
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the MuJoCo renderer and viewer."""
        self._env.close()

    # ── pass-through helpers ──────────────────────────────────────────────────

    @property
    def env(self) -> EnvironmentSoArm:
        """Direct access to the underlying EnvironmentSoArm (e.g. for load_obj)."""
        return self._env

    @property
    def grasp_mode(self) -> str:
        return self._env.grasp_mode
