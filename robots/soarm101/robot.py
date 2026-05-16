"""SOARM101Robot — standardized robot interface inspired by HuggingFace LeRobot.

Wraps EnvironmentSoArm and exposes:
  - Standardized joint/gripper state (JointState, GripperState)
  - Standardized action vector (GraspAction)
  - Camera via CameraBase (simulated or real)
  - Episode-level data collection via EpisodeWriter

This class does NOT retrain policies, redesign the benchmark, or replace the
existing MuJoCo pipeline.  It is a thin adapter on top of env_soarm.py.

Usage (simulation):
    from owg_robot.env_soarm import EnvironmentSoArm
    from robots.soarm101.robot import SOARM101Robot

    env    = EnvironmentSoArm(obj_names=["YcbBanana"], vis=False)
    robot  = SOARM101Robot(env=env)

    state  = robot.get_state()
    frame  = robot.camera.capture()
    action = GraspAction(eef_pos=np.array([0.0, -0.4, 0.85]),
                         yaw=0.0, opening_m=0.08, obj_height=0.05)
    result = robot.execute_grasp(action, obj_id=1, obj_name="Banana")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from datasets.episode import (
    Episode, EpisodeStep,
    GraspAction, GripperState, JointState,
)
from cameras.base import CameraBase, RGBDFrame
from cameras.simulated import SimulatedCamera

# env_soarm constants we re-export for convenience
from owg_robot.env_soarm import (
    ARM_JOINTS,
    GRIP_OPEN,
    GRIP_CLOSED,
    GRASP_MODE_PHYSICS,
    GRASP_MODE_DEMO_ATTACH,
)


# ── robot state snapshot ──────────────────────────────────────────────────────

@dataclass
class RobotState:
    """Full robot state at one instant."""
    joint_state:   JointState
    gripper_state: GripperState
    eef_pos:       np.ndarray          # (3,) float32 in robot-base frame
    timestamp:     float

    def as_obs_vector(self) -> np.ndarray:
        """10-dim flat obs: joint(5) + gripper(2) + eef(3)."""
        return np.concatenate([
            self.joint_state.as_vector(),
            self.gripper_state.as_vector(),
            self.eef_pos,
        ]).astype(np.float32)


# ── main class ────────────────────────────────────────────────────────────────

class SOARM101Robot:
    """
    Standardized interface for SO-ARM101 in MuJoCo simulation.

    The class can also be subclassed for real hardware by overriding
    `get_state()`, `move_joints()`, and `move_gripper()`.

    Parameters
    ----------
    env : EnvironmentSoArm
        Already-constructed MuJoCo environment.
    camera : CameraBase | None
        Camera backend.  Defaults to SimulatedCamera wrapping env.
    writer : EpisodeWriter | None
        Dataset writer.  If None, no data is saved.
    record_steps : bool
        If True and writer is set, log per-step state to EpisodeStep list.
    record_frames : bool
        If True (and record_steps=True), attach RGB-D frames to each step.
    """

    # SO-ARM101 gripper limits (metres)
    GRIPPER_OPEN_M   = GRIP_OPEN    # 1.0 — treated as fully open for state reporting
    GRIPPER_CLOSED_M = GRIP_CLOSED  # 0.05

    def __init__(
        self,
        env,
        camera:        Optional[CameraBase]    = None,
        writer=None,   # Optional[EpisodeWriter] — avoid circular import
        record_steps:  bool = False,
        record_frames: bool = False,
    ):
        self._env          = env
        self.camera: CameraBase = camera if camera is not None else SimulatedCamera(env)
        self._writer       = writer
        self._record_steps = record_steps
        self._record_frames = record_frames

        # active episode buffer
        self._episode: Optional[Episode] = None
        self._step_idx: int = 0

    # ── state ─────────────────────────────────────────────────────────────────

    def get_state(self) -> RobotState:
        """Read current joint positions, gripper opening, and EEF position."""
        env = self._env

        # arm joint positions from qpos
        joint_pos = np.array(
            [env.data.qpos[adr] for adr in env._arm_qpos_adr],
            dtype=np.float32,
        )

        # gripper opening: read the gripper actuator position
        try:
            grip_adr = env.model.joint(env._GRIP_JOINT_NAME if hasattr(env, "_GRIP_JOINT_NAME")
                                        else "gripper").qposadr[0]
            opening  = float(env.data.qpos[grip_adr])
        except Exception:
            opening  = float(GRIP_OPEN)

        is_closed = opening < 0.15   # heuristic: below ~15 % open → considered closed

        eef_pos = env.data.site_xpos[env._eef_site_id].copy().astype(np.float32)

        return RobotState(
            joint_state   = JointState(positions=joint_pos),
            gripper_state = GripperState(opening_m=opening, is_closed=is_closed),
            eef_pos       = eef_pos,
            timestamp     = time.time(),
        )

    # ── motion ────────────────────────────────────────────────────────────────

    def move_to_home(self) -> None:
        """Move arm to home configuration and open gripper."""
        self._env.reset_robot()

    def move_eef(self, target_pos: np.ndarray, max_step: int = 200) -> bool:
        """
        Move EEF to target_pos in robot-base frame.

        Returns True if target reached within IK tolerance.
        """
        return self._env.move_ee(target_pos, max_step=max_step)

    def open_gripper(self, step: int = 80) -> None:
        self._env.move_gripper(self.GRIPPER_OPEN_M, step=step)

    def close_gripper(self, step: int = 100) -> bool:
        return self._env.auto_close_gripper(step=step)

    def move_joints(self, joint_positions: np.ndarray) -> None:
        """
        Command joints directly (radians).

        joint_positions: (5,) matching ARM_JOINTS order.
        """
        env = self._env
        for act_id, q in zip(env._arm_act_ids, joint_positions):
            env.data.ctrl[act_id] = float(q)
        env._steps(40)

    # ── observation ───────────────────────────────────────────────────────────

    def get_obs(self, pointcloud: bool = True) -> dict:
        """Return full observation dict from the environment."""
        return self._env.get_obs(pointcloud=pointcloud)

    def capture_frame(self) -> RGBDFrame:
        """Capture one RGB-D frame from the robot's camera."""
        return self.camera.capture()

    # ── episode recording ─────────────────────────────────────────────────────

    def begin_episode(
        self,
        episode_id:    int,
        obj_name:      str,
        obj_id:        int,
        yaw_mode:      str  = "xyz_only",
        execution_mode: str = GRASP_MODE_PHYSICS,
    ) -> Episode:
        self._episode = Episode(
            episode_id=episode_id,
            obj_name=obj_name,
            obj_id=obj_id,
            yaw_mode=yaw_mode,
            execution_mode=execution_mode,
        )
        self._step_idx = 0
        return self._episode

    def log_step(
        self,
        action: Optional[GraspAction] = None,
        success: Optional[bool] = None,
    ) -> Optional[EpisodeStep]:
        """Snapshot current state into the active episode's step list."""
        if not self._record_steps or self._episode is None:
            return None

        state = self.get_state()
        frame_rgb   = None
        frame_depth = None
        if self._record_frames:
            frame = self.camera.capture()
            frame_rgb   = frame.rgb
            frame_depth = frame.depth

        step = EpisodeStep(
            step_idx      = self._step_idx,
            timestamp     = time.time(),
            joint_state   = state.joint_state,
            gripper_state = state.gripper_state,
            eef_pos       = state.eef_pos,
            rgb           = frame_rgb,
            depth         = frame_depth,
            action        = action,
            success       = success,
        )
        self._episode.steps.append(step)
        self._step_idx += 1
        return step

    def end_episode(
        self,
        success:   bool,
        dz:        float,
        fell_off:  bool,
        obj_pos_before:  Optional[np.ndarray] = None,
        obj_quat_before: Optional[np.ndarray] = None,
        pc_stats_before: Optional[np.ndarray] = None,
        depth_mean_before: Optional[float]    = None,
        obj_pos_after:   Optional[np.ndarray] = None,
        obj_quat_after:  Optional[np.ndarray] = None,
        grasp_action:    Optional[GraspAction] = None,
    ) -> Optional[Episode]:
        if self._episode is None:
            return None

        ep = self._episode
        ep.success            = success
        ep.dz                 = dz
        ep.fell_off           = fell_off
        ep.obj_pos_before     = obj_pos_before
        ep.obj_quat_before    = obj_quat_before
        ep.pc_stats_before    = pc_stats_before
        ep.depth_mean_before  = depth_mean_before
        ep.obj_pos_after      = obj_pos_after
        ep.obj_quat_after     = obj_quat_after
        ep.grasp_action       = grasp_action

        if self._writer is not None:
            self._writer.write(ep)

        self._episode = None
        return ep

    # ── grasp execution ───────────────────────────────────────────────────────

    def execute_grasp(
        self,
        action:    GraspAction,
        obj_id:    int,
        obj_name:  str,
        yaw_mode:  str = "xyz_only",
    ) -> Dict[str, Any]:
        """
        Execute a grasp and return an outcome dict.

        Delegates to EnvironmentSoArm.pick_obj_by_id() / _execute_grasp()
        via the existing API so all grasp logic stays in env_soarm.py.

        Returns
        -------
        dict with keys: success, dz, fell_off, obj_pos_before, obj_pos_after
        """
        env    = self._env
        pos    = action.eef_pos
        roll   = action.yaw
        opening = action.opening_m
        height  = action.obj_height

        obs_before = env.get_obs(pointcloud=True)
        obj_pos_b  = env.get_obj_pos(obj_id)
        obj_quat_b = env.get_obj_orn(obj_id)

        from data.transition_logger import compute_pc_stats
        pc_stats = compute_pc_stats(obs_before, obj_id)
        depth_b  = float(obs_before["depth"].mean())

        # Execute via the existing env API
        success = env.pick_obj_by_id(
            obj_id, pos=list(pos), roll=roll,
            gripper_opening_length=opening,
            obj_height=height,
        )

        obj_pos_a  = env.get_obj_pos(obj_id)
        obj_quat_a = env.get_obj_orn(obj_id)
        dz         = float(obj_pos_a[2] - obj_pos_b[2])
        fell_off   = bool(obj_pos_a[2] < 0.70)

        return {
            "success":           bool(success),
            "dz":                dz,
            "fell_off":          fell_off,
            "obj_pos_before":    obj_pos_b,
            "obj_quat_before":   obj_quat_b,
            "pc_stats_before":   pc_stats,
            "depth_mean_before": depth_b,
            "obj_pos_after":     obj_pos_a,
            "obj_quat_after":    obj_quat_a,
        }

    # ── info ──────────────────────────────────────────────────────────────────

    @property
    def joint_names(self) -> List[str]:
        return list(ARM_JOINTS)

    @property
    def execution_mode(self) -> str:
        return self._env.grasp_mode

    def __repr__(self) -> str:
        return (
            f"SOARM101Robot("
            f"mode={self.execution_mode}, "
            f"camera={self.camera.camera_id}, "
            f"record_steps={self._record_steps})"
        )
