"""RobotBackend — abstract interface shared by MuJoCo sim and SO-ARM101 hardware.

Implementors
------------
    MujocoBackend      robots/mujoco_backend.py
    SOARMRealBackend   robots/soarm_real_backend.py

Coordinate convention
---------------------
    All positions are in robot-base frame (metres).
    All angles are radians.
    joint_positions follows ARM_JOINTS order:
        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]

Decoupling guarantee
--------------------
    The planner (owg/policy.py, tango_robot/ui.py) and benchmark
    (benchmark/runner.py) are NOT expected to import from this module.
    They continue to use EnvironmentSoArm directly.  This interface lives
    below the planner: it abstracts only the grasp-execution and
    state-reading primitives needed by MujocoBackend, SOARMRealBackend,
    and the trajectory recorder/replayer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from cameras.base import RGBDFrame
from datasets.episode import GraspAction

if TYPE_CHECKING:
    # Imported at type-check time only to avoid a circular dependency
    # (trajectory.py uses RobotBackend duck-typing, not a direct import).
    from robots.trajectory import Trajectory


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class GraspResult:
    """Outcome of a single execute_grasp() call.

    trajectory is None unless a TrajectoryRecorder was attached to the
    backend before the call (see MujocoBackend.attach_recorder).
    """
    success:    bool
    dz:         float              # vertical displacement of object (m); >0 = lifted
    eef_pos:    np.ndarray         # (3,) float32 — EEF position at end of lift
    trajectory: Optional[Trajectory] = field(default=None, repr=False)


# ── abstract backend ──────────────────────────────────────────────────────────

class RobotBackend(ABC):
    """Minimal robot interface for grasp execution and state reading.

    Both simulation and hardware backends implement this contract so that
    trajectory recording (TrajectoryRecorder) and replay (TrajectoryReplayer)
    work identically across backends.

    Abstract methods must be overridden.  The concrete helper get_state_vector()
    is available on all backends without override.
    """

    # ── state reads ───────────────────────────────────────────────────────────

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Current arm joint positions. Returns (5,) float32 in radians."""
        ...

    @abstractmethod
    def get_gripper_opening(self) -> float:
        """Current gripper opening in metres [GRIP_CLOSED=0.05, GRIP_OPEN=1.0]."""
        ...

    @abstractmethod
    def get_eef_pos(self) -> np.ndarray:
        """EEF position in robot-base frame. Returns (3,) float32."""
        ...

    # ── motion commands ───────────────────────────────────────────────────────

    @abstractmethod
    def reset(self) -> None:
        """Move arm to home configuration and open gripper."""
        ...

    @abstractmethod
    def move_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        """Command arm joints to q (radians). q.shape must be (5,).

        blocking=True  — waits until motion settles (sim: _steps; hw: poll).
        blocking=False — sends command and returns immediately.
        """
        ...

    @abstractmethod
    def set_gripper(self, opening: float, blocking: bool = True) -> None:
        """Command gripper to opening (metres).

        Values are clipped to [GRIP_CLOSED, GRIP_OPEN] by the implementor.
        blocking semantics match move_joints.
        """
        ...

    # ── high-level execution ──────────────────────────────────────────────────

    @abstractmethod
    def execute_grasp(self, action: GraspAction) -> GraspResult:
        """Execute a full grasp cycle: pre-grasp → descend → close → lift.

        action.eef_pos   — target XYZ for descent
        action.yaw       — wrist-roll / jaw orientation (radians)
        action.opening_m — gripper opening at approach
        action.obj_height — estimated object height for IK

        Returns GraspResult; result.trajectory is populated only when a
        TrajectoryRecorder is attached (MujocoBackend.attach_recorder).
        """
        ...

    # ── perception ────────────────────────────────────────────────────────────

    @abstractmethod
    def capture_frame(self) -> RGBDFrame:
        """Capture one overhead RGB-D frame. Must be non-blocking."""
        ...

    # ── lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def close(self) -> None:
        """Release resources (MuJoCo renderer, hardware connection, etc.)."""
        ...

    # ── convenience ───────────────────────────────────────────────────────────

    def get_state_vector(self) -> np.ndarray:
        """Flat state snapshot: joint(5) + gripper(1) + eef(3) = 9-dim float32.

        Consistent with the 10-dim obs vector in SOARM101Robot (which adds
        gripper is_closed as a second gripper channel).  Useful for logging
        and world-model feature extraction without importing higher-level types.
        """
        return np.concatenate([
            self.get_joint_positions(),
            [self.get_gripper_opening()],
            self.get_eef_pos(),
        ]).astype(np.float32)
