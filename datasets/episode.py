"""Episode and EpisodeStep dataclasses — canonical in-memory format.

Matches the Transition layout in data/transition_logger.py but adds:
  - per-step RGB-D frames (optional, large)
  - joint_state / gripper_state vectors
  - action vector in standardized form
  - episode-level metadata
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── joint / gripper state ─────────────────────────────────────────────────────

@dataclass
class JointState:
    """Arm joint positions for SO-ARM101 (5 joints)."""
    positions:    np.ndarray          # (5,) float32 radians: shoulder_pan … wrist_roll
    velocities:   Optional[np.ndarray] = None   # (5,) float32 — None if unavailable
    names: tuple = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

    def as_vector(self) -> np.ndarray:
        return self.positions.astype(np.float32)


@dataclass
class GripperState:
    """SO-ARM101 gripper state (single DOF)."""
    opening_m:  float                 # current opening in metres [0, 0.10]
    is_closed:  bool = False          # True when actively commanded closed

    def as_vector(self) -> np.ndarray:
        return np.array([self.opening_m, float(self.is_closed)], dtype=np.float32)


# ── action ────────────────────────────────────────────────────────────────────

@dataclass
class GraspAction:
    """Standardized grasp action vector for SO-ARM101."""
    eef_pos:        np.ndarray        # (3,) float32  target XYZ in robot-base frame (m)
    yaw:            float             # wrist-roll angle (radians)
    opening_m:      float             # gripper opening width (m)
    obj_height:     float             # estimated object height used by IK (m)

    def as_vector(self) -> np.ndarray:
        return np.array(
            [*self.eef_pos, self.yaw, self.opening_m, self.obj_height],
            dtype=np.float32,
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "GraspAction":
        v = np.asarray(v, dtype=np.float32)
        return cls(
            eef_pos=v[:3],
            yaw=float(v[3]),
            opening_m=float(v[4]),
            obj_height=float(v[5]),
        )


# ── step ─────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeStep:
    """One timestep of a grasp episode."""
    step_idx:     int
    timestamp:    float

    # observation
    joint_state:  JointState
    gripper_state: GripperState
    eef_pos:      np.ndarray          # (3,) float32 EEF XYZ in robot-base frame
    obj_pos:      Optional[np.ndarray] = None   # (3,) float32 — None if unknown
    obj_quat:     Optional[np.ndarray] = None   # (4,) float32 [w,x,y,z]
    rgb:          Optional[np.ndarray] = None   # (H,W,3) uint8 — stored if record_frames=True
    depth:        Optional[np.ndarray] = None   # (H,W) float32

    # action (set after the step is taken)
    action:       Optional[GraspAction] = None

    # outcome (set at episode end)
    success:      Optional[bool] = None
    dz:           Optional[float] = None
    fell_off:     Optional[bool] = None

    def obs_vector(self) -> np.ndarray:
        """Flat observation: joint(5) + gripper(2) + eef(3) = 10 dims."""
        return np.concatenate([
            self.joint_state.as_vector(),
            self.gripper_state.as_vector(),
            self.eef_pos,
        ]).astype(np.float32)


# ── episode ───────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """Complete grasp episode."""
    episode_id:    int
    obj_name:      str
    obj_id:        int
    yaw_mode:      str
    execution_mode: str               # "physics" | "demo_attach"
    timestamp:     float = field(default_factory=time.time)

    # pre-grasp snapshot (mirrors Transition)
    obj_pos_before:  Optional[np.ndarray] = None    # (3,)
    obj_quat_before: Optional[np.ndarray] = None    # (4,)
    pc_stats_before: Optional[np.ndarray] = None    # (9,)
    depth_mean_before: Optional[float]   = None

    # action
    grasp_action:  Optional[GraspAction] = None

    # post-grasp snapshot
    obj_pos_after:  Optional[np.ndarray] = None
    obj_quat_after: Optional[np.ndarray] = None

    # outcome
    success:       Optional[bool]  = None
    dz:            Optional[float] = None
    fell_off:      Optional[bool]  = None

    # per-step log (populated when record_steps=True)
    steps: List[EpisodeStep] = field(default_factory=list)

    # arbitrary extras (e.g. VLM prompt / scores)
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "episode_id":     self.episode_id,
            "obj_name":       self.obj_name,
            "yaw_mode":       self.yaw_mode,
            "execution_mode": self.execution_mode,
            "success":        self.success,
            "dz":             self.dz,
            "fell_off":       self.fell_off,
            "timestamp":      self.timestamp,
        }
