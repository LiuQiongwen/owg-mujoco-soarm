"""Trajectory recording and replay for sim-to-real transfer.

Workflow
--------
Simulation (record)
    recorder = TrajectoryRecorder()
    # MujocoBackend calls recorder.snap(self) inside its _steps hook
    traj = recorder.end(success=True, dz=0.12)
    traj.save("trajs/banana_seed42.json")

Hardware (replay)
    traj = Trajectory.load("trajs/banana_seed42.json")
    replayer = TrajectoryReplayer(speed=0.8)   # 80 % of sim speed
    replayer.replay(traj, real_backend)

Data format
-----------
    Trajectory is saved as JSON (human-readable).  Arrays are stored as plain
    Python lists so the files are inspectable with any text editor.
    Shape / dtype are recovered on load: joint_pos → (5,) float32,
    eef_pos → (3,) float32.

Circular-import note
--------------------
    This module does NOT import from robots.base.  RobotBackend is referenced
    only via duck-typing (snap / replay accept any object with the matching
    method signatures).  This keeps the import graph acyclic:
        base.py  →  trajectory.py  (TYPE_CHECKING only, no runtime import)
        trajectory.py  →  (nothing from robots/)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    """Robot state at a single timestep within a grasp trajectory."""
    t:          float          # seconds elapsed since trajectory start
    joint_pos:  np.ndarray     # (5,) float32 — shoulder_pan … wrist_roll
    gripper:    float          # metres: GRIP_CLOSED(0.05) … GRIP_OPEN(1.0)
    eef_pos:    np.ndarray     # (3,) float32 — XYZ in robot-base frame

    def to_dict(self) -> dict:
        return {
            "t":         self.t,
            "joint_pos": self.joint_pos.tolist(),
            "gripper":   self.gripper,
            "eef_pos":   self.eef_pos.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrajectoryPoint:
        return cls(
            t         = float(d["t"]),
            joint_pos = np.array(d["joint_pos"], dtype=np.float32),
            gripper   = float(d["gripper"]),
            eef_pos   = np.array(d["eef_pos"],   dtype=np.float32),
        )


@dataclass
class Trajectory:
    """Ordered sequence of TrajectoryPoints covering one grasp execution.

    metadata keys (all optional, populated by TrajectoryRecorder.end):
        action      — GraspAction.as_vector() serialised as list
        obj_name    — e.g. "Banana"
        seed        — integer random seed used in sim
        backend     — "mujoco" | "soarm_real"
        timestamp   — ISO-8601 string (set automatically by recorder)
        success     — bool outcome
        dz          — float lift height in metres
    """
    points:   List[TrajectoryPoint]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def duration(self) -> float:
        """Total elapsed time from first to last point (seconds)."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].t - self.points[0].t

    @property
    def n_points(self) -> int:
        return len(self.points)

    # ── serialization ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Serialise to JSON.  Parent directories are created if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "points":   [p.to_dict() for p in self.points],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        """Deserialise from a JSON file written by save()."""
        payload = json.loads(Path(path).read_text())
        return cls(
            metadata = payload.get("metadata", {}),
            points   = [TrajectoryPoint.from_dict(d) for d in payload["points"]],
        )

    def __repr__(self) -> str:
        obj   = self.metadata.get("obj_name", "?")
        ok    = self.metadata.get("success")
        ok_str = {True: "ok", False: "fail", None: "?"}[ok]
        return (
            f"Trajectory(n={self.n_points}, "
            f"duration={self.duration:.2f}s, "
            f"obj={obj}, success={ok_str})"
        )


# ── recorder ─────────────────────────────────────────────────────────────────

class TrajectoryRecorder:
    """Accumulates per-step robot state during a grasp execution.

    Designed to be called from MujocoBackend's internal step loop via a hook,
    but works with any object that exposes:
        get_joint_positions() -> np.ndarray  (5,)
        get_gripper_opening() -> float
        get_eef_pos()         -> np.ndarray  (3,)

    Lifecycle
    ---------
        recorder.begin(metadata={...})   # called before grasp starts
        recorder.snap(backend)           # called once per sim/hw step
        traj = recorder.end(success, dz) # called after grasp completes
    """

    def __init__(self):
        self._points: List[TrajectoryPoint] = []
        self._t0:     float = 0.0
        self._meta:   Dict[str, Any] = {}
        self._active: bool = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def begin(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start a new recording session, discarding any previous buffer."""
        self._points = []
        self._t0     = time.time()
        self._meta   = dict(metadata or {})
        self._meta.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._active = True

    def snap(self, backend) -> None:
        """Capture current robot state into the trajectory buffer.

        backend: duck-typed — must have get_joint_positions(),
                 get_gripper_opening(), get_eef_pos().
        No-op when not active (safe to call unconditionally in step loops).
        """
        if not self._active:
            return
        self._points.append(TrajectoryPoint(
            t         = time.time() - self._t0,
            joint_pos = np.asarray(backend.get_joint_positions(),
                                   dtype=np.float32).copy(),
            gripper   = float(backend.get_gripper_opening()),
            eef_pos   = np.asarray(backend.get_eef_pos(),
                                   dtype=np.float32).copy(),
        ))

    def end(
        self,
        success: Optional[bool]  = None,
        dz:      Optional[float] = None,
    ) -> Trajectory:
        """Finalise the recording and return the completed Trajectory.

        success / dz are written into metadata when provided.
        Calling end() twice without a begin() in between returns an empty
        Trajectory (points=[]) — harmless.
        """
        self._active = False
        if success is not None:
            self._meta["success"] = success
        if dz is not None:
            self._meta["dz"] = float(dz)
        return Trajectory(points=list(self._points), metadata=dict(self._meta))

    # ── inspection ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    def __len__(self) -> int:
        """Number of points accumulated so far."""
        return len(self._points)


# ── replayer ─────────────────────────────────────────────────────────────────

class TrajectoryReplayer:
    """Replays a recorded Trajectory on any RobotBackend.

    Timing
    ------
    Each point's .t value encodes seconds elapsed since trajectory start.
    The replayer converts these to wall-clock targets and sleeps as needed,
    so the motion speed matches the original recording when speed=1.0.

    Parameters
    ----------
    speed : float
        Playback speed multiplier (>1.0 = faster, <1.0 = slower).
        Default 1.0 reproduces the original timing.

    on_step : callable(idx, TrajectoryPoint) | None
        Optional callback invoked after each command is sent.
        Useful for live monitoring or recording a second trajectory.
    """

    def __init__(self, speed: float = 1.0):
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed!r}")
        self.speed = speed

    def replay(
        self,
        traj:    Trajectory,
        backend,
        on_step: Optional[Callable[[int, TrajectoryPoint], None]] = None,
    ) -> None:
        """Replay traj on backend.

        backend: duck-typed — must have reset(), move_joints(), set_gripper().
        Calls backend.reset() before the first point so the arm is at home.
        """
        if not traj.points:
            return

        backend.reset()

        t_wall_start  = time.time()
        t_traj_origin = traj.points[0].t

        for idx, pt in enumerate(traj.points):
            t_target = t_wall_start + (pt.t - t_traj_origin) / self.speed
            wait     = t_target - time.time()
            if wait > 0:
                time.sleep(wait)

            backend.move_joints(pt.joint_pos, blocking=False)
            backend.set_gripper(pt.gripper,   blocking=False)

            if on_step is not None:
                on_step(idx, pt)

    def replay_from_file(
        self,
        path:    str | Path,
        backend,
        on_step: Optional[Callable[[int, TrajectoryPoint], None]] = None,
    ) -> Trajectory:
        """Load a Trajectory from path and replay it.  Returns the Trajectory."""
        traj = Trajectory.load(path)
        self.replay(traj, backend, on_step=on_step)
        return traj
