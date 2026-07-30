"""Trajectory recording and replay for Piper sim-to-real transfer.

Piper-specific analog of robots/trajectory.py's TrajectoryRecorder/
TrajectoryReplayer (SO-ARM101) -- flagged as "the natural next piece" in
piper_real_backend.py's own docstring, not built until now. Follows the
SAME pattern deliberately, without importing robots/ (CLAUDE.md forbids
tango_robot/ importing robots/; piper_real_backend.py already established
this "same pattern, no cross-import, no shared ABC" convention because
Piper's 6-DoF/gripper conventions genuinely differ from SO-ARM101's 5-DoF
ones).

Workflow
--------
Simulation (record)
    recorder = PiperTrajectoryRecorder()
    recorder.begin(metadata={"obj_name": "cracker", "trial_id": 903})
    result = run_pick_and_place(env, "cracker", step_hook=recorder.snap)
    traj = recorder.end(success=result["success"], dist_to_tray=result["dist_to_tray"])
    traj.save("piper_trajs/cracker_903.json")

Hardware (replay) -- NOT hardware-tested, see piper_real_backend.py
    traj = PiperTrajectory.load("piper_trajs/cracker_903.json")
    replayer = PiperTrajectoryReplayer(speed=0.5)   # start slow on real hardware
    with PiperRealBackend() as backend:
        replayer.replay(traj, backend)

Recording boundary matches the causal-validity commit marker
--------------------------------------------------------------
    run_pick_and_place's step_hook is only invoked from inside move_to*
    functions, which are all called AFTER CAUSAL_VALIDITY_COMMIT_POINT()
    (see piper_pick_and_place.py) -- the initial 30-step settle loop
    (letting a just-spawned object's pose stabilize) is deliberately NOT
    recorded. This is the right boundary for a different reason here too:
    a trajectory meant for real-hardware replay should start from the
    actual grasp attempt, not from simulation-only spawn-settling that has
    no real-hardware equivalent.

GRIPPER UNIT MISMATCH -- RESOLVED (2026-07-22), was based on a stale assumption
--------------------------------------------------------------------------------
    This section previously claimed TrajectoryPoint.gripper stores the RAW
    +-1 action-space value (GRIPPER_OPEN=-1.0/GRIPPER_CLOSE=1.0). Checked
    directly against `env.sim.data.ctrl` at runtime (2026-07-22) and that
    claim was WRONG -- it described the *external* action interface
    (run_pick_and_place's action argument), not what actually lands in
    `env.sim.data.ctrl[-1]`, which `PiperTrajectoryRecorder.snap()`
    records. `piper_gripper.xml`'s finger actuators are POSITION
    actuators (ctrlrange -0.05 to -0.004 metres per finger, joint7/joint8)
    -- `env.sim.data.ctrl[-1]` is joint8's ABSOLUTE target position in
    metres, already resolved from the +-1 action by robosuite's
    PiperGripper.format_action (piper_gripper.py) before it ever reaches
    `ctrl`. Confirmed empirically: post-grasp `ctrl[-1] == -0.05` (the
    documented "closed" bound). This is exactly the quantity
    piper_gripper.py's own comment already flagged as fixed ("this value
    ... is an ABSOLUTE real-units joint position, not a normalized [-1,1]
    action") -- that fix just never got propagated to this file's
    docstring until now.

    Converting a recorded `gripper` value (single-finger qpos, metres, in
    [FINGER_QPOS_OPEN, FINGER_QPOS_CLOSED] = [-0.05, -0.004]) to
    PiperRealBackend's full-span-metres convention
    (GRIP_OPEN_M=0.12/GRIP_CLOSED_M=0.0) is now a straightforward linear
    map between two ALREADY-established calibration anchors (the XML's own
    ctrlrange bounds, and the empirically-measured true open span from
    piper_real_backend.py) -- see `finger_qpos_to_span_m()` below. No new
    hardware measurement was needed; the two facts just hadn't been
    connected before. `PiperTrajectoryReplayer.replay()` now performs this
    conversion instead of raising `NotImplementedError`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

JOINTS = [f"robot0_joint{i}" for i in range(1, 7)]
EEF_SITE = "robot0_eef_site"

# Per-finger position-actuator ctrlrange, piper_assets/piper_gripper.xml
# (joint7/joint8, coupled via equality constraint) -- FINGER_QPOS_OPEN is
# the "open" bound despite being the more-negative number (sign convention
# verified empirically, see piper_gripper.py's PiperGripper.format_action
# comment).
FINGER_QPOS_OPEN = -0.05
FINGER_QPOS_CLOSED = -0.004

# Empirically-measured full fingertip-to-fingertip span at FINGER_QPOS_OPEN,
# matching PiperRealBackend's GRIP_OPEN_M/GRIP_CLOSED_M convention
# (piper_real_backend.py -- "true open span measured at 12.01cm in sim").
GRIP_OPEN_M = 0.12
GRIP_CLOSED_M = 0.0


def finger_qpos_to_span_m(finger_qpos: float) -> float:
    """Convert a recorded single-finger qpos (metres, FINGER_QPOS_OPEN..
    FINGER_QPOS_CLOSED) to full gripper opening span (metres,
    GRIP_CLOSED_M..GRIP_OPEN_M) -- linear interpolation between the two
    already-established calibration anchors, see module docstring's
    "GRIPPER UNIT MISMATCH -- RESOLVED" section."""
    frac = (finger_qpos - FINGER_QPOS_CLOSED) / (FINGER_QPOS_OPEN - FINGER_QPOS_CLOSED)
    frac = min(max(frac, 0.0), 1.0)
    return GRIP_CLOSED_M + frac * (GRIP_OPEN_M - GRIP_CLOSED_M)


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class PiperTrajectoryPoint:
    """Robot state at a single sim step within a recorded grasp trajectory."""
    t:          float          # seconds elapsed since recording began (post-commit)
    joint_pos:  np.ndarray     # (6,) float32 -- joint1 ... joint6, radians
    gripper:    float          # single-finger qpos, metres, see finger_qpos_to_span_m()
    eef_pos:    np.ndarray     # (3,) float32 -- XYZ, world frame (sim)
    phase:      str = ""       # solve_and_move's phase name at capture time
                                # (2026-07-18, added for cr_cfm/data.py -- lets
                                # segment extraction use the real phase boundary
                                # instead of a z-height heuristic). "" for points
                                # captured before this field existed, or if the
                                # step_hook passed wasn't phase-aware.

    def to_dict(self) -> dict:
        return {
            "t":         self.t,
            "joint_pos": self.joint_pos.tolist(),
            "gripper":   self.gripper,
            "eef_pos":   self.eef_pos.tolist(),
            "phase":     self.phase,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PiperTrajectoryPoint":
        return cls(
            t         = float(d["t"]),
            joint_pos = np.array(d["joint_pos"], dtype=np.float32),
            gripper   = float(d["gripper"]),
            eef_pos   = np.array(d["eef_pos"],   dtype=np.float32),
            phase     = str(d.get("phase", "")),
        )


@dataclass
class PiperTrajectory:
    """Ordered sequence of PiperTrajectoryPoints covering one grasp execution.

    metadata keys (all optional, populated by PiperTrajectoryRecorder.end):
        obj_name, trial_id, seed, backend ("piper_mujoco" | "piper_real"),
        timestamp, success, dist_to_tray
    """
    points:   List[PiperTrajectoryPoint]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].t - self.points[0].t

    @property
    def n_points(self) -> int:
        return len(self.points)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "points":   [p.to_dict() for p in self.points],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PiperTrajectory":
        payload = json.loads(Path(path).read_text())
        return cls(
            metadata = payload.get("metadata", {}),
            points   = [PiperTrajectoryPoint.from_dict(d) for d in payload["points"]],
        )

    def __repr__(self) -> str:
        obj    = self.metadata.get("obj_name", "?")
        ok     = self.metadata.get("success")
        ok_str = {True: "ok", False: "fail", None: "?"}[ok]
        return (
            f"PiperTrajectory(n={self.n_points}, "
            f"duration={self.duration:.2f}s, obj={obj}, success={ok_str})"
        )


# ── recorder ─────────────────────────────────────────────────────────────────

class PiperTrajectoryRecorder:
    """Accumulates per-step robot state during a Piper grasp execution.

    Pass recorder.snap directly as run_pick_and_place's step_hook=
    parameter -- snap's signature (env) -> None matches step_hook's
    expected call convention exactly.

    Lifecycle
    ---------
        recorder.begin(metadata={...})              # before run_pick_and_place
        result = run_pick_and_place(..., step_hook=recorder.snap)
        traj = recorder.end(success=..., dist_to_tray=...)
    """

    def __init__(self):
        self._points: List[PiperTrajectoryPoint] = []
        self._t0:     float = 0.0
        self._meta:   Dict[str, Any] = {}
        self._active: bool = False
        self._phase:  str = ""

    def begin(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._points = []
        self._t0     = time.time()
        self._meta   = dict(metadata or {})
        self._meta.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._meta.setdefault("backend", "piper_mujoco")
        self._active = True
        self._phase  = ""

    def set_phase(self, name: str) -> None:
        """Called by piper_pick_and_place.py's solve_and_move (if the
        step_hook passed has this method -- checked via hasattr, not
        required) right before each phase's motion begins. Purely a
        side-channel for phase labeling; does not change step_hook's own
        (env) -> None call signature used everywhere else."""
        self._phase = name

    def snap(self, env) -> None:
        """Capture current sim state. Call signature matches
        piper_pick_and_place.py's step_hook convention exactly -- pass this
        method directly as step_hook= without a wrapper.
        No-op when not active (safe to pass unconditionally)."""
        if not self._active:
            return
        qpos_adr = [env.sim.model.joint(n).qposadr[0] for n in JOINTS]
        joint_pos = np.array([env.sim.data.qpos[a] for a in qpos_adr], dtype=np.float32)
        # Gripper: last action-space component actually applied is not
        # directly readable from sim state (data.ctrl reflects the
        # post-controller-scaling value, not the pre-scaling action) --
        # recording the actuator ctrl signal itself as the closest available
        # proxy. See module docstring's GRIPPER UNIT MISMATCH section.
        gripper_ctrl = float(env.sim.data.ctrl[-1]) if env.sim.data.ctrl.size else 0.0
        eef_pos = np.array(env.sim.data.site(EEF_SITE).xpos, dtype=np.float32).copy()
        self._points.append(PiperTrajectoryPoint(
            t         = time.time() - self._t0,
            joint_pos = joint_pos,
            gripper   = gripper_ctrl,
            eef_pos   = eef_pos,
            phase     = self._phase,
        ))

    def end(
        self,
        success:      Optional[bool]  = None,
        dist_to_tray: Optional[float] = None,
    ) -> PiperTrajectory:
        self._active = False
        if success is not None:
            self._meta["success"] = success
        if dist_to_tray is not None:
            self._meta["dist_to_tray"] = float(dist_to_tray)
        return PiperTrajectory(points=list(self._points), metadata=dict(self._meta))

    @property
    def is_active(self) -> bool:
        return self._active

    def __len__(self) -> int:
        return len(self._points)


# ── replayer ─────────────────────────────────────────────────────────────────

class PiperTrajectoryReplayer:
    """Replays a recorded PiperTrajectory on any backend exposing
    reset(), move_joints(q, blocking), set_gripper(opening, blocking) --
    i.e. PiperRealBackend's actual public interface (piper_real_backend.py).

    NOT hardware-tested: the gripper unit conversion is now resolved (see
    module docstring's "GRIPPER UNIT MISMATCH -- RESOLVED" section) but
    has never driven a real gripper -- gripper_agnostic=True remains
    available to replay arm motion only, useful as a first, lower-risk
    hardware test before trusting the gripper conversion on a real unit.
    """

    def __init__(self, speed: float = 1.0):
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed!r}")
        self.speed = speed

    def replay(
        self,
        traj: PiperTrajectory,
        backend,
        on_step: Optional[Callable[[int, PiperTrajectoryPoint], None]] = None,
        gripper_agnostic: bool = False,
    ) -> None:
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
            if not gripper_agnostic:
                backend.set_gripper(finger_qpos_to_span_m(pt.gripper), blocking=False)

            if on_step is not None:
                on_step(idx, pt)

    def replay_from_file(
        self,
        path: str | Path,
        backend,
        on_step: Optional[Callable[[int, PiperTrajectoryPoint], None]] = None,
        gripper_agnostic: bool = False,
    ) -> PiperTrajectory:
        traj = PiperTrajectory.load(path)
        self.replay(traj, backend, on_step=on_step, gripper_agnostic=gripper_agnostic)
        return traj
