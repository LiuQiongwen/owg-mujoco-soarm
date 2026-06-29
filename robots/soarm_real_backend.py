"""SOARMRealBackend — RobotBackend for the physical SO-ARM101 over Feetech serial bus.

Hardware requirements
---------------------
    - SO-ARM101 follower arm connected via USB-to-serial (e.g. /dev/ttyUSB0)
    - lerobot installed in the active Python environment:
          pip install lerobot        # or: conda install -c conda-forge lerobot
    - scservo_sdk installed (pulled in by lerobot's feetech extras)
    - Robot calibration performed with: lerobot-calibrate --robot.type=so101_follower

Coordinate conventions
-----------------------
    Joint positions
        MuJoCo uses radians.  The Feetech bus (lerobot, DEGREES mode) uses
        degrees centred at the calibration midpoint, which should coincide with
        the URDF joint zero.  Conversion: degrees = radians × 180 / π.

    Gripper
        MuJoCo: opening_m ∈ [0.0, 0.10] metres (0 = closed, 0.10 = open).
        Feetech: RANGE_0_100 (0 = closed, 100 = open after calibration).
        Conversion: lerobot_val = clip(opening_m / 0.10 × 100, 0, 100).

    EEF position
        The real hardware has no direct EEF readback.  Forward kinematics
        is NOT implemented here; get_eef_pos() returns the last value
        written via set_eef_pos(), which replay scripts supply via the
        on_step callback.  Callers that need live FK should subclass and
        override get_eef_pos() using a URDF kinematics solver.

Typical sim-to-real workflow
-----------------------------
    # 1. Record trajectory in simulation (see scripts/record_trajectory.py)
    traj = Trajectory.load("trajs/banana_42.json")

    # 2. Replay on hardware
    cam     = RealSenseCamera()
    backend = SOARMRealBackend(port="/dev/ttyUSB0", camera=cam)
    backend.connect()

    replayer = TrajectoryReplayer(speed=0.7)
    replayer.replay(
        traj, backend,
        on_step=lambda i, pt: backend.set_eef_pos(pt.eef_pos),
    )

    backend.close()

execute_grasp() is intentionally not implemented
-------------------------------------------------
    Online IK on real hardware is outside this module's scope.  Direct grasp
    execution on hardware should go through TrajectoryReplayer so that the
    motion profile is identical to the validated simulation trajectory.
    Call replayer.replay(traj, backend) instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cameras.base import CameraBase, RGBDFrame
from datasets.episode import GraspAction
from robots.base import GraspResult, RobotBackend

# ── lerobot availability guard ────────────────────────────────────────────────

try:
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    _LEROBOT_AVAILABLE = True
except ImportError:
    _LEROBOT_AVAILABLE = False

# ── SO-ARM101 motor configuration ─────────────────────────────────────────────
# Joint order matches tango_robot/env_soarm.py ARM_JOINTS for direct array indexing.

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_NAME    = "gripper"

# Motor IDs as wired on the default SO-ARM101 follower arm.
# Verify against your physical wiring if IDs differ.
_MOTOR_IDS = {
    "shoulder_pan":  1,
    "shoulder_lift": 2,
    "elbow_flex":    3,
    "wrist_flex":    4,
    "wrist_roll":    5,
    "gripper":       6,
}

# Home position in degrees, derived from MuJoCo HOME_QPOS (radians):
#   [0.0, -0.4, 0.8, -0.4, 0.0] × 180/π = [0.0, -22.9, 45.8, -22.9, 0.0]
# Adjust if the physical arm's calibration zero differs from the URDF zero.
HOME_DEG = np.degrees(np.array([0.0, -0.4, 0.8, -0.4, 0.0], dtype=float))

# Gripper home: fully open (100 in RANGE_0_100 mode).
HOME_GRIPPER = 100.0

# Gripper travel range in metres — must match MujocoBackend._GRIP_TRAVEL_M.
GRIP_TRAVEL_M = 0.10

# Settling time when blocking=True.
_ARM_SETTLE_S     = 1.0   # seconds to wait after sending joint goal
_GRIPPER_SETTLE_S = 0.5


class SOARMRealBackend(RobotBackend):
    """RobotBackend for the physical SO-ARM101 via Feetech serial bus.

    Parameters
    ----------
    port : str
        Serial port, e.g. "/dev/ttyUSB0" on Linux or "COM3" on Windows.
    camera : CameraBase
        Overhead camera backend (typically RealSenseCamera from cameras/).
    calibration : dict | Path | None
        Motor calibration.  Either a pre-loaded dict or a path to a JSON
        file saved by lerobot-calibrate.  When None the bus will raise at
        connect() time if the servos are uncalibrated.
    max_relative_target : float | None
        Safety clamp: maximum allowed position change per command (degrees).
        None disables clamping.  Recommended: 30–60° for initial testing.
    """

    def __init__(
        self,
        port:                str,
        camera:              CameraBase,
        calibration:         Optional[Any]   = None,   # dict | Path | None
        max_relative_target: Optional[float] = None,
    ):
        if not _LEROBOT_AVAILABLE:
            raise ImportError(
                "lerobot is required for SOARMRealBackend. "
                "Install it with: pip install lerobot"
            )

        self._port   = port
        self._camera = camera
        self._max_relative_target = max_relative_target

        # Load calibration from file if a Path was given
        _cal = _load_calibration(calibration) if isinstance(calibration, (str, Path)) else calibration

        # Build FeetechMotorsBus with DEGREES mode for arm, RANGE_0_100 for gripper
        motors: dict[str, Motor] = {
            name: Motor(_MOTOR_IDS[name], "sts3215", MotorNormMode.DEGREES)
            for name in ARM_JOINT_NAMES
        }
        motors[GRIPPER_NAME] = Motor(_MOTOR_IDS[GRIPPER_NAME], "sts3215", MotorNormMode.RANGE_0_100)

        self._bus = FeetechMotorsBus(
            port=self._port,
            motors=motors,
            calibration=_cal,
        )

        # EEF position cache — updated externally (e.g. on_step in replay scripts).
        # Not computed from FK; returned as-is by get_eef_pos().
        self._eef_pos: np.ndarray = np.zeros(3, dtype=np.float32)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the serial port and handshake with all motors.

        Call this before any state read or motion command.
        Raises ConnectionError if any motor does not respond.
        """
        self._bus.connect()

    def close(self) -> None:
        """Disable torque and close the serial port."""
        try:
            self._bus.disconnect(disable_torque=True)
        except Exception:
            pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # ── state reads ───────────────────────────────────────────────────────────

    def get_joint_positions(self) -> np.ndarray:
        """Read current arm joint positions via sync_read. Returns (5,) float32 in radians.

        Converts from lerobot DEGREES (calibrated, centred at URDF zero) to radians.
        """
        obs = self._bus.sync_read("Present_Position", ARM_JOINT_NAMES, normalize=True)
        deg = np.array([obs[n] for n in ARM_JOINT_NAMES], dtype=np.float64)
        return np.radians(deg).astype(np.float32)

    def get_gripper_opening(self) -> float:
        """Read current gripper opening. Returns metres in [0.0, GRIP_TRAVEL_M=0.10].

        Converts from lerobot RANGE_0_100 (0=closed, 100=open) to metres.
        """
        val = self._bus.read("Present_Position", GRIPPER_NAME, normalize=True)
        return float(np.clip(val / 100.0 * GRIP_TRAVEL_M, 0.0, GRIP_TRAVEL_M))

    def get_eef_pos(self) -> np.ndarray:
        """Return cached EEF position. Returns (3,) float32 in robot-base frame.

        The real arm has no direct EEF sensor.  Update this cache by calling
        set_eef_pos(pt.eef_pos) in the on_step callback of TrajectoryReplayer.
        """
        return self._eef_pos.copy()

    def set_eef_pos(self, pos: np.ndarray) -> None:
        """Update the cached EEF position (called by replay scripts via on_step)."""
        self._eef_pos = np.asarray(pos, dtype=np.float32).copy()

    # ── motion commands ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Move arm to home configuration and open gripper.

        Blocks for _ARM_SETTLE_S + _GRIPPER_SETTLE_S seconds to allow the
        servos to reach the target positions.
        """
        goal = {name: float(deg) for name, deg in zip(ARM_JOINT_NAMES, HOME_DEG)}
        goal[GRIPPER_NAME] = HOME_GRIPPER

        if self._max_relative_target is not None:
            goal = _apply_relative_clamp(
                self._bus, ARM_JOINT_NAMES, goal, self._max_relative_target
            )

        self._bus.sync_write("Goal_Position", goal)
        time.sleep(_ARM_SETTLE_S + _GRIPPER_SETTLE_S)
        self._eef_pos = np.zeros(3, dtype=np.float32)

    def move_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        """Command arm joints to q (radians). q.shape must be (5,).

        blocking=True  sleeps _ARM_SETTLE_S to let servos reach the goal.
        blocking=False sends the goal command and returns immediately.

        Note: hardware servos are always moving asynchronously after this call.
        TrajectoryReplayer manages inter-point timing externally via wall-clock
        sleep, so use blocking=False during replay.
        """
        q_deg  = np.degrees(np.asarray(q, dtype=float))
        goal   = {name: float(d) for name, d in zip(ARM_JOINT_NAMES, q_deg)}

        if self._max_relative_target is not None:
            goal = _apply_relative_clamp(
                self._bus, ARM_JOINT_NAMES, goal, self._max_relative_target
            )

        self._bus.sync_write("Goal_Position", goal)
        if blocking:
            time.sleep(_ARM_SETTLE_S)

    def set_gripper(self, opening: float, blocking: bool = True) -> None:
        """Command gripper to opening (metres). Clipped to [0.0, GRIP_TRAVEL_M].

        blocking=True  sleeps _GRIPPER_SETTLE_S.
        blocking=False sends the command and returns immediately.
        """
        opening  = float(np.clip(opening, 0.0, GRIP_TRAVEL_M))
        lerobot_val = opening / GRIP_TRAVEL_M * 100.0
        self._bus.sync_write("Goal_Position", {GRIPPER_NAME: lerobot_val})
        if blocking:
            time.sleep(_GRIPPER_SETTLE_S)

    # ── high-level execution ──────────────────────────────────────────────────

    def execute_grasp(self, action: GraspAction) -> GraspResult:
        """Not implemented for direct online use.

        Online IK on real hardware is outside this module's scope.
        Use TrajectoryReplayer to replay a validated simulation trajectory:

            replayer = TrajectoryReplayer(speed=0.8)
            replayer.replay(
                traj, backend,
                on_step=lambda i, pt: backend.set_eef_pos(pt.eef_pos),
            )
        """
        raise NotImplementedError(
            "SOARMRealBackend.execute_grasp() is not implemented. "
            "Use robots.trajectory.TrajectoryReplayer to replay a trajectory "
            "recorded in simulation. See the module docstring for an example."
        )

    # ── perception ────────────────────────────────────────────────────────────

    def capture_frame(self) -> RGBDFrame:
        """Capture one overhead RGB-D frame from the attached camera."""
        return self._camera.capture()

    # ── pass-through helpers ──────────────────────────────────────────────────

    @property
    def bus(self) -> "FeetechMotorsBus":
        """Direct access to the FeetechMotorsBus (for advanced configuration)."""
        return self._bus

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_calibration(path: str | Path) -> dict:
    """Load a lerobot calibration JSON and return a MotorCalibration dict."""
    import json
    data = json.loads(Path(path).read_text())
    cal  = {}
    for motor_name, entry in data.items():
        cal[motor_name] = MotorCalibration(
            id           = entry["id"],
            drive_mode   = entry.get("drive_mode", 0),
            homing_offset= entry.get("homing_offset", 0),
            range_min    = entry["range_min"],
            range_max    = entry["range_max"],
        )
    return cal


def _apply_relative_clamp(
    bus,
    joint_names: list[str],
    goal:        dict[str, float],
    max_delta:   float,
) -> dict[str, float]:
    """Clamp each joint's goal to ±max_delta degrees from its current position.

    Reads present positions once and clips each target so the arm cannot move
    more than max_delta degrees in a single command.  This prevents large jumps
    if the trajectory is loaded starting from an unexpected arm configuration.
    """
    present = bus.sync_read("Present_Position", joint_names, normalize=True)
    clamped = dict(goal)
    for name in joint_names:
        cur = present.get(name)
        if cur is not None:
            clamped[name] = float(
                np.clip(goal[name], cur - max_delta, cur + max_delta)
            )
    return clamped
