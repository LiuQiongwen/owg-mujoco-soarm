"""PiperRealBackend -- interface skeleton for the physical AgileX Piper arm.

API VERIFIED AGAINST piper_sdk 0.6.1, STILL NOT HARDWARE-TESTED (2026-07-22):
`piper_sdk` is now installed (`pip install piper_sdk`, v0.6.1) and every
method call below has been checked against the actual installed package's
source (`piper_sdk/interface/piper_interface_v2.py`) and its official
`piper_sdk/demo/V2/*.py` examples -- unit conversions, method names, and
parameter orders are no longer guesses. What remains unverified is
everything that can only be confirmed with a physically connected arm
(timing, settle behavior, whether the documented units match this specific
unit's firmware). Do not skip a cautious first connection just because the
API layer is now solid.

*** CRITICAL DISCREPANCY FOUND (2026-07-22), NOT YET RESOLVED ***
The entire CR-CFM/wrist-fix investigation (`cr_cfm/IMPROVEMENT_PLAN.md`)
assumed joint6 (wrist roll)'s hardware limit is +-3.14 rad (+-180 deg),
matching `robot_arm.xml`'s MuJoCo joint range. `piper_sdk`'s own `JointCtrl`
docstring lists joint6's default limit as **+-2.09439 rad (+-120 deg)** --
a full 60 degrees narrower. This is a *software* limit (CAN ID 0x474,
`MotorAngleLimitMaxSpdSet`, persisted to flash, not a live-adjustable
value) and the SDK's own demo (`piper_set_motor_angle_limit.py`) widens it
to only +-170 deg (1700 * 0.1 deg) as an example -- still short of the
sim's assumed +-180 deg, and the demo's number is illustrative, not a
claim about the true mechanical maximum. Before any real-hardware wrist-fix
test: (a) query the arm's actual configured joint6 limit
(`SearchMotorMaxAngleSpdAccLimit` / `GetAllMotorAngleLimitMaxSpd`), (b)
do not assume the sim's `pick_wrist_friendly_orientation(joint6_limit=3.14)`
threshold transfers unchanged -- recompute against whatever the real
arm's actual configured limit turns out to be, since a mismatch here
would silently invalidate the fix's real-world applicability.

Before the first real connection:
  1. `pip install piper_sdk` (done, 2026-07-22 -- v0.6.1).
  2. Resolve the joint6-limit discrepancy above.
  3. Start with `max_relative_target` set very small (a few degrees) and
     a conservative `move_spd_rate_ctrl` (this file now defaults to 30%,
     not the demo's 100%), matching SOARMRealBackend's own
     `max_relative_target` safety-clamp convention
     (robots/soarm_real_backend.py) -- do not skip this the way that
     module's own docstring warns against.

Why this lives in tango_robot/piper_robosuite/, not robots/
-------------------------------------------------------------
    robots/base.py's RobotBackend ABC is explicitly scoped to SO-ARM101
    (5-DOF ARM_JOINTS, its own coordinate/gripper conventions baked into
    the docstring) and CLAUDE.md states tango_robot/ must not import
    robots/. Piper is 6-DoF with its own already-established conventions
    (READY_QPOS, GRIPPER_OPEN/GRIPPER_CLOSE in piper_pick_and_place.py) --
    forcing it into the SO-ARM101-shaped ABC would misrepresent both. This
    module follows the SAME safety-conscious PATTERN (relative-delta
    clamping, execute_grasp deliberately NOT implemented for live IK on
    hardware, trajectory-replay-only workflow) without inheriting the
    SO-ARM101-specific interface.

execute_grasp() is intentionally not implemented, same reasoning as
SOARMRealBackend
-------------------------------------------------------------------
    Online IK on real hardware is outside this module's scope. Real
    execution should replay a qpos sequence already validated in
    simulation (piper_pick_and_place.run_pick_and_place's ArmIK solves),
    not re-solve IK live against physical state. PiperTrajectoryRecorder/
    PiperTrajectoryReplayer (piper_trajectory.py, 2026-07-17) now exist,
    analogous to robots/trajectory.py's SO-ARM101 pair -- verified
    end-to-end in simulation (record via run_pick_and_place's step_hook=,
    save/load round-trips exactly, replay's joint-motion path tested
    against a mock backend). NOT hardware-tested: replay() deliberately
    raises NotImplementedError on the gripper-command step rather than
    guessing a unit conversion -- the recorded gripper value is the sim's
    raw action-space ctrl signal, not PiperRealBackend.set_gripper's
    metres convention, and no utility in this codebase reads real gripper
    width from sim state to derive one safely. See piper_trajectory.py's
    module docstring for the exact unresolved conversion; resolving it (or
    building a metres-reading utility) is required before a real replay,
    and joint-only replay is available now via replay(..., gripper_agnostic=True)
    for validating arm motion timing independent of that blocker.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

# ── piper_sdk availability guard (same pattern as SOARMRealBackend's
#    _LEROBOT_AVAILABLE guard). Installed 2026-07-22 (v0.6.1); the guard
#    stays for anyone running this module without it.
try:
    from piper_sdk import C_PiperInterface_V2  # type: ignore
    _PIPER_SDK_AVAILABLE = True
except ImportError:
    _PIPER_SDK_AVAILABLE = False

# Joint order matches piper_pick_and_place.py's JOINTS list
# (["robot0_joint1", ..., "robot0_joint6"]) for direct array indexing --
# the "robot0_" MuJoCo body-naming prefix is sim-only, real joint indices
# below are 1-6 per piper_sdk's own JointCtrl(joint_1, ..., joint_6) signature,
# CONFIRMED against piper_sdk/interface/piper_interface_v2.py (2026-07-22).
N_ARM_JOINTS = 6

# JointCtrl/GetArmJointMsgs both use integer units of 0.001 degree (confirmed
# against piper_interface_v2.py's own docstrings, e.g. "joint_1 (int): 关节1
# 角度,单位0.001度" and matching piper_ctrl_joint.py's own
# `factor = 57295.7795  # 1000*180/3.1415926`). rad <-> count: count = rad *
# RAD_TO_MILLIDEG, rad = count / RAD_TO_MILLIDEG.
RAD_TO_MILLIDEG = 1000.0 * 180.0 / math.pi  # == 57295.7795...

# JointCtrl's own docstring table lists joint6 (wrist roll)'s DEFAULT
# software limit as +-2.09439 rad (+-120 deg) -- narrower than the sim's
# +-3.14 rad assumption. See the CRITICAL DISCREPANCY note in the module
# docstring before trusting any wrist-fix threshold on real hardware.
WRIST_ROLL_SDK_DEFAULT_LIMIT_RAD = 2.09439

# GripperCtrl/GetArmGripperMsgs both use integer units of 0.001 mm (confirmed
# against piper_interface_v2.py's docstring, "gripper_angle (int): 夹爪范围,
# 以整数表示, 单位0.001mm", matching piper_ctrl_joint.py's own
# `joint_6 = round(position[6]*1000*1000)` where position[6] is metres).
# metres <-> count: count = metres * M_TO_MICROMETRE, metres = count / M_TO_MICROMETRE.
M_TO_MICROMETRE = 1_000_000.0

# Default gripper effort/torque for GripperCtrl's gripper_effort arg (unit
# 0.001 N/m, range 0-5000 == 0-5 N/m) -- matches piper_ctrl_joint.py's own
# demo value (1000 == 1 N/m), not an invented number.
GRIPPER_EFFORT_DEFAULT = 1000

# Gripper convention, matching piper_pick_and_place.py's GRIPPER_OPEN=-1.0 /
# GRIPPER_CLOSE=1.0 sim convention, re-expressed here in metres to match
# the now-corrected sim opening range (piper_controller_config.py's
# use_action_scaling fix, README's "ROOT CAUSE FOUND AND FIXED" entry --
# true open span measured at 12.01cm in sim, NOT the originally-modeled
# 7.6cm/10cm figures). Still unverified against the real gripper's actual
# mechanical travel -- the sim's open-span number came from a MuJoCo
# geom-to-geom measurement, not a hardware spec sheet; the API-level unit
# conversion above (metres <-> 0.001mm count) is now confirmed regardless
# of what the true max travel turns out to be.
GRIP_OPEN_M = 0.12
GRIP_CLOSED_M = 0.0

# MotionCtrl_2's move_spd_rate_ctrl (0-100, percent of max joint speed).
# piper_ctrl_joint.py's own demo uses 100; deliberately conservative here
# for a first real connection -- raise once real behavior is characterized.
DEFAULT_MOVE_SPD_RATE_PCT = 30

_ARM_SETTLE_S = 1.0
_GRIPPER_SETTLE_S = 0.5
_ENABLE_POLL_TIMEOUT_S = 5.0


class PiperRealBackend:
    """Interface skeleton for the physical Piper arm over piper_sdk/CAN.

    NOT hardware-tested -- see module docstring before first connection.

    Parameters
    ----------
    can_name : str
        Activated CAN interface name, e.g. "can0". The CAN adapter must
        already be brought up (`sudo ip link set can0 up type can
        bitrate 1000000` or the vendor-provided activation script) BEFORE
        constructing this class -- `can_auto_init=True` (this class's
        setting, confirmed as the constructor's own default) only
        auto-initializes the SDK's internal CAN bus object, it is not a
        substitute for bringing the physical interface up first; still
        unverified against a real adapter since this has never been run
        against hardware.
    max_relative_target : float | None
        Safety clamp: maximum allowed position change per command
        (radians). None disables clamping. Follows
        SOARMRealBackend's own max_relative_target convention exactly --
        that module's docstring recommends starting conservative (its
        SO-ARM101 default suggestion was 30-60 degrees ~ 0.5-1.0 rad;
        Piper is a larger/heavier arm, consider starting smaller, e.g.
        0.05-0.1 rad, until real behavior is characterized).
    """

    def __init__(self, can_name: str = "can0", max_relative_target: Optional[float] = None):
        if not _PIPER_SDK_AVAILABLE:
            raise ImportError(
                "piper_sdk is required for PiperRealBackend. Install it with: "
                "pip install piper_sdk -- NOT yet installed in the `tango` env "
                "as of 2026-07-16 (this class has never been run against real "
                "hardware; see this module's docstring before first connection)."
            )
        self._can_name = can_name
        self._max_relative_target = max_relative_target
        self._piper: Optional["C_PiperInterface_V2"] = None
        self._connected = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the CAN connection and enable the arm.

        judge_flag/start_sdk_joint_limit/start_sdk_gripper_limit all left
        at their CONFIRMED constructor defaults (True) -- i.e. the SDK's
        own built-in safety checks stay ON, deliberately not replicating
        an example that turns them off. EnablePiper() (not a bare
        EnableArm() call) is the SDK's own documented convenience method,
        used identically in every piper_sdk/demo/V2/*.py script: it calls
        EnableArm(7) internally and reports whether all motors ended up
        enabled, which is why every demo polls it in a loop rather than
        calling it once. Still NOT hardware-tested -- the poll timeout
        below is this module's own conservative addition (no official
        demo bounds the loop), since polling forever on a real bus fault
        would hang silently.
        """
        self._piper = C_PiperInterface_V2(
            can_name=self._can_name,
            judge_flag=True,
            can_auto_init=True,
            dh_is_offset=1,
            start_sdk_joint_limit=True,
            start_sdk_gripper_limit=True,
        )
        self._piper.ConnectPort()
        deadline = time.monotonic() + _ENABLE_POLL_TIMEOUT_S
        while not self._piper.EnablePiper():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"PiperRealBackend: EnablePiper() did not report all-enabled "
                    f"within {_ENABLE_POLL_TIMEOUT_S}s -- check CAN bus/power "
                    "before retrying."
                )
            time.sleep(0.01)
        self._connected = True
        time.sleep(_ARM_SETTLE_S)

    def close(self) -> None:
        """Disable the arm and release the CAN connection."""
        if self._piper is not None:
            try:
                self._piper.DisablePiper()
            except Exception:
                pass
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── state reads ───────────────────────────────────────────────────────

    def get_joint_positions(self) -> np.ndarray:
        """Current arm joint positions. Returns (6,) float32 in radians.

        GetArmJointMsgs().joint_state.joint_1..joint_6, integer units of
        0.001 degree -- confirmed against piper_interface_v2.py's own
        docstring (2026-07-22). Field names/units are now verified; actual
        real-time values (noise, latency, whether the physical arm reports
        cleanly) are not, since this has never run against hardware."""
        self._require_connected()
        js = self._piper.GetArmJointMsgs().joint_state
        counts = np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=np.float64,
        )
        return (counts / RAD_TO_MILLIDEG).astype(np.float32)

    def get_gripper_opening(self) -> float:
        """Current gripper opening in metres [GRIP_CLOSED_M, GRIP_OPEN_M].

        GetArmGripperMsgs().gripper_state.grippers_angle, integer units of
        0.001 mm -- confirmed against piper_interface_v2.py's own docstring
        (2026-07-22)."""
        self._require_connected()
        count = self._piper.GetArmGripperMsgs().gripper_state.grippers_angle
        return float(count) / M_TO_MICROMETRE

    # ── motion commands ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Move arm to READY_QPOS (matching piper_pick_and_place.py's sim
        convention) and open the gripper."""
        from tango_robot.piper_robosuite.piper_pick_and_place import READY_QPOS
        self.move_joints(READY_QPOS, blocking=True)
        self.set_gripper(GRIP_OPEN_M, blocking=True)

    def move_joints(self, q: np.ndarray, blocking: bool = True) -> None:
        """Command arm joints to q (radians). q.shape must be (6,).

        JointCtrl(joint_1..joint_6), integer units of 0.001 degree --
        confirmed against piper_interface_v2.py's own docstring and
        piper_ctrl_joint.py's demo (`factor = 57295.7795`), 2026-07-22.
        MotionCtrl_2(0x01, 0x01, DEFAULT_MOVE_SPD_RATE_PCT, 0x00) must
        precede JointCtrl to select CAN-command / MOVE-J / position-
        velocity mode -- confirmed as the exact call every JointCtrl demo
        makes first. joint6's WRIST_ROLL_SDK_DEFAULT_LIMIT_RAD caveat in
        the module docstring applies here: __CalJointSDKLimit clips
        server-side to whatever the arm's configured limit is, silently,
        so a q[5] beyond that limit will not error, it will just not
        reach the commanded angle."""
        q = np.asarray(q, dtype=float)
        assert q.shape == (N_ARM_JOINTS,), f"expected {N_ARM_JOINTS} joint values, got {q.shape}"
        self._require_connected()

        if self._max_relative_target is not None:
            current = self.get_joint_positions()
            q = np.clip(q, current - self._max_relative_target, current + self._max_relative_target)

        counts = np.round(q * RAD_TO_MILLIDEG).astype(int)
        self._piper.MotionCtrl_2(0x01, 0x01, DEFAULT_MOVE_SPD_RATE_PCT, 0x00)
        self._piper.JointCtrl(*counts.tolist())
        if blocking:
            time.sleep(_ARM_SETTLE_S)

    def set_gripper(self, opening: float, blocking: bool = True) -> None:
        """Command gripper to opening (metres). Clipped to [GRIP_CLOSED_M, GRIP_OPEN_M].

        GripperCtrl(gripper_angle, gripper_effort, gripper_code, set_zero),
        confirmed against piper_interface_v2.py's own docstring and
        piper_ctrl_joint.py's demo (`piper.GripperCtrl(abs(joint_6), 1000,
        0x01, 0)`), 2026-07-22. gripper_code=0x01 enables the gripper motor
        (0x00 would disable it, not open/close it -- a common
        misread of this field)."""
        opening = float(np.clip(opening, GRIP_CLOSED_M, GRIP_OPEN_M))
        self._require_connected()
        count = round(opening * M_TO_MICROMETRE)
        self._piper.GripperCtrl(count, GRIPPER_EFFORT_DEFAULT, 0x01, 0)
        if blocking:
            time.sleep(_GRIPPER_SETTLE_S)

    # ── high-level execution ─────────────────────────────────────────────

    def execute_grasp(self, *args, **kwargs):
        """Not implemented for direct online use -- same reasoning as
        SOARMRealBackend.execute_grasp(): online IK on real hardware is
        outside this module's scope. Real execution should replay a
        qpos sequence already validated in simulation, not re-solve IK
        live. A Piper-specific trajectory recorder/replayer (analogous to
        robots/trajectory.py, but for Piper's 6-DoF convention) does not
        exist yet -- natural next step once this backend is hardware-
        verified, not built in this architecture-only pass."""
        raise NotImplementedError(
            "PiperRealBackend.execute_grasp() is intentionally not implemented. "
            "Real hardware execution should replay a joint-position sequence "
            "already solved and validated in simulation (see "
            "piper_pick_and_place.run_pick_and_place's ArmIK solves), not "
            "re-solve IK live against physical state. A trajectory recorder/"
            "replayer for Piper does not exist yet -- build that first."
        )

    # ── internal ──────────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("PiperRealBackend: call connect() before any state read or motion command.")
