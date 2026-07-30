# Real-Hardware Backend Architecture — T-RO Paper Section Draft

**Scope, stated plainly**: this section documents `piper_real_backend.py`'s design. It has
**never been run against physical hardware** — the Piper arm was not connected to the
development machine while this was written (confirmed via AskUserQuestion before writing any
of it). Nothing below should be read as a real-hardware result. It is an architecture and safety
design contribution, presented as such.

## Why this exists in the paper at all

The narrowed T-RO scope explicitly asked to include real hardware, not stay simulation-only.
Given the arm is not currently connected, the honest way to satisfy that without fabricating
results is to contribute the **backend architecture and safety design** needed for real
execution, clearly scoped as "ready for hardware verification," not "verified on hardware."

## Design summary

- **Pattern, not inheritance**: follows the same safety-conscious pattern as the project's
  existing `robots/soarm_real_backend.py::SOARMRealBackend` (relative-delta clamping on every
  motion command, `execute_grasp()` deliberately left unimplemented for live IK, lazy/guarded
  SDK import) — but does **not** inherit `robots/base.py::RobotBackend`, because that ABC is
  scoped to SO-ARM101's 5-DOF joint/gripper conventions and `CLAUDE.md` forbids `tango_robot/`
  importing `robots/`. Piper is 6-DoF with its own established conventions
  (`READY_QPOS`, `GRIPPER_OPEN`/`GRIPPER_CLOSE` in `piper_pick_and_place.py`); forcing it into
  the SO-ARM101-shaped interface would misrepresent both rather than reuse cleanly.

- **Availability guard**: `piper_sdk` is not installed in the `tango` conda environment
  (confirmed via `pip show piper_sdk` before writing this module). Import is wrapped in a
  try/except exactly like `SOARMRealBackend`'s `_LEROBOT_AVAILABLE` guard, so the rest of the
  codebase can import this module without the dependency being present, and instantiation
  raises a clean `ImportError` rather than failing at import time. Verified: the module imports
  cleanly and raises the expected error on instantiation without the SDK.

- **Safety-first refusal to guess units**: every hardware-touching method
  (`get_joint_positions`, `get_gripper_opening`, `move_joints`, `set_gripper`) raises
  `NotImplementedError` with an explicit `VERIFY:` comment rather than guessing a plausible
  unit/scale convention. This was a deliberate choice, not an oversight — a web search could not
  confirm `JointCtrl`/`GripperCtrl`'s exact parameter order, units (radians vs. degrees vs.
  scaled integer counts, common in CAN-bus arm SDKs), or full call signature (e.g. `GripperCtrl`
  appears to take force/speed/mode parameters beyond a single position value) from documentation
  alone, and the SDK's own repository carries an explicit warning that protocol misuse can
  damage the arm. Given real physical stakes, false confidence was judged worse than an
  explicit, loud placeholder.

- **`max_relative_target` safety clamp**: identical convention to `SOARMRealBackend`, applied in
  `move_joints` before any command reaches the (currently unimplemented) SDK call. Recommended
  starting value is smaller than the SO-ARM101 default (0.05–0.1 rad vs. 0.5–1.0 rad) given
  Piper is a larger/heavier arm — this is a documented recommendation, not yet empirically tuned
  against real hardware.

- **`execute_grasp()` intentionally unimplemented**: same reasoning as `SOARMRealBackend` — real
  hardware execution should replay a joint-position sequence already solved and validated in
  simulation (`piper_pick_and_place.run_pick_and_place`'s `ArmIK` solves), not re-solve IK live
  against physical state. A Piper-specific trajectory recorder/replayer (analogous to
  `robots/trajectory.py`'s `TrajectoryRecorder`/`TrajectoryReplayer`) does not exist yet — this
  is the concrete next engineering step once hardware is connected, explicitly flagged rather
  than silently deferred.

## Explicit VERIFY checklist (must be resolved before first physical connection)

1. Install `piper_sdk`, read `piper_sdk/demo/V2/` in the actual installed package.
2. Confirm `JointCtrl`/`GripperCtrl`'s exact parameter order, units, and full signature against
   a real example script — do not trust this module's placeholder scale assumptions.
3. Confirm `GetArmJointMsgs()`/`GetArmGripperMsgs()`'s return schema and units.
4. Re-evaluate `judge_flag`, `start_sdk_joint_limit`, `start_sdk_gripper_limit` — currently
   `False` in `connect()` (copied from the SDK's own example), but the module explicitly flags
   these as probably wanting to be `True` for a first real connection, not blindly reused.
5. Confirm the CAN interface activation step (`can_auto_init` may or may not handle this —
   unconfirmed) and the `EnableArm`/`DisableArm` method names (guessed by symmetry, unconfirmed).
6. Build the Piper trajectory recorder/replayer before attempting any `execute_grasp`-equivalent
   real motion.

## How to frame this in the paper

Present as: "we provide a hardware backend architecture and an explicit, itemized safety
verification checklist for physical deployment, following the same relative-delta-clamped,
replay-not-live-IK pattern validated on our SO-ARM101 hardware track — physical validation is
future work, gated on the checklist above." This is an honest, defensible framing: it's a real
engineering contribution (a safety-conscious interface design + a concrete pre-flight checklist
distilled from an SDK whose documentation genuinely does not specify enough to proceed safely
without it), not a disguised simulation-only result.
