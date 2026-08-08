Piper real-hardware bring-up — mandatory gates

**Status: BLOCKED.** No OWG/critic component may dispatch any waypoint to
physical hardware until Gates 1–6 pass and are recorded.

These are hard gates, not a checklist of suggestions. Each exists because
of a specific, already-documented discrepancy between what sim assumes and
what the hardware is documented to do — not as defensive boilerplate.

Sources: `piper_real_backend.py` header, `piper_pick_and_place.py`'s
`clip_action_to_real_limits()` docstring, `PIPER_FINDINGS_SUMMARY.md`, and
this investigation's sim measurements (`calib/piper_baseline_v1.json`).
Every physical quantity below must be measured **on the device**; nothing
here may be satisfied by reading either side's static definition.

## Gate 1 — Real joint limits, signs, units

**The critical one.** Sim assumes joint6 ∈ ±3.14 rad (from
`robot_arm.xml`); `piper_sdk`'s `JointCtrl` docstring gives the hardware
*default* as ±2.09439 rad — a software limit set over CAN
(`MotorAngleLimitMaxSpdSet`). That is ~60° of total range in dispute.

- Query the device for its **actual** current limits; record as ground
  truth in a versioned calib file.
- Do **not** resolve this by trusting either static definition.
- Then re-check every sim-validated trajectory against the real envelope.
  Wrist motions that are legal in sim may be rejected or silently truncated
  on hardware.
- Also verify per-joint sign convention and unit scaling
  (joints 0.001°, gripper 0.001mm per the backend header) against measured
  motion, not documentation.

**Pass:** measured limits/signs/units recorded; all candidate trajectories
verified inside the measured envelope.

## Gate 2 — Enforced clipping with violation logging

`clip_action_to_real_limits()` must be **on** for every hardware path.

Evidence it is necessary, not precautionary: on trial 1007 a commanded
action reached 151.7 rad (48× the real range), with 275/1690 steps out of
range; the universal clip reduced that to 0/1690. MuJoCo's own `jnt_range`
enforcement makes the clip look redundant **in sim only** — a real motor
controller has no such guarantee.

- Log every clipped action: joint index, original value, clipped value,
  timestamp.
- A nonzero clip count during qualification is a **stop condition**, not a
  statistic to report later.

**Pass:** clipping active on all paths; clip log empty across a full dry
run.

## Gate 3 — Independent gripper calibration

`REAL_GRIP_OPEN_M = 0.12` (hardware) vs **100.0mm measured in sim**
(`calib/piper_baseline_v1.json`). A 20mm gap must not be assumed to be a
units artifact.

Measure on device:
- command value → actual jaw opening (swept, not two points)
- maximum opening
- minimum closure
- linearity of the mapping
- what command reproduces sim's 100.0mm

**Pass:** measured command↔opening curve recorded; the sim↔real
correspondence stated explicitly rather than inferred.

## Gate 4 — TCP / capture frame re-verification on hardware

This investigation retracted a 65.6mm "TCP offset" that turned out to be a
measurement artifact of a vertex-selection heuristic. The sim conclusion is
that `robot0_eef_site` is a sound grasp reference — that conclusion is
**sim-only** and must be independently re-established on hardware before
any grasp target is trusted.

**Pass:** physical eef/finger geometry measured and compared against the
sim model.

## Gate 5 — Low-speed free-space waypoints

No object, no contact, reduced speed. Verifies the full command path end to
end with all of the above active.

**Pass:** commanded vs achieved joint positions within a recorded
tolerance; zero clip events; no faults.

## Gate 6 — Fixed grasp smoke test

Single hard-coded, sim-validated grasp on a known object. Note
`PiperRealBackend.execute_grasp()` is intentionally **not** implemented:
the design is to replay a validated sim trajectory, not to solve IK live on
hardware. Preserve that.

**Pass:** repeatable grasp/lift/place with no clip events and no faults.

## Gate 7 — sim-real paired candidate evaluation

Only after 1–6. This is where the OWG/critic path may first reach
hardware, and only under supervision.

## Relationship to the main line

Milestone 2 (robustness-labelled critic) proceeds in simulation and is
**not** blocked by any of this. The two tracks are parallel.

World-model work should wait for whichever of sim/real gives a trustworthy
*temporal* execution chain — which is the current bottleneck on both sides,
not a reason to accelerate either.
