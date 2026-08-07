# Jaw opening calibration — steps ① and ② (2026-08-07)

Agreed order (this session, after the pad-fidelity diagnostic): split "fix
opening calibration" into three separable measurements before touching any
control code, because the earlier framing conflated three different things
under one broken constant:

```
① q [rad] -> true pad opening [m]        pure kinematics, no contact, no solver
② commanded opening -> actuator response  no object, tests the ACTUATOR alone
③ commanded opening + object -> response  what the pad-fidelity diagnostic
                                          already showed is contact/solver-
                                          dependent (100% excessive penetration
                                          on legacy successes)
```

① and ② don't depend on contact solver parameters, so they're safe to freeze
now and won't need redoing after any future `solref`/`solimp` retuning. ③ is
deliberately deferred.

**No core code changed this round.** `tango_robot/env_soarm.py` has zero diff.
Both scripts are new, read-only, and produce calibration artifacts under
`calib/` — `move_gripper()`, `GRIP_CLOSED`, `GRIP_OPEN` are untouched.

## ① q ↔ true opening (`scripts/build_jaw_opening_lut.py` → `calib/jaw_opening_lut.json`)

201-point dense LUT over the joint's full mechanical range, geometry only (no
MuJoCo scene, no physics stepping — `JawMetrology.true_opening_m` is a pure
function of the hinge angle since both fingers are rigid).

```
joint range          -0.1745 .. 1.7453 rad  ->  2.1  .. 95.7  mm
legacy control window  0.05  .. 1.0    rad  -> 19.4  .. 70.9  mm
legacy code claims                             0.0   .. 100.0 mm
```

Matches every prior measurement in this investigation exactly (step 4's audit,
the jaw_metrology test suite's pinned values). Confirms nothing new; formalizes
it as a reusable artifact instead of a number embedded in a report.

## ② free-space actuator validation (`scripts/validate_free_space_actuator.py` → `calib/jaw_free_space_actuator.json`)

Commands `move_gripper()` (legacy, unmodified) at 11 requested openings from
5–100 mm with **no object in the scene**, waits 200 steps (2.5× `move_gripper`'s
own default), and compares the actuator's own `ctrl` target against the
`qpos` it actually settles at.

```
max |target_q - settled_q|  = 0.06 mrad  (~0.003°)
actuator saturation          none (peak force 0.065 / ±3.35 N·m, <2%)
convergence                  clean at every commanded opening
```

**The actuator reaches its own commanded target cleanly in free space.** This
rules out a third, independent defect (actuator gain/damping/settle-time
failing to track its own setpoint) as a contributor to anything seen with an
object present. A caught methodology bug along the way: the first pass flagged
several rows "still moving" using a velocity threshold (`|qvel| > 1e-3 rad/s`)
— a lightly-damped position controller has small residual velocity oscillation
around its setpoint indefinitely (observed up to 0.003 rad/s) without that
meaning anything is unresolved. Re-judged on position tracking error instead
(the physically meaningful quantity), the "still moving" cases disappeared:
position error was already ≤0.06 mrad when velocity looked non-zero.

The same run independently reconfirms the known linear-map defect through a
completely different code path (live simulation, not the geometric LUT):
`req=5mm -> measured 22.9mm`, `req=100mm -> measured 70.9mm` — consistent with
① to within the discretization of the requested-opening sweep.

## What this establishes, and what it doesn't

**Established**: any gap between commanded and actual opening seen WITH an
object present (the pad-fidelity diagnostic's 12–14 mm penetration on
Hammer/MediumClamp/Banana) is attributable to contact/solver behavior, not to
the actuator failing to track its own free-space target. That was a live
possibility before this check and is now ruled out.

**Not yet done**: `move_gripper()`'s API still lies about units (declares
metres, computes via a broken linear map), and `GRIP_CLOSED`/`GRIP_OPEN` still
bound the operational range to 19.4–70.9 mm true opening rather than the
mechanical 2.1–95.7 mm. Both remain deferred:

- Wiring ①'s LUT into `move_gripper()`/`get_gripper_opening()` so the API's
  stated units become true — this is the actual "step 1" fix, still pending.
  Safe to do independent of ③ (contact/solver work), since ① doesn't depend on
  contact.
- Whether to widen `GRIP_CLOSED` toward the joint's true lower limit — this
  was already deferred to "step 2" in the original 4→3→1→2 plan, for the
  reason restated here: expanding the operational range moves the jaw into
  territory the collider has never correctly described, and per the
  pad-fidelity diagnostic, even the CURRENT range does not correctly describe
  contact yet either.
