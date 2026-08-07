# Piper: correction to earlier failure attribution, + first production integration (2026-08-07)

Two things in this doc: a **correction to a headline claim** made in two
earlier committed docs from this same investigation, and the first
production code change to `piper_pick_and_place.py` (previously every pass
was strictly zero-diff).

## CORRECTION: "transit_high is the dominant failure mode" was not supported

`docs/PIPER_TCP_CORRECTION_AB_20260807.md` and
`docs/PIPER_TRANSIT_HIGH_AND_BILATERAL_AUDIT_20260807.md` both framed
`ik_no_converge:transit_high` as *the dominant failure mode* of the Piper
pipeline. **That causal claim is withdrawn.**

The label came from `_failure_stage()` in
`scripts/piper_tcp_correction_ab.py`, a helper written for that A/B, which
returns the **first non-converged phase of a failed trial**. `transit_high`
is the pipeline's first IK phase, and it essentially never converges — so
it was assigned as the "failure stage" of virtually every failure
tautologically, regardless of what actually went wrong.

Checked directly against the orientation A/B/C's own T0 (legacy) arm:

```
transit_high CONVERGED among successes: 0/8
transit_high CONVERGED among failures : 0/5

transit_high err_cm, successes: [0.42, 0.58, 0.61, 0.64, 0.67, 0.71, 0.75, 2.73]
transit_high err_cm, failures : [0.68, 0.73, 0.82, 0.83, 3.02]
mean: 0.89cm (success) vs 1.21cm (failure)
```

`transit_high` fails to converge in **100% of successful trials too**, and
the error distributions overlap almost entirely. The label has **zero
discriminating power** between success and failure.

**What survives unchanged** (these were direct measurements, not
inferences, and remain valid):

- `transit_high` genuinely never converges — real defect.
- `SAFE_TRANSIT_Z = 1.05m` genuinely sits on a reachability cliff: 13/13
  candidates comfortably reachable at 0.85–0.95m, 2/13 at 1.05m, 0/13 at
  1.10m+ (`outputs/piper_transit_reachability_sweep.jsonl`).
- Joints genuinely pin at exactly their `REAL_JOINT_LIMITS` bounds there.
- The clearance-vs-reachability tension (Cracker's top is ~1.02m, so
  clearance *demands* roughly the height reachability *denies*) is real.

**What does not survive**: that any of this is what makes trials fail. On
the evidence, it is a real geometric defect the pipeline largely tolerates
— a ~0.6–0.9cm error at a hover waypoint that downstream phases absorb.

**Consequence for the plan**: region-based transit remains a legitimate
*correctness/robustness* fix, but it has **no demonstrated success-rate
benefit**, and should not be sold as fixing "the main blocker". It was
therefore **not implemented in this pass** — implementing it first would
have meant shipping a non-trivial planner change on a justification that
had just collapsed.

**And the real failure cause for Piper is currently unknown.** That is the
honest state. Identifying it is now the highest-value open question, ahead
of any further hardening.

## Production change 1: T_eef_capture formalized (the one change shipped)

`tango_robot/piper_robosuite/piper_pick_and_place.py` — adds
`T_EEF_CAPTURE_LOCAL`, `capture_pose_from_eef()`,
`eef_target_for_capture_target()`, `grasp_capture_target()`, and routes
only the **grasp-semantic** call sites through them (descend,
descend_refresh, cr_cfm descend, two-stage commit, wrist-friendly
orientation probe, candidate IK scoring). Transit/hover/tray waypoints
still use eef-site semantics by design.

### The trap this avoids, which is the important part

`GRASP_HEIGHT_OFFSET = 0.0`'s own comment says it centres the grasp on the
object's CoM, and documents a 2026-07-14 experiment preferring it over
`+0.02`. But it is an **eef-site** offset, and eef_site is 65.6mm from the
fingertip midpoint. So with a level grasp that experiment really compared:

```
fingertips at CoM + 65.6mm   -> stable hold      (kept, as "0.0")
fingertips at CoM + 85.6mm   -> slipped free     (rejected, as "+0.02")
```

It **never tested fingertips at the CoM**. The validated physical grasp
height is CoM + 65.6mm.

Naively applying the capture correction while leaving that constant at 0.0
would have moved every level grasp **65.6mm down**, to a height never
validated and away from the only known-good one. That is very likely the
mechanism behind the Cracker bilateral-contact degradation observed in
`docs/PIPER_TCP_CORRECTION_AB_20260807.md` (9/10 -> 5/10), which the
tilt-based explanations tried and failed to account for — and it would have
been shipped as an invisible regression.

`GRASP_CAPTURE_HEIGHT_OFFSET = +0.0656` restates the validated height in
the correct frame. The two offsets cancel exactly, so:

- **level grasps: algebraically identical to legacy** (the refactor is a
  pure re-description);
- **tilted grasps: deliberately different**, and now correct — legacy
  missed the intended capture point by `2·0.0656·sin(θ/2)` (34.0mm at 30°).

The practical value is that grasp height is now an **explicit, correctly-
framed, tunable parameter** instead of an accidental by-product of a
mislabelled reference — re-tuning it is now a meaningful experiment rather
than a confounded one.

### Verification

- `tests/test_piper_capture_frame.py` — 6 unit tests: pure-local-Z
  property, inverse-pair round trip over 25 random rotations, **exact**
  level-grasp equivalence to legacy, the constants' consistency, tilted
  divergence with the correct chord-length miss, explicit height-offset
  pass-through. All pass. (One test initially failed on my own wrong
  arithmetic — I had asserted the tilted miss equals the full 65.6mm; it is
  the chord length between two equal-magnitude vectors, 34.0mm at 30°. The
  production code was right; the assertion was fixed.)
- End-to-end: 8 (object, seed) pairs re-run against the committed
  pre-change baseline in `outputs/piper_tcp_correction_ab.jsonl` — **8/8
  identical**.
- Downstream importers checked: `piper_pick_and_place_mink.py` still
  imports `GRASP_HEIGHT_OFFSET` and keeps legacy semantics (deliberately
  left at 0.0, unconverted).

## Also found: the "~7.6cm max opening" figure in this file is stale

While scoping the mechanical width gate (item 3 of the freeze list), the
file turned out to carry three different opening figures: `~7.6cm` (in the
`OBJECT_NARROW_AXIS` comments), `~12cm` ("true ~12cm open span" elsewhere),
and `REAL_GRIP_OPEN_M = 0.12` in the real backend. Measured against the
**live composed robosuite model**: the actuator's own ctrlrange floor
(-0.05) gives an inner-face gap of exactly **0.1000 m**.

This matters beyond tidiness: the `can` is documented as *"effectively
circular (~8.6cm diameter) — WIDER than the gripper's ~7.6cm max opening
regardless of yaw, so no orientation fixes this"*, i.e. written off as
geometrically ungraspable. Against a true 100mm opening an 86mm can is
**within** the mechanical limit (though tight — 7mm total clearance).

**The width gate was therefore not implemented in this pass.** Building it
on the stale 7.6cm would wrongly reject objects; building it on the
measured 100mm silently reverses a standing "this object is ungraspable"
conclusion that other analyses may depend on. That reversal should be a
deliberate, visible decision, not a side effect of adding a gate.

## Status

| item | state |
|---|---|
| 1. `T_eef_capture` formalized, grasp-phases only | **shipped**, behaviour-preserving for level grasps, tested |
| 2. Region-based transit | **not shipped** — justification withdrawn (see correction); real defect, no demonstrated benefit |
| 3. Mechanical width gate | **not shipped** — blocked on the stale-7.6cm-vs-measured-100mm decision |
| 4. Contact physics | untouched, as agreed |
| 5. `Lift` success semantics | untouched, as agreed |
| 6. Bilateral dynamic probe | not started (explicitly non-blocking) |
| joint6 re-run | correctly still deferred |

## Open questions, in priority order

1. **What actually causes Piper grasp failures?** The previous answer was
   an artefact. No current candidate explanation.
2. Does the `can`'s graspability conclusion change under a true 100mm
   opening? (cheap to test empirically — one batch of trials)
3. Is region-based transit worth shipping on correctness grounds alone,
   given no measured success benefit?
