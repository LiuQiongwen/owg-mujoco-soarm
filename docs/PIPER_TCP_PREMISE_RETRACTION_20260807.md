# RETRACTION: the Piper "65.6mm TCP offset" was a measurement artifact (2026-08-07)

This retracts the central premise of several earlier docs in this
investigation. The production change built on it has been reverted
(`8734c28` reverts `a63c6a7`); `piper_pick_and_place.py` is back to its
original state.

## What was claimed

`docs/PIPER_GRIPPER_AUDIT_20260807.md` reported a **FAIL**: that
`robot0_eef_site` (what IK targets) sits **65.6mm from the true fingertip
midpoint**, i.e. that every Piper grasp in the project's history was aimed
at the wrong point. `docs/PIPER_CAPTURE_FRAME_CALIBRATION_20260807.md` then
calibrated this as a rigid transform `T_eef_capture`, and
`docs/PIPER_TCP_CORRECTION_AB_20260807.md` ran a paired A/B on correcting
it.

## What is actually true

The 65.6mm number is real and rigid — but it does **not** measure what it
was labelled as. It measures the distance from `eef_site` to the **far end
of the finger mesh from its link origin**, which for this gripper is the
**top** of the finger, not the pinching tip.

Measured directly, in a live `can` grasp, at the moment the gripper closes:

```
can    z: [0.7994, 0.9020]      (the object)
finger z: [0.8526, 0.9292]      (both finger collision meshes)
eef_site z: 0.8576
vertical finger/object overlap: +49.4mm
```

`eef_site` (0.8576) sits **inside** the finger span, near its lower end,
and the fingers genuinely straddle the object with 49.4mm of overlap. The
claimed "true fingertip midpoint" (eef + 65.6mm = 0.9236) is up near the
finger's **top** edge (0.9292) — a point that never contacts anything.

**So `eef_site` was a reasonable grasp reference all along.** There was no
65.6mm aiming defect. The original code was not mis-aiming every grasp.

## Root cause of the error

Both `scripts/audit_piper_gripper.py` and
`scripts/calibrate_piper_capture_frame.py` locate the "fingertip" as:

```python
r = np.linalg.norm(vw - body_pos, axis=1)
return vw[r >= np.quantile(r, 0.75)]     # "the tip region"
```

i.e. the 25% of mesh vertices **farthest from the link's body origin**. For
a finger whose body origin sits at the *distal* end, that heuristic selects
the *proximal* end — the opposite of the intended tip. It was never
validated against the actual contact region, and its output was carried
forward as ground truth through four subsequent passes.

The rigidity result itself (std 0.000mm across 5 arm poses × 3 openings) is
still valid — it just proves the finger mesh is rigidly mounted, which was
never in doubt, and is **not** evidence that the quantity being measured is
the grasp reference.

## What this invalidates

- **`PIPER_GRIPPER_AUDIT_20260807.md`'s "TCP / grasp reference: FAIL"** —
  withdrawn. On current evidence this row should read PASS.
- **`PIPER_CAPTURE_FRAME_CALIBRATION_20260807.md`** — the transform is
  real but mislabelled; it is not a grasp reference.
- **`PIPER_TCP_CORRECTION_AB_20260807.md`** — the A/B correctly measured
  that "correcting" the reference didn't help, and its conclusion ("the
  offset explains none of the historical failures") now has a much simpler
  explanation: there was no offset to correct. The Cracker bilateral drop
  it observed (9/10 → 5/10) was caused by the intervention moving grasps
  65.6mm away from a working reference.
- **`PIPER_CORRECTION_AND_INTEGRATION_20260807.md`'s** account of
  `GRASP_HEIGHT_OFFSET` — its claim that the 2026-07-14 tuning "never
  tested fingertips at the CoM" is void. `GRASP_HEIGHT_OFFSET = 0.0`'s
  original comment is fine as written.

The production change is reverted rather than kept-and-re-documented,
because its only remaining behavioural effect was on tilted grasps, where
it "corrected" toward a reference now known to be wrong.

## Also retracted this pass: the "grasp height above object" separator

Within this same pass I proposed that `can` and `banana` fail because the
grasp height sits above them (headroom −21.4mm / −30.9mm), and noted the
historical rates matched perfectly (can 0/25, banana 0/5, versus 32–73% for
the others). **That was falsified by the intervention test in the same
session**: running `can` at the baseline height today gives **6/6
successes**, and at a mid-height target also 6/6.

Two conclusions, both solid:

1. **`can` is NOT ungraspable.** It succeeds 6/6 today. The standing claim
   in `piper_pick_and_place.py`'s comments — *"~8.6cm diameter … WIDER than
   the gripper's ~7.6cm max opening … no orientation fixes this"* — is
   wrong on both numbers: the can measures **65.6mm** wide at mid-height,
   and the live model's opening is **100.0mm**. This is worth fixing in the
   comments (not done here).
2. **The historical 0/25 predates current fixes.** Most likely the gripper
   double-scaling bug (fixed 2026-07-15,
   `piper_controller_config.py`), which reduced effective travel to ~0.1mm
   and would have made every object fail. Historical per-object rates from
   before that date should not be used as a baseline for anything.

The perfect agreement between my geometric prediction and the historical
rates was coincidence — both `can` and `banana` were at 0% for an unrelated
reason. This is exactly the failure mode the previous correction warned
about (a plausible story matching the data for the wrong reason), and it
survived one round of validation before the intervention test killed it.

## Genuinely established, and still standing

- Live composed model's max inner-face opening: **100.0mm** (measured).
- `clamp` is genuinely too wide: 150mm at grasp height vs 100mm opening.
- Object geometry table: `outputs/piper_object_grasp_geometry.json`.
- `SAFE_TRANSIT_Z = 1.05m` sits on a real reachability cliff (13/13
  reachable at 0.85–0.95m, 2/13 at 1.05m) — still a real defect, still with
  no evidence it causes task failures.
- `transit_high`'s failure-mode attribution is still withdrawn (previous
  doc).
- The SO-101 106.7mm "local support width" figure is **SO-101's**, measured
  against SO-101's 95.7mm jaw. It does not transfer to Piper and was never
  Piper evidence.

## Where this leaves Piper

Three consecutive proposed failure mechanisms — transit_high
non-convergence, the TCP offset, and grasp-height-above-object — have each
been withdrawn after direct testing. **The cause of Piper's grasp failures
(cracker 33%, mustard 55%, pear 73%) remains unknown**, and no current
hypothesis is standing.

The right next step is the outcome-conditioned trace that motivated this
pass in the first place: collect matched success/failure rollouts under the
*current* code, record continuous per-phase quantities, and look for
variables that actually separate the two — rather than proposing a
mechanism and then seeking confirmation for it. Three retractions in a row
argue strongly for that ordering.

## Process note

The reason all three claims were caught rather than shipped is that each
was eventually subjected to a test it could fail: a discriminating-power
check, a same-seed intervention, and a direct geometric measurement at the
moment of contact. The pattern worth keeping: **a measurement heuristic
(here, "farthest vertices = the tip") must itself be validated against the
thing it claims to measure before anything is built on it.** Four passes
were built on one unvalidated `np.quantile` line.
