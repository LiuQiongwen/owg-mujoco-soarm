# Piper TCP correction A/B: does the 65.6mm reference fix change historical conclusions? (2026-08-07)

Follows `docs/PIPER_CAPTURE_FRAME_CALIBRATION_20260807.md`'s confirmation that
`T_eef_capture` is a genuine rigid transform. Answers the scoped question
this A/B was set up for, and only that question: **does correcting the
65.6mm `robot0_eef_site`-to-fingertip-midpoint offset change Piper's
historical grasp conclusions?** Descend controller, force-compliance,
joint6 rules, candidate sampling, and every other pipeline behavior were
held frozen — only the IK target reference changes between arms.

Zero diff on `tango_robot/piper_robosuite/` and `tango_robot/piper_assets/`,
confirmed by `git status`/`git diff` before and after every run. The
correction is injected as a subclass of `ArmIK` (`CorrectedArmIK` in
`scripts/piper_tcp_correction_ab.py`) monkeypatched into
`piper_pick_and_place.ArmIK` for the duration of one trial, then restored —
`run_pick_and_place`'s own `ik = ArmIK(env)` resolves the name at call
time, so this needed no edit to the 87KB file, per the "thin wrapper /
transform injection" instruction this was scoped against.

Reproduce:
```bash
conda run -n tango python scripts/piper_tcp_correction_ab.py
```

## A real implementation bug, caught before trusting any result

The first version of this script patched `ArmIK.solve()` globally, applying
the capture-frame correction to *every* IK call in the 8-phase pipeline —
`transit_high`, `approach`, `lift`, `transit_above_tray`, `lower_into_tray`,
not just the actual grasp-commit target. Those are independently-computed
hover/transit/tray waypoints that were never expressed in capture-frame
terms; shifting them by 65.6mm along each call's own (often very
differently-oriented) `target_mat` axis is not what "correct the grasp
reference" means. First run: Cracker collapsed from 7/10 to 0/10 successes
with one IK solve diverging to 1283mm error — a signature of the bug, not a
physics finding (elongated, strongly-oriented objects like Cracker have
`grasp_mat` vary a lot more across phases than round Pear, so the bug's
damage was very object-shape-dependent, which is what made the first run's
result suspicious enough to check rather than report).

Fixed by using `run_pick_and_place`'s own `step_hook`/`_set_phase` extension
point (a legitimate, already-existing observation hook — passing a
`PhaseTracker` through it changes no production behavior) to tag every
`solve()` call with its phase name, and restricting the correction to calls
where the phase name starts with `"descend"` — i.e. `descend`,
`descend_retry*`, `descend_refresh`, confirmed via grep to be the only
call sites where `obj_pos + [0,0,GRASP_HEIGHT_OFFSET]` (a fresh read of the
candidate grasp target) is converted to an IK target. Re-running the
Cracker/seed=1042 case that previously diverged to 1283mm now converges
cleanly with the target 0.6mm from the true capture center.

## Method

- Objects: Cracker (`PIPER_FINDINGS_SUMMARY.md`'s strongest validated
  wrist-fix effect, p=1.8e-5/p=0.027, n=152) and Pear (its validated null
  result) — the two objects the prior investigation already characterizes,
  so any correction effect is measured against a known baseline.
- Scene composition matches `piper_experiment_runner.py`'s own convention
  exactly (`pear`/`can`/`mustard` trio together, everything else solo) —
  required, since `PiperMultiObjectScene`'s placement sampler cannot fit
  large objects like Cracker alongside 6 others (documented
  `RandomizationError` risk in `piper_multi_object_scene.py`).
- 10 seeds/object, paired (`np.random.seed(trial_id)` before each
  `env.reset()`, identical for P0/P1 at the same trial id) — a small-scale
  directional first pass, not a replacement for a full n=152 confirmatory
  run.
- `use_oriented_grasp=True`, `wrist_friendly_orientation=True`,
  `candidate_selection=None` — the validated wrist-fix baseline condition,
  held identical across both arms.
- Recorded per trial: `candidate_target`, `actual_capture_center_at_pregrasp`,
  `capture_position_error_m`, `success`, `failure_stage`, `joint6`,
  `min_object_distance_m`, `bilateral_contact` (read directly from
  `data.contact`, snapshotted right after the close command and before lift
  motion begins — checking at the trial's final state would catch the
  object already released during `open`/`retract`, which isn't the
  question).

## Sanity check: did the correction actually take effect?

```
                n    mean(mm)   median(mm)   range(mm)
legacy         20      97.46        65.60    [65.58, 702.94]
corrected      20      57.17         0.24    [0.02, 1133.87]
```

Median is the right summary here, not mean — one seed (cracker 1049) fails
IK convergence on `descend_refresh` for BOTH arms (the object has already
been displaced by that point in the trial), inflating both means with a
genuinely-unreachable outlier unrelated to the correction. Median tells the
real story: **65.60mm → 0.24mm**, confirming the correction is applied
correctly and only where intended.

## Q1: do old (legacy) successes remain successes under correction?

```
cracker: 7/8 legacy successes remained successes (87.5%)
pear:    5/6 legacy successes remained successes (83.3%)
```

Mostly yes. One trial per object flips from success to failure under
correction. At n=10/object this could be sampling noise rather than a real
regression — not enough data to distinguish, and not the primary question
this pass was scoped to answer.

## Q2 (the important one): how many old failures were purely reference misalignment?

```
cracker: 0 of 2 legacy failures explained by the reference error
pear:    0 of 4 legacy failures explained by the reference error
```

**Zero.** Correcting the 65.6mm offset did not rescue a single historical
failure in this sample. Looking at `failure_stage`: nearly every failure
(both arms, both objects) is tagged `ik_no_converge:transit_high` — IK
failing to converge on the very first phase, a pure safety-height hover
point that has nothing to do with the grasp/capture reference at all (it is
not a `"descend"`-prefixed phase, so it is byte-for-byte identical between
arms — confirmed by the paired failures firing at the *same* seed in both
conditions). The dominant historical failure mode is orthogonal to the TCP
reference question.

## Q3: does the joint6/wrist-fix conclusion still hold post-correction?

```
                    n    mean joint6 (rad)   std     range
cracker legacy     10       -0.007          1.420   [-1.475, +3.140]
cracker corrected  10       +0.235          1.481   [-1.475, +2.658]
pear    legacy     10       +0.073          1.088   [-1.267, +1.541]
pear    corrected  10       +0.424          1.154   [-1.118, +2.033]
```

Not directly re-testable at n=10 (the original result needed n=152 for
significance) — but the underlying joint6 distribution measurably shifts
under correction, roughly +0.24 to +0.35 rad more positive on average for
both objects, with a wider range. This is mechanistically expected (the
corrected target sits 65.6mm further along a different effective reach
geometry, so DLS IK settles on a different joint6 solution) but it means
`PIPER_FINDINGS_SUMMARY.md`'s wrist-fix threshold — calibrated against
**legacy, uncorrected** joint6 values — should not be assumed to transfer
unchanged. **This needs a dedicated re-run at the original n=152 scale
against corrected targets before the existing p=1.8e-5/p=0.027 result is
cited again** — this A/B establishes that the joint6 distribution moved,
not whether the effect survives.

## Bonus finding: bilateral engagement drops under correction (Cracker)

```
                bilateral contact rate (post-close, pre-lift)
cracker legacy      9/10
cracker corrected   5/10
pear    legacy     10/10
pear    corrected   9/10
```

Despite near-identical final success rates (`Lift._check_success` is purely
height-based, so a one-sided pin/wedge grasp that never achieves true
bilateral contact can still count as "success"), Cracker's bilateral
engagement rate drops sharply under the corrected, millimeter-accurate
target — 90% to 50%. This directly reproduces this thread's earlier SO-101
finding: **bilateral engagement is a contact-sequence-dependent problem,
not fixed (and here, actively worsened) by more accurate static aim.**
Geometrically correct targeting changes *where* first contact happens
during closure, which can shift the outcome away from a symmetric pincer
grasp even when the final height-based success metric doesn't move. Not
something this pass modifies or investigates further — flagged because it
is a real, cross-platform-reproduced signal, and because a success rate
that looks stable can still be hiding a shift toward a less robust grasp
mode.

## Bottom line

The 65.6mm TCP reference correction is real, correctly implemented, and
verified to take effect (sanity check passes cleanly). It does **not**
explain any of Piper's historical grasp failures in this sample — the
dominant failure mode (`transit_high` IK non-convergence) is unrelated to
it entirely — and success rate is roughly unchanged to slightly worse at
this small n. Its downstream effects are on joint6 distribution (real,
needs the wrist-fix result re-validated before further citation) and
bilateral engagement quality (real, worse for Cracker, echoes the SO-101
line's contact-sequence finding). Per the agreed plan, no production
integration point is decided in this pass — that's the next, still-open
decision.

## Not done in this pass

- No formal significance test (n=10/object was explicitly a small directional
  pass, not a replacement for the n=152 confirmatory design already used for
  the wrist-fix result).
- No re-run of the wrist-fix/joint6 statistical comparison at scale against
  corrected targets — flagged as needed, not executed here.
- No investigation into *why* `transit_high` fails to converge as often as
  it does (the dominant failure mode found here, orthogonal to this A/B's
  scope).
- No decision on the production integration point for `T_eef_capture`.
