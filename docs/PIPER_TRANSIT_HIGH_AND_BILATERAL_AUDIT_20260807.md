# Piper: transit_high reachability + Cracker bilateral-contact trace (2026-08-07)

Two targeted follow-ups to `docs/PIPER_TCP_CORRECTION_AB_20260807.md`, per
the agreed plan: characterize the dominant blocker (`transit_high`
non-convergence) and the Cracker bilateral-engagement drop *before* deciding
a production integration point for `T_eef_capture`. Both read-only, zero
diff on `tango_robot/piper_robosuite/` and `tango_robot/piper_assets/`,
confirmed by `git status`/`git diff`.

Reproduce:
```bash
conda run -n tango python scripts/audit_piper_transit_high_ik.py
conda run -n tango python scripts/trace_piper_cracker_contact_sequence.py
```

## Part 1: transit_high is a genuine kinematic reachability wall, not a search problem

Audited all 7 (object, seed) pairs from the A/B where `ik_no_converge:
transit_high` fired, both arms, plus a within-run reproducibility check.

**Finding: every failing case, every one of the 7 solve attempts (1 primary
+ 6 `FALLBACK_SEEDS`), pins some joint at EXACTLY its declared
`REAL_JOINT_LIMITS` bound (margin = +0.000rad).**

```
pear (all 4 failing seeds):     joint2 pinned at its LOWER limit (0.0 rad)
cracker/1045, cracker/1048:     joint5 (near-target attempts) or
                                 joint4/joint1 (fallback attempts) pinned
```

This is the opposite of what `solve_multi_seed`'s fallback mechanism is
built to fix. Fallback seeds exist to escape a bad LOCAL basin when a
solution exists somewhere else reachable from a different start point --
but here, all 7 independent starting points (`READY_QPOS` plus 6 seeds
spanning most of joint1's range and a few distinct elbow postures) converge
to the SAME joint pinned at the SAME limit. That's the signature of a
target that is not reachable at all within `REAL_JOINT_LIMITS`, not a
seed-diversity gap. `FALLBACK_SEEDS` itself never varies joint2 below 0.2 --
consistent with (though not the root cause of) this: even a perfectly
diverse fallback set couldn't rescue a solution that requires joint2 < 0.

**Why:** `solve_and_move("transit_high", ...)` doesn't pass `target_mat`,
so it defaults to whatever `grasp_mat` the chosen candidate currently holds
-- not `DOWN_ORIENTATION`. Measured `target_mat`'s angle from
`DOWN_ORIENTATION` across the failing seeds: 4.2-57.0 degrees, several well
past 30 degrees. `transit_high` is nominally just "get to a safe hover
height" (`SAFE_TRANSIT_Z=1.05m`, chosen specifically to clear every
object's top surface -- see the constant's own header comment), but as
currently written it is ALSO required to match the final grasp orientation
at that height, for an object whose candidate happens to need a
significantly tilted grasp. Matching a tilted 6D pose at a specific height,
with a joint-limited 6-DOF arm, is a substantially harder reachability ask
than matching that height alone -- and every failing case here pins exactly
the joint (wrist joint5, or joint2/elbow-adjacent joints for the harder
fallback attempts) you'd expect to run out of range on first when forcing a
tilted orientation at a stretched-out hover height.

**This is independent of the TCP correction.** 6 of 7 seeds show byte-for-
byte identical `transit_high` targets between legacy and corrected arms (as
expected -- `transit_high` isn't a `"descend"`-prefixed phase, so
`CorrectedArmIK` passes it through unmodified), and both arms fail or
succeed together at those 6. One seed (cracker/1045) is the exception --
see the reproducibility note below for why, which turns out to be
unrelated to the correction too.

**Not a fix, a scoping finding.** No change made here. If `transit_high`'s
`target_mat` were relaxed to `DOWN_ORIENTATION` (or dropped to
position-only) instead of inheriting the candidate's tilted `grasp_mat`,
that would be a testable hypothesis for reducing this failure mode -- but
that's a change to `run_pick_and_place`'s own logic, out of scope for this
audit pass, and not decided here.

## Reproducibility check: floating-point noise floor, characterized precisely

Re-running the same arm twice at the same seed showed `max|qpos_diff|`
ranging from ~1e-15 (most seeds) up to ~2.8e-2 (cracker/1045) -- the
1e-15 cases are pure floating-point noise from non-associative summation
order in multithreaded BLAS inside the DLS solve's `np.linalg.solve` calls
(confirmed by hand: differs at the 15th significant digit, ~1e-13 relative,
5+ orders of magnitude below the 5mm/0.02rad convergence tolerances -- this
noise alone never flips a converged/not-converged outcome, confirmed:
`outcome stable` in all 7 reproducibility checks here). This resolves a
concern raised while building the harness (an early exact-dict-equality
check misreported this same noise as "non-deterministic" before being
fixed to compare with tolerance).

The one seed with a much larger 2.8e-2 rerun difference (cracker/1045) is
the SAME seed whose transit_high target differed between legacy and
corrected arms ("same transit_high target in both arms: False" -- the only
such case in all 7). Both anomalies trace to the same likely source: env
physics settling (`env.reset()`'s under-gravity drop/settle, itself subject
to the same multithreaded-BLAS-style floating-point noise) lands close
enough to a decision boundary in `wrist_friendly_orientation`'s
candidate-orientation selection that a ~1e-13-level perturbation
occasionally flips which orientation gets chosen -- not every seed, just
this one. Worth knowing (candidate/orientation selection is not perfectly
seed-locked in rare cases) but not a threat to the A/B's paired design at
the scale run so far (6/7 audited seeds, and by extension the A/B's
overall pattern, showed stable, matching outcomes).

## Part 2: Cracker's bilateral-contact drop traces to a real geometric mechanism, not noise

Traced the "descend_refresh" through "lift" window (final re-approach +
250-step close command) for Cracker at the 3 seeds where the corrected arm
succeeded overall but registered `bilateral_contact=False` (1041, 1042,
1046), both arms, for direct comparison.

```
              legacy                          corrected
seed   first_touch  @step   rot/trans    first_touch  @step(s)   rot/trans
1041   right         82      0.3deg/1.7mm  right (only)  1        0.4deg/7.0mm
1042   left          81      0.9deg/0.5mm  left  (only)  1        0.1deg/5.3mm
1046   right         82      0.2deg/2.5mm  right (only)  2        0.8deg/8.8mm
```

Under legacy, both fingers make contact within a few steps of each other
(step 81-86, a genuine near-simultaneous bilateral squeeze), and the object
barely moves (0.5-2.5mm translation across the whole window).

**Under corrected, the story is qualitatively different, not just later:**
one finger registers contact essentially IMMEDIATELY (step 1-2, i.e. it is
already touching the object at the very start of the descend_refresh
window, before the deliberate close command even begins) -- and the OTHER
finger never appears in MuJoCo's contact list even once across the entire
~330-step trace (confirmed directly: `left_dist_m` is `None` -- not
"far but present," genuinely absent from the broad-phase contact list --
for every single step of seed 1041's corrected trace). Object translation
during the window is also 3-5x higher (5.3-8.8mm vs 0.5-2.5mm) --
consistent with one finger contacting and nudging/pushing the object before
the second ever gets a chance to engage, rather than a controlled
simultaneous squeeze.

**Most likely mechanism (not fully isolated in this pass, flagged for
follow-up rather than claimed as proven):** the correction is a pure
translation along the eef_site's LOCAL Z axis (`[0,0,-0.0656]`, confirmed
rigid and rotation-identity in `docs/PIPER_CAPTURE_FRAME_CALIBRATION_
20260807.md`). For a perfectly level (`DOWN_ORIENTATION`) grasp this stays
purely vertical in world frame and shouldn't touch horizontal centering at
all. But Cracker's oriented candidates here have `target_mat` tilted
30-57 degrees off `DOWN_ORIENTATION` (measured directly in Part 1's audit,
same object). Composed with a tilted `target_mat`, the same local-Z offset
picks up a WORLD-FRAME HORIZONTAL component -- meaning the correction may be
shifting the aim point sideways relative to the object's midline by an
amount that scales with tilt, not just fixing the vertical/depth alignment
it was calibrated for. The old, uncorrected 65.6mm-short aim may have
accidentally been giving both fingers more symmetric clearance to converge
from; the geometrically-precise correction removes that margin and, for
tilted grasps specifically, may be landing the effective capture point
closer to one fingertip's true position than the object's actual center
between the fingers.

This is a plausible, mechanistically-grounded hypothesis consistent with
every data point gathered so far (legacy's near-simultaneous, low-
disturbance contact vs. corrected's immediate one-sided contact with
elevated object displacement, on the SAME 3 seeds), but it has not been
directly confirmed here (would need, e.g., checking whether
`local_offset` composed through each trial's actual tilted `target_mat`
correlates with the sign/magnitude of the observed touch-side and
displacement -- not done in this pass). Reported as the leading hypothesis,
not a settled conclusion.

## Updated status

| item | status |
|---|---|
| `T_eef_capture` calibration | done |
| TCP correction takes effect | verified |
| TCP offset explains historical failures | no, ruled out |
| legacy successes preserved | mostly |
| `transit_high` IK | **root cause found: joint-limit-pinned, all 7 seeds, independent of TCP correction; likely driven by forcing a tilted grasp_mat onto a pure hover waypoint** |
| bilateral engagement (Cracker) | **leading mechanism identified: correction may add a tilt-dependent horizontal component via a local-frame offset composed with non-level target_mat; not yet directly confirmed** |
| joint6 historical conclusion | needs re-run against corrected targets |
| production integration | still not decided -- this pass narrows the remaining unknowns further but doesn't resolve them |

## Not done in this pass

- No fix to `transit_high`'s orientation target (a real candidate fix --
  relaxing it to `DOWN_ORIENTATION` or position-only -- but that's a
  `run_pick_and_place` logic change, a different, larger decision).
- No direct correlation test of tilt-angle vs. bilateral-touch-side/
  displacement to confirm the horizontal-component hypothesis.
- No larger-n confirmatory run of either finding (n=7 seeds / n=3 seeds
  respectively -- directional audits, not confirmatory statistics).
