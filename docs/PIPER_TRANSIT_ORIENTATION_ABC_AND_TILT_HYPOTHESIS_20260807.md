# Piper: transit_high orientation A/B/C + tilt-hypothesis falsification (2026-08-07)

Two further follow-ups to `docs/PIPER_TRANSIT_HIGH_AND_BILATERAL_AUDIT_
20260807.md`, per the explicit next-step request: (1) test whether relaxing
`transit_high`'s (and other non-grasp phases') orientation target rescues
the 7 known-failing seeds without damaging known successes, and (2)
directly verify the "tilted grasp_mat gives the capture-frame correction a
horizontal world-frame component" hypothesis with an actual geometric
decomposition, rather than leaving it as an unconfirmed leading guess.

Both read-only, zero diff on `tango_robot/piper_robosuite/` and
`tango_robot/piper_assets/`. Per the explicit scope: neither touches TCP
correction, contact mechanics, or joint6 rules -- the orientation A/B/C
uses legacy (uncorrected) target_pos throughout, varying only which
target_mat non-grasp phases receive.

## Part 1: orientation relaxation does NOT cleanly fix transit_high -- position reachability, not orientation, is the dominant constraint

Built `scripts/piper_transit_orientation_abc.py`: three arms, all using
LEGACY (uncorrected) `target_pos` throughout (TCP correction untouched, per
scope) --

```
T0 legacy:          every phase's target_mat = candidate's grasp_mat (today, unmodified)
T1 neutral_hover:    ONLY transit_high forced to DOWN_ORIENTATION
T2 relaxed_hover:    every non-descend phase forced to DOWN_ORIENTATION,
                     grasp_mat restored only at descend/descend_refresh
```

Run on the 7 known-failing pairs + 6 known-successful pairs (13 total),
each object/seed combination.

**A single-seed smoke test (pear/1042) looked promising going in** (T0/T1
both failed, T2 succeeded) -- but the full 13-pair sample tells a more
sobering story:

```
transit_high convergence:  T0 0/13   T1 1/13   T2 1/13   (barely moved)
overall success:           T0 8/13   T1 5/13   T2 7/13   (NET REGRESSION)
```

**T1 (the narrow, transit_high-only fix -- the literal reading of "just
stop forcing candidate orientation onto the hover waypoint") is a net
loss**: it rescued zero genuinely new failures (the one T1-success among
the 7 failing pairs, pear/1044, was already succeeding under T0) while
flipping one previously-working case (cracker/1047) to failure. **T2 (the
broader relaxation) is closer to a wash**: it rescued 3 genuinely new cases
(pear 1042/1045/1049) but flipped 2 previously-working cases to failure
(cracker/1041, pear/1047) -- a real trade, not a clean win. One case
(pear/1046) is worth flagging specifically: T1/T2 DID make `transit_high`
itself converge (the only 2 of 39 trials where it did) -- but this just
relocated the unreachability to the NEXT high-transit waypoint instead of
resolving it: `transit_above_tray` then failed to converge in its place.

**transit_high's own convergence rate barely moving (0/13 -> 1/13,
regardless of which relaxation) is the most informative single number
here.** If orientation were the dominant blocker, T1/T2 forcing a neutral,
easy-to-reach orientation should have unpinned most of the joint-limit
cases found in the prior audit. It didn't. Spot-checking the position error
itself: several cases show `err_cm` essentially UNCHANGED between T0 and
T1/T2 (e.g. pear/1042: 0.73cm in all three conditions) -- meaning for these
specific targets, the position alone (independent of what orientation is
requested) is already infeasible given `REAL_JOINT_LIMITS` at
`SAFE_TRANSIT_Z=1.05m` and this candidate's XY. The prior doc's framing
("transit_high fails because it's forced to match a tilted orientation")
was a genuine, real contributing factor for SOME cases (a few `err_cm`
values do drop measurably under relaxation, e.g. cracker/1048: 0.82cm ->
0.68cm), but this A/B/C shows it is not the PRIMARY driver for most of the
13 seeds tested. Position reachability at this specific transit height is
the larger open question.

**One more caveat worth reporting plainly, not smoothing over**: T0 in
this run doesn't perfectly reproduce every legacy outcome from the
original A/B at the same seed (e.g. cracker/1043 showed `success=True` in
the original A/B's legacy arm, `False` here under an logically-identical
T0). This is consistent with -- not contradictory to -- the reproducibility
characterization in `docs/PIPER_TRANSIT_HIGH_AND_BILATERAL_AUDIT_
20260807.md`: same-seed reruns carry a small (~1e-13 relative) floating-
point noise floor from multithreaded BLAS inside the DLS solver, confirmed
there to leave 6/7 sampled outcomes stable but not guaranteed to leave
ALL of them stable -- for cases sitting exactly on a convergence-tolerance
or decision-boundary knife-edge, it can occasionally flip a result. The
per-condition comparisons above use each run's OWN T0 as the paired
baseline (not the original A/B's numbers) specifically to control for
this, but the discrepancy itself is worth flagging as a standing caveat on
any single-trial (n=1 per seed) reachability claim at this scale.

## Part 2: the tilt hypothesis is falsified for the traced seeds -- and was imprecisely stated to begin with

Built `scripts/piper_bilateral_geometry_decomposition.py` to directly
compute, for Cracker's 3 traced seeds (1041/1042/1046, from
`outputs/piper_cracker_contact_trace.jsonl`):

```
delta_world = R_grasp @ [0, 0, -0.0656]
```

**Caught before trusting the result: the hypothesis as stated in the prior
doc is mathematically impossible, not just unconfirmed.** `LOCAL_OFFSET` is
a pure translation along the eef frame's own local Z axis. Projecting it
back into that SAME local frame: `R_grasp^T @ delta_world = R_grasp^T @
R_grasp @ [0,0,-0.0656] = [0,0,-0.0656]` exactly, for ANY rotation matrix
`R_grasp` (that's what `R^T R = I` means). The offset has EXACTLY ZERO
component along the local X (jaw-closing, confirmed by direct geometry:
`finger7`="left" sits at local X=-0.010m, `finger8`="right" at X=+0.010m in
the gripper's own frame) or local Y axis, in the gripper's own frame,
always -- this isn't something tilt could ever change. Verified this
identity numerically in the script (`local_check` assertion) before
reporting anything further.

What tilt COULD change is the offset's WORLD-frame horizontal component
(tilting the whole approach axis away from vertical naturally gives a
horizontal projection) -- that was the more defensible part of the
hypothesis, and it's directly testable. Measured for all 3 seeds:

```
seed=1041: gravity=+65.60mm  horizontal=0.00mm
seed=1042: gravity=+65.60mm  horizontal=0.00mm
seed=1046: gravity=+65.60mm  horizontal=0.00mm
```

**Zero horizontal component, all 3 seeds -- the offset is purely vertical.**
This is because `grasp_mat` for THESE specific seeds is essentially level
(≈`DOWN_ORIENTATION`), not tilted -- the 30-57 degree tilts reported in
`docs/PIPER_TRANSIT_HIGH_AND_BILATERAL_AUDIT_20260807.md` were measured on
a DIFFERENT seed set (the 7 transit_high-*failing* seeds), and citing that
figure for these bilateral-trace seeds without re-checking was an error in
the prior doc's hypothesis -- caught here by actually computing it rather
than reusing a number from an unrelated seed set.

**The tilt/horizontal-leak mechanism does not explain Cracker's bilateral
drop.** Whatever the real mechanism is, it isn't this.

One measurement caveat, reported rather than silently fixed: the script
also compares `|P - object|` (candidate target vs. object position) as a
"how well-centered is the candidate itself" sanity check. First run showed
an implausible 300-400mm gap -- traced to reading the object's position
AFTER `run_pick_and_place` had already finished (i.e., after the object
was placed in the tray, ~30-40cm from its spawn point), not at the moment
`descend_refresh` actually fired. Fixed via a phase-tracker snapshot at the
right moment, which brought the number down to ~32mm -- but that residual
32mm turns out to be a near-tautology: it's very close to
`|OBJECT_CENTROID_OFFSET_LOCAL["cracker"]|` (the file's own existing
body-origin-to-true-centroid correction), which rotation preserves the
magnitude of regardless of the object's actual orientation -- so this
particular comparison isn't informative evidence either way and shouldn't
be read as a finding.

### Revised hypothesis, not yet confirmed

Given the offset is confirmed PURELY vertical for these seeds, the
mechanism has to be about DEPTH along gravity, not horizontal centering.
`GRASP_HEIGHT_OFFSET = 0.0` targets the object's own CoM height
specifically -- per that constant's own header comment, tuned for
slip-resistance during lift, not for bilateral symmetry. Legacy's
uncorrected 65.6mm-too-shallow error means legacy's ACTUAL fingertip
contact point was historically ~6.5cm HIGHER than the nominal CoM-height
target (closer to Cracker's top), while corrected reaches the CoM height
essentially exactly. If Cracker's cross-section (or surface features --
box seams, printed panel edges, the exact CoM-height cross-section vs. a
point 6.5cm higher) isn't uniform along its height, a 6.5cm difference in
WHERE ALONG THE HEIGHT the fingers actually close could plausibly produce
a different (worse, for CoM-height) bilateral-symmetry outcome even with
zero horizontal displacement. This is a testable, mechanistically
plausible hypothesis given what's now confirmed -- not yet directly
checked (would need e.g. cross-sectional geometry at both heights, or a
grasp-height sweep), flagged for a future pass rather than investigated
further here.

## Updated status

| axis | status |
|---|---|
| `T_eef_capture` rigid transform | done |
| TCP correction takes effect | verified |
| TCP offset explains historical failures | no, ruled out |
| `transit_high` root cause (joint-limit pinning) | confirmed (prior pass) |
| `transit_high` fix via orientation relaxation | **tested, does not cleanly work -- T1 net regression, T2 a wash; position reachability at SAFE_TRANSIT_Z=1.05m is the larger open question, not primarily orientation** |
| bilateral engagement (Cracker) tilt hypothesis | **falsified for the traced seeds (offset confirmed purely vertical); revised depth/CoM-height hypothesis proposed, not yet tested** |
| joint6 historical conclusion | still needs re-run against corrected targets |
| production integration | still not decided |

## Honest bottom line

Both of this pass's leading hypotheses from the prior audit -- "relax
transit_high's orientation" and "tilt gives the offset a horizontal
component" -- did not survive direct testing as originally stated. That's
a real result, not a null one: it rules out two specific, cheap
explanations and narrows what's left. What remains open:

- `transit_high`'s failures are more about position reachability at
  `SAFE_TRANSIT_Z=1.05m` than about orientation-forcing -- a different,
  and probably harder, question (is 1.05m too aggressive for some XY
  positions given `REAL_JOINT_LIMITS`? would a lower or per-object transit
  height help?).
- Cracker's bilateral drop's mechanism is still open -- the depth/CoM-height
  hypothesis above is the current best guess, untested.

Per the standing instruction, no production integration decision is made
here either. Two credible next probes, not yet started: a `SAFE_TRANSIT_Z`
sweep (does lowering it change the 0/13 -> 1/13 convergence picture more
than orientation did?), and a grasp-height sweep on Cracker to test the
depth hypothesis directly.

## Not done in this pass

- No fix applied anywhere -- both scripts are diagnostic-only, matching
  every prior pass in this thread.
- No `SAFE_TRANSIT_Z` sweep (the natural next test given Part 1's result).
- No direct test of the depth/CoM-height hypothesis from Part 2.
- No larger-n confirmatory run of either result (n=13 pairs / n=3 seeds
  respectively).
