# Piper: SAFE_TRANSIT_Z reachability sweep + swept-volume bilateral features (2026-08-07)

Two "minimal version" diagnostics, per the literature-informed direction
(RM4D/reachability-map style probing for problem A, GraspGen-X/Contact-
GraspNet-style local-geometry features for problem B) -- built as cheap,
deterministic first passes rather than the full learned/formal machinery,
matching the explicit "不用一上来上 learned...不需要马上训练一个
GraspGen-X" scoping. Both read-only, zero diff on
`tango_robot/piper_robosuite/` and `tango_robot/piper_assets/`.

Reproduce:
```bash
conda run -n tango python scripts/piper_transit_reachability_sweep.py
conda run -n tango python scripts/piper_bilateral_swept_features.py
```

## Part A: SAFE_TRANSIT_Z=1.05m sits almost exactly on the arm's reachability cliff -- for all 13 candidates, not just the failing ones

Swept transit height z in [0.85, 1.15]m (0.05m steps) at each of the 13
tracked candidates' own (x, y), scoring by IK convergence and minimum
`REAL_JOINT_LIMITS` margin, orientation fixed to `DOWN_ORIENTATION`
throughout (isolating the pure position-reachability question, since the
prior orientation A/B/C already showed orientation isn't the dominant
factor for most seeds).

```
convergence rate by z, across all 13 candidates:
  z=0.85   13/13
  z=0.90   13/13
  z=0.95   13/13
  z=1.00   11/13
  z=1.05    2/13   <-- SAFE_TRANSIT_Z
  z=1.10    0/13
  z=1.15    0/13
```

This is about as clean a result as this whole investigation has produced.
Every single candidate is comfortably reachable (margins 0.37-0.61rad,
i.e. 20+ degrees of joint room) at z=0.85-0.95m, margins collapse sharply
by z=1.00m, and by z=1.05m all but 2 of 13 candidates have already lost
convergence entirely, joints pinned exactly at their limits. This is not a
property of a few unlucky failing seeds -- it's a property of the
`SAFE_TRANSIT_Z=1.05m` constant itself, given this Piper arm's
`REAL_JOINT_LIMITS`, essentially independent of which object or candidate
pose it's evaluated at. The orientation A/B/C's "position reachability, not
orientation, is the dominant constraint" reading is now directly
confirmed and quantified.

**Important tension, not a simple "just lower it" fix.** `SAFE_TRANSIT_Z`'s
own header comment in `piper_pick_and_place.py` explains why 1.05m was
chosen: it's meant to clear every object's top surface, specifically
Cracker's (documented there as the tallest, top surface at ~1.02m) --
`SAFE_TRANSIT_Z` was created BECAUSE a lower transit height let the arm
sweep through Cracker mid-transit and launch it off the table. Dropping
to the comfortably-reachable 0.90-0.95m band this sweep found would put
transit height AT OR BELOW Cracker's own top surface again, plausibly
reintroducing the exact collision bug `SAFE_TRANSIT_Z` was written to fix.
**A single global constant cannot satisfy "clears every object" and
"comfortably reachable for every candidate XY" simultaneously here** -- the
two constraints are in real tension, not just imprecisely tuned. This is
exactly what motivates a region/reachability-aware approach (candidate
XY-dependent transit height, chosen as the highest-margin reachable point
that still clears that specific object's top) over either a fixed height
or a naive per-object constant -- consistent with the TSR framing ("transit
is a safe region, not a hard point") this was scoped against, now backed
by concrete numbers showing why a single fixed value doesn't work.

**Not done here**: no XY-dependent/region-based transit height was
implemented or tested -- this sweep establishes the reachability profile
and the clearance/reachability tension, not a fix. The natural next
experiment (not run) is checking whether a per-candidate "highest
reachable z below object top + margin" choice both converges AND avoids
re-triggering the sweep-through-object bug, ideally validated the same way
the original bug was found (a step-by-step contact trace during transit,
matching this thread's own established methodology).

## Part B: static swept-volume gap features do NOT meaningfully predict bilateral touch order

Measured Piper's finger inner-face X extents directly (open -> closed):
left (finger7) -0.0500 -> -0.0040m, right (finger8) +0.0500 -> +0.0040m --
close to but not identical to the existing 14mm/104mm tip-gap LUT (that
used a stricter tip-vertex quantile filter; this uses the raw mesh AABB
extent along the closing axis, a coarser measure, documented as a
limitation).

For Cracker's 3 traced seeds, transformed the object's mesh into the
candidate's local frame (origin = the corrected capture-frame target,
which `T_eef_capture` reaches to ~0.2mm), restricted to a heuristic
contact band near local Y=0/Z=0, and computed each finger's closing gap
(travel distance from its open position to the object's nearest surface
point in that band):

```
seed=1041: left_gap=49.970mm  right_gap=49.977mm  imbalance=6.8um   predicted=left   observed=right  MISMATCH
seed=1042: left_gap=49.874mm  right_gap=49.927mm  imbalance=53.7um  predicted=left   observed=left   MATCH
seed=1046: left_gap=49.987mm  right_gap=49.982mm  imbalance=5.6um   predicted=right  observed=right  MATCH
```

2/3 matched -- but the imbalance magnitudes (6.8, 53.7, 5.6 MICROMETERS)
are the real story here, not the match count. At n=3 with imbalances this
small relative to any plausible mesh/measurement precision, 2/3 is not
distinguishable from chance. **This static, pre-close geometric snapshot
does not carry a real, robust signal for which side touches first** -- both
gaps sit right at the object's near-symmetric center-plane (both finger
gaps are ~49.9-50.0mm, i.e. essentially the FULL open-half-width, meaning
the object surface in-band sits almost exactly on the candidate's own
centerline either way), so the actual determination of which side touches
must come from something this snapshot doesn't capture: real closing
dynamics (however small a difference in contact timing, compliance, or
sub-millimeter object settling during the approach), not the static
pre-close geometry.

**This is a negative result worth keeping, not a failed script.** It
directly reinforces -- via an entirely different method, on a different
platform -- the standing finding from this thread's original SO-101 work:
bilateral engagement is fundamentally contact-SEQUENCE-dependent, not
predictable from a static capture-frame snapshot alone, however carefully
that snapshot is computed. A cheap deterministic geometric feature was a
reasonable thing to try before reaching for a learned swept-volume
representation (GraspGen-X-style) -- but this result suggests a static
feature, however framed, may hit a hard ceiling here, and a genuinely
dynamics-aware signal (e.g. instrumented short closing-motion rollouts, not
a pre-close snapshot) may be necessary for a feature that actually predicts
bilateral outcome.

## Updated status

| axis | status |
|---|---|
| `T_eef_capture` rigid transform | done |
| TCP correction takes effect / explains history | verified / ruled out |
| `transit_high` root cause | **confirmed and quantified: SAFE_TRANSIT_Z=1.05m sits on the arm's reachability cliff for essentially all candidates, in real tension with object-clearance requirements -- not a per-seed or orientation issue** |
| bilateral engagement (Cracker) | static geometric features (tilt-offset interaction, swept-volume gap) both tested and both ruled out / uninformative; mechanism appears genuinely dynamics-dependent |
| joint6 historical conclusion | still needs re-run against corrected targets |
| production integration | still not decided |

## Not done in this pass

- No region-based / XY-dependent transit-height implementation (the
  natural next step Part A's tension points to).
- No dynamics-aware bilateral feature (e.g. short closing-motion rollout
  signal) to replace the falsified static swept-volume-gap approach.
- No engagement with the specific external literature cited (RM4D,
  RichMap, TSR, GraspGen-X, GraspGen, Contact-GraspNet, CARP) beyond using
  their high-level framing to motivate these two minimal diagnostics --
  none of those papers/repos were independently fetched or verified in
  this pass.
