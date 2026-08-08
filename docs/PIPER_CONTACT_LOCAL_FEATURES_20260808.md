Phase A: contact-local geometry is NOT the mediator of the P2 effect (2026-08-08)

Recomputed P2's geometry on the finger contact region instead of the eef
target, then joined to the existing 180 P2 outcomes. No new rollouts — the
aim point and the geometry around it are pre-execution quantities, so
`env.reset()` plus candidate computation is sufficient.

```bash
conda run -n tango python scripts/piper_contact_local_features.py
```

## Two feature bugs fixed, and the fix is validated

1. **Wrong reference frame (the P2 bug).** Measured the finger contact
   region directly: it spans **[−71.5, +5.0] mm** along the eef site's own
   local Z. P2 used a ±15mm band around the aim target, which sits almost
   entirely outside that region — hence pear's all-zero widths.
2. **Wrong surface definition.** A first version took the *innermost*
   surface on each side. That is meaningless for a closed mesh: the surface
   wraps at the top and bottom of the cross-section, so points near x≈0
   always exist and both innermost values collapse to ~0 (it reported
   0.02mm widths on a 66mm pear). Parallel jaws close onto the **extremes**,
   so the extent across the band is the right quantity.

The corrected widths validate against known object dimensions:

| object | measured support width | known narrow-axis bbox |
|---|---|---|
| cracker | 67.2mm | 71.7mm |
| mustard | 57.9mm | 58.1mm |
| pear | 66.5mm | 66.5mm |

So the features are now trustworthy, which is what makes the following
negative result meaningful rather than another artifact.

## The result: the geometry does not move at all, but the outcome does

| object | offset −15 → +15mm | support width | opening margin | centring err | antipodal | success |
|---|---|---|---|---|---|---|
| pear | | **66.54 → 66.54** | 33.46 → 33.46 | 0.72 → 0.72 | 1.00 → 1.00 | **3/12 → 12/12** |
| mustard | | **57.90 → 57.90** | 42.10 → 42.10 | −5.88 → −5.88 | 0.87 → 0.90 | 6/12 → 9/12 |
| cracker | | 67.45 → 67.10 | 32.55 → 32.90 | −2.80 → −3.11 | 0.99 → 0.99 | flat |

**Every contact-local geometric feature is constant across the intervention
for pear and mustard, while success varies from 3/12 to 12/12.** A variable
that does not change cannot mediate an effect that does.

This is physically sensible rather than another measurement failure: the
jaws' contact region genuinely spans ±20mm in Y, so shifting the aim ±15mm
along the object's long axis slides that window without changing what the
jaws can actually enclose. The candidate grabs the same cross-section
either way.

**Conclusion: the whole contact-local geometry family — support width,
opening margin, centring error, antipodal score, left/right surface — is
ruled out as the mediator of the P2 effect.** Combined with P2's own
finding that success does not track local cross-section width in mustard,
the cross-section hypothesis is now falsified from two independent
directions.

## What remains, and it points somewhere different

The one quantity that *does* move monotonically with the intervention is
`rel_dist_at_descend` — pear 46.5 → 25.5mm, mustard 52.9 → 48.0mm — tracking
success exactly, alongside `gripper_q_at_close`.

If the jaws see identical geometry at every offset but the arm ends up
measurably closer to the object at some offsets than others, the difference
is in **where the arm actually arrives**, not in what it is aiming at. That
makes the surviving candidate mediator **kinematic** (reachability /
tracking accuracy at the descend target), not geometric.

This has a concrete consequence for which literature applies. For *this*
effect the relevant direction is the workspace/reachability one (Task Space
Regions, reachability maps) rather than the grasp-geometry one
(Contact-GraspNet, Dex-Net antipodal sampling) — the latter was the natural
reading of P2's asymmetry, and it does not survive this measurement. The
contact-local framing may still matter for a candidate critic in general;
it just is not what produced the P2 effect.

It also rhymes with earlier, independently-measured findings on this
platform: `SAFE_TRANSIT_Z = 1.05m` sitting on a reachability cliff, and
joints pinning at `REAL_JOINT_LIMITS` across the workspace. Those were
withdrawn as *failure* explanations, but the underlying reachability
structure was always real.

## Status

- Cross-section / contact-local geometry as the P2 mediator: **falsified**.
- Kinematic reachability at the descend target: **the surviving
  hypothesis**, untested.
- Next test, cheap and direct: for the same (object, seed, offset) grid,
  record the descend IK residual, joint-limit margin, and commanded-vs-
  achieved eef error. If those move monotonically with the offset while the
  geometry does not, the mediator is identified.
- Pear centroid-offset question remains open and underpowered (n=12); it is
  now less likely to be a geometry bug and more likely another face of the
  same reachability effect.
