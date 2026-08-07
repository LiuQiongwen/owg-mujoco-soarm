# Geometry-derived capture reference: first (static) version tested, found insufficient (2026-08-07)

Follows `docs/BILATERAL_ENGAGEMENT_MECHANISM_20260807.md`, which found a
hand-searched +10mm offset fixes Hammer's bilateral engagement and explicitly
warned against turning that into an object-specific lookup table. This
implements the requested first version of a *derived* (not searched) capture
reference and tests whether it reproduces the manual result.

Zero production-code diff, confirmed by `git status`/`git diff` on
`tango_robot/` before and after. Reuses (does not modify)
`tango_robot.jaw_metrology.object_collision_verts`/`closing_axis` and
`scripts/microbenchmark_blocked_closure_codac.py`'s frozen-pose machinery.

Reproduce:

```bash
conda run -n tango python scripts/derive_capture_reference.py
```

## Method implemented

Per the agreed first-version design: two-stage IK, not a lookup table.

1. Bootstrap: solve IK once targeting the object's raw centroid (the old
   reference).
2. At that bootstrap pose, find the object's local collision-vertex support
   along the closing axis, in a slab centred on the PAD geom midpoint (not
   `tip_points()` — the contact onset audit found that reference wrong).
3. `capture_point` = the midpoint between the two along-axis extremes of that
   local support — geometry-derived, not searched.
4. Solve IK a second time targeting `capture_point`, freeze, run the
   standard S1_stiff_pads closure trial.

## Result: the derived correction is far too small

| object | derived offset (closing-axis mm) | local support width | bilateral? |
|---|---|---|---|
| HammerC | **−1.20 mm** | 38.7 mm | No (final: +27.5 / −0.07mm) |
| BananaC | +1.56 mm | 46.1 mm | No (final: +39.4 / +1.6mm) |
| TomatoSoupCanC | +3.21 mm | 106.7 mm | No (final: distmax-saturated, IK converged nowhere near the object) |

**None of the three achieve bilateral contact.** Most tellingly: Hammer's
derived correction is −1.2mm — an order of magnitude smaller than, and
opposite in sign trend from, the +10mm that worked when searched by hand.
Hammer's `final_dist_fixed_m` did improve somewhat under the derived
reference (35.0mm baseline → 27.5mm) — a real, non-zero effect in a
sensible direction — but nowhere near what's needed.

TomatoSoupCanC's IK converging to a position ~1m from the object (the
`mj_geomDistance` distmax cap) is consistent with what was already known:
its local support width (106.7mm) exceeds the jaw's own mechanical range —
this object may not be gripper-capturable at this scale at all, independent
of any aiming refinement.

## What this rules out, and what it points to

Ruling out is the useful part of a negative result. This tests, and rejects,
the hypothesis that Hammer's engagement failure is **substantially a static
geometric mis-centering** — if it were, a geometrically-computed local
support midpoint should have landed close to the empirically-found +10mm
sweet spot. It didn't, by almost an order of magnitude.

This strengthens, by elimination, the second mechanism
`BILATERAL_ENGAGEMENT_MECHANISM_20260807.md`'s trajectory instrumentation
already flagged: **object lateral drift and rotation continue growing for
hundreds of steps after first contact** (Hammer: 5.2mm at step 20 → 7.9mm at
step 380). A purely static aim-point correction cannot address a
still-accumulating dynamic effect. The +10mm manual offset likely worked not
because it found "the true static centre" more precisely than this
derivation did, but because shifting the aim far enough changed how the
*closure dynamics* played out (e.g., changing which surface region the
moving pad first contacts, and therefore the direction/magnitude of the push
it imparts) — a fundamentally different mechanism than "correcting a
mis-centred aim point."

## Consequence for the design

The user's own message anticipated this possibility explicitly: "SO-101 不是
理想平行夹爪 ... 更严格的版本应该进一步考虑 fixed pad surface / moving pad
swept surface ... 而不是两个静态平面之间的中点." This result is the evidence
that the upgrade is necessary, not optional — a static two-surface-midpoint
capture reference, computed correctly and cheaply, is not sufficient for
Hammer, and by extension should not be assumed sufficient for arbitrary
asymmetric objects in general.

**Not implemented here** (substantially larger scope, not started without
explicit direction): a closure-trajectory-aware capture reference that
models the moving pad's swept path through its closing arc rather than a
single fixed-angle position, and finds where that swept surface would first
and most-symmetrically engage the object — the "second version" this result
shows is needed, not merely nice-to-have.

## Where this leaves the thread's open items

Per the 5-item plan from the prior message:

1. ~~Geometry-derived capture reference~~ — attempted (first/static version),
   tested, found insufficient. Needs the swept/dynamic upgrade before it can
   replace the searched offset.
2. Verify on Hammer/Banana/Tomato — done as part of (1); Tomato additionally
   confirmed likely uncapturable at this scale regardless of aiming.
3. Contact-force/drift instrumentation on Banana's remaining failure — not
   started.
4. Formal opening API — not started (deferred since steps ①②, still pending).
5. Real SO-101 blocked-closure anchor — not started.

Bilateral engagement fidelity remains an open problem with a partially
understood mechanism (static mis-centering is now a ruled-out primary cause;
closure dynamics is the standing hypothesis, not yet directly instrumented
with contact-force data).
