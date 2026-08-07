# Blocked-closure microbenchmark extended to Hammer/TomatoSoupCan/Banana (2026-08-07)

Zero production-code diff, confirmed by `git status`/`git diff` on
`tango_robot/` before and after. Same pattern as every experiment in this
thread: throwaway script patches an already-compiled `MjModel`; `move_gripper`,
`GRIP_CLOSED`/`GRIP_OPEN`, `_solve_ik_jaw_pos_only`, `object_collision_verts`
all used unmodified.

Reproduce:

```bash
conda run -n tango python scripts/microbenchmark_blocked_closure_codac.py
```

## Design

Extends `scripts/microbenchmark_blocked_closure.py` (fixed IK solved once and
frozen, no candidate/approach/park-restore/weld) from the two symmetric
fixtures to the three real CoACD objects. Per
`docs/CONTACT_ONSET_AUDIT_20260807.md`'s finding that `object_local_thickness_m`
centres on the same mesh-tip reference surface shown to disagree with the
actual pad geometry, this uses a corrected version
(`local_collision_support_width`) centred on the **pad geom midpoint**
instead, computed independently in this script rather than by editing
`tango_robot/jaw_metrology.py`.

Per object: drop at a fixed (no RNG) position, settle under gravity, solve IK
once (targeting the settled object's centroid), freeze that arm qpos and the
settled object pose, then reuse both verbatim across every subsequent config
trial for that object.

## Repeatability holds

HammerC / S1_stiff_pads × 5 repeats: `steady_pad_dist_fixed_m` identical to
15 decimal places across all 5. Confirms the fixture benchmark's determinism
finding extends to real CoACD objects under this design.

## A new, genuine finding: bilateral engagement is not guaranteed by pad stiffness alone

| config | HammerC steady (fixed/moving, mm) | TomatoSoupCanC | BananaC |
|---|---|---|---|
| S0_baseline | +41.9 / −5.6 | +0.2 / −4.7 | +37.5 / −3.3 |
| S1_stiff_pads | +34.7 / **−0.07** | +168.6 / +166.7 | +23.3 / **−0.07** |
| S1b_7.5ms | +34.2 / **−0.08** | +11.7 / **+0.01** | +21.9 / **−0.08** |

**The moving-side pad achieves excellent contact under S1/S1b on Hammer and
Banana** (−0.07 to −0.08mm, matching the fixture's near-ideal result) — the
penetration-reduction finding transfers to real objects on the side that does
engage. **But the fixed-side pad never reaches the object at all, on any
object, under any config** (21.9–169mm away). This is not a config effect —
it's identical in character across S0/S1/S1b, which is the tell that
something other than contact stiffness is responsible.

## This is not the earlier IK/freeze bug

Checked directly before writing this up (not assumed): at the moment of
freeze, before any closing motion, both pads sit almost symmetrically around
the object —

```
pad_fixed  to centroid: 43.80 mm
pad_moving to centroid: 43.62 mm
```

— and the jaw midpoint matches the object centroid to sub-millimetre
precision. **The setup is correct.** The asymmetry develops during the
400-step closing simulation itself: HammerC's `obj_displacement_m` over that
window is ~9.7mm, which closely matches the ~9mm change in the fixed-pad
distance (43.8mm → 34.7mm under S1). The object is being pushed or rotated by
the approaching moving pad in a way that does not centre it between the two
pads — plausible for an asymmetric CoACD mesh resting at an arbitrary
orientation, where the moving pad's first contact point can impart a sideways
or rotational component rather than a clean squeeze.

## Why this matters

This is exactly the "unilateral near-contact cannot be bilateral" pattern
`tango_robot/pad_fidelity.py`'s classifier was built to catch (property 2 in
its test suite) — and it shows up here on real, unmodified geometry even
under the contact configuration (S1/S1b) that fixed the fixture's
penetration problem cleanly. **Stiffer, priority-overridden pad contact
solves "does the pad that touches the object penetrate it" — it does not
solve "do both pads touch the object at all."** Those are different failure
modes, addressed by different fixes: contact stiffness for the first,
approach/candidate geometry (or, per this data, possibly closing-dynamics
compliance/friction on the pad, unexplored here) for the second.

TomatoSoupCanC's numbers are noisier (steady distance ranges from +0.2mm to
+169mm across configs) and its known large size (106.7mm local support width
— wider than the jaw's own mechanical range in places) likely compounds
whatever is happening here with a second effect; not disentangled in this
pass.

## What this does and doesn't change

**Does not undermine** the S0-vs-S1/S1b penetration finding — every prior
measurement of that (pad-fidelity diagnostic, attribution experiment, box
fixture) was already scoped to trials where bilateral engagement occurred, so
this doesn't contradict them, it adds a case those measurements didn't
probe: what happens on the side that DOESN'T engage.

**Does add** a concrete, evidenced caveat for anyone about to treat S1b as
settled: penetration reduction and bilateral engagement are separate
properties, and this thread has now only validated the first.

## Not done here

- Root-causing exactly why the moving pad's approach pushes/rotates the
  object asymmetrically (would need per-step object trajectory/contact-force
  logging during the closing window — a further, more detailed instrument
  than this pass built).
- TomatoSoupCanC's specific noise.
- Anything about real hardware — still the validation step nothing in
  simulation can substitute for, per this thread's standing recommendation.
