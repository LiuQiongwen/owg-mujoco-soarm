# Contact onset audit: why did the box show "+6.71mm past the surface"? (2026-08-07)

Answers the diagnostic question raised about the blocked-closure
microbenchmark: is `+6.71mm` (S1/S1b's apparent stop-point past the box's
known 30mm surface) a geometry/reference problem or a contact-stiffness/
actuator-equilibrium problem? No new simulation needed — the answer was in
data already collected, once the right two numbers were compared directly.
Zero production-code diff, as with everything in this thread.

## The four candidate explanations, checked

1. **Pad collision box effective surface position wrong.** Not the cause —
   see #4.
2. **Fixture's real collision extent isn't actually 30mm.** Ruled out:
   `register_primitive_geom`'s box is a MuJoCo primitive with `size=(0.05,
   0.05, 0.015)` — full thickness `2×0.015 = 30mm` is exact by construction,
   nothing to mismeasure.
3. **MuJoCo `margin`/`gap` triggering early "contact."** Ruled out by direct
   read: `geom_margin` and `geom_gap` are `0.0` on both pad geoms.
4. **`true_opening_m`'s reference surface doesn't match the actual pad
   collision surface.** **Confirmed — this is the cause.**

## The mismatch, quantified

`JawMetrology.true_opening_m(q)` (`tango_robot/jaw_metrology.py`) is built
from a "tip" point selection on the **original finger meshes** — the top
quartile (`TIP_QUANTILE=0.75`) of face-sampled points by distance from the
hinge, selected when this module was built for step 4's opening calibration
(`docs/JAW_OPENING_CALIBRATION_STEP1_20260807.md`), before pad geometry
existed.

The pads that actually collide in `measured_pads*` modes
(`tango_robot/jaw_pads.py`) are derived **independently**, from a different
point selection (a distal gripping band, `BAND_LO/BAND_HI = 0.058–0.086`,
angular-extreme selection per radius bin) and offset outward by `PAD_PROUD +
PAD_HALF_THICK`. Nothing links the two derivations, and nothing re-validated
that they agree after pads were introduced in step 3.

They don't agree, and not by a constant offset:

| q (rad) | `true_opening_m` (mesh-tip LUT) | pad-box `mj_geomDistance` | delta |
|---|---|---|---|
| −0.1745 | 2.06 mm | −3.80 mm | −5.86 mm |
| −0.0145 | 14.64 mm | 7.75 mm | −6.89 mm |
| 0.1454 | 26.37 mm | 19.53 mm | −6.84 mm |
| 0.3054 | 37.98 mm | 31.25 mm | **−6.73 mm** |
| 0.4654 | 46.71 mm | 42.61 mm | −4.10 mm |
| 0.6254 | 54.34 mm | 53.34 mm | −1.00 mm |
| 0.7854 | 61.68 mm | 63.14 mm | +1.46 mm |
| 0.9454 | 68.66 mm | 71.83 mm | +3.17 mm |
| 1.2654 | 81.13 mm | 87.68 mm | +6.55 mm |
| 1.7453 | 95.74 mm | 107.58 mm | +11.83 mm |

The delta crosses zero around q≈0.7 rad and grows in both directions —
**not a constant, not correctable by a single offset.** The box's blocked
equilibrium under S1 sits at q=0.2875, right where delta≈−6.7 to −6.8mm —
which is exactly the "+6.71mm past the known surface" the microbenchmark
reported. Mystery solved: `true_opening_m` was reporting a DIFFERENT
surface's position, off by almost exactly the observed discrepancy.

## The correction

The microbenchmark's `min_pad_dist_fixed/moving_m` and
`steady_pad_dist_fixed/moving_m` fields were **never affected** by this —
they come from `env._pad_to_obj_dist(gids)`, which queries the actual pad
geoms directly via `mj_geomDistance` (the same method verified equal to
`contact.dist` to within 0.03mm back in step 3). Only the derived
`"gap vs known thickness"` summary column, built from `settled_true_opening_m
- known_thickness_m`, was contaminated. The raw measurements were correct
all along; only one presentational column was misleading.

Reading the correct fields directly from
`outputs/microbenchmark_config_compare.jsonl` (no rerun needed):

| config | steady pad distance (fixed / moving) | min pad distance (fixed / moving) |
|---|---|---|
| S0_baseline | **−7.90 / −8.45 mm** (genuine ~8mm compression) | −8.53 / −8.98 mm |
| S1_stiff_pads | **−0.07 / −0.07 mm** (essentially exact contact) | −1.01 / −0.26 mm |
| S1b_7.5ms | **−0.08 / −0.08 mm** (essentially exact contact) | −1.44 / −0.63 mm |

**S1/S1b do not overshoot past the surface.** They settle within 0.1mm of
the box's true surface — close to ideal rigid-body contact. The small gap
between `min` (slightly deeper, a brief transient as the position actuator's
own approach decelerates against the object) and `steady` (settling back to
~0) is ordinary underdamped settling, not a residual error. There is no "new
problem on the over-blocking side" — the earlier framing (built on the wrong
reference surface) manufactured a problem that measured penetration data
does not show.

## What this changes elsewhere in this thread

Any table in this thread's docs built from `settled_true_opening_m -
known_thickness_m` for a **fixture** inherits this same reference-surface
distortion and should be read with that caveat:

- `SOLVER_CONTACT_ATTRIBUTION_20260807.md`'s "fixtures only: settled opening
  vs known thickness" table
- `SOLREF_STABILITY_SWEEP_20260807.md`'s fixture-thickness comparisons
- `BLOCKED_CLOSURE_MICROBENCHMARK_20260807.md`'s "gap vs known thickness"
  columns (corrected above)

**Not affected**, because they never used `true_opening_m` for this purpose:
the pad-fidelity diagnostic's `pad_obj_dist_fixed/moving_m` fields and the
`EXCESSIVE_PENETRATION_DOMINANT` classifier (built entirely from
`_pad_to_obj_dist`), and every **named-object** (Hammer/TomatoSoupCan/Banana)
penetration number in this thread, which came from the production
`pad_obj_dist_fixed/moving_m` path, not the fixture-specific
known-thickness shortcut. The headline findings of every prior doc in this
thread — S0 shows persistent excessive penetration, S1 substantially
eliminates it, S1 and S1b behave identically — all stand and are, if
anything, understated by the corrected numbers above (S1's true behavior is
better than what was reported).

## Not fixed here

`true_opening_m`/`JawMetrology`'s tip-point selection is not corrected to
match the pad geometry — that would touch `tango_robot/jaw_metrology.py`,
production code, out of scope for this thread's "don't modify any official
API" constraint. Flagging it as a real, load-bearing defect for whoever next
touches jaw metrology: **`true_opening_m` should either be rederived from
the same point selection `jaw_pads.py` uses, or renamed/documented as
specifically the "step-4 mesh-tip opening," not "the opening," so it stops
being reached for as a general-purpose ground truth in pad-mode contexts
where it silently measures the wrong surface.**
