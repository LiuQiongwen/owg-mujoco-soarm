# Diagnostic-script sanity rules (Piper investigation, 2026-08)

Rules promoted from repeated, concrete failures in this codebase. Each has
already prevented at least one wrong conclusion.

## R1 — Suspicious-value rule

**If a measured quantity comes out suspiciously close to an input parameter,
to zero, to a geometric constant, or to an obvious physical bound, assume
the instrumentation or reference frame is wrong before explaining it as
physics.**

Caught three times in one investigation:

| observation | looked like | actually was |
|---|---|---|
| descend "tracking error" ≈ 15.0mm at a ±15mm offset | controller undershoot | metric differenced against the *unoffset* target — it was measuring the injected offset |
| pear `local_width` = 0.0 in every trial | pear ungraspable / degenerate | z-band defined around the eef target, which sits below pear's mesh |
| 0.02mm "width" on a 66mm pear | near-zero support | took the *innermost* mesh surface per side; a closed mesh always has vertices near x≈0 |

## R2 — Measure before the intervention's own effect normalises it

Sampling a relative pose *after* the gripper closes measures the grasp, not
the aim: closure pulls the object into the jaws and flattens the very
variation being tested. Observed: object-relative lateral/longitudinal
placement read 1.20mm / 20.1mm at both −15mm and +15mm offsets, while
pre-close first-contact terms over the same trials varied by 4–6mm.

Sample at the last state *upstream* of the mechanism under test.

## R3 — A constant cannot mediate a varying effect

Before proposing any mediator, check it actually moves under the
intervention. This single check eliminated four candidate families for the
P2 effect (contact-local geometry, IK solvability/joint margin, joint
tracking error, frame/phase-timing semantics).

## R4 — Decompose along named axes, not norms

A Euclidean distance hides which direction changed. The finger contact
envelope is asymmetric ([−71.5, +5.0] mm along eef-local Z), so
longitudinal placement is a distinct question from closing-axis placement
and is invisible in a norm.

## R5 — n=12 per cell is not a confirmatory sample here

Established in P1.1: under an i.i.d. rate of 0.825, observing 4/12 has
probability ~1e-4, yet that occurred — outcome is strongly driven by the
spawn draw. Small paired designs across *ordered* levels (P2's 5-level
sweep, 50/0 concordant) remain valid; single n=12 comparisons do not.

## R6 — A null criterion must first be achievable by the unmodified system against itself

Before requiring `A ≡ B`, measure `A ≡ A`. An equivalence threshold that
the system cannot meet when compared to a rerun of itself is a
specification error, not a failing result.

Concrete case: Phase 2Y pre-registered a "byte-identical trajectory"
validity gate (`max|ΔEEF| < 1e-12`). The instrument failed it — but so did
the *unmodified baseline compared to itself*, at the same order of
magnitude. The gate was unmeetable by any pair of runs on this platform.

Measured baseline-vs-baseline floor for the Piper sim (n=10 paired reruns,
`calib/phase2y_noise_floor.json`):

| metric | observed max | frozen threshold (×1.25) |
|---|---|---|
| max EEF position deviation | 1.310e-3 m | 1.638e-3 m |
| RMS EEF position deviation | 1.595e-4 m | 1.994e-4 m |
| max EEF orientation deviation | 0.246° | 0.307° |
| position at `descend_refresh` | 5.140e-4 m | 6.425e-4 m |

Two properties worth carrying: **success outcomes matched 10/10** while
trajectories did not, so outcome reproducibility and trajectory
reproducibility are different claims. And the divergence is *bifurcating,
not uniform* — 7/10 seeds reproduce `descend_refresh` position to ~1e-11 m
(numerically exact) while 3 diverge to 1e-5…5e-4 m, consistent with tiny
noise being amplified across a threshold (IK branch, contact onset) rather
than accumulating smoothly.

Corollary: report RMS alongside max. Max is spike-dominated — here it is
8–40× the RMS on the same trajectory pair.

## R7 — Collision groups come from physics attributes, never from names

Two distinct faults that look identical in logs (`inf` distance, empty set),
both hit during Phase 2Y Gate 4:

1. **Name-based matching is unsound.** Piper's `robot0_g0_vis` …
   `g6_vis` geoms carry `contype=1, conaffinity=1` and **do** collide,
   despite the `_vis` suffix. A matcher keyed on `robot0_link*` found
   nothing and reported arm distance as `inf` — one step from being written
   up as "the Piper arm has no collision geometry". Always select by
   `geom_contype` / `geom_conaffinity`, then filter by name.
2. **"Group absent" ≠ "matcher failed".** The Piper gripper genuinely has
   no palm collision geometry (its only colliding geoms are
   `finger7/8_collision`), so an empty palm group is a real model property
   (N/A). An empty *arm* group was a matcher bug. Treating both as failure
   blocks valid runs; treating both as N/A silently skips real checks. They
   must be distinguished explicitly, with the model-property case justified
   in code.

Corollary: a distance reading against a group you have not verified is
physical may be measuring a **visualisation marker**. Phase 2Y's earlier
`palm = −0.023 m` was distance to `right_eef_target_*`, a render-only geom.

## R8 — A treatment must be legal under the simulator's model-update semantics

"I modified a variable" is not the same as "the simulator supports modifying
that variable at runtime". Before any runtime model mutation, establish
whether the field is safe to change post-compile, whether it invalidates
collision-acceleration structures, and whether it requires `mj_setConst` or
a recompile.

Live case: Phase 2Y's finger-shift instrument mutates `model.geom_pos`
after compile (`piper_phase2y_smoke.py:74`,
`piper_phase2y_qualify.py:115`). MuJoCo's documentation lists
`geom_pos/geom_quat/geom_size/geom_rbound/geom_aabb` among unsafe runtime
modifications, because compile-time collision structures are derived from
them. If that applies here, Gate 3's pre-contact trajectory divergence is
an artifact of an illegal mutation rather than a physical effect — and no
amount of enlarging the baseline noise envelope would have revealed it.

**Decisive empirical test** (does not require trusting documentation):
build one compile-time variant with the finger geoms shifted in XML, and
compare it against the runtime-mutated model from an identical state. If
the two diverge, runtime mutation is unsafe on this model, confirmed
directly.

## R9 — Rules must be executable guards, not documentation

R1–R8 are only binding where they are implemented in code. A rule written
in a document does not protect a one-off query.

Demonstrated the hard way: R7 (select collision geoms by
`contype`/`conaffinity`, never by name) was written up after it caused a
near-false conclusion about the Piper arm — and was then violated in an
ad-hoc clearance query days later, which selected a non-colliding
`table_visual` geom, returned exactly `0.000000`, and invalidated a
margin measurement. The gated scripts obeyed R7; the throwaway query did
not, because nothing forced it to.

`scripts/piper_collision_geoms.py` now provides `collision_geoms()`,
`require_nonempty()` and `min_distance()`. Diagnostics must use these
rather than matching geom names directly.

Corollary: when a rule is added, ask what code change makes it
unavoidable. If the answer is "none", expect it to be violated in the next
ad-hoc script.
