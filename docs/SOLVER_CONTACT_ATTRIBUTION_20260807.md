# Small-scale solver/contact attribution experiment (2026-08-07)

> **Correction (same day, later):** the follow-up stability sweep
> (`docs/SOLREF_STABILITY_SWEEP_20260807.md`) found the "harmless, appeared
> after all data was written" characterization of the QACC warning below was
> too narrow — a properly-scoped recheck found a genuinely reproducible
> warning on at least one specific trial, though NOT the whole-attempt
> "1876 m/s" reading initially reported for it, which turned out to be a
> measurement artifact (see that doc for the full account). The
> `EXCESSIVE_PENETRATION_DOMINANT` 6/10→0/10 finding below is unaffected and
> stands; do not treat this doc's implied "S1 is a safe default" framing as
> validated — it isn't, yet.
>
> **Second correction:** `docs/CONTACT_ONSET_AUDIT_20260807.md` found this
> doc's "fixtures only: settled opening vs known thickness" table (built from
> `settled_true_opening_m`) is contaminated by a reference-surface mismatch
> between `true_opening_m` and the actual pad collision geometry, unrelated
> to contact physics. The `EXCESSIVE_PENETRATION_DOMINANT` classifier and
> every named-object number are unaffected (never used `true_opening_m` for
> this). The fixture table's numbers understate how well S1 performs; see the
> audit doc for corrected figures.

Answers the question left open by the pad-fidelity diagnostic: is the 100%
excessive-penetration rate on legacy successes attributable to **contact
softness** (fixable by solver configuration) or something else (actuator
authority — already ruled out in step ②, object mass/inertia, mesh geometry)?

**Zero production-code diff.** `EnvironmentSoArm`, `_build_scene_xml`,
`register_primitive_geom`, `move_gripper`, `GRIP_CLOSED`/`GRIP_OPEN` are used
completely unmodified. The only non-default behavior is patching numeric
fields (`model.opt.*`, `model.geom_solref/solimp/priority` on the two pad
geoms only) on an already-compiled `MjModel` from inside the throwaway
experiment script — MuJoCo reads these live at every step, so this needs no
MJCF change and vanishes when the process exits. Confirmed by `git status`/
`git diff` on `tango_robot/` before and after this work: no tracked file
changed.

Reproduce:

```bash
conda run -n tango python scripts/experiment_solver_contact_attribution.py
```

## Design

Two canonical fixtures with **exactly known thickness**, via the existing
`register_primitive_geom`/`load_primitive` mechanism (no manifest edits): a
30 mm-thick box and a 40 mm-diameter cylinder. Plus the three real objects
already implicated (Hammer/TomatoSoupCan/Banana). 2 seeds each = 10 scenes,
under 4 contact configurations (10 mechanically identical trials each = 40
total):

| config | pad solref | pad solimp | pad priority | cone | impratio |
|---|---|---|---|---|---|
| S0_baseline | [0.02, 1.0] (stock) | [0.9, .95, .001, .5, 2] (stock) | 0 | pyramidal | 1.0 |
| S1_stiff_pads | [0.005, 1.0] | [.95, .999, .0001, .5, 2] | **1** | pyramidal | 1.0 |
| S2_elliptic_cone | same as S1 | same as S1 | 1 | **elliptic** | 1.0 |
| S3_high_impratio | same as S1 | same as S1 | 1 | elliptic | **10.0** |

S0 is not "an unconfigured baseline" — Newton solver, pyramidal cone, and
impratio=1.0 were confirmed by direct inspection to already be this codebase's
running defaults, not something the literature review needed to recommend.
`geom_priority=1` on the pads matters mechanically: MuJoCo mixes two geoms'
contact parameters by simple average unless one has higher priority, in which
case its parameters win outright. Without it, stiffening only the pads would
be diluted toward whatever the object's own (default, soft) parameters are.

Only the pad geoms' own parameters are touched — not a global re-tune of every
object's collision geometry. That's a deliberate scope limit: this project
controls its own gripper, not every asset in the manifest.

## A methodology bug caught and fixed mid-run

The first pass produced **byte-identical results across all four configs** —
every metric matched to sub-mm precision, which is not a real physics finding
（solver/cone/priority changes this large cannot legitimately produce zero
effect). Root cause: `EnvironmentSoArm.__init__`'s `obj_names` kwarg is
silently swallowed by its `**kwargs` catch-all and never read — every prior
script in this investigation (and, it turns out, apparently every script
generally) actually populates the pool via an explicit `load_obj()` call
after construction, which triggers `_rebuild_model()` the first time it sees
a new pool name. The first version of this script applied its config patches
*before* that `load_obj()` call, so `_rebuild_model()` silently discarded them
and rebuilt a fresh, default-parameter model for every trial regardless of
which config was nominally active. Fixed by moving `apply_config()` to run
*after* `load_obj()`. Re-verified on one trial before rerunning the full
matrix that the four configs now produce genuinely different `model.opt`/
`geom_solref` state.

A second bug (state leakage sharing one `EnvironmentSoArm` across every
object and config) was caught even earlier, before the first full run: a
>200 mm nonsense pad-distance reading appeared on one specific
(object, config, seed) combination that did not reproduce when that same
combination was re-run in isolation. Not root-caused (out of scope for this
experiment), but resolved by constructing a fresh `EnvironmentSoArm` per
trial, eliminating that whole class of bug by construction at the cost of
~40 constructions instead of 1 — acceptable for "small scale."

## Results

### `geometric_verdict` (pad-fidelity diagnostic's own classifier, unmodified), 10 trials/config

| config | NO_ENGAGEMENT | PLAUSIBLE_ENGAGEMENT | EXCESSIVE_PENETRATION_DOMINANT | AMBIGUOUS |
|---|---|---|---|---|
| S0_baseline | 0 | 0 | **6** | 4 |
| S1_stiff_pads | 1 | 3 | **0** | 6 |
| S2_elliptic_cone | 1 | 4 | **0** | 5 |
| S3_high_impratio | 2 | 3 | **0** | 5 |

**S1 alone (pad-only stiffening + priority override) took persistent
excessive penetration from 6/10 to 0/10.** S2/S3 hold that at zero without
clearly improving further on this small sample — the additional levers
(elliptic cone, higher impratio) did not show a clear additional effect here.

### Fixtures: settled opening vs. known thickness (mm; 0 = pads stop exactly at the surface)

| config | Box (30mm) | Cylinder (40mm) |
|---|---|---|
| S0_baseline | −6.7 | −15.3 |
| S1_stiff_pads | −1.6 | −7.6 |
| S2_elliptic_cone | −0.7 | −8.1 |
| S3_high_impratio | −0.7 | −3.9 |

Monotonic-ish improvement from S0 to S3 on unambiguous, mesh-independent
ground truth (these numbers don't depend on any collision-geometry naming
convention — they come straight from the `q -> true_opening_m` LUT). Residual
compression remains even at S3 (−0.7 to −3.9 mm) — stiffening pad-only
parameters clearly helps but does not fully eliminate penetration on these
two fixtures in this sample.

## Answer to the motivating question

**Yes, substantially attributable to contact softness, not something more
fundamental.** Step ② already ruled out actuator authority (the actuator
tracks its own free-space target to 0.06 mrad). This experiment shows that
changing only the pad geoms' contact stiffness — without touching control,
without touching any other object's parameters — collapses persistent
excessive penetration from 6/10 to 0/10 on this sample, and reduces (though
does not zero) the fixtures' measured compression through known-thickness
objects.

## An honest caveat: an instability warning appeared

```
WARNING: Nan, Inf or huge value in QACC at DOF 10. The simulation is unstable. Time = 2.4080.
```

Appeared exactly once, in the log **after** "wrote 40 trials" had already
printed — i.e. after every trial's data was recorded, during the last env
instance's teardown, not during a measured grasp. It did not corrupt any
recorded row. Not chased to a specific cause. Flagged because S1-S3's pad
solref time constant (5 ms) is only 2.5× the model's 2 ms timestep — within
MuJoCo's documented stability floor (≥2×) but close to it. **Before adopting
any of these configs as a real default, this warrants its own dedicated
stability sweep** (vary timestep alongside solref, more seeds, longer
horizons) — this experiment's job was attribution (is softness the cause?),
not certifying a production-ready configuration.

## What this does not do

Does not change any control code, does not decide which config (if any)
becomes a new default, does not re-run any historical experiment under a
different contact configuration. That decision — and the stability
validation it would require first — is explicitly out of scope here, per the
instruction this experiment was scoped against ("不改任何正式 API").
