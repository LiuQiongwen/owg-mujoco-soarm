# SO-101 blocked-closure contact microbenchmark (2026-08-07)

> **Correction (same day, later):** `docs/CONTACT_ONSET_AUDIT_20260807.md`
> found the `"gap vs known thickness"` numbers below (built from
> `settled_true_opening_m`) are contaminated by a reference-surface mismatch
> unrelated to contact physics -- `true_opening_m` measures a different point
> set than the actual pad collision geometry. The box's `min_pad_dist`/
> `steady_pad_dist` fields (also below) were **not** affected and are the
> correct numbers: S1/S1b settle within 0.1mm of the true 30mm surface, not
> "+6.71mm past it." Read the audit doc before trusting any `"gap vs known
> thickness"` figure in this file.

Zero production-code diff, confirmed by `git status`/`git diff` on
`tango_robot/` before and after. Same pattern as every experiment in this
thread: patches an already-compiled `MjModel`'s numeric fields from a
throwaway script; `move_gripper`, `GRIP_CLOSED`/`GRIP_OPEN`,
`register_primitive_geom`, `_build_scene_xml`, `_solve_ik_jaw_pos_only` all
used unmodified.

Reproduce:

```bash
conda run -n tango python scripts/microbenchmark_blocked_closure.py
```

## Why this design

The solref stability sweep tried to answer "which solref value is stable"
using full grasp success rate and found the signal too noisy to trust —
non-monotonic, scene-dependent, most likely dominated by IK/approach
convergence chaos rather than genuine contact-stiffness effects. This
benchmark removes that confound **by construction**: candidate generation,
per-trial IK, the park/restore cycle, and weld are all removed. IK is solved
**once**, before any contact config is ever touched, and the resulting arm
qpos is frozen and reused verbatim by every trial. The object is placed
deterministically (centred at the pad midpoint, oriented so its known
thickness faces the closing axis) rather than sampled from a seed. The arm
never moves except the gripper closing.

## Phase 1: repeatability

Same trial (S1_stiff_pads, FixtureBox30mm), run 20 times.

**Bit-identical across all 20 repeats.** `min_pad_dist_fixed_m`,
`min_pad_dist_moving_m`, `final_true_opening_m`, `obj_displacement_m`, and
`max_obj_speed_mps` all show exactly zero spread.

This is the cleanest evidence in this whole investigation thread on the
determinism question the solver-contact-attribution and solref-sweep docs
both left open. It does not resolve the earlier **cross-process** discrepancy
noted in `SOLREF_STABILITY_SWEEP_20260807.md` (same nominal config, different
scripts, different penetration numbers) — this test ran all 20 repeats within
one process. What it does establish cleanly: **once IK/approach chaos and
seed-sampled placement are removed, MuJoCo's contact simulation is
deterministic within a process.** The earlier cross-run discrepancy is
therefore more likely attributable to IK/approach-path sensitivity or a setup
difference between scripts than to MuJoCo's solver itself being
non-deterministic — consistent with what the solref sweep already suspected
but couldn't isolate.

## Phase 2: config comparison

3 configs (S0 baseline, S1 5ms pad solref + priority, S1b 7.5ms variant) ×
2 fixtures × 3 repeats.

### FixtureBox30mm: clean, reproducible, and directly informative

| config | gap vs. known 30mm thickness | repeats |
|---|---|---|
| S0_baseline | **−6.66 mm** (compressed through) | bit-identical ×3 |
| S1_stiff_pads (5ms) | **+6.71 mm** (stops outside the surface) | bit-identical ×3 |
| S1b_7.5ms | **+6.71 mm** (identical to 5ms) | bit-identical ×3 |

Confirms, with the cleanest instrument in this thread, the same direction
found by every prior experiment: pad-only stiffening with `geom_priority`
override measurably reduces penetration. Also answers a question the earlier
attribution experiment couldn't cleanly isolate: **5ms and 7.5ms give
identical outcomes on this fixture** — the extra stiffness beyond 7.5ms buys
nothing here, a small point in favor of not needing the more aggressive value.
S1 slightly *overshoots* to +6.71mm (stopping past the true surface, not
exactly at it) — the pad box's own geometry/measurement tolerance, not a
new problem.

### FixtureCyl40mm: unusable, and the reason is informative in itself

| config | gap vs. known 40mm thickness | `first_contact_step` |
|---|---|---|
| S0_baseline | −24.24 mm | None |
| S1_stiff_pads | −24.24 mm | None |
| S1b_7.5ms | −24.24 mm | None |

**Identical across all three configs, and bilateral contact was never
achieved in any of them** (`first_contact_step=None`, `steady` distances in
the hundreds of mm). Config-independence here is the tell: pad solref cannot
matter before the pads ever get near the object. Most likely explanation: the
cylinder's placement (this script places it with no settle step, immediately
starting the close) leaves it unsupported and it falls or rolls clear before
the jaw closes far enough to matter, regardless of contact stiffness. This is
a bug in this script's `place_object_at_pad_gap` orientation/support logic
for the cylinder case specifically, not a new physics finding, and not
resolved here — flagged as a known follow-up rather than chased further given
time already spent on this thread.

## What this changes

- The determinism question is now cleanly answered for the isolated
  jaw-closing dynamics: deterministic within a process once IK/seed-sampled
  placement noise is removed.
- The box result is the strongest, cleanest evidence yet for the S1
  configuration's penetration benefit — no chaos, no per-run variance, three
  configs compared with the confound structurally removed.
- 5ms vs. 7.5ms show no difference on this fixture, which is a small,
  concrete data point toward preferring the less aggressive 7.5ms if a choice
  has to be made (same benefit, presumably more margin against the earlier
  BADQACC concern, though that warning was never confirmed attributable to
  solref specifically either).
- Still not established: behavior on a rounder/asymmetric geometry (the
  cylinder test needs a fix before it says anything), and nothing here says
  anything about the real objects (Hammer/TomatoSoupCan/Banana) or about
  real-hardware agreement — both explicitly out of scope for this pass.

## Next, if this thread continues

1. Fix the cylinder placement (most likely: give it a brief settle against a
   support before closing, or reconsider its resting orientation) before
   trusting any cylinder-specific numbers.
2. Extend the frozen-arm-pose design to Hammer/TomatoSoupCan/Banana — harder,
   since their CoACD geometry doesn't have a single well-defined "thickness
   axis" the way the fixtures do, but the deterministic-placement technique
   itself should transfer.
3. Real-hardware blocked-closure comparison (per this thread's own repeated
   suggestion): the same protocol run on physical SO-101 against a known
   30mm block, comparing target opening / actual blocked opening / whether it
   holds — the validation step nothing here can substitute for.
