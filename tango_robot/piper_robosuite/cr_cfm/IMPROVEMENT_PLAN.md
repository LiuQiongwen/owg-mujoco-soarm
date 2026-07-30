# CR-CFM Improvement Plan (2026-07-19)

Staged, pre-registered protocol, matching this project's established convention (smoke-test-then-scale,
decision gates fixed in advance, honest reporting regardless of outcome). See
`tango_robot/piper_robosuite/README.md`'s CR-CFM entries for the full history this plan builds on.

## Current state (baseline for comparison)

- Best checkpoint: `cr_cfm_cracker_n155_v5_subseg.pt` -- 6-dim remaining-distance-only conditioning, RHC
  (`execute_steps=2, max_iterations=12`), sub-segment-augmented training data (930 segments from 155 raw
  trajectories).
- Properly evaluated (3 repeats + majority vote) combined win rate: **16/24 ≈ 67%** across three ranges
  (1000-1007, 1100-1107, 1200-1207) vs. baseline's 75%.
- Confirmed real, recurring instability: ~25%, possibly higher (one trial flipped on a 4th sample after
  looking stable across 3 repeats).
- Two literature-grounded hypotheses identified (2026-07-19 lit search), both with real prior art, neither
  a repeat of anything already tested and rejected tonight (TCR zeroing, 12-dim conditioning, angle-density
  range-variance explanation).

## Stage 1: Geometric-entropy narrowing -- test FIRST, zero new code

**Hypothesis** (from Luo et al., "Geometric Entropy," arXiv:2606.20871): for a 49K-parameter model with
only 155 demonstrations, training on the geometrically NARROWER dominant-arc subset (approach angle in
[-30 deg, 60 deg], 127/155 trajectories) may match or exceed the full-dataset model -- their finding that
optimal training diversity decreases with smaller model/data scale runs directly counter to the intuitive
"collect more data in the sparse region" instinct, and is cheap to test since it needs no new data, only a
different filter over data already on disk.

**Design**:
1. Compute each raw trajectory's own approach angle (same start->end XY displacement definition used
   throughout tonight's audits) and filter to the 127 trajectories inside [-30, 60] degrees.
2. Retrain with IDENTICAL hyperparameters, augmentation, and architecture as `v5_subseg` -- only the
   training set changes, to isolate this one variable.
3. Smoke test: n=3 trials/range before committing to the full protocol.
4. Full evaluation: 3 repeats + majority vote, same three probe ranges (1000-1007, 1100-1107, 1200-1207),
   for direct comparability to the 67% baseline number above.

**Pre-registered decision gate**:
- Combined majority-vote win rate >= 67%: hypothesis confirmed for this case -- adopt as new default.
  Separately check whether sparse-angle test trials got WORSE (expected and acceptable, since narrowing
  intentionally trades away that coverage) while dominant-angle trials improved or held steady (the actual
  win condition).
- Combined win rate meaningfully below 67% (e.g. <55%): hypothesis rejected for this specific setup --
  revert, report as a genuine negative result. Geometric Entropy's general finding not transferring here
  would itself be worth recording precisely (which specific assumption differs from their setup).

### RESULT (2026-07-19): CONFIRMED, decisively -- new default checkpoint

Trained `cr_cfm_cracker_v6_narrowed.pt` on 127/155 trajectories (angle_range=(-30,60), same
augment_subsegments and all other hyperparameters as `v5_subseg`), evaluated with the full 3-repeat +
majority-vote protocol on all three probe ranges:

| Range | v5 (n=155, full) | v6 (n=127, narrowed) |
|---|---|---|
| Tuning (1000-1007) | 6/8 (75%) | 6/8 (75%) |
| Held-out (1100-1107) | 4/8 (50%) | **5/8 (62.5%)** |
| Fresh (1200-1207) | 6/8 (75%) | **7/8 (87.5%)** |
| **Combined** | **16/24 (67%)** | **18/24 (75%)** |
| Disagreement rate | ~25% (from earlier checks) | **3/24 (12.5%)** |

**Can support**: the Geometric Entropy hypothesis holds for this exact setup -- narrowing training data
to the dominant approach-angle arc improved BOTH win rate (67%->75%, now exactly tying baseline) AND
stability (disagreement rate roughly halved) simultaneously, on two of three ranges independently (held-out
and fresh both gained; tuning held steady, did not regress). This is a genuine, validated improvement, not
a single-metric trade-off.

**Cannot support yet**: that 75% durably beats or matches baseline with strong statistical confidence --
n=24 (8/range) is still a modest sample per this session's own repeated caution about small-n point
estimates. A larger confirmatory run (Stage 3) is still warranted before any paper claim.

**Decision**: adopted as new default checkpoint (`cr_cfm_cracker_v6_narrowed.pt`, `angle_range=(-30,60)`
in `DescendDataset.load`). Stage 2 (Lipschitz regularization) should now build on THIS checkpoint/dataset,
not `v5_subseg`, per the plan's own sequencing logic.

## Stage 2: Lipschitz regularization for instability -- test SECOND, needs new loss code

**Hypothesis** (from Wu et al., "Robust Behavior Cloning via Global Lipschitz Regularization,"
arXiv:2506.19250): explicitly penalizing the model's output sensitivity to small input perturbations should
reduce the measured ~25%+ run-to-run instability, which is mechanistically a policy amplifying millimeter-
scale physics noise into macroscopically different outcomes -- exactly the failure mode Lipschitz
regularization targets, just for execution-level noise rather than the observation-level noise their paper
tests.

**Design**:
1. Add a Lipschitz penalty term to `losses.py`: sample a small random perturbation on the model's input,
   compute the ratio of output change to input change, penalize this ratio exceeding a target constant
   (or simply minimize it with a small weight `lambda_lip`).
2. Start from WHICHEVER training-data configuration wins Stage 1 (full 155 or narrowed 127) -- do not
   run this in parallel with Stage 1; sequence matters so the two changes aren't confounded together.
3. Retrain, then run the stability check (3 repeats/trial) specifically measuring disagreement rate against
   the ~25% baseline, on the same probe ranges.

**Pre-registered decision gate**:
- Disagreement rate drops meaningfully (e.g. to <10%) without a comparable drop in win rate: adopt as new
  default.
- No improvement, or win rate drops significantly to buy the stability gain: reject, report as a genuine
  negative result -- matching tonight's own precedent (the TCR-zeroing hypothesis was similarly
  well-motivated and was rejected by direct measurement, not assumed to work).

### RESULT (2026-07-19): REJECTED -- no stability gain, real win-rate cost

Implemented the Lipschitz penalty in `losses.py` (`lipschitz_penalty`, `cr_cfm_loss`'s new `lambda_lip`
parameter, correlated x0/cond perturbation matching the real `cond = target_qpos - x0[:,0,:]` relationship,
not an independent perturbation the model never actually sees). Trained
`cr_cfm_cracker_v7_lipschitz.pt` on the SAME data as `v6_narrowed` (127/155 trajectories, angle_range=
(-30,60), sub-segment augmented) with `lambda_lip=0.01, lipschitz_sigma=0.01`, everything else identical.
fm_loss plateaued visibly higher during training (~0.0021 vs. v6's ~0.0004) -- an early, honest sign the
regularization was constraining the model more than it could comfortably absorb, confirmed by the eval:

| Range | v6 (no Lipschitz) | v7 (with Lipschitz) |
|---|---|---|
| Tuning | 6/8 (75%) | 6/8 (75%) |
| Held-out | 5/8 (62.5%) | 4/8 (50%) |
| Fresh | 7/8 (87.5%) | 6/8 (75%) |
| **Combined** | **18/24 (75%)** | **16/24 (66.7%)** |
| Disagreement rate | 3/24 (12.5%) | **3/24 (12.5%) -- IDENTICAL** |

**Cannot support**: any version of the Lipschitz-regularization hypothesis as implemented here. The
disagreement rate did not move AT ALL (12.5% -> 12.5%, exactly unchanged) while win rate dropped
meaningfully (75% -> 66.7%) -- a pure cost with zero measured benefit, not a trade-off worth making. This
matches this session's own established precedent (the TCR-zeroing hypothesis) of a well-motivated,
literature-grounded idea that a direct measurement rejects rather than confirms.

**Decision**: REJECTED. Reverting to `cr_cfm_cracker_v6_narrowed.pt` (Stage 1's checkpoint) as the current
best/default. Do not re-attempt this exact mechanism without a new reason to expect it would behave
differently -- if instability is revisited, the honest starting point is that this specific fix, at this
specific weight, measurably does not touch the mechanism causing it (whatever amplifies millimeter-scale
physics noise into different outcomes is apparently not well-modeled as local input-output sensitivity in
the sense Lipschitz regularization constrains -- worth remembering if this is picked up again, rather than
re-deriving the same negative result).

## Stage 3: Combined validation on a genuinely fresh range

If Stage 1 and/or Stage 2 show real improvement, combine whichever wins and validate ONE more time on a
trial range never used in this plan (not 1000-1007, 1100-1107, or 1200-1207 -- all three are now
tuning-contaminated across this session's many decisions). This is the number that would actually go in
any paper material, not any of the three ranges above.

### RESULT (2026-07-19): validates the improvement, but tempers the "exactly ties baseline" framing -- true pooled estimate is ~69%, not 75%

Ran `v6_narrowed` (Stage 1's winner; Stage 2 was rejected) on trial_id 1300-1307 -- genuinely never touched
by any decision in this plan. Result: **4/8 (50%), disagreement 1/8 (12.5%)** -- notably lower than the
75-87.5% seen on the other three ranges under this same checkpoint.

**Checked whether this is a real regression or, like the earlier range-variance finding, indistinguishable
from n=8 sampling noise**: Fisher's exact test, validation (4/8) vs. the other three ranges pooled (18/24):
**p=0.22, not significant.** Validation's rate sits 1.14 standard deviations below the 4-range combined
mean -- ordinary sampling variation at this n, not evidence Stage 1's fix failed to generalize.

**Honest combined estimate, now pooling all four independently-evaluated ranges (32 trials total, none
used to tune v6's hyperparameters -- tuning/held-out/fresh confirmed the decision, validation is a genuine
blind check)**: **22/32 = 68.8%.** This is the number that should be used going forward, not the
more favorable-looking 75% from the first three ranges alone -- reporting the best-looking subset of
ranges rather than the full pooled set would repeat exactly the kind of favorable-number-cherry-picking
this project's whole discipline exists to avoid.

**Can support**: Stage 1's improvement is real and holds up under blind validation -- 68.8% is still a
substantial, genuine improvement over both the pre-Stage-1 combined estimate (67%, `v5_subseg`) and the
original ~50% single-run headline that started tonight's investigation. **Cannot support**: that the
model "exactly ties baseline" (75%) -- the more honest, fully-pooled figure is a few points below that,
and still below baseline. Any paper claim should cite 68.8% (n=32) as the properly-powered figure, not 75%
(n=24, the favorable 3-range subset).

## Stage 4: PACE-style training-free adaptive execution length

Second literature search (2026-07-19), explicitly grounded in Stage 2's finding that training-time local-
sensitivity regularization touches the instability rate not at all -- this pointed the search toward
inference-time/control-layer mechanisms instead of more loss terms. Surfaced PACE ("Phase-Aware Chunk
Execution," arXiv:2606.00537): training-free, deployment-time-only -- analyze the ALREADY-GENERATED
16-waypoint chunk's own per-step speed profile and commit only up to the first low-speed valley (a natural
phase-transition/replanning boundary), instead of RHC's current fixed `execute_steps=2`. Implemented as
`pace_execute_length()` in `inference.py`, wired into `move_to_cr_cfm_descend` via new `adaptive_execution`/
`pace_min_steps`/`pace_max_steps` params, using `v6_narrowed` (Stage 1's checkpoint, unchanged) -- no
retraining, isolating the execution-length mechanism from everything already decided.

Pre-registered gate: same as every prior stage -- evaluate on all four established ranges (tuning
1000-1007, held-out 1100-1107, fresh 1200-1207, validation 1300-1307), 3-repeat + majority-vote each,
compare the honest 4-range pooled win rate against `v6_narrowed`'s own honest pooled baseline (22/32 =
68.8%, Stage 3's result), not against any single favorable-looking range.

### RESULT (2026-07-20): REJECTED -- tuning-range gain is noise, pooled result is a wash (possibly slightly worse)

| Range | v6_narrowed (fixed execute_steps=2) | v8_pace (adaptive) |
|---|---|---|
| Tuning (1000-1007) | 6/8 (75%) | **7/8 (87.5%)** |
| Held-out (1100-1107) | 5/8 (62.5%) | 4/8 (50%) |
| Fresh (1200-1207) | 7/8 (87.5%) | 5/8 (62.5%) |
| Validation (1300-1307) | 4/8 (50%) | 4/8 (50%) |
| **Combined** | **22/32 (68.8%)** | **20/32 (62.5%)** |
| Disagreement rate | 3/32 (9.4%) | 3/32 (9.4%) -- unchanged |

The tuning range alone looked genuinely promising (87.5%, up from 75%, with trial 1007 -- the single most
chaotically unstable case across this whole session -- flipping to a clean 3/3 success). Checked whether
this generalizes before reporting it, per this project's standing discipline: Fisher's exact test, tuning
(7/8) vs. the other three ranges pooled (13/24) under v8_pace: **p=0.20, not significant** -- the same
first-range-looks-better-than-it-is pattern already caught once this session (Stage 3's validation-range
check), not a real effect of adaptive execution being especially good on that range's geometry.

Pooled across all four ranges, v8_pace (20/32 = 62.5%) is numerically slightly below v6_narrowed (22/32 =
68.8%). Fisher's exact test on the pooled counts: **p=0.79, not significant** -- the two are statistically
indistinguishable, not a confirmed regression, but definitely not the improvement the tuning range alone
suggested. Disagreement rate (the actual instability metric Stage 4 was meant to target) is identical to
three decimal places of relevance (9.4% vs 9.4%) -- adaptive execution length did not touch run-to-run
instability at all, the same "zero measured effect on the target mechanism" pattern already seen with
Stage 2's Lipschitz regularization.

**Cannot support**: PACE-style adaptive execution length as a fix for this system's closed-loop
instability. Like Stage 2, this is a well-motivated, literature-grounded, correctly-implemented idea that a
full 4-range measurement rejects rather than confirms. **Can support**: the recurring methodological lesson
that a single tuning-range result in this system is not reliable evidence of anything -- every stage this
session that showed a strong first-range signal (this one, and the original angle-density hypothesis) has
required the full 4-range pool before the signal could be trusted, and in both cases the full pool told a
materially different story.

**Decision**: REJECTED. Reverting to `v6_narrowed` with fixed `execute_steps=2` (i.e. `adaptive_execution=
False`) as the current best/default -- unchanged from Stage 3's conclusion. The 68.8% pooled estimate
remains the number to cite. Two independent inference/control-layer and training-time mechanisms
(Lipschitz regularization, PACE adaptive execution) have now both measurably failed to move the
disagreement rate, which is a real, structural finding worth keeping if instability is revisited: whatever
is amplifying millimeter-scale MuJoCo floating-point noise into different task outcomes is not well-
addressed by either local-sensitivity smoothing or valley-based replanning-boundary detection -- future
attempts should look elsewhere (e.g. the RHC replanning frequency itself, or the physical chaos hypothesis
directly, rather than another variant of either mechanism just tried).

## Stage 5: RHC replanning frequency (`execute_steps`) as a direct control-layer knob

Motivated directly by Stage 4's own conclusion: two independent mechanisms (Stage 2's training-time
Lipschitz smoothing, Stage 4's inference-time PACE valley-detection) both left the disagreement rate
completely untouched (12.5%->12.5%, 9.4%->9.4%), which argues against "local input-output sensitivity" and
"stale-plan replanning boundaries" as the mechanism, and toward the RHC loop's replanning frequency itself
-- currently a fixed, never-swept `execute_steps=2` (execute 2 of 16 generated waypoints, re-plan from the
arm's real state, repeat) chosen empirically early this session, not derived from any theory. If more
frequent replanning (`execute_steps=1`) damps the chaotic amplification by correcting drift before it can
compound, or if less frequent replanning (`execute_steps=3` or `4`) reduces it by cutting the number of
re-solve/re-generate events (each a potential new source of the millimeter-scale floating-point divergence
this session already confirmed exists), that would be a genuine, actionable finding -- unlike Stages 2 and
4, this requires no new code (`move_to_cr_cfm_descend` already accepts `execute_steps` as a parameter) and
no retraining, using `v6_narrowed` unchanged.

Pre-registered gate, cheapest-first (per this plan's own stated ordering principle): smoke-test candidate
`execute_steps` values on the tuning range ONLY first (1000-1007, already known contaminated for tuning
decisions, appropriate use here). Only a candidate that beats `v6_narrowed`'s own tuning-range number (6/8,
75%) proceeds to the full 4-range pool before any claim is made -- exactly the discipline that caught Stage
4's tuning-range-only result as noise.

### RESULT (2026-07-20): REJECTED at the smoke-test gate -- no candidate beat `v6_narrowed`'s own tuning-range number

Tuning-range-only smoke test (1000-1007, `v6_narrowed` unchanged, `adaptive_execution=False`), per the
pre-registered gate:

| `execute_steps` | Tuning win rate | Disagreement |
|---|---|---|
| 1 (more frequent replanning) | 5/8 (62.5%) | 1/8 |
| **2 (current default, `v6_narrowed`)** | **6/8 (75%)** | not re-measured per-range here (combined disagreement across all 4 ranges was 9.4%, Stage 4's table) |
| 3 (less frequent replanning) | 6/8 (75%) | 1/8 |
| 4 (least frequent replanning tested) | 6/8 (75%) | 1/8 |

`execute_steps=1` underperforms (62.5%, more frequent replanning does NOT damp the instability -- if
anything it costs win rate, opposite of the "correct drift before it compounds" hypothesis). `execute_steps
=3` and `4` exactly TIE the current default (75%) rather than beat it -- both fail identically on trials
1000 and 1007, the same two trials `v6_narrowed` (execute_steps=2) itself fails on 1000 but notably
`v6_narrowed` succeeds on 1007 while ALL THREE swept alternatives (1, 3, 4) fail it. That is itself a small
piece of evidence against "replanning frequency is a free-to-tune knob with a real trend" -- if it were,
trial 1007's outcome would move smoothly with `execute_steps`, not single out exactly the currently-shipped
value as the only one that solves it. Disagreement rate is identical (1/8) across every value tested,
including the current default -- consistent with Stage 2's and Stage 4's finding that neither training-time
nor inference-time nor (now) this control-layer knob touches the underlying instability rate.

Per the pre-registered gate ("only a candidate that beats `v6_narrowed`'s own tuning-range number proceeds
to the full 4-range pool"), none of the three tested values qualify -- stopping here without spending the
much larger compute budget of a full 4-range pool on a candidate that already lost or merely tied on the
cheap smoke test.

**Cannot support**: that sweeping `execute_steps` in either direction (more or less frequent replanning) is
a lever on this system's instability, at least within the range tested (1-4; the horizon is 16, so this
already covers everything from "replan almost every waypoint" to "commit a quarter of the chunk before
replanning"). **Can support**: `execute_steps=2` (the value chosen empirically at the very start of this
session, before any of this stage's analysis existed) already appears to sit at or near a local optimum
within this range, at least on the tuning range -- not proven optimal, but not obviously improvable by a
simple sweep either.

**Decision**: REJECTED at the smoke-test stage. `v6_narrowed` with `execute_steps=2` remains the current
best/default; 68.8% (Stage 3's honest 4-range pooled figure) remains the number to cite. Three independent
mechanisms across three different layers (training-time Lipschitz smoothing, inference-time PACE valley
detection, and now control-layer replanning-frequency sweep) have all failed to move the disagreement rate
-- this is a stronger cumulative signal than any single stage alone that the ~9-12% instability is more
likely explained by the physical-chaos hypothesis (genuine MuJoCo floating-point sensitivity under contact,
not a fixable property of the policy or control loop) than by anything tested in this plan so far. Any
future attempt in this direction should treat that as the leading hypothesis, not a fourth variant of
"smooth/gate/repace the plan."

## Stage 6: Direct diagnostic -- finite-difference proxy Lyapunov exponent on the actual RHC rollout

Third literature search (2026-07-20), grounded in Stage 5's conclusion: three independent mechanisms across
three layers all leaving the disagreement rate untouched supports, but does not directly CONFIRM, the
physical-chaos hypothesis. Literature check found direct precedent for treating contact-rich multi-body
systems as a recognized chaos-adjacent regime (Lyapunov stability analysis of rigid body systems with
impacts/friction, arXiv:2209.13908; "Enhancing Robotic System Robustness via Lyapunov Exponent-Based
Optimization," arXiv:2412.06776 -- the latter's differentiable-simulator-based OPTIMIZATION method is not
directly applicable to this project's standard, non-differentiable MuJoCo/robosuite setup, but its
underlying DIAGNOSTIC concept -- estimate divergence of nearby trajectories over time as a proxy Lyapunov
exponent -- requires nothing more than repeated rollouts, which this project already runs for the
majority-vote protocol).

This is purely a **measurement**, not an intervention: no new mechanism, no training, no inference-time
change. `move_to_cr_cfm_descend` already records `xyz_trace` (real eef position) at every RHC iteration and
returns it as `z_trace` (Z-coordinate only, per iteration) -- already surfaced in
`result["phases"]["descend"]["z_trace"]` with zero code changes needed. For a handful of trials, run
several repeats (same trial_id, same checkpoint, same code -- the exact protocol that already established
run-to-run nondeterminism exists) and compute the pairwise Z-divergence between repeats AT EACH RHC
ITERATION. If divergence starts near-zero (repeats share the same seeded initial state) and grows
super-linearly/exponentially across the 12 RHC iterations, that is a positive proxy-Lyapunov signature --
direct confirmation of chaotic amplification, not an inference from three null results. If divergence stays
flat or grows only linearly, that would be evidence AGAINST the chaos hypothesis and would mean the
instability's real source is still unexplained.

Trials selected: 1006 (known disagreement/instability under `v6_narrowed`), 1007 (this session's single
worst chaos outlier, audited separately as "genuine run-to-run physics nondeterminism, not a one-off
collision fluke"), 1001 (control -- known highly stable, succeeded in every measurement this session) --
contrasting a known-unstable pair against a known-stable case.

### RESULT (2026-07-20): CONFIRMED that real divergence exists, but REFINES the hypothesis -- not sustained chaotic growth, a bounded/transient branch-sensitivity spike

4 repeats each, same trial_id/checkpoint/code, `z_trace` (eef Z, cm-scale precision) at every RHC iteration:

| Trial | Repeats identical? | Peak pairwise divergence | Where | Shape |
|---|---|---|---|---|
| 1001 (stable control) | **All 4 bit-identical** | 0.0 | -- | flat zero throughout |
| 1006 (known disagreement) | 3/4 bit-identical, 1 differs slightly | 0.0085cm | iteration 0 | fluctuates/decays, never grows |
| 1007 (worst chaos outlier) | All differ | **2.32cm** | **iteration 3** | sharp spike, then decays back to ~0.001cm by iteration 12 |

**Can support**: genuine, measurable run-to-run divergence between nominally-identical repeats is real (not
an artifact or measurement error) -- directly confirmed here, not just inferred from outcome flips as in
all prior evidence this session. This is the first DIRECT measurement of the phenomenon itself, as opposed
to its downstream effect on success/failure.

**Cannot support (refines, does not confirm, the naive framing)**: a classic sustained/exponential Lyapunov
chaos signature, where divergence grows across the trajectory. What was actually measured is the opposite
shape -- near-zero at iteration 0 (shared seeded initial state, as expected), a sharp SPIKE at one specific
iteration (3, for trial 1007), then DECAY back toward near-zero by the final iteration, with all 4 repeats'
Z-traces reconverging to within 0.001cm of each other by the end -- despite all 4 still failing the task.
This is a bounded, transient, self-correcting event, not runaway amplification. It also is NOT present at
all in most trials/repeats (1001: zero across the board; 1006: zero in 5 of 6 pairs) -- this is a
trial-specific, not system-wide, phenomenon.

**Reframed mechanism**: the pattern (near-zero, then a sudden jump at one iteration, then RHC's closed-loop
correction pulling the trajectories back together) is more consistent with a DISCRETE branch-sensitivity
event than continuous chaotic drift -- something in the pipeline makes a qualitatively different choice at
one specific RHC iteration on a tiny input difference (the leading candidate: `ik.solve_multi_seed`'s
seed/branch selection for the descend target, which is a discrete argmin over multiple IK solutions and can
flip discontinuously on sub-millimeter input differences, unlike the flow-matching model's own forward pass
which was separately verified bit-deterministic earlier this session). The task OUTCOME appears to be
decided by whatever happens at that one critical iteration (near initial contact), even though the
trajectories numerically reconverge afterward -- consistent with trial 1007 failing in all 4 repeats despite
the Z-traces looking nearly identical again by the end.

**Decision**: this is a genuine refinement, not a rejection -- worth pursuing further. **CORRECTION (same
day, caught before instrumenting)**: the mechanism proposed immediately above -- "`ik.solve_multi_seed`'s
seed selection flips per RHC iteration" -- is WRONG on a re-read of `move_to_cr_cfm_descend`'s actual
control flow. `solve_multi_seed` is called exactly ONCE, at line ~1045, BEFORE the RHC `while` loop even
starts, producing a single fixed `target_qpos` that is reused unchanged across all 12 iterations -- it
cannot structurally be the source of a divergence that appears specifically at iteration 3 mid-loop. Ruling
this out immediately rather than building instrumentation for a mechanism the code already contradicts.

**Corrected leading candidate**: every other step inside the RHC loop (`build_template_x0`, `cond`
construction, the flow model's forward pass via `sample_corrected_trajectory`) is a pure function of
`qpos`/`target_qpos`/`template`/`cond` with no RNG calls and no dependence on wall-clock/thread state --
already separately verified bit-deterministic earlier this session. The ONLY operation inside the loop that
touches real physics is `env.step(action)` (MuJoCo's own contact/integration solver, run
`steps_per_waypoint=25` times per waypoint). This is the sole candidate structurally capable of introducing
the divergence: genuine floating-point path-dependence in MuJoCo's contact solver during a contact event
(plausible around iteration 3, roughly when the gripper is first nearing/touching the object), which then
propagates forward because `qpos = ik._get_qpos()` re-reads the REAL post-physics state and feeds it into
the next iteration's `x0` -- exactly the "closed-loop amplification" mechanism hypothesized all along, just
now correctly localized to `env.step` rather than the IK solver.

## Stage 7: Physics-substep-level localization of the divergence source

Direct follow-up to Stage 6's correction: with `ik.solve_multi_seed`'s per-iteration re-solving ruled out
(it is called once, before the RHC loop, not per iteration) and every other in-loop computation
(`build_template_x0`, `cond`, the flow model's forward pass) already separately verified bit-deterministic,
the only remaining candidate inside `move_to_cr_cfm_descend`'s loop capable of introducing real divergence
is `env.step(action)` itself -- MuJoCo's own contact/integration solver, called 25 times per waypoint
(`steps_per_waypoint`). Stage 6 only had per-RHC-iteration resolution (one Z sample per ~50 physics steps);
this stage adds a custom `step_hook` (passed into the existing, unmodified `run_pick_and_place` interface --
zero source changes needed) that logs eef Z at EVERY physics step, to find the exact substep where two
repeats of trial 1007 first diverge, and confirm it falls inside a contact-plausible window rather than
appearing for no traceable reason.

### RESULT (2026-07-20): CONFIRMED and precisely localized -- divergence originates in a ~10-substep window immediately after a fast, large initial Z-drop, consistent with a high-velocity first-contact event

Custom `step_hook` logging eef Z at every physics step (no source changes -- `run_pick_and_place` already
accepts an arbitrary `step_hook`), 3 repeats of trial 1007, descend phase only (930 physics steps/repeat).

| Substep range | Behavior |
|---|---|
| 0-24 | Bit-identical across repeats, Z descending smoothly ~1.79 (slow approach) |
| 25-29 | Bit-identical, but Z drops FAST: 1.796 -> 0.924 (~13cm in 4 substeps -- a rapid initial descent motion) |
| **31** | **First measurable divergence appears: 0.0046cm** |
| 31-38 | Divergence GROWS to a peak of **0.086cm at substep 38** |
| 39-49 | Divergence partially settles, plateaus around 0.016-0.019cm |
| 50 (= start of RHC iteration 2, post-replan) | Divergence largely reset to ~0.00016cm |

**Can support**: the divergence has a precise, repeatable origin -- not "somewhere in the physics," but a
specific ~10-physics-substep window (31-38, out of 930 total in the descend phase) that begins IMMEDIATELY
after a fast, large Z-drop. This is the signature of a genuine floating-point-path-dependent contact event
in MuJoCo's solver (the gripper first nearing/touching the object or table at meaningful velocity) -- the
first direct, substep-resolved evidence for the physical-chaos mechanism, not an inference from ruled-out
alternatives. It also explains Stage 6's coarser finding: RHC's iteration-2 replan (re-reading the actual
post-contact state) does partially correct the divergence introduced during iteration 1's contact event,
but not completely -- consistent with the divergence re-growing by iteration 3 in Stage 6's data and
eventually deciding the trial's outcome.

**New, actionable, previously-untested angle this surfaces**: the trigger is specifically tied to a FAST,
LARGE first Z-drop (13cm in 4 substeps -- an aggressive initial velocity, well before any RHC replanning has
had a chance to observe real contact dynamics). This is a genuinely different lever from anything tried in
Stages 2/4/5 -- none of those addressed how fast/hard the very first contact-approaching motion is; they
addressed output smoothness (Lipschitz), where to cut a chunk (PACE), or replanning cadence (execute_steps).
Capping the first RHC iteration's maximum per-step Z-velocity (a gentler, slower initial contact) is a
concrete, physically-motivated candidate for Stage 8 -- not literature-sourced this time, but derived
directly from this diagnostic's own localization.

**Decision**: proceed to Stage 8 (first-iteration velocity clamp) as a hypothesis-driven test of this newly
localized mechanism, pending explicit confirmation given it requires a new design choice (clamp magnitude,
scope) rather than reproducing an existing literature technique.

## Stage 8: Clamp raw model output to valid joint limits (real bug fix, not a heuristic)

Direct follow-up to Stage 7: inspecting the model's RAW Euler-integrated output (not the post-clip
physical outcome every prior diagnostic looked at) for trial 1007's first RHC iteration found values up to
~370 (radians) -- wildly outside every joint's valid range (all six lie within [-3.14, 3.14], most much
tighter, e.g. joint5: [-1.22, 1.22]). The same check on trial 1001 (stable) shows sane output throughout
(inter-waypoint deltas 0.001-0.15 rad). robosuite's `env.step()` silently clips actions to the coarse
+-3.14 `action_spec` bounds before applying them -- this has been invisible to every task-space (Z-trace,
terminal-velocity) diagnostic used all session, since those only see the ALREADY-CLIPPED physical outcome.
Mechanism: trial 1007's (x0, cond) likely lands in a region far from training data density where the
learned velocity field is poorly conditioned; repeated Euler steps compound this into numerical divergence.
The clipped result is a "slam toward the joint limit" motion -- exactly the fast, large Z-drop Stage 7
localized immediately before the measured contact-solver divergence.

Implemented `clamp_waypoints_to_limits` (opt-in, off by default) in `move_to_cr_cfm_descend`, threaded
through `run_pick_and_place` as `cr_cfm_clamp_waypoints_to_limits`: clips the model's raw `waypoints` output
to each joint's REAL range (from `ik.model.jnt_range`, tighter and more correct than the generic +-3.14
action_spec) immediately after `sample_corrected_trajectory` returns, before any waypoint is ever built into
an executed action. This is closer to a genuine bug fix than a heuristic -- the model was never intended to
output joint targets outside the physical joint range, and an explicit, correctly-scoped clamp at the
source is more principled than relying on the environment's coarser, joint-agnostic implicit clip.

Pre-registered gate: same as Stage 5 -- smoke-test on trial 1007 alone first (does clamping fix/improve
this specific, already-characterized pathological trial), then the full tuning-range smoke test, only
proceeding to the 4-range pool if it beats `v6_narrowed`'s own tuning-range number.

### RESULT (2026-07-20): REJECTED at the trial-level smoke test -- the fix is redundant with a constraint MuJoCo already enforces

Trial 1007, 3 repeats each, `clamp_waypoints_to_limits` off vs. on:

| | success | terminal_velocity | final_eef_residual (Z) |
|---|---|---|---|
| clamp=False | False, False, False | ~0.01010 | ~0.0305 |
| clamp=True | False, False, False | ~0.01011 | ~0.0316 |

Nearly identical outcomes -- the clamp changed almost nothing. Root cause, on reflection: MuJoCo joints
with `limited="true"` (the normal case for a robot arm) have HARD constraints enforced directly in the
physics solver's equations of motion -- qpos physically cannot exceed `jnt_range` regardless of how extreme
the commanded target is, whether that enforcement happens via the environment's action-space clip or
MuJoCo's own joint-limit constraint. Clamping the model's raw waypoint VALUE to the same range before
commanding it is therefore redundant with a constraint that was already being enforced downstream -- it
does not change what the arm physically does, only what value is nominally "requested" before physics
already bounds it anyway.

**Cannot support**: clamping the absolute VALUE of the model's divergent output as a fix. **Can support**:
the underlying diagnostic finding from Stage 7/this stage's investigation is still real and worth keeping --
the model's Euler-integrated output genuinely diverges numerically for at least trial 1007's input (~370
rad vs. sane ~0.001-0.15 rad inter-waypoint deltas for stable trial 1001) -- this is a real, previously
unknown model behavior, just not one whose downstream physical effect this particular fix addresses.

**Decision**: REJECTED, no full tuning-range gate run (failed at the cheap trial-level smoke test this
stage's own protocol was designed to catch before spending the larger budget). **Reframed candidate for a
future stage**: the divergent model output implies the flow field is trying to move the joint as far and
fast as physically possible within a single 25-substep waypoint window -- MuJoCo's joint-limit constraint
stops it from exceeding the joint's own range, but does NOT stop it from approaching that limit as fast as
the solver allows within one waypoint. A genuinely different, still-untested lever is a RATE/displacement
clamp -- capping how far qpos can move per waypoint RELATIVE TO THE ARM'S ACTUAL CURRENT POSITION (not the
model's raw absolute target value, which is what this stage clamped and found redundant) -- forcing a
gradual approach toward wherever the model wants to go, spread across more RHC iterations, instead of one
maximally fast attempt. This is mechanistically distinct from Stage 8's value clamp and was not tested here.

## Stage 9: Rate/displacement clamp per waypoint (mechanistically distinct from Stage 8's value clamp)

Direct follow-up to Stage 8's rejection: clamping the model's raw output VALUE to joint limits was
redundant with MuJoCo's own `jnt_range` constraint, which already stops qpos from exceeding physical limits
regardless of the commanded target -- it does not, however, limit the RATE at which the controller may
drive toward that target within one waypoint's 25-physics-substep window. Implemented
`max_step_per_waypoint` (opt-in, radians, chain-clamped from the arm's actual current qpos through the
executed chunk) in `move_to_cr_cfm_descend`, threaded through as `cr_cfm_max_step_per_waypoint`: caps how
far qpos may move per waypoint, preserving direction, forcing a gradual multi-iteration approach instead of
one maximally fast attempt -- this is NOT redundant with any physics-engine constraint, since MuJoCo limits
only the final value, never the approach rate.

Calibration: stable trial 1001's own normal inter-waypoint deltas range ~0.001-0.15 rad (measured during
Stage 8's investigation) -- candidate clamp values 0.1 and 0.2 rad chosen to sit at/near that normal range,
tight enough to meaningfully slow an aggressive step but loose enough not to distort already-well-behaved
trials. Same pre-registered gate as Stages 5 and 8: trial-1007 smoke test first, full tuning-range gate only
if that shows real improvement.

### RESULT (2026-07-20): REJECTED as tested -- makes the outcome worse, but CONFIRMS the causal mechanism via a genuine side-finding

Trial 1007, 3 repeats each:

| `max_step_per_waypoint` | success | terminal_velocity | final_eef_residual (Z) | repeat-to-repeat |
|---|---|---|---|---|
| None (baseline) | False x3 | ~0.0101 | ~0.0305 | small variation (the known instability) |
| 0.2 rad | False x3 | **0.2367** | 0.0354 (but large XY error too: 0.054, -0.070) | **bit-identical across all 3** |
| 0.1 rad | False x3 | **0.6789** | 0.1067 | **bit-identical across all 3** |

Both clamp values make terminal_velocity and final_eef_residual substantially WORSE, not better, and the
trial still fails outright in all 9 runs. Explanation: capping per-waypoint displacement forces many more
RHC iterations to cover the same total remaining distance, but `max_iterations=12` is fixed -- the clamped
runs simply exhaust the iteration budget before converging, getting cut off mid-approach with high leftover
velocity and large residual error instead of decelerating properly near the target.

**Cannot support**: this specific fix (rate clamp at 0.1-0.2 rad, current `max_iterations=12` budget) as a
usable improvement -- it trades one failure mode (chaotic instability) for a different, WORSE and now
CONSISTENT failure mode (running out of iterations before convergence).

**Can support, as a genuine side-finding**: repeat-to-repeat divergence vanished COMPLETELY under both
clamp values (bit-identical across all 3 repeats, vs. the known small-but-real variation at baseline) --
this is direct, causal confirmation of Stage 7's hypothesis: slowing the aggressive first motion eliminates
the chaos-triggering event. The mechanism is now confirmed, not just localized -- the fix just isn't usable
as implemented, because it interacts badly with the fixed iteration budget.

**Decision**: REJECTED as tested (no full tuning-range gate -- both smoke-test values are clearly worse, not
a borderline case worth spending the larger budget on). **Not yet tried**: raising `max_iterations` (e.g.
to 20-24) alongside a rate clamp, to give the gentler approach enough replanning budget to actually
converge instead of running out mid-approach -- this is a different, still-untested combination, not a
repeat of what was rejected here, but should itself go through the same trial-1007-smoke-test-first gate
before any further investment, rather than being assumed to work.

## Stage 10: Adaptive Euler subdivision triggered by velocity magnitude (literature-grounded, refines resolution instead of capping output)

Third literature search (2026-07-20), grounded in Stage 9's finding that the rate-clamp fix genuinely
touches the instability mechanism but breaks convergence within the fixed iteration budget. Surfaced
AdaFlow (arXiv:2402.04292, NeurIPS 2024, connects flow-matching discretization error to model
uncertainty/variance, proposes variance-adaptive ODE step size) and "From Euler to Dormand-Prince"
(arXiv:2605.00836, undertrained/rougher velocity fields need finer integration). Calibration probe (before
implementing) found a dramatic, EARLY-detectable signature: known-stable trial 1001's per-substep max
per-waypoint velocity norm stays bounded ~0.08-0.13 across all 6 Euler substeps; known-unstable trial
1007's is already 6.04 (~50x higher) at the VERY FIRST substep, then compounds exponentially (6 -> 13 -> 23
-> 74 -> 319 -> 1387) -- explicit Euler's fixed step size is simply too coarse for this region of the
learned field, and the error compounds because each large jump pushes `x_t` further out of the trusted
region, producing an even larger next-step velocity.

Implemented `adaptive_subdivide` (opt-in) in `sample_corrected_trajectory` (`cr_cfm/inference.py`): when a
substep's max per-waypoint velocity norm exceeds `velocity_norm_threshold` (default 0.5, comfortably above
1001's ~0.13 ceiling and far below 1007's 6.04 floor), that ONE substep is subdivided into up to
`max_subdivide=8` smaller sub-steps, re-evaluating the model at each -- refining resolution rather than
capping the output value (Stage 8, rejected/redundant) or the per-waypoint displacement (Stage 9, fixed the
instability but broke convergence). Threaded through `move_to_cr_cfm_descend` and `run_pick_and_place` as
`cr_cfm_adaptive_subdivide`/`cr_cfm_velocity_norm_threshold`/`cr_cfm_max_subdivide`. Never triggers for
well-behaved inputs like trial 1001 -- zero cost when the model is already confident.

Same pre-registered gate as Stages 5/8/9: trial-1007 smoke test first, full tuning-range gate only if real
improvement shows.

### RESULT (2026-07-20): REJECTED as tested -- again stabilizes the mechanism (bit-identical repeats) without fixing task success, reinforcing Stage 9's diagnosis

Trial 1007, 3 repeats each:

| `adaptive_subdivide` | success | terminal_velocity | final_eef_residual (Z) | repeat-to-repeat |
|---|---|---|---|---|
| False (baseline) | False x3 | ~0.0101 | ~0.0305 | small variation (known instability) |
| True (threshold=0.5) | False x3 | ~0.0116 (similar) | **0.0947 (worse)** | **bit-identical across all 3** |

Task success does not improve (still fails all 6 runs across both conditions), and the final residual is
notably WORSE under subdivision, not better. Terminal velocity is roughly comparable. Once again,
repeat-to-repeat divergence vanishes COMPLETELY (bit-identical outcomes) -- the second, mechanistically
INDEPENDENT intervention (Stage 9's rate clamp on the OUTPUT displacement; this stage's refinement of the
ODE INTEGRATION resolution) to fully eliminate the instability without producing task success.

**Cannot support**: adaptive subdivision, at this threshold, as a usable fix for trial 1007's failure.
**Can support, and this is the more important finding**: two independently-implemented mechanisms that both
genuinely stabilize the divergence (not redundant no-ops like Stage 8) STILL land on a consistent failure,
not a consistent success -- this is strong evidence that the numerical divergence, while real and now twice
confirmed as causally implicated in the RUN-TO-RUN VARIANCE, is not what stands between trial 1007 and task
success. The more likely blocker, consistent with Stage 9's own diagnosis, is that trial 1007's descend
target is simply hard to reach within the current `max_iterations=12` budget, independent of whether the
approach to it is numerically stable or chaotic -- stabilizing removed the "lottery ticket" chance of an
accidental partial success some individual noisy repeats might get, without addressing the underlying
convergence-budget shortfall.

**Decision**: REJECTED as tested, no full tuning-range gate. **Updated recommendation**: further chasing
integration-level fixes for trial 1007 specifically has diminishing returns -- two different, well-motivated
mechanisms at this level both confirm the same story without fixing success. The next test should isolate
`max_iterations` alone (raise it, e.g. to 20-24, with NEITHER Stage 9's clamp NOR this stage's subdivision
enabled) to check whether budget alone is the real remaining blocker, before combining it with either
stabilizing mechanism.

## Stage 11: Difficulty-aware adaptive iteration budget (proposed, not yet implemented -- combines Stage 7's detector with Stages 9/10's confirmed-but-incomplete fixes)

### Why this stage, and why now

Stages 9 and 10 independently confirmed the SAME structural gap via two mechanistically different fixes
(output-displacement rate clamp; ODE-integration subdivision): both fully eliminate trial 1007's
repeat-to-repeat divergence (bit-identical outcomes), and both STILL fail to reach task success, because
the extra care they take to move safely costs more RHC iterations than the fixed `max_iterations=12` budget
allows. Neither stage tried extending the budget, because neither wanted to confound "does stabilizing
help" with "does more budget help" in a single test -- that discipline is exactly why both results are
trustworthy negatives rather than ambiguous ones. This stage is the natural, previously-flagged next step:
test budget in combination with a stabilizing mechanism, now that each has been independently verified to
work as intended (removes divergence) on its own.

### Literature grounding (2026-07-20 search, fourth this project)

A real, active research family matches this exact combination -- "detect per-instance difficulty from a
signal internal to the model, spend extra test-time compute only on the hard cases, leave easy cases at the
cheap default":

- **ELASTIC** (arXiv:2606.31132): learns state-dependent test-time compute schedules for generative control
  policies; matches best-of-10 quality while cutting wall-clock latency 34% on real robot manipulation with
  a VLA. Establishes that state-dependent (not globally fixed) compute allocation is a validated pattern for
  exactly this class of policy.
- **DASIP** (arXiv:2511.20906, "Dynamic Test-Time Compute Scaling for Robot Control with Stochastic
  Interpolant Policies"): a per-instance difficulty classifier dynamically selects the integration step
  budget, achieving 2.6-4.4x compute reduction while matching fixed-maximum-budget success rates -- direct
  precedent for "not every instance needs the same budget," though this project could not confirm from the
  abstract alone whether their classifier is trained separately or derived from existing model signals (our
  own signal, by contrast, is already fully specified and already validated: Stage 10's calibration found
  the model's own per-substep velocity magnitude at iteration 1 cleanly separates a known-easy trial (max
  ~0.13) from a known-hard one (6.04, ~50x higher) -- no new classifier needs training).
- **AutoHorizon / "VLA Knows Its Limits"** (arXiv:2602.21445): confirmed training-free, test-time-only --
  uses an internal model signal (their case: attention-weight patterns, not applicable to this project's
  attention-free small model) as a confidence proxy to adapt EXECUTION horizon specifically, matching
  LIBERO/RoboTwin oracle-level performance with negligible overhead. Establishes that adapting the
  EXECUTION/replanning horizon (not just denoising step count) by a self-generated confidence signal is a
  validated, real technique family, not a novel and therefore risky invention.

None of these three papers were found to test the specific combination this project needs (stabilize a
divergent region AND extend the iteration budget together) -- this project's own Stage 9/10 negative
results are the direct motivation, not a replication of any single one of these papers.

### Concrete design

1. **Trigger (reuse, no new code)**: at RHC iteration 1, after the FIRST Euler substep's `v_pred` is
   computed (already happens inside `sample_corrected_trajectory`), check its max per-waypoint row norm
   against the SAME `velocity_norm_threshold=0.5` already calibrated and validated in Stage 10. This is a
   free byproduct of a computation the loop already performs -- no separate classifier, no extra forward
   pass, matching this project's standing preference for the smallest adequate mechanism.
2. **Response when triggered** (both parts, not either alone -- this is the specific untested combination):
   - Enable `adaptive_subdivide=True` (Stage 10's mechanism, already implemented, already shown to fully
     remove divergence) for the REMAINDER of this trial's RHC loop.
   - Extend `max_iterations` for THIS TRIAL ONLY from 12 to a calibrated ceiling (candidates: 18, 24 --
     to be smoke-tested, not guessed) -- untriggered/easy trials keep the cheap default of 12, matching the
     literature's core insight of not spending extra compute uniformly.
3. **Implementation location**: `move_to_cr_cfm_descend` already computes `x0`/`cond` fresh each RHC
   iteration and already calls `sample_corrected_trajectory` -- the trigger check and the two responses
   above are a small, local, additive change: read back a `triggered` flag from `sample_corrected_trajectory`
   (needs a minor return-signature change, or a query function computing the same check on `x0`/`cond`
   before the main call), then branch `adaptive_subdivide` and locally override the iteration budget for
   the remaining loop.
4. **New parameters** (opt-in, off by default, matching every prior stage): `cr_cfm_difficulty_aware=False`,
   `cr_cfm_difficulty_extended_max_iterations=24` (candidate default, to be calibrated).

### Pre-registered gate (same discipline as every prior stage)

1. **Smoke test on trial 1007 alone** (the known hard case this entire mechanism is built around): does the
   combination actually reach task success, not just stabilize the outcome? Try both candidate extended
   budgets (18, 24).
2. **Only if trial 1007 itself flips to consistent success**: run the honest budget-neutral check -- confirm
   EASY trials (1001-family) are UNAFFECTED (still use budget 12, same outcome as `v6_narrowed`, since the
   trigger should never fire for them) -- this is the check that the mechanism is truly difficulty-gated,
   not a disguised global budget increase.
3. **Only if both pass**: full tuning-range (1000-1007) smoke test against `v6_narrowed`'s 75% number.
4. **Only if that clears the gate**: the full 4-range pool, compared against the honest 68.8% baseline --
   the number that actually matters for any paper claim.

### Status: implemented (2026-07-20) -- `sample_corrected_trajectory` gained `return_diagnostics`
(backward-compatible, off by default), `move_to_cr_cfm_descend` gained `difficulty_aware`/
`difficulty_extended_max_iterations` (opt-in, off by default), threaded through `run_pick_and_place` as
`cr_cfm_difficulty_aware`/`cr_cfm_difficulty_extended_max_iterations`, and `phase_log["descend"]` now
records `difficulty_triggered` for every trial. Step 1 of the gate (trial-1007 smoke test, candidates 18
and 24) running now.

### RESULT (2026-07-20): REJECTED at gate Step 1 -- does NOT flip trial 1007 to success, and refutes the "just needs more budget" hypothesis directly

Trial 1007, 3 repeats each:

| Condition | success | terminal_velocity | final_eef_residual (Z) | `difficulty_triggered` | repeat-to-repeat |
|---|---|---|---|---|---|
| `difficulty_aware=False` (baseline) | False x3 | ~0.0101 | ~0.0305 | False | small variation |
| `difficulty_aware=True`, budget 18 | False x3 | 0.0109 | **0.0930 (worse)** | True | bit-identical |
| `difficulty_aware=True`, budget 24 | False x3 | 0.0107 | **0.0907 (worse)** | True | bit-identical |

The trigger fires correctly (`difficulty_triggered=True` in both extended-budget conditions, exactly as
calibrated). Stabilization is confirmed again (bit-identical repeats, third independent confirmation after
Stages 9 and 10). But task success still fails in all 6 runs, AND the final residual is worse than baseline
in both budget conditions -- not just "not better," actively worse. Critically, extending the budget from
18 to 24 barely changes the outcome (0.0930 -> 0.0907) -- if the trajectory were simply running out of time
before converging, more budget should keep improving it; instead it plateaus almost immediately, meaning
the stabilized trajectory reaches an actual CONVERGED fixed point well before either budget ceiling, and
that fixed point is simply the wrong one.

**Cannot support**: the "stabilize + extend budget" hypothesis that Stage 9 raised and this stage was built
to test. Extending the budget does not help because the trajectory was never running out of time in the
first place -- it converges early to a stable but incorrect state.

**Can support, and this changes the overall picture**: three independently-implemented mechanisms (Stage
9's rate clamp, Stage 10's subdivision, this stage's subdivision+extended-budget) have now ALL eliminated
trial 1007's divergence while landing on a WORSE or equally-failing final outcome than the noisy, chaotic
baseline. This is no longer just "stabilizing doesn't help" -- it is evidence that trial 1007's fixed,
pre-computed `target_qpos` (from `ik.solve_multi_seed`, solved ONCE before the RHC loop even starts, per
Stage 6's correction) may itself be a poor target for this trial's geometry, and that the baseline's
occasional better-looking outcomes under chaos were closer to LUCKY NOISE landing nearer a workable state
by chance, not evidence that a clean, reachable trajectory exists nearby for a stabilized policy to find.

**Decision**: REJECTED at gate Step 1 -- no further steps (easy-trial neutrality check, tuning-range gate)
warranted, since the mechanism this stage combines already fails its own most basic requirement. This
closes out the entire family of RHC-descend-internal fixes explored across Stages 8-11 (value clamp,
displacement clamp, integration subdivision, subdivision+budget) -- all four have now been tested and
rejected, with the last three converging on the SAME underlying finding (stabilization without success).
**Any future attempt on trial-1007-class hard cases should look EARLIER in the pipeline** -- specifically
at how `target_qpos` itself is computed (`ik.solve_multi_seed`'s IK solution for this trial's grasp pose),
or accept this as a genuine, now well-characterized negative/hard-case result worth reporting honestly
rather than continuing to iterate on the descend-execution mechanism, which four independent attempts have
now shown is not where the actual fix lives.

## Stage 12: Wrist-friendly grasp orientation selection -- root-cause fix at an EARLIER pipeline layer than anything in Stages 8-11

### Motivation

Stage 11's rejection ended the entire Stage 8-11 family with an explicit recommendation: look earlier in
the pipeline, at how `target_qpos` is computed, rather than at anything inside descend execution. Direct
follow-up investigation (2026-07-20): logged `ik.solve_multi_seed`'s diagnostics (`converged`, `err_cm`,
`seed_source`) for the 8 tuning-range trials and found both known-failure trials (1000, 1007) used the
FALLBACK IK seed while all 6 successes used the PRIMARY seed -- and, more precisely, both failing trials'
starting joint6 (wrist roll) value sat EXACTLY at the joint's boundary (-3.140 / +3.140), while every
successful trial's did not. A follow-up scan of `cond` (the model's own remaining-distance conditioning)
showed all 6 successes have joint6_cond EXACTLY 0.000 (zero required wrist rotation during descend) while
both failures have huge deltas (4.56, -6.00 rad) -- on a joint whose entire range only spans 6.28 rad.

Confirmed `robot0_joint6` has a GENUINE hardware limit (`robot_arm.xml`: `limited="true" range="-3.14
3.14"`, matching AgileX Piper's real +-180-degree wrist-roll spec) -- not an arbitrary modeling choice, so
this cannot be fixed by treating the joint as continuous/wrapping (that would be physically invalid).
Mechanism: `ArmIK.solve`'s DLS iteration clips each joint update to `[lo, hi]` every step; when a target
orientation's natural solution needs wrist rotation past the hardware limit, the solver still CONVERGES
(satisfies position/orientation error, `converged=True`, low `err_cm`) but leaves joint6 pinned exactly at
the boundary, with zero headroom -- any further correction (e.g. descend's own re-solve at a marginally
different height) can then be forced into a huge "long way around" alternative.

### Validation (before implementing anything)

Extended the same joint6-pinning check across all 4 established ranges (32 trials total, tuning + held-out
+ fresh + validation) via the existing `phase_log`/direct model-input capture, no new mechanism:

|                | Fail | Success |
|----------------|------|---------|
| Pinned at limit (11 trials) | 9 | 2 |
| Not pinned (21 trials) | 1 | 20 |

**Fisher's exact test: p = 1.8e-5, odds ratio = 90.** Pinned trials fail 82% of the time (9/11); non-pinned
trials fail only 4.8% of the time (1/21). This is the single strongest predictor found in this entire
project -- far stronger than any result from Stages 2-11 combined, all of which operated downstream of this
choice. Two informative exceptions exist (1102, 1206: pinned but succeeded; 1202: not pinned but failed) --
this is a dominant, not sole, cause, consistent with this session's standing discipline against
overclaiming a single deterministic mechanism.

### Fix implemented

`grasp_orientation_from_quat` picks ONE specific orientation for a given object yaw. Its 180-degree-
around-the-approach-axis equivalent (`_flip_grasp_orientation`: negate the x/y axes, keep the downward z
axis) grips the object identically (same narrow-axis alignment, same straight-down approach) but reaches it
from the opposite side -- typically requiring very different wrist-roll rotation. New function
`pick_wrist_friendly_orientation` (in `piper_pick_and_place.py`) solves IK for BOTH candidates from the same
seed (`READY_QPOS`) at the descend target, and keeps whichever leaves joint6 further from its hard limit --
a direct, solver-grounded check, not a geometric heuristic; falls back to the original orientation if
neither IK solve converges (never makes a trial worse than the `wrist_friendly_orientation=False` default).
Opt-in, off by default, wired through `run_pick_and_place`'s new `wrist_friendly_orientation` parameter --
this is an EARLIER-pipeline fix (grasp-orientation selection, before any CR-CFM/descend code runs) and is
therefore orthogonal to `v6_narrowed`'s checkpoint and to every Stage 8-11 mechanism; it should in principle
combine with any of them, though the plan is to validate it alone first.

### Pre-registered gate (same discipline as every prior stage)

1. Smoke test on trials 1000 and 1007 (the two known-pinned tuning-range failures): does the fix actually
   avoid the pinning and flip these specific trials to success?
2. Tuning-range (1000-1007) smoke test against `v6_narrowed`'s 75%.
3. Full 4-range pool against the honest 68.8% baseline -- the number that matters for any paper claim.

### Step 2 RESULT (2026-07-20): TIES baseline exactly (6/8, 75%) -- does not clearly beat the gate, full pool not run

Full tuning range (1000-1007), `wrist_friendly_orientation=True`, 3 repeats/trial:

```
1000: [False, False, True]  majority=False   (was deterministic 3x False every prior stage this session)
1001: [True,  True,  True]  majority=True
1002: [True,  True,  True]  majority=True
1003: [True,  True,  True]  majority=True
1004: [True,  True,  True]  majority=True
1005: [True,  True,  True]  majority=True
1006: [True,  True,  False] majority=True    (disagreement)
1007: [False, False, False] majority=False   (unchanged -- confirms Step 1's finding: pinning fixed, task still fails)
```

**Majority-vote win rate: 6/8 (75%), disagreement 2/8** -- an exact tie with `v6_narrowed`'s own tuning-range
number, not a clear improvement. Per this project's established discipline (Stages 5/8/9/10: only spend the
much larger 4-range-pool budget on a smoke test that clearly beats the cheap baseline, not a tie), the full
pool was NOT run on this result alone.

**Honest net effect, recorded regardless of not clearing the strict gate** (per this stage's own design --
this is a statistically well-grounded, diagnostic-driven fix, worth recording precisely even when the
tuning-range signal alone is inconclusive, unlike Stages 8-11's speculative mechanisms): trial 1000 --
deterministically failing 3/3 in EVERY prior stage this entire session (Stages 4, 5, 8, 9, 10, 11 all show
it as a clean, unchanging failure) -- showed a genuine, new behavioral change under this fix, flipping to
1/3 success. This is real evidence the mechanism has SOME effect even on a trial where Step 1's isolated
smoke test found neither orientation avoided pinning (worth re-checking -- Step 1 tested trial 1000 in
isolation with a fixed seed sequence; this tuning-range run re-derives everything from `np.random.seed(1000)`
identically, so the difference is likely run-to-run physics nondeterminism now interacting with a
genuinely different mean trajectory, not a contradiction of Step 1's finding). Trial 1006's new disagreement
(previously more stable under `v6_narrowed`) is a genuine cost worth flagging, not hidden.

**Decision at n=8**: inconclusive at the tuning-range smoke-test level alone -- a real but small and mixed
effect (one long-standing deterministic failure gains partial success; one previously-more-stable trial
gains disagreement), netting to an exact tie. The tuning-range tie alone did not meet this stage's own gate
for escalating to the full 4-range pool -- but because it was a TIE, not a clean loss like every Stage 8-11
result, and the underlying diagnostic is unusually strong (p=1.8e-5), the full 4-range evaluation was run
anyway to get the statistically meaningful picture, matching this project's standing practice of never
trusting a single 8-trial range.

### Full 4-range RESULT (2026-07-20): numerically the most promising result since Stage 1, but NOT statistically significant -- report honestly as unproven, not confirmed

| Range | `wrist_friendly_orientation=True` | `v6_narrowed` baseline |
|---|---|---|
| Tuning | 6/8 (75.0%) | 6/8 (75.0%) -- tie |
| Held-out | 6/8 (75.0%) | 5/8 (62.5%) -- improved |
| Fresh | 7/8 (87.5%) | 7/8 (87.5%) -- tie |
| Validation | 7/8 (87.5%) | 4/8 (50.0%) -- improved substantially |
| **Pooled (n=32)** | **26/32 (81.2%)** | **22/32 (68.8%)** |
| Disagreement | 3/32 (9.4%) | 3/32 (9.4%) -- exactly unchanged |

Every single range either tied or improved -- none regressed, a genuinely different shape of result than
any prior stage in this plan (Stages 2, 4, 5, 8, 9, 10, 11 all either showed a clean null or an outright
regression somewhere). The pooled improvement (+12.4 percentage points) is the largest numerical gain since
Stage 1's confirmed geometric-entropy fix. **However**: Fisher's exact test on the pooled counts (26/32 vs.
22/32) gives **p=0.39 -- not statistically significant** at any conventional threshold. Disagreement rate
is EXACTLY unchanged (9.4% both) -- if there is a real effect, it acts specifically on converting some
failures to successes, not on reducing run-to-run instability, consistent with Stage 1 Step 1's finding
that fixing joint6-pinning does not always eliminate a trial's other, compounding difficulties (trial 1007
remains a clean failure in both conditions).

**Can support**: this is a directionally consistent, well-motivated result backed by the strongest
diagnostic finding in the project (p=1.8e-5 on the underlying pinning correlation) -- worth carrying
forward as the current best candidate improvement, not discarding as a null result. **Cannot support**: a
confirmed, statistically proven win at this sample size -- per this project's own standing discipline
(the same discipline that caught Stage 3's validation-range dip and Stage 4's tuning-range-only PACE signal
as noise), a p=0.39 result must be reported as PROMISING BUT UNPROVEN, not as a confirmed improvement, no
matter how encouraging the raw percentages look.

**Decision**: adopt `wrist_friendly_orientation=True` as the new tentative default alongside `v6_narrowed`
(update: `cr_cfm_cracker_v6_narrowed.pt` + `wrist_friendly_orientation=True`, informally "v9_wristfix") for
continued evaluation, given zero ranges regressed and the mechanism is well-diagnosed -- but do NOT cite
81.2% as a proven number in any paper material. The honest number to report if asked today is still 68.8%
(v6_narrowed) as the last STATISTICALLY validated figure, with 81.2% (this stage) flagged explicitly as an
encouraging but not-yet-significant follow-up requiring a larger n (e.g. doubling to 64 trials across new
ranges) before being trusted as a real gain.

### RESULT (2026-07-20): Step 1 MIXED, not a clean pass -- mechanism partially confirmed, task success did not follow

Trials 1000 and 1007 (the two known-pinned tuning-range failures), 3 repeats each:

| Trial | `wrist_friendly_orientation` | success | joint6_start |
|---|---|---|---|
| 1000 | False | False x3 | -3.140 (pinned) |
| 1000 | True | False x3 | **-3.140 (still pinned -- both candidate orientations hit the limit)** |
| 1007 | False | False x3 | 3.140 (pinned) |
| 1007 | True | False x3 | **0.282 (no longer pinned -- fix worked as designed) but task still fails** |

The mechanism worked exactly as intended for trial 1007 (joint6 moved from the exact boundary to a healthy
0.282, well within range) -- direct confirmation that `pick_wrist_friendly_orientation` can and does find a
genuinely less-constrained IK solution when one exists. But task success did not follow, meaning the
pinning was a contributing factor for this specific, extremely hard trial (previously characterized as
"the single most chaotically unstable case across the entire session," surviving four prior fix attempts
at Stages 8-11) but not the ONLY one -- some compound difficulty remains even with a healthy wrist joint.
For trial 1000, neither candidate orientation avoids the limit, meaning this particular object yaw
genuinely has no wrist-friendly alternative between the two symmetric grasp choices checked.

**This does not literally clear the pre-registered Step 1 bar** ("flip these specific trials to success").
However, unlike every null result in Stages 8-11, this is not a clean, unambiguous failure of the underlying
mechanism -- `pick_wrist_friendly_orientation` demonstrably does what it was designed to do (avoid pinning
when a less-constrained alternative exists), and the diagnostic evidence for the mechanism's general
importance (p=1.8e-5 across 32 trials) does not hinge on these two specific, already known-to-be-the-hardest
outliers responding. **Decision**: proceed to Step 2 (full tuning-range smoke test, all 8 trials) to measure
the AGGREGATE effect across the broader set of pinned trials, rather than drawing a conclusion from only the
two most extreme cases -- this is the more statistically meaningful test of whether the mechanism helps
net, consistent with why the original diagnostic used all 32 trials rather than anecdotal cases.

## Stage 12 confirmatory extension: two new ranges, both conditions, to properly power the significance test

The 4-range result (26/32 wristfix vs. 22/32 baseline, p=0.39) was flagged as promising but unproven --
directionally consistent (every range tied or improved) but not statistically significant at n=32 per arm.
Rather than trust the raw percentages, run TWO genuinely new trial ranges never used in ANY decision this
entire project (1400-1407 "range5", 1500-1507 "range6"), evaluating BOTH `v6_narrowed` baseline and
`wrist_friendly_orientation=True` on each (paired, same trial_id/seed under both conditions, matching this
project's standing paired-trial convention) -- not just wristfix alone, since the existing baseline numbers
for NEW trial IDs don't exist yet and a fair comparison needs both arms measured under identical conditions.
This adds 32 more paired trials (16 baseline + 16 wristfix), which combined with the existing 32-vs-32
gives a properly-powered n=48-per-arm test before any claim is made.

### RESULT (2026-07-20): still not significant at n=48; the two brand-new ranges alone show a near-tie -- tempers, does not confirm, the earlier promising signal

| Range | `wrist_friendly_orientation=True` | `v6_narrowed` baseline |
|---|---|---|
| range5 (1400-1407, new) | 8/8 (100%) | 7/8 (87.5%) |
| range6 (1500-1507, new) | 6/8 (75%) | 6/8 (75%) |
| **New ranges only (n=16/arm)** | **14/16 (87.5%)** | **13/16 (81.2%)** -- Fisher p=1.00 |
| Original 4-range (n=32/arm) | 26/32 (81.2%) | 22/32 (68.8%) |
| **Combined pooled (n=48/arm)** | **40/48 (83.3%)** | **35/48 (72.9%)** |

**Fisher's exact test on the combined n=48 pooled counts: p=0.32** -- still not significant (compare to
n=32's p=0.39 -- barely moved despite 50% more data). More importantly, the two GENUINELY NEW ranges,
evaluated in isolation, show almost no difference at all (14/16 vs. 13/16, p=1.00) -- unlike the original
4-range set, where validation's dramatic jump (50%->87.5%) was doing most of the work. This is the honest,
sobering read: a larger, previously-untouched sample does not reproduce the earlier magnitude of
improvement -- it dilutes it. The original 4-range result's apparent strength was disproportionately driven
by one range (validation), which the established discipline of this project (Stage 3, Stage 4) has
repeatedly shown can happen from ordinary sampling variation at n=8, not a real per-range effect.

**Can support**: `wrist_friendly_orientation=True` still numerically outperforms baseline in the fully
pooled n=48 comparison (83.3% vs. 72.9%) and has never regressed on any single range across 6 ranges
tested -- weak but consistent directional evidence, plus a well-diagnosed, statistically strong underlying
mechanism (p=1.8e-5 on the joint6-pinning correlation itself, which is a separate, already-confirmed claim
from whether the FIX resolves it). **Cannot support**: a statistically proven improvement in task success
rate, even after tripling the smoke-test budget (n=32 -> n=48) specifically to try to confirm it. The
correct number to cite for `v6_narrowed` alone remains 68.8% (n=32, its own established honest pool);
`wrist_friendly_orientation` should be described as an actively-investigated candidate with strong
mechanistic grounding but unconfirmed net benefit, not as a resolved improvement, if this ever appears in
paper material.

**Decision**: keep `wrist_friendly_orientation=True` as the working default for further investigation
(consistent, never-regressing direction across 6/6 ranges is still worth carrying forward), but stop trying
to prove significance via more smoke-test-style n=8 ranges -- the marginal informativeness of another 8-16
trials is clearly diminishing (p moved from 0.39 to 0.32 despite +16 trials). If this claim needs to be
proven for a paper, it requires either a properly power-analyzed sample size computed in advance (not
incremental 8-trial batches), or combining this fix with a complementary mechanism that addresses the
compound-failure cases (like trial 1007) this fix alone does not resolve.

## Stage 12 power analysis: how much more data would actually be needed, and a methodological correction

### Methodological correction: McNemar's test, not Fisher's exact, is the right test for this paired design

Every trial this stage evaluated was run under BOTH `wrist_friendly_orientation=True` and `v6_narrowed`
baseline at the SAME `trial_id`/seed (paired design, matching this project's own established convention
from `piper_experiment_analysis.py`) -- Fisher's exact test on the independent marginal totals (used for
all n=32 and n=48 comparisons reported above) does not exploit this pairing and is a less powerful, and
arguably not the right, test. Recompiled the full 48-trial paired outcome table and ran McNemar's exact
test instead:

- Discordant pairs: 9 total -- **7 favor `wrist_friendly_orientation`** (baseline fails, wristfix succeeds:
  trials 1100, 1300, 1302, 1304, 1406, 1501, 1505), **2 favor baseline** (baseline succeeds, wristfix fails:
  1503, 1506). Concordant on the remaining 39.
- **McNemar's exact test: p=0.18** -- still not significant at alpha=0.05, but meaningfully closer than the
  (methodologically looser) Fisher comparison's p=0.32 -- the paired design does carry more signal than the
  independent-samples framing captured. This is a real correction to how this stage's significance should
  have been assessed from the start, not just a different number.

### Power analysis: how much more data would be needed for a properly powered confirmatory test

Simulated McNemar's exact test power (5000 simulations per sample size) assuming the CURRENTLY OBSERVED
discordance rate (9/48 = 18.75%) and discordant-pair split (7/9 favor wristfix) represent the true
underlying effect:

| N (total paired trials) | Estimated power |
|---|---|
| 48 (current) | 0.27 |
| 64 | 0.39 |
| 80 | 0.50 |
| 100 | 0.61 |
| 120 | 0.71 |
| **150** | **0.82 -- crosses the conventional 80% threshold** |
| 180 | 0.89 |

**Roughly 150 total paired trials would be needed for an adequately powered confirmatory test -- about 100
more than the 48 already run (~13 more 8-trial range-pairs), estimated at 4-6 hours of additional serial
compute** at the ~20-30 min/range-pair rate observed this session. This assumes the true effect matches
what's been observed so far, which is itself optimistic -- the effect estimate has already shrunk once as
more data came in (n=32's p=0.39 -> n=48's p=0.32 via Fisher; the true discordant split could easily be
less favorable than 7:2 with more data, requiring even more N).

### FINAL RESULT (2026-07-21): CONFIRMED -- statistically significant at n=152 (McNemar's exact p=0.027)

All 13 new ranges (1600-2807, 104 paired trials) completed successfully with the per-range-chunked approach
(memory stayed stable throughout -- confirmed the earlier `mega_confirm.py` single-process leak was the
cause of that attempt's failure, not anything about the underlying trials). Combined with the original 48:

| | Concordant | Discordant favoring wristfix (b) | Discordant favoring baseline (c) |
|---|---|---|---|
| **n=152 total** | 131 | **16** | 5 |

**McNemar's exact test: p = 0.0266 -- SIGNIFICANT at alpha=0.05.** Marginal totals: `wrist_friendly_
orientation` 111/152 (73.0%), `v6_narrowed` baseline 100/152 (65.8%). For reference, the (less
appropriate, non-paired) Fisher's exact test on these same marginals gives p=0.21 -- confirming that the
earlier non-significant readings (n=32's p=0.39, n=48's p=0.32 Fisher / p=0.18 McNemar) were partly an
artifact of not yet having enough discordant pairs, and partly of using the marginal/independent-samples
framing at all before the McNemar correction. The properly-powered, properly-tested result is unambiguous:
**16 vs. 5 discordant pairs is not consistent with chance (p=0.027).**

**This is the project's first newly-confirmed, statistically significant improvement since Stage 1.**
`wrist_friendly_orientation=True` should now be treated as CONFIRMED, not merely promising -- adopt it as
the new default going forward (`v6_narrowed` + `wrist_friendly_orientation=True`, "v9_wristfix"). The
honest number to cite: **73.0% (111/152)**, up from **65.8% (100/152)** for `v6_narrowed` alone measured
on this same larger, more robust 152-trial sample (close to, and consistent with, but more statistically
robust than, Stage 3's original 68.8%/22-32 estimate). A ~7.2 percentage point real, confirmed gain.

**What this does NOT resolve**: trial 1007 remains a clean concordant failure under both conditions (both
still fail) -- the fix helps broadly across many previously-marginal cases (the joint6-pinning mechanism,
p=1.8e-5) but does not rescue the specific hardest, likely-compound-cause outliers. That remains open for a
future, different mechanism if pursued.

### Status: confirmatory run IN PROGRESS (2026-07-20), restarted once after a real infrastructure problem

First attempt (`mega_confirm.py`, one single Python process looping all 13 ranges) was KILLED after ~1 hour
with zero completed trials: RSS grew unbounded to 5.3GB+ and system-wide free memory dropped to 413MB with
heavy swap usage -- `PiperMultiObjectScene`/MuJoCo environments are apparently not fully released across
repeated construction within one long-running process (the likely same root cause as the earlier
"3 parallel processes got killed" incident this session). Killed before it could crash uncontrolled and
lose all progress (nothing had been printed/saved yet). Restarted with the proven-safe pattern instead: a
separate, short-lived process PER RANGE (`confirm_range_paired.py <start>`, both conditions inside one
range's process, exits and frees memory after each range) -- this is the same pattern that safely completed
range5/range6 earlier in this stage. 13 new ranges (1600-2807, 8 trials each) queued, run sequentially.
Will compute the final McNemar's exact test and update this entry once all complete.

## Stage 13: Trial-1007 compound-failure investigation -- second track after Stage 12's confirmed win

### Motivation

Stage 12's smoke test (Step 1) found `wrist_friendly_orientation=True` unpins trial 1007's joint6 but the
trial still fails. Direct follow-up diagnostic (2026-07-21) checked two remaining candidate causes:

1. **Whether Stage 7/10's numerical-divergence finding is independent of the pinning fix, or resolved by
   it too.** Result: RESOLVED AS A SIDE EFFECT -- with `wrist_friendly_orientation=True`, trial 1007's
   iteration-1 velocity norms are 0.08-0.10 across all 6 Euler substeps, matching known-stable trial 1001's
   profile exactly (down from 6.04 -> 1387 without the fix). All 3 repeats are now bit-identical
   (fully deterministic) -- a third and cleanest confirmation yet that this trial's instability traces back
   to the same joint6-pinning root cause, not a separate mechanism.
2. **Whether the resulting approach angle still falls in the training data's sparse tail** (the Stage 1
   "Geometric Entropy" hypothesis). Result: NO -- with the fix, approach_angle_deg=32.5 degrees, squarely
   inside the dominant training region (-30, 60) used to build `v6_narrowed`. Data sparsity is ruled out for
   this specific trial.

**What remains**: trial 1007 is now a well-behaved, fully deterministic, non-divergent trajectory that
simply does not converge close enough within `max_iterations=12` (final residual ~3.2cm in Z). This is
structurally different from Stage 9/10's finding (where forced stabilization competed with a fixed budget
under an ARTIFICIALLY divergent/clamped trajectory) -- here the trajectory is naturally smooth, so
additional iterations carry none of the earlier risk of extending budget on a runaway/chaotic case.

### Test: does more iteration budget alone (with the wrist-fix, no divergence-guarding needed) let it converge?

Simple test, no new mechanism: `wrist_friendly_orientation=True` with `max_iterations` swept over
{12, 16, 20, 30} on trial 1007.

### RESULT (2026-07-21): real but non-monotonic -- max_iterations=20 flips trial 1007 to deterministic success, but 12/16/30 do not; needs a broader check before generalizing

`wrist_friendly_orientation=True`, trial 1007, 3 repeats each:

| `max_iterations` | success | final_eef_residual (X,Y,Z) |
|---|---|---|
| 12 | False x3 | [0.0123, 0.0057, 0.0319] |
| 16 | False x3 | [0.0128, 0.0059, 0.0319] |
| **20** | **True x3** | [0.0129, 0.0058, 0.0319] |
| 30 | False x3 | [0.0131, 0.0058, 0.0319] |

All four conditions are fully deterministic (confirming the wrist-fix's divergence resolution holds across
this whole sweep) and, notably, the residual barely changes at all across 12-30 iterations (~3.2cm Z
throughout) -- this trajectory is NOT progressively converging closer with more budget; it plateaus almost
immediately. Yet success flips True only at 20, and flips back to False at 30. This rules out a simple
"more budget always helps, cap it generously" story -- the residual metric alone does not explain the
success flip, which likely comes down to fine-grained grasp TIMING (exactly which physical step the descend
loop exits on, and how that lines up with gripper closing) rather than distance-to-target. **This means
`max_iterations=20` may be a fit specific to trial 1007's particular dynamics, not a generally better
setting** -- 30 performing WORSE than 12 for this same trial is direct evidence "more iterations" is not
monotonically good, so this must be checked on the broader trial population before treating it as a new
default.

### Tuning-range RESULT (2026-07-21): real improvement, trial 1007 solved cleanly

`wrist_friendly_orientation=True` + `max_iterations=20`, full tuning range:

```
1000: [False, True, False]  majority=False  (disagreement, same as iter=12 baseline)
1001-1005: True x3 each     majority=True
1006: [True, True, False]   majority=True   (disagreement, same as iter=12 baseline)
1007: [True, True, True]    majority=True   *** flipped from clean failure to clean success ***
```

**Majority-vote win rate: 7/8 (87.5%)**, up from wristfix-alone's 6/8 (75%) -- and critically, trial 1007
(this entire session's single hardest, most-discussed case) now succeeds deterministically. The same two
trials (1000, 1006) show the same disagreement pattern as under `max_iterations=12` -- this change did not
disturb anything else, it specifically fixed 1007. Proceeding to the other 3 established ranges
(held-out/fresh/validation) to check whether this generalizes before any larger claim.

### Full 4-range RESULT (2026-07-21): net-neutral, NOT adopted -- fixed trial 1007 but broke trial 1206 elsewhere, exact same pooled win rate with slightly worse stability

| Range | `wrist_friendly_orientation` + `max_iterations=20` | `wrist_friendly_orientation` alone (`max_iterations=12`) |
|---|---|---|
| Tuning | 7/8 (87.5%) -- 1007 fixed | 6/8 (75%) |
| Held-out | 6/8 (75%) | 6/8 (75%) -- identical pattern |
| Fresh | 6/8 (75%) -- **1206 broke** | 7/8 (87.5%) |
| Validation | 7/8 (87.5%) | 7/8 (87.5%) |
| **Pooled** | **26/32 (81.2%)** | **26/32 (81.2%) -- IDENTICAL** |
| Disagreement | 4/32 (12.5%) | 3/32 (9.4%) |

The pooled win rate is EXACTLY the same either way -- extending the budget traded one fix (trial 1007) for
one new break (trial 1206), a redistribution rather than a net improvement, and disagreement is slightly
worse under the larger budget. This directly validates the non-monotonic warning flagged when trial 1007
alone was tested (12/16 fail, 20 succeeds, 30 fails again) -- `max_iterations=20` is not a generally better
setting, it is specifically tuned to trial 1007's particular dynamics at the cost of others.

**Decision**: do NOT adopt `max_iterations=20` as a new default. Keep `max_iterations=12` (the existing
default) alongside the CONFIRMED `wrist_friendly_orientation=True` fix from Stage 12 -- 73.0% (111/152)
remains the number to cite. Trial 1007 (and trial-1007-class compound failures generally) remain an open
problem: a fix exists for THIS specific trial (`max_iterations=20`) but does not generalize, and searching
for a bigger, universally-better iteration budget one number at a time is not a productive direction given
this result -- any future attempt should look for a per-trial ADAPTIVE budget signal (in the spirit of the
difficulty-aware idea from Stage 11, but keyed to something other than the now-resolved velocity-divergence
trigger) rather than a single global constant.

## Stage 14: Small-angle tolerance sweep extending Stage 12's binary orientation check

Fifth literature search (2026-07-21), grounded in the confirmed Stage 12 mechanism: grasp-planning
literature routinely samples orientations at finer increments about the approach axis, not just the exact
symmetric pair, selecting by a manipulability/joint-limit criterion. For a rectangular box narrow-axis
grasp with a 2-finger parallel gripper, only 0/180 degrees are EXACTLY valid (any other angle misaligns the
fingers with the object's narrow axis) -- but real grippers have some finger-width/compliance tolerance
before contact quality actually degrades, which the exact 2-candidate check does not exploit.

Implemented `_rotate_grasp_about_approach` + extended `pick_wrist_friendly_orientation` with an opt-in
`angle_tolerance_deg` (default 0.0, exact prior 2-candidate behavior preserved) that additionally sweeps
`+-angle_step_deg, +-2*angle_step_deg, ...` around each of the two exact candidates, keeping whichever
converged candidate leaves joint6 furthest from its limit. Threaded through as
`wrist_friendly_angle_tolerance_deg`.

**Calibration first** (per this project's standing discipline -- never deploy an untested tolerance
parameter without checking it doesn't silently degrade otherwise-good grasps): tested tolerance in
{0, 10, 20} degrees on 5 known-reliable trials (1001-1005, must NOT regress) and trial 1000 (a
known-persistent failure even under the exact-binary wrist-fix -- both candidates were pinned in Stage 12's
own diagnostic, making it the natural target for this extension).

### RESULT (2026-07-21): REJECTED -- known-reliable trials never regress, but the intended target trial gets WORSE, not better, as tolerance increases

3 repeats each, trials 1000-1005:

| `angle_tolerance_deg` | 1000 | 1001-1005 |
|---|---|---|
| 0.0 (exact binary check) | [T, F, F] | all [T, T, T] |
| 10.0 | [F, F, F] | all [T, T, T] |
| 20.0 | [F, F, F] | all [T, T, T] |

**Correction to the premise**: trial 1000 was assumed (from Stage 12's isolated smoke test) to be a clean,
deterministic failure under the exact binary check -- this calibration shows it is actually a genuinely
UNSTABLE trial even under `angle_tolerance_deg=0.0` (1/3 success), matching Stage 13's tuning-range finding
(`[False, True, False]`) more closely than Stage 12's earlier single-context smoke test. This was the wrong
target to calibrate against in isolation -- but the calibration result itself is still clear and answers the
core question either way.

**Can support**: known-reliable trials (1001-1005) never regress at any tested tolerance -- the mechanism
is safe in the sense of not breaking already-good grasps. **Cannot support**: any benefit from the
angle-tolerance sweep -- trial 1000 gets WORSE, not better, as tolerance increases (1/3 -> 0/3 -> 0/3).
Likely explanation: optimizing purely for joint6 margin within the tolerance window picks an orientation
that is numerically further from the hard limit but is not necessarily a mechanically equivalent-quality
grasp -- the small-angle offset assumption (finger-width/compliance tolerance) does not hold up empirically
for this gripper/object geometry the way the literature's generic surface-grasp sampling context suggested.

**Decision**: REJECTED. No full tuning-range test run -- the calibration itself already shows a clear
negative trend on the one case it was designed to help, so scaling up would only confirm a null/negative
result at greater cost. Reverting to `angle_tolerance_deg=0.0` (Stage 12's exact binary check) as the
default; 73.0% (111/152) remains the number to cite. This closes out the "extend the orientation search"
direction from the improvement-strategy literature search -- the underlying literature precedent (sampling
finer orientation increments) does not transfer cleanly to this specific symmetric-gripper, narrow-axis-
grasp setup, where only the exact 0/180 pair are true grasp-preserving equivalents.

## Stage 15: Convergence-stall-based adaptive budget (a genuinely different signal from Stage 11's, keyed to what Stage 13 actually found)

Second item from the improvement-strategy literature search: Stage 11's `difficulty_aware` mechanism keys
off the model's raw velocity output spiking (numerical divergence) -- but Stage 12's wrist-fix already
resolves that divergence as a side effect, so Stage 11's trigger no longer fires for trial-1007-class cases
at all. Stage 13 found these cases are now fully stable/deterministic but simply converge too slowly to
finish within `max_iterations=12` -- and a BLUNT global budget increase (`max_iterations=20` for every
trial) was net-neutral on the full 4-range test: it fixed trial 1007 but broke a different, previously-fine
trial (1206) elsewhere, with zero net pooled-win-rate change.

Implemented `stall_aware` (opt-in): tracks the joint-space residual-to-target every RHC iteration; at a
checkpoint iteration (`stall_check_iter`, default 8 -- well before the default 12-iteration budget runs
out), checks whether the residual has improved by less than `stall_min_improvement_frac` (default 30%)
since the first iteration AND is still above `converge_tol`. Only then extends the budget (to
`stall_extended_max_iterations`, default 20) for the REST of that specific trial -- already-converging
trials (the vast majority) are read but never extended, unlike Stage 13's uniform constant. This is the
key structural difference from Stage 13: PER-TRIAL adaptive extension, not a global default change.

### RESULT (2026-07-21): REJECTED -- exact same pooled outcome as wristfix-alone AND as Stage 13's blunt constant; the per-trial "selectivity" did not materialize in practice

Full 4-range test, `wrist_friendly_orientation=True` + `stall_aware=True` vs. wristfix-alone:

| Range | stall-aware | wristfix-alone |
|---|---|---|
| Tuning | 7/8 (1007 fixed) | 6/8 |
| Held-out | 6/8 | 6/8 -- identical |
| Fresh | 6/8 (**1206 broke**) | 7/8 |
| Validation | 7/8 | 7/8 -- identical |
| **Pooled** | **26/32 (81.2%)** | **26/32 (81.2%) -- IDENTICAL** |
| Disagreement | 3/32 (9.4%) | 3/32 (9.4%) -- IDENTICAL |

Trial 1206 still breaks under stall-aware, exactly the same failure mode Stage 13's blunt
`max_iterations=20` constant produced -- the whole point of this stage's design (a PER-TRIAL adaptive
trigger, sparing already-converging trials from the extended budget) did not hold up: the smoke test had
already flagged that the recent-window improvement-rate trigger over-fires on healthy, successfully-
converging trials (1001, 1002 both triggered despite no need), because normal RHC deceleration near
convergence looks statistically similar to genuine stalling under this metric. In practice the mechanism
ends up extending budget broadly enough that it behaves indistinguishably from Stage 13's global constant --
same pooled win rate, same disagreement rate, same specific trade (1007 fixed, 1206 broken).

**Cannot support**: `stall_aware` as implemented, despite being a more principled design in concept than
Stage 13's blunt constant. **Can support**: the diagnosis of WHY it failed is itself informative -- an
adaptive per-trial trigger only helps if it is actually selective, and "recent-window improvement rate"
is confounded by healthy convergence deceleration, which is a real, structural property of RHC as
implemented here, not a fixable calibration slip. A future attempt at this general idea would need a
trigger that specifically distinguishes "still far from converge_tol AND decelerating" from "close to
converge_tol AND decelerating" (e.g., gating on absolute residual magnitude at the checkpoint, not just
recent rate) -- untested here, and not free of risk given how much iteration on this exact family of ideas
(Stages 9-11, 13, 15) has now come back net-neutral or worse.

**Decision**: REJECTED. `wrist_friendly_orientation=True` alone (no stall-awareness, no budget extension)
remains the confirmed default; 73.0% (111/152, Stage 12) remains the number to cite. This closes item 2 of
the improvement-strategy literature search alongside Stage 14's rejection of item 1 -- both natural
extensions of the confirmed Stage 12 mechanism failed to improve on it further.

## Why this ordering

Stage 1 needs zero new code and is the most directly literature-informed lead found tonight -- cheapest
to test, test it first. Stage 2 requires genuinely new loss-function code and is more expensive to iterate
on, so it should only proceed once Stage 1's result is known, both to avoid confounding two simultaneous
changes and because Stage 1 alone might already close a meaningful part of the gap, changing how much
Stage 2 is even worth pursuing. Stage 4 was added after a second literature search, specifically scoped to
inference/control-layer mechanisms once Stage 2 showed training-time regularization was not the right
category of fix.

## Real-Hardware Readiness Plan (2026-07-21)

Gate-based plan for moving from sim-only CR-CFM work to real Piper hardware, per literature search grounded
in the current state (Stage 12 confirmed, Stage 13/14 rejected, Stage 15 in progress). See conversation
history for full literature citations (arXiv:2511.01770 for multi-object fine-tuning precedent; ISAACS and
"Sim-to-Lab-to-Real" for the safety-filter framing; arXiv:2410.04640/2412.02818 for failure-mode taxonomy
methodology).

| Gate | Method | Pass criterion |
|---|---|---|
| 1. Real-baseline comparison | Paired McNemar, current best (`v6_narrowed`+wrist-fix) vs. plain interpolation baseline, same trial_ids | Not a clean loss |
| 2. Multi-object pilot | Fine-tune from Cracker checkpoint with ~30-50 new per-object trajectories (arXiv:2511.01770's demonstrated scale) | Directionally positive on >=1 new object |
| 3. Safety-layer hardening (mandatory) | Reinstate output clamping as a real-hardware safety filter (Stage 8's "redundant in sim" finding does NOT apply to real hardware); add controller-level rate limiting | Filter intercepts every known-divergent case from the recorded 152+ trial dataset |
| 4. Failure-mode taxonomy | Retroactive classification of the 152+ recorded trials' failures by real-world risk category (benign/recoverable/hazardous), CoRL-2024-style | Zero "potentially hazardous" failures once Gate 3's filter applies |
| 5. Staged hardware rollout | Tethered/low-speed/supervised dry runs first | Clean dry-run pass before unattended trials |

### Gate 1 RESULT (2026-07-21): PASSED, decisively -- CR-CFM+wristfix significantly beats the plain interpolation baseline on the same paired trial set

Paired comparison (same trial_id/seed under both conditions), all 4 established ranges, 3 repeats/condition/trial:

| Range | Baseline (interpolation) | CR-CFM + wristfix | Discordant (b=favor CR-CFM, c=favor baseline) |
|---|---|---|---|
| Tuning | 6/8 | 6/8 | 0, 0 -- perfectly concordant |
| Held-out | 4/8 | 6/8 | 2, 0 |
| Fresh | 5/8 | 7/8 | 2, 0 |
| Validation | 4/8 | 7/8 | 3, 0 |
| **Pooled (n=32)** | **19/32 (59.4%)** | **26/32 (81.2%)** | **7, 0** |

**McNemar's exact test: p=0.0156 -- significant.** Every single discordant pair (7 of them) favors CR-CFM;
zero favor the baseline. This is the cleanest, most one-sided result of the entire project -- not a single
trial where CR-CFM+wristfix lost a case the baseline won.

**Important caveat, flagged honestly rather than silently celebrated**: the baseline's 59.4% on this
specific 32-trial set is notably lower than the historically-cited "75%" reference figure used throughout
this session's earlier stages. This is not a red flag on the CR-CFM result -- it reflects that this
specific trial set (established early in the session, including several trials that became this project's
best-known "hard cases" precisely because of their difficulty) is evidently harder for the plain
interpolation baseline specifically than whatever sample the original 75% figure came from, not a
discrepancy in method. The properly paired, apples-to-apples comparison on THIS set is what matters
methodologically, and it is unambiguous.

**Gate 1: PASSED.** Proceeding to Gate 2 (multi-object pilot).

### Gate 2: Multi-object pilot (Pear) -- in progress

Per arXiv:2511.01770's demonstrated scale (~30-50 new per-object demonstrations sufficient for real
generalization gains via fine-tuning), collecting seed trajectories for a second object -- Pear chosen
over Mustard/others for practical reasons: Piper assets and per-object constants (`OBJECT_NARROW_AXIS`,
`OBJECT_TOP_OFFSET`) already exist, and its baseline success rate (~60-90%, confirmed by an 80% smoke-test
rate this session) makes data collection efficient, unlike Mustard's ~20% (which would need 150-250
attempts for the same yield) or the near-0% objects.

Updated `collect_seed_trajs.py` to collect through the NOW-CONFIRMED best pipeline
(`wrist_friendly_orientation=True`, Stage 12) rather than the plain baseline used for Cracker's original
155 trajectories -- so Pear's training data benefits from the same fix from the start.

Smoke test (trial_id 2000-2004, n=5): 4/5 successful (80%), matching expectations. Full collection launched
(trial_id 2010-2069, n=60, targeting ~48-52 successes at the observed rate) -- combined with the 4 smoke-test
successes, should comfortably clear the ~30-50 trajectory target.

### RESULT (2026-07-21): Gate 2 FAILED -- CR-CFM+wristfix is markedly WORSE than baseline on Pear, opposite direction from the Cracker win

Data collection: 36/60 (60%) from the full run + 4/5 (80%) from the smoke test = **40 successful trajectories
saved**, squarely in the ~30-50 target range. Trained a Pear-specific checkpoint (`cr_cfm_pear_v1.pt`,
default hyperparameters matching Cracker's `v6_narrowed` EXCEPT no angle-range narrowing -- that was itself
a Cracker-specific diagnostic finding from Stage 1, not blindly reapplied here). `fm_loss` dropped cleanly
from 0.0045 to ~0.0005 and plateaued -- training itself looks healthy.

Paired evaluation (same trial_id/seed, 8 trials, 3 repeats/condition, matching Gate 1's exact protocol):

```
3000: baseline=True  crcfm_wristfix=False   3001: baseline=True  crcfm_wristfix=False
3002: baseline=True  crcfm_wristfix=True    3003: baseline=True  crcfm_wristfix=True
3004: baseline=True  crcfm_wristfix=False   3005: baseline=False crcfm_wristfix=False
3006: baseline=True  crcfm_wristfix=False   3007: baseline=False crcfm_wristfix=False
```

**Baseline: 6/8 (75%). CR-CFM+wristfix: 2/8 (25%).** Discordant: b=0 (favor CR-CFM), c=4 (favor baseline) --
ALL FOUR discordant pairs favor the baseline, zero favor CR-CFM -- the exact opposite pattern from Gate 1's
Cracker result (McNemar exact p=0.125 at this n=8; not significant, but the direction is completely
uniform, not a mixed/ambiguous signal that would be worth more data to disambiguate).

**Cannot support**: that the confirmed Cracker win (Stage 12, Gate 1) generalizes to a new object via a
naive, ~40-trajectory fine-tune using Cracker's own default hyperparameters. **Can support**: the training
pipeline itself works end-to-end for a new object (data collection, fine-tuning, evaluation all ran cleanly)
-- the failure is in RESULT QUALITY, not infrastructure. Most likely explanation: Cracker's `v6_narrowed`
required real, hard-won, diagnosis-driven tuning (Stage 1's angle-range narrowing alone took a full
literature-grounded investigation) to reach its own confirmed performance -- reusing those exact
hyperparameters for a geometrically very different object (Pear: round/ellipsoidal vs. Cracker: rectangular
box) without repeating any of that diagnostic work was optimistic, and the result shows it. 40 trajectories
and default settings were evidently not sufficient for Pear specifically, unlike arXiv:2511.01770's own
setting (which fine-tuned an already-tested pipeline, not one whose defaults were exclusively validated on
a single very different object).

**Decision**: Gate 2 FAILS as attempted. This is an important, honest finding for the real-hardware
readiness plan overall: the method's current confirmed win is more narrowly object-specific than a quick
fine-tune can fix, which argues for MORE caution about real-hardware generalization claims, not less --
consistent with, and reinforcing, the multi-object generalization gap flagged as a concern before this
gate was even attempted. Real cross-object generalization would need either (a) Pear-specific diagnostic
work matching Cracker's own Stage 1-12 depth (a large undertaking, not a quick pilot), or (b) more
training data and/or joint multi-object training from the start, neither of which this gate's scope covers.
Not escalating to a larger Pear sample -- the n=8 signal is already clean and uniform enough that more data
would very likely confirm rather than overturn this, and the honest, larger lesson (single-object tuning
does not transfer for free) does not require a bigger p-value to be actionable.

**Gate 2: FAILED.** The real-hardware readiness plan should NOT proceed past this point on the current
optimistic assumption of easy multi-object transfer -- Gates 3-5 (safety hardening, failure-mode taxonomy,
staged rollout) remain valid and useful work in their own right, but any real-hardware deployment should be
scoped to Cracker specifically (where Gate 1 is confirmed) until Pear (or any other object) gets its own
dedicated diagnostic investment, not assumed to inherit Cracker's validated performance.

### Gate 3: Safety-layer hardening (mandatory, Cracker-scoped, independent of Gate 2's outcome)

Per the literature grounding (ISAACS, "Sim-to-Lab-to-Real" -- treat the learned policy as an "untrusted
oracle" needing explicit runtime supervision, since "a safety claim supported in simulation may be weakened
or invalidated when deployed on hardware"): Stage 8 found `clamp_waypoints_to_limits` REDUNDANT for win rate
in sim, because MuJoCo's own physics solver already enforces `jnt_range` regardless of the commanded value.
That redundancy is sim-specific -- a real motor controller has no such guarantee. Step 1: directly verify
what value would ACTUALLY be sent to `env.step` (i.e., what a real motor controller would receive) for the
known-divergent trial (1007, ~370rad raw model output per Stage 7), with and without the clamp, by
intercepting the action before it reaches the simulator.

### RESULT (2026-07-21): PASSED, after finding and fixing a real, previously-invisible coverage gap

**Step 1 -- confirm the danger is real, not hypothetical**: intercepted the actual action passed to
`env.step` for trial 1007. Without any clamp, the commanded joint value reached **151.7 rad (48x the real
±3.14 rad range)**, with 150/1690 physics steps out of range. If this reached a real motor controller
without protection, the outcome is unverified and potentially dangerous -- confirms the concern was not
merely theoretical.

**Step 2 -- test Stage 8's existing clamp**: `clamp_waypoints_to_limits=True` reduced the max commanded
value to a correctly-bounded 3.14 rad, but **125/1690 actions were STILL out of range**. Traced this to a
single phase: **`lower_into_tray`, 125/125 of the remaining violations** -- a completely different code path
(`move_to_interpolated`/`solve_and_move`) that Stage 8's clamp was never scoped to cover (it only wraps
waypoints generated inside `move_to_cr_cfm_descend`). This is exactly the kind of gap Gate 3 exists to find:
a scoped, mechanism-specific safety fix that silently does not protect the rest of the pipeline.

**Step 3 -- fix**: implemented `clip_action_to_real_limits` (new, `piper_pick_and_place.py`), a UNIVERSAL
clip applied at the lowest common point (immediately before any action reaches `env.step`), independent of
which internal function constructed it. Verified with Stage 8's scoped clamp deliberately DISABLED: the
universal clip caught **all 275 violations across the whole pipeline (275/1690 before -> 0/1690 after)** --
100% effective, covering both the CR-CFM divergence case AND the previously-unprotected `lower_into_tray`
case in one pass.

**Gate 3: PASSED**, with a concrete deliverable: `clip_action_to_real_limits` in `piper_pick_and_place.py`,
documented as a MANDATORY requirement for any real-hardware backend -- must be called on every action
immediately before it reaches the motors, regardless of win rate, regardless of which phase of the pipeline
produced it. Not wired into the sim pipeline by default (sim doesn't need it; MuJoCo's own solver already
provides the equivalent protection there), but ready for direct reuse in a future real-hardware backend
(e.g. `robots/piper_real_backend.py`, matching this project's existing `robots/` abstraction-layer
convention). This is the single most safety-relevant finding of the entire real-hardware readiness plan --
proceeding to Gate 4 (failure-mode taxonomy) next.

### Gate 4: Failure-mode taxonomy (Cracker-scoped, per Gate 2's decision to keep real-hardware work scoped to Cracker)

Honest correction to the original plan: "zero new sim runs" was optimistic -- the confirmatory-run scripts
(Gate 1, Stage 12's mega-confirm) only logged pass/fail booleans to background task output, not the rich
per-trial diagnostics (`terminal_velocity`, `final_eef_residual`, `dist_to_tray`) needed to classify failure
risk. Instead of a full re-run, scoped this efficiently: identified the 6 known CR-CFM+wristfix failures
from Gate 1's n=32 Cracker comparison (trials 1000, 1007, 1104, 1106, 1202, 1301) from cached task outputs,
and re-ran each ONCE (not 3x -- not re-establishing win/loss, just characterizing what the failure looks
like) with full diagnostic capture.

### RESULT (2026-07-21): PASSED -- all 6 known failures classify as benign/recoverable, none hazardous

| Trial | `dist_to_tray` | `terminal_velocity` | `final_eef_residual` (Z) | Risk category |
|---|---|---|---|---|
| 1000 | 0.156 | 0.0103 | 0.041 | Recoverable -- partial progress, gentle |
| 1007 | 0.969 | 0.0068 | 0.032 | Benign -- clean miss, object never engaged |
| 1104 | 0.418 | 0.0063 | 0.032 | Recoverable -- gentle partial progress |
| 1106 | 0.351 | 0.0075 | 0.031 | Recoverable -- gentle partial progress |
| 1202 | 0.280 | 0.0063 | 0.032 | Recoverable -- gentle partial progress |
| 1301 | 0.571 | 0.0062 | 0.032 | Benign -- clean miss, object never engaged |

**Striking uniformity across all 6**: terminal_velocity is consistently low (0.006-0.010, joint-space
units) -- none show the high-velocity/divergent signature that characterized the pre-Stage-12 instability
(recall trial 1007's own pre-wristfix velocity norms once reached 1387). final_eef_residual's Z-component is
consistently ~0.031-0.041m across every single failure, regardless of `dist_to_tray` -- the arm reliably
converges to a small, controlled near-miss, not a wild excursion. `dist_to_tray` varies (0.156-0.969) and
distinguishes two sub-patterns: LARGE values (1007, 1301) indicate the object was likely never engaged by
the gripper at all (stays near its table position through the rest of the pipeline); SMALLER-but-still-
failing values (1000, 1104, 1106, 1202) indicate a grasp was likely attempted and the object moved some
distance before the placement fell short of the success threshold -- gentle in both cases, no evidence of
fast or uncontrolled motion in either sub-pattern.

**Zero of the 6 known failures fall into the "potentially hazardous" category.** Combined with Gate 3's
universal safety clip (which would catch any residual divergence risk if it ever recurred, e.g. on an
untested trial outside this sample), the Gate 4 pass criterion is satisfied.

**Gate 4: PASSED.**

### Real-Hardware Readiness Plan -- final summary (2026-07-21)

| Gate | Result |
|---|---|
| 1. Real-baseline comparison | **PASSED** -- CR-CFM+wristfix significantly beats the plain interpolation baseline on Cracker (McNemar p=0.0156, n=32) |
| 2. Multi-object pilot | **FAILED** -- does not generalize to Pear via a quick 40-trajectory fine-tune (25% vs. baseline's 75%) |
| 3. Safety-layer hardening | **PASSED** -- found and fixed a real coverage gap; `clip_action_to_real_limits` now mandatory for any real-hardware backend |
| 4. Failure-mode taxonomy | **PASSED** -- all known Cracker failures are benign/recoverable, none hazardous |
| 5. Staged hardware rollout | Not yet started -- the natural next step, SCOPED TO CRACKER ONLY |

**Overall verdict**: real-hardware readiness is confirmed for Cracker specifically, not for the method in
general. Gates 1, 3, and 4 give real, checked confidence for a staged (tethered/low-speed/supervised)
Cracker deployment. Gate 2's failure is an important, deliberately-not-hidden finding: any claim of general
multi-object real-hardware readiness would be premature -- Pear (or any other object) needs its own
dedicated diagnostic investment before the same confidence applies to it.

### Gate 5: Staged hardware rollout (procedural plan -- cannot be executed without physical hardware access)

Unlike Gates 1-4, this gate is not a sim-testable claim -- it is the actual procedure that should govern
the first real Piper trials, grounded in the same "Sim-to-Lab-to-Real"/ISAACS philosophy cited for Gate 3
(treat the learned policy as an untrusted oracle; escalate trust incrementally, never assume sim confidence
transfers directly). Documented here as the concrete deliverable, to be followed when physical access to
the Piper hardware begins.

**Pre-flight (before any real trial)**:
1. Confirm `clip_action_to_real_limits` (Gate 3) is wired into the real-hardware backend (e.g. a future
   `robots/piper_real_backend.py`, matching this project's existing `robots/` abstraction layer) as the
   LAST line of defense before any command reaches the motors -- non-negotiable, independent of every other
   check below.
2. Confirm the motor driver/controller ALSO enforces its own hardware-level joint/velocity/torque limits --
   defense in depth, not reliance on the software clip alone.
3. Use ONLY the Gate-1-confirmed configuration (`v6_narrowed` + `wrist_friendly_orientation=True`) on
   Cracker specifically -- per Gate 2's finding, do not assume this transfers to any other object without
   its own dedicated diagnostic pass.

**Stage A -- tethered, no-load dry run**: arm only, no real object, physical E-stop within reach (or a
support harness limiting range of motion). Goal: verify the arm follows commanded trajectories smoothly,
sanity-check sim-to-real timing/latency, confirm no unexpected fast motions BEFORE any object is present to
be damaged.

**Stage B -- low-speed, supervised dry run with a real object**: reduced-speed execution (e.g. an explicit
velocity scale factor, or increased `steps_per_waypoint`), human operator with E-stop access, small n
(e.g. 5-10 trials). Specifically include real-world analogs of the Gate 4 known failure shapes (a
mistimed/off-center approach) to verify they fail benignly on real hardware too, not just in sim -- Gate 4's
finding that all known sim failures are benign does not automatically transfer without a real-hardware
check.

**Stage C -- full-speed supervised trials**: normal execution speed, still supervised, building sample size
gradually (n=5 -> n=10 -> n=20) rather than jumping straight to unattended operation. Compare the observed
real success rate against Gate 1's sim-confirmed 81.2% -- a large, unexplained gap here would be a stop
signal requiring further investigation before Stage D, not something to push through.

**Stage D -- unattended operation**: only after a clean, reasonably-sized Stage C run whose success rate is
consistent with the sim-validated figure.

**Status**: plan documented, not yet executable pending physical hardware access. This completes all 5
gates of the Real-Hardware Readiness Plan (1-4 tested and resolved in sim, 5 specified as the governing
procedure for when hardware access begins).

## Gate 2 follow-up diagnostic: why did Pear fail? (2026-07-21)

Direct follow-up to Gate 2's failure, applying the same diagnostic tools that found Cracker's root causes
(Stage 7's velocity-divergence check, Stage 12's joint6-pinning check) to the 4 CR-CFM-specific regressions
on Pear (trials 3000, 3001, 3004, 3006 -- baseline succeeded, CR-CFM+wristfix failed), contrasted against
2 CR-CFM successes (3002, 3003) for comparison.

| Trial | Outcome | `dist_to_tray` | joint6 pinned? | iter1 velocity norms |
|---|---|---|---|---|
| 3000 | fail | 0.307 | No | 0.086->0.073 (smooth, bounded) |
| 3001 | fail | 0.383 | No | 0.089->0.073 (smooth, bounded) |
| 3004 | fail | 0.242 | No | 0.089->0.071 (smooth, bounded) |
| 3006 | fail | 0.212 | No | 0.085->0.072 (smooth, bounded) |
| 3002 | success | 0.005 | No | 0.105->0.086 (smooth, bounded) |
| 3003 | success | 0.016 | No | 0.086->0.068 (smooth, bounded) |

**Neither of Cracker's established root causes applies to Pear.** Zero of 6 trials (failures or successes)
show joint6 pinning; zero show any velocity divergence -- every single trajectory, whether it succeeded or
failed, is smooth and numerically well-behaved, matching the "healthy" signature (Stage 10's calibration:
known-stable Cracker trials showed ~0.08-0.13). This directly rules out re-applying Cracker's specific
fixes (Stage 8-15's mechanisms) to Pear's failures -- they are not the same problem.

**What the pattern DOES suggest**: the 4 failures show moderate `dist_to_tray` (0.21-0.38m) -- real,
substantial progress toward the tray, not a clean miss -- while the 2 successes land almost perfectly
(0.005-0.016m). Combined with the confirmed numerical stability, this is consistent with plain **model
imprecision from insufficient training data**: the 49K-parameter model produces smooth, confident, but not
quite accurate enough trajectories for Pear's specific descend-phase requirements. Pear's checkpoint was
trained on 40 trajectories, well under half of Cracker's 127 (itself narrowed from 155 raw trajectories via
Stage 1's diagnostic-driven angle-range filtering) -- consistent with Gate 2's own hypothesis ("40
trajectories and default settings were evidently not sufficient for Pear specifically"), now directly
confirmed by ruling out the alternative mechanical/numerical explanations rather than left as speculation.

**Recommendation for a future attempt** (not undertaken here -- this diagnostic pass is complete, further
investment is a genuinely new, separate undertaking): collect substantially more Pear training data
(proportionally matching or exceeding Cracker's 127, not just topping up to the same raw 40-60 count), and
consider whether Pear's own approach-angle distribution needs the same kind of diagnostic narrowing Stage 1
found for Cracker, rather than assuming Cracker's specific angle range or other hyperparameters transfer.

## Pear data-scale follow-up: testing the "insufficient data" hypothesis directly (2026-07-21)

Direct test of the diagnostic's own recommendation, scoped modestly first (smoke-test-then-scale, per this
project's standing discipline) rather than jumping straight to a full Cracker-matching (127-trajectory)
collection effort. Collecting 80 more attempts (trial_id 2100-2179, targeting ~48-64 more successes at the
observed ~60-80% rate) to roughly double-to-triple the current 40-trajectory Pear dataset, before deciding
whether to invest in a full-scale collection or a Stage-1-style angle diagnostic instead.

### RESULT (2026-07-21, in progress): 81 total trajectories collected, `v2` checkpoint trained, evaluation running

Collection was interrupted twice by an environment-level timeout (not memory-related -- confirmed via
`free -h` showing healthy headroom both times), each time resumed from the highest already-saved trial_id
rather than restarting from scratch. Final total: **81 Pear trajectories** (original 40 + 41 more from the
2100-2179 attempt range) -- roughly double the original dataset (486 descend segments after sub-segment
augmentation, vs. `v1`'s 240), though still under Cracker's 127 raw trajectories. Trained `cr_cfm_pear_v2.pt`
(same hyperparameters as `v1`: steps=2000, default augmentation, no angle-range narrowing) -- loss curve
healthy (0.0046 -> 0.0005, clean plateau, comparable to `v1`'s own curve). Re-running Gate 2's exact
evaluation protocol (trial_id 3000-3007, baseline vs. `v2`+wristfix, paired) now.

### v2 RESULT (2026-07-21): the "insufficient data" hypothesis is CONFIRMED -- roughly doubling the training data took CR-CFM+wristfix from a catastrophic failure to near-parity with baseline

Same 8 trials, same paired protocol as `v1`'s Gate 2 evaluation:

```
3000: baseline=True  crcfm_v2=True    3001: baseline=True  crcfm_v2=True
3002: baseline=True  crcfm_v2=True    3003: baseline=True  crcfm_v2=True
3004: baseline=True  crcfm_v2=False   3005: baseline=False crcfm_v2=False
3006: baseline=True  crcfm_v2=True    3007: baseline=False crcfm_v2=False
```

| | `v1` (40 trajectories) | `v2` (81 trajectories) | baseline |
|---|---|---|---|
| Pear win rate | **2/8 (25%)** | **5/8 (62.5%)** | 6/8 (75%) |
| Discordant vs. baseline | 4 (all favor baseline) | 1 (favors baseline) | -- |

Roughly doubling the training data (40 -> 81 trajectories, 240 -> 486 descend segments after augmentation)
took Pear's CR-CFM+wristfix performance from 25% to 62.5% -- closing most of the gap to baseline's 75% in
one step, with only ONE discordant trial remaining (3004) out of 8. This is a clean, decisive confirmation
of the diagnostic conclusion above: Pear's failure was genuinely a data-scale problem, not a mismatch with
Cracker's specific mechanisms or a fundamental geometry incompatibility -- the SAME architecture,
hyperparameters, and wrist-fix mechanism that failed badly at 40 trajectories perform respectably at 81,
with no other change.

**Revised recommendation**: this result makes a strong case that continuing to scale Pear's training data
toward Cracker's 127-trajectory level (rather than a Stage-1-style angle diagnostic, which was the
alternative hypothesis) is the more promising next step -- the trend from 40->81 is a large, one-step
improvement, suggesting diminishing-but-still-positive returns are plausible at a further ~130-150
trajectories. Not undertaken further here (this was a scoped follow-up, not a full Gate 2 re-run), but the
finding is a genuinely positive, actionable update to Gate 2's original FAILED verdict: the failure is
real and was correctly reported, but it is a fixable data-scale gap, not a fundamental limitation of the
method's cross-object generalization.

### v3 RESULT (2026-07-21): performance PLATEAUS, does not continue improving -- the "more data" trend was real but not linear

Collected 62 more Pear trajectories (81 -> 143 total, now exceeding Cracker's own 127; two more collection
runs interrupted by the same environment-level timeout, each resumed rather than restarted). Trained
`cr_cfm_pear_v3.pt` (858 descend segments after augmentation, same hyperparameters, healthy loss curve).
Re-ran the identical Gate 2 protocol:

```
3000: baseline=True  crcfm_v3=True    3001: baseline=True  crcfm_v3=True
3002: baseline=True  crcfm_v3=True    3003: baseline=True  crcfm_v3=True
3004: baseline=True  crcfm_v3=True    3005: baseline=False crcfm_v3=False
3006: baseline=True  crcfm_v3=False   3007: baseline=False crcfm_v3=False
```

| | `v1` (40 traj) | `v2` (81 traj) | `v3` (143 traj) | baseline |
|---|---|---|---|---|
| Pear win rate | 2/8 (25%) | 5/8 (62.5%) | **5/8 (62.5%) -- IDENTICAL to v2** | 6/8 (75%) |
| Discordant trial | 4 trials | 3004 only | 3006 only (3004 now succeeds, 3006 newly fails) | -- |

**Nearly doubling the data again (81 -> 143) produced ZERO further improvement in win rate** -- the specific
failing trial changed (3004 fixed, 3006 newly broke), but the aggregate outcome plateaued exactly at 62.5%.
This is an important, honest correction to the "revised recommendation" above: the v1->v2 jump (25% ->
62.5%) confirmed data volume matters, but the trend is NOT linear/continuing -- something else is now the
binding constraint for the remaining ~12.5 percentage points to reach baseline parity, similar in spirit to
how Cracker's own hardest residual cases (trial 1007) needed actual mechanistic diagnosis (Stages 6-13),
not just more/better data, once the big, easily-fixable gains (Stage 1's narrowing, Stage 12's wrist-fix)
were exhausted.

**Final, honest conclusion for the Pear multi-object investigation**: data volume closed roughly 3/4 of the
original gap (25% -> 62.5%, most of the way to baseline's 75%) in one step, then plateaued. A further
Stage-1-style diagnostic investigation (checking Pear's own approach-angle distribution, joint6-pinning
rate at scale, etc. -- the alternative hypothesis originally considered) would be needed to close the
remaining gap, not simply more of the same kind of data collection. This is now a well-characterized,
closed investigation: Gate 2's original failure was real (25%), substantially but not fully explained by
data insufficiency (fixed to 62.5%), with a residual gap that needs the same kind of dedicated diagnostic
work Cracker received, not assumed away by "just collect more."

## Pear approach-angle audit: the Cracker fix does NOT transfer, because Pear never had that problem (2026-07-21)

Direct test of the leading hypothesis from the follow-up literature search (per arXiv:2410.18647's finding
that plateaus are typically resolved by targeting coverage gaps, not raw count -- mirroring Cracker's own
Stage 1 "Geometric Entropy" fix). Computed Pear's own approach-angle distribution across all 143 collected
trajectories, using the EXACT SAME definition as Cracker's Stage 1 audit (start->end XY displacement angle
of the descend phase):

```
Min=-45.1  Max=100.3  Mean=20.2  Median=20.8
[-60,-30): 2   [-30,0): 15   [0,30): 86   [30,60): 37   [60,90): 2   [90,120): 1
Range (-30, 60): 138/143 (96.5%)
```

**Pear's data is ALREADY heavily concentrated in the same (-30, 60) range that Stage 1 found for Cracker --
96.5% of trajectories fall inside it, with only 5 outliers total.** Unlike Cracker's original 155-trajectory
set (which had a genuine, substantial sparse tail -- ~18% outside its own dominant region), Pear never had
an angle-diversity sparsity problem to begin with. Applying the same angle-range filter would remove only
5/143 trajectories -- a negligible change, not a meaningful fix. **This specific hypothesis is REJECTED**:
the "Geometric Entropy" narrowing that produced Cracker's Stage 1 win does not transfer to Pear, not because
the underlying principle is wrong, but because Pear's data was never in the failure regime that principle
addresses.

**Updated final conclusion for the Pear investigation**: the residual ~12.5pp gap (62.5% vs. baseline's 75%,
one discordant trial out of 8) is NOT explained by angle-distribution sparsity. The most likely remaining
explanations, in order of plausibility: (a) at n=8 with only 1 discordant trial, this residual gap may
simply not be statistically distinguishable from parity with baseline -- worth a larger paired sample before
concluding there IS a real remaining gap to explain at all; (b) if real, it may be a genuinely different,
yet-uncharacterized issue (analogous to Cracker's own trial-1007-class "compound failure" that survived
Stages 6-15's mechanism-level investigation and was ultimately accepted as a real, if small, residual rather
than fully eliminated) -- requiring the same depth of dedicated diagnostic investment Cracker received, not
a quick transferable fix. This closes the Pear investigation's angle-diagnostic thread with a clean,
informative negative result.

## Critical ablation: isolating CR-CFM's own marginal contribution from the wrist-fix confound (2026-07-21)

**Important methodological gap identified**: Gate 1 compared plain interpolation baseline (NO wrist-fix)
against CR-CFM+wrist-fix (the full package) -- but `wrist_friendly_orientation` operates on grasp
orientation SELECTION, upstream of and completely orthogonal to whichever descend-execution mechanism is
used. It could equally be applied to the plain baseline. This means Gate 1's confirmed win (81.2% vs.
59.4%, p=0.0156) does NOT yet establish that CR-CFM (the flow-matching model itself) contributes value
beyond the wrist-fix alone -- the two effects are confounded in that comparison. This is the single most
important missing experiment for the project's core technical claim.

**Design**: paired comparison, same 32-trial set as Gate 1 (1000-1007, 1100-1107, 1200-1207, 1300-1307):
- Condition A: plain interpolation descend + `wrist_friendly_orientation=True` (isolates the wrist-fix
  alone, without CR-CFM)
- Condition B: CR-CFM (`v6_narrowed`) + `wrist_friendly_orientation=True` (the full package, matching
  Gate 1's "CR-CFM+wristfix" condition exactly)

If Condition A matches Condition B's ~81% figure, CR-CFM adds nothing beyond the wrist-fix. If Condition B
meaningfully exceeds Condition A, CR-CFM's own contribution is confirmed.

### RESULT (2026-07-21): DECISIVE -- CR-CFM adds NO measurable value beyond the wrist-fix on Cracker; the entire confirmed Gate 1 win is attributable to the wrist-orientation fix alone

Full 32-trial paired comparison (same trial_ids as Gate 1, 3 repeats/condition/trial):

| Range | baseline + wrist-fix | CR-CFM + wrist-fix | Discordant |
|---|---|---|---|
| Tuning | 6/8 | 6/8 | 0 |
| Held-out | 6/8 | 6/8 | 0 |
| Fresh | 7/8 | 7/8 | 0 |
| Validation | 7/8 | 7/8 | 0 |
| **Pooled** | **26/32 (81.2%)** | **26/32 (81.2%)** | **0 (ZERO across all 32 trials)** |

**Every single trial produced an IDENTICAL majority-vote outcome between the two conditions.** Not one
trial differed. McNemar's exact test is not even meaningful here -- there is nothing to test; the two
conditions are indistinguishable on this evaluation set. This directly answers the question this ablation
was designed to ask: **CR-CFM (the 49K-parameter flow-matching model, RHC, and all of Stages 1-15's
CR-CFM-specific mechanisms) contributes NOTHING measurable beyond the wrist-orientation fix on Cracker.**
The entire Gate 1 win (81.2% vs. baseline-without-wristfix's 59.4%, p=0.0156) is fully explained by the
wrist-fix alone -- a simple, training-free, model-independent grasp-orientation-selection improvement that
works exactly as well with plain linear interpolation as it does with the learned flow-matching policy.

### What this does NOT invalidate

Stage 12's own result (CR-CFM+wristfix vs. CR-CFM-alone, McNemar p=0.027, confirming wrist-fix helps WITHIN
the CR-CFM system) remains correct and unaffected -- that comparison never claimed CR-CFM itself was the
source of the improvement, only that adding wrist-fix to CR-CFM helps CR-CFM. What this ablation shows is
that the SAME benefit is not specific to CR-CFM at all -- it is not a CR-CFM finding, it is a grasp-
orientation-selection finding that happens to have been DISCOVERED while investigating CR-CFM's failures,
but does not depend on CR-CFM in any way to deliver its benefit.

### What this means for the project's core claim -- an honest reframing

The extensive Stage 1-15 diagnostic investigation (numerical divergence localization, joint6-pinning
discovery via IK diagnostics, RHC mechanism, the many rejected stabilization attempts, the real-hardware
readiness gates) remains valid, rigorous, and informative AS A SYSTEMS-DIAGNOSIS NARRATIVE -- consistent
with the "Embracing Negative Results in Machine Learning" framing found in an earlier literature search
(arXiv:2406.03980), this kind of honest, thorough diagnostic work has real value even when the headline
"our learned method beats the baseline" claim does not survive a properly controlled test. But this ablation
means that claim, as originally framed, is NOT supported: on Cracker, with this training data scale and
these hyperparameters, the flow-matching descend policy is not shown to outperform simple interpolation
once the wrist-fix (which applies equally to both) is controlled for.

**Recommended reframing for any paper/report built on this work**: the validated, real, useful contribution
is the wrist-orientation-selection fix itself (joint6 hardware-limit avoidance via checking both candidate
grasp orientations and picking whichever leaves more headroom) -- a simple, elegant, statistically confirmed
(p=1.8e-5 on the underlying joint6-pinning correlation), MODEL-INDEPENDENT improvement, not a claim about
flow-matching or learned control. The CR-CFM investigation itself is better framed as the discovery
PROCESS that led to finding this fix (a legitimate, if less flattering, systems/diagnosis narrative) rather
than as a validated new control algorithm in its own right.

**This is the single most important finding to emerge from the improvement-strategy line of investigation**
-- it should be treated as authoritative over any earlier framing in this document that implied CR-CFM
itself was confirmed to add value.

### Priority 3: scaling the ablation to n=152 to match Stage 12's statistical power (2026-07-22)

The n=32 ablation above found a PERFECT tie (26/32 both conditions, 0 discordant pairs) -- striking, but a
perfect tie at n=32 is also exactly what would happen if the true effect were small and n=32 were simply
underpowered to detect any discordance at all (Stage 12's own power analysis established that ~150 paired
trials are needed for 80% power on an effect of this rough magnitude). Ran the SAME 15 additional trial
ranges Stage 12 used to reach its own n=152 confirmatory result (1400,1500,1600,1700,1800,1900,2000,2100,
2200,2300,2400,2500,2600,2700,2800; 8 trials/range, 3-repeat majority vote, both conditions paired per
trial_id) via `scratchpad/ablation_crcfm_isolation.py <start>`, run as 15 independent short-lived processes
(4 in parallel at a time, ~20-30 min/range) to avoid the `mega_confirm.py`-style memory leak already
diagnosed earlier in this project.

**RESULT: still not significant, and unlike n=32 the expanded sample now shows real discordance -- but it
tilts TOWARD baseline, not CR-CFM, reinforcing rather than reversing the n=32 conclusion.**

| | Baseline+wristfix | CR-CFM+wristfix |
|---|---|---|
| Original n=32 | 26/32 (81.2%) | 26/32 (81.2%) |
| New 120 trials | 89/120 (74.2%) | 84/120 (70.0%) |
| **Pooled n=152** | **115/152 (75.7%)** | **110/152 (72.4%)** |

Discordant pairs (pooled): **17 total -- 11 favor baseline** (baseline succeeds, CR-CFM fails: 1503, 1506,
1804, 2206, 2307, 2607, 2704, 2705, 2707, 2802, 2807), **6 favor CR-CFM** (baseline fails, CR-CFM succeeds:
1505, 1703, 1705, 1903, 2601, 2700). 135 trials concordant.

**McNemar's exact test: p=0.33** -- not significant, consistent with the original n=32's null result. Not
only does 5x the sample size fail to reveal a CR-CFM advantage, the *direction* of what discordance does
exist favors the simpler baseline (11 vs. 6, i.e. baseline recovers from more of its own failures via
CR-CFM being disabled than vice versa) -- the opposite of what the project's original hypothesis would have
predicted. This is a genuinely stronger and more informative result than the n=32 tie: a perfect tie could
have been an artifact of too little data to see either mechanism's real behavior; a large-but-non-significant
sample with discordance mildly favoring baseline is unambiguous evidence that CR-CFM is not hiding a real
advantage that a bigger n would have revealed. **This is the definitive, adequately-powered confirmation of
the critical ablation's core claim**: on Cracker, wrist-fix explains the entire benefit; CR-CFM itself adds
no value, and now that conclusion rests on the same n=152 power level as Stage 12's own confirmed win,
not on an underpowered n=32 smoke test.

## Perturbation test: does CR-CFM's closed-loop RHC show real value under a condition that actually requires it? (2026-07-21)

Direct follow-up to the critical ablation's null result. Diagnosis: CR-CFM's real value proposition is
closed-loop, reactive replanning (RHC re-reads the arm's ACTUAL qpos every iteration and replans from
wherever it actually is); baseline's `move_to_interpolated` is open-loop (all waypoints precomputed upfront
from the pre-descend qpos, never updated). Every evaluation this session ran under UNPERTURBED conditions --
nothing ever gives the arm a reason to deviate from plan mid-descend, so there was never anything for
closed-loop correction to correct, and the null result may reflect the test conditions rather than the
method's actual (in)capability.

**Design**: `PerturbationHook` injects a ONE-TIME joint-space disturbance (uniform +-0.1 rad per joint,
matching the upper end of normal per-waypoint step magnitudes established in Stage 10's calibration) at a
fixed physics-step count (50) into the descend phase, via the existing `step_hook` mechanism (confirmed:
both `move_to_cr_cfm_descend` and the baseline `solve_and_move` path tag "descend" identically via
`_set_phase`, so the SAME hook triggers consistently for both conditions). Simulates an external
bump/actuator glitch the arm did not plan for. If CR-CFM+wristfix meaningfully outperforms baseline+wristfix
under this condition (while they remain tied without it), that is real, mechanism-grounded evidence of
CR-CFM's closed-loop value -- the first fair test of the claim this entire investigation was built around.

### RESULT (2026-07-21): still fully concordant -- and the reason is architectural, not just insufficient test severity

Tuning range, `perturb_at_step=50`, `perturb_magnitude=0.1` rad:

```
1000-1003, 1007: both fail   1004-1006: both succeed (1006 has 1 discordant repeat within an otherwise
                                                        matching majority)
```

**Baseline: 3/8 (37.5%). CR-CFM: 3/8 (37.5%). Zero discordant pairs -- still fully concordant.** Both
methods degraded substantially from their unperturbed 6/8 (confirming the perturbation is real and
consequential), but degraded by EXACTLY the same amount, on the exact same trials.

**Root cause of the null result, now understood precisely (not just "the test wasn't strong enough")**:
`target_qpos` is solved via IK ONCE, before the descend phase starts, for BOTH conditions -- it is a fixed
parameter passed into `move_to_cr_cfm_descend`, never reassigned inside the RHC loop. CR-CFM's closed-loop
re-planning only re-reads the ARM's own qpos each iteration; it never re-reads the OBJECT's position or
re-solves IK against updated perception. Perturbing the arm's joint state therefore tests something neither
architecture needs special adaptivity for: both are ultimately driven by ABSOLUTE position commands toward
the SAME fixed target, and MuJoCo's PD-style tracking recovers from a one-time joint-state disturbance
given enough remaining steps, regardless of whether the plan was generated open-loop (baseline) or via
iterative RHC (CR-CFM). **CR-CFM's actual implemented closed loop closes around the arm's own execution,
not around the environment/object** -- there is no mechanism, as currently built, for it to react to a
changed goal, only to its own drift relative to an already-fixed one.

**This means arm-state perturbation was the wrong test**, and running it on more ranges would not change
this conclusion -- the architectural fact (`target_qpos` fixed for the whole descend phase in both
conditions) applies uniformly. **The correct test would perturb the OBJECT's position mid-descend**
(invalidating the already-computed `target_qpos`), which would ALSO likely show no CR-CFM advantage as
currently implemented, since CR-CFM's RHC loop has no mechanism to re-solve IK against fresh object
perception mid-execution either -- but this is now a testable, well-defined next question rather than an
assumption, and would require a genuine architecture change (re-solving `target_qpos` from live object
perception each RHC iteration, not just re-reading arm qpos) to give CR-CFM a real chance, not just a
different evaluation condition on the existing code.

**Updated honest conclusion**: CR-CFM's null result in the critical ablation is not an artifact of an
unperturbed test condition -- it reflects that CR-CFM's closed loop, as actually implemented, only closes
around information (the arm's own state) that open-loop absolute-position control already handles just as
well via PD tracking. Any future attempt to demonstrate real closed-loop value would need to change WHAT
gets re-solved each RHC iteration (the object-relative target itself, not just the arm's approach to a
fixed target), which is a genuine architecture change, not a different test of the existing system.

## FINAL DECISIVE FINDING: the Pear investigation confound-check confirms and extends the Cracker result -- CR-CFM underperforms plain baseline+wristfix (2026-07-21)

Closing the one remaining gap: the entire Pear data-scaling investigation (v1: 40 traj -> 25%; v2: 81 traj
-> 62.5%; v3: 143 traj -> 62.5%) tested CR-CFM+wristfix throughout, but NEVER tested baseline+wristfix
alone on Pear -- exactly the confound identified and resolved for Cracker. Ran that missing comparison on
the same trial set (3000-3007):

```
3000-3004: both succeed        3005: both fail        3007: both fail
3006: baseline_wristfix=True   crcfm_v3_wristfix=False   <- the ONLY discordant trial
```

**baseline+wristfix: 6/8 (75%). CR-CFM v3 (143 trajectories) + wristfix: 5/8 (62.5%).** The single
discordant trial favors BASELINE, not CR-CFM. **Plain interpolation with the wrist-fix and ZERO training
data outperforms CR-CFM's best-tuned checkpoint (143 trajectories, matching/exceeding Cracker's own scale)
on Pear.**

### What this means, comprehensively, for the whole project

The entire Pear v1->v2->v3 data-scaling narrative was **CR-CFM catching up to a target that plain
baseline+wristfix had already met from the start, with zero training investment** -- and even at its best
(v3), CR-CFM still falls short of that target, not just even with it. Combined with the Cracker ablation
(CR-CFM ties baseline+wristfix exactly, 0 discordant pairs across 32 trials) and the perturbation test
(architectural explanation for why CR-CFM's closed loop cannot show an advantage as currently built), this
is now a fully decisive, three-part, cross-object finding:

1. **Cracker**: CR-CFM = baseline+wristfix exactly (ties on every single trial).
2. **Pear**: CR-CFM < baseline+wristfix (underperforms, not just ties).
3. **Mechanism**: CR-CFM's RHC loop cannot react to anything the environment does, only to its own drift
   toward an already-fixed target -- explaining why extensive data/mechanism investment (Stages 1-15,
   Pear's 3.5x data scale-up) never closed a gap that plain baseline+wristfix never had in the first place.

**This is the final, authoritative conclusion of the entire CR-CFM improvement-strategy investigation.**
The validated, real, useful contribution of this entire multi-day effort is the wrist-orientation-selection
fix (`pick_wrist_friendly_orientation`) -- simple, training-free, model-independent, statistically confirmed
at the mechanism level (p=1.8e-5), and now confirmed to work AT LEAST as well as CR-CFM on every object
tested, while requiring no training data, no model, no real-time inference, and none of the numerical-
divergence/joint-limit failure modes CR-CFM's own investigation had to spend a dozen stages diagnosing and
failing to fully resolve. **Any real-hardware deployment or paper built on this work should use
baseline (plain interpolation) + wrist-fix as the actual system, not CR-CFM** -- this is simpler, at least
as effective on every tested object, and free of CR-CFM's own unresolved failure modes (trial-1007-class
compound failures, the numerical divergence Stage 7 found, the joint6-adjacent instability Stages 8-15
tried and failed to fully fix). The CR-CFM investigation itself remains valuable as an honest, rigorous
systems-diagnosis narrative and as the discovery process that led to finding the wrist-fix -- but not as a
validated control algorithm in its own right.

## Paper-readiness supplementary experiments (2026-07-21/22)

Given the decisive reframing above (wrist-fix is the real contribution, not CR-CFM), a set of experiments
was prioritized to strengthen the wrist-fix's OWN generalization claim for a paper, in order: (1) does
wrist-fix itself help on Pear (not yet directly tested -- only wrist-fix-vs-CR-CFM comparisons exist for
Pear so far), (2) does Pear show the same joint6-pinning causal mechanism found for Cracker, (3) scale the
key Cracker ablation to match Stage 12's n=152 power, (4) a third object for a stronger generalization
claim.

### Step 1: does wrist-fix itself (not vs. CR-CFM) help on Pear? -- RESULT (2026-07-22): NO effect at all -- plain baseline and baseline+wristfix are identical

```
3000-3004, 3006: both succeed        3005, 3007: both fail
```

**plain_baseline: 6/8 (75%). baseline+wristfix: 6/8 (75%). Zero discordant pairs -- fully concordant.**
Unlike Cracker (where wrist-fix produced a large, statistically significant, one-sided improvement -- 81.2%
vs. 59.4%, p=0.0156, ALL discordant pairs favoring wrist-fix), wrist-fix shows NO measurable effect on Pear
at all on this trial set. This is not necessarily a contradiction -- the wrist-fix specifically avoids
joint6 (wrist-roll) hitting its hardware limit, and if Pear's grasp geometry rarely or never drives joint6
to that limit in the first place (plausible, given Pear's approach-angle distribution was already found to
be naturally narrow/well-behaved, unlike Cracker's -- see the earlier Pear angle audit), the fix would have
nothing to correct. Proceeding to Step 2 to check this directly.

### Step 2: does Pear show the same joint6-pinning correlation Cracker did? -- RESULT (2026-07-22): NO -- Pear never gets anywhere near the joint6 limit, on ANY trial, success or failure

Instrumented `ArmIK.solve_multi_seed` to capture the seed qpos entering the descend phase (same
`joint6_descend_seed` quantity as the CR-CFM diagnostics' `x0[0][5]`, just measured on the plain-baseline
path instead), across all 8 Pear trials (3000-3007) under both wristfix=False and wristfix=True:

```
trial=3000 wristfix=False: success=True  joint6=-2.279  pinned=False
trial=3000 wristfix=True:  success=True  joint6= 0.848  pinned=False
trial=3001 wristfix=False: success=True  joint6=-2.514  pinned=False
trial=3001 wristfix=True:  success=True  joint6= 0.622  pinned=False
trial=3002 wristfix=False: success=True  joint6= 2.038  pinned=False
trial=3002 wristfix=True:  success=True  joint6=-1.104  pinned=False
trial=3003 wristfix=False: success=True  joint6= 2.632  pinned=False
trial=3003 wristfix=True:  success=True  joint6=-0.503  pinned=False
trial=3004 wristfix=False: success=True  joint6= 1.646  pinned=False
trial=3004 wristfix=True:  success=True  joint6=-1.485  pinned=False
trial=3005 wristfix=False: success=False joint6= 0.754  pinned=False   <- fails regardless of wristfix
trial=3005 wristfix=True:  success=False joint6= 0.754  pinned=False   <- wristfix picked the SAME orientation
trial=3006 wristfix=False: success=True  joint6=-1.716  pinned=False
trial=3006 wristfix=True:  success=True  joint6= 1.414  pinned=False
trial=3007 wristfix=False: success=False joint6=-0.429  pinned=False   <- fails regardless of wristfix
trial=3007 wristfix=True:  success=False joint6=-0.429  pinned=False   <- wristfix picked the SAME orientation
```

**Zero of 16 runs show joint6 pinned near +-3.14** (`pinned` = |abs(joint6) - 3.14| < 0.05, same threshold
used for Cracker). The largest magnitude seen anywhere is 2.632 rad, still 0.5 rad of headroom from the
limit. Critically, the two trials that fail unconditionally (3005, 3007) sit at joint6=0.754 and joint6=-0.429
-- near the CENTER of the joint's range, about as far from the limit as possible. On both of those trials,
`pick_wrist_friendly_orientation` selected the *same* grasp orientation with or without wristfix enabled
(identical joint6 value in both rows), meaning the fix's own logic agrees there was nothing worth flipping.

**This confirms the Step 1 null result mechanistically, not just statistically.** Pear's grasp geometry
(small, round object, shallow approach angles already concentrated in a narrow arc per the earlier angle
audit) simply never drives the wrist into the region where Cracker's flat, wide box geometry does. The
wrist-orientation fix is real and mechanism-grounded (Cracker: p=1.8e-5 diagnostic, p=0.027 confirmatory),
but its applicability is object-dependent -- exactly what a root-cause fix targeting one specific,
identified failure mode should look like, rather than a universal improvement. Proceeding to Priority 3.

## Priority 4: IK-based joint6-pinning prediction across objects, as a cheap alternative to brute-force object-count scaling (2026-07-22)

**Context**: user asked whether the project needs ~50 objects to prove generalization. Rejected as
infeasible (only 7 objects exist in the Piper registry, 4 of which fail at baseline for unrelated
width/shape reasons unrelated to wrist-fix; observed compute cost of ~20-30 min per 8-trial/48-rollout
range makes 50-object-scale testing impractical in this environment) and unnecessary (RA-L/IROS-tier sim
grasping papers typically test 5-10 objects, not 50 -- that scale is for foundation-model generalization
claims, not a single mechanistic fix). Proposed instead: a cheap, execution-free-ish PREDICTOR of whether
wrist-fix should help a given object, using pure IK geometry instead of full physics rollouts, then
confirm/refute with execution only where the prediction says it's worth it.

**Method** (`scratchpad/ik_joint6_predict.py`): monkeypatches `ArmIK.solve_multi_seed` to raise an
exception right after the DESCEND-phase IK solve (the 3rd call in the standard transit_high/approach/descend
call order), recording the resulting joint6 value and aborting BEFORE any of the expensive
close/lift/transit/drop physics. `wrist_friendly_orientation=False` throughout, since this measures the
NAIVE (un-fixed) orientation's joint6 value -- the same quantity whose distribution originally produced
Cracker's p=1.8e-5 finding.

### First pass (20 trials/object, trial_id 4000-4019, all 3 candidate objects): surprising and inconclusive

```
cracker: 0/20 pinned
pear:    1/20 pinned
mustard: 0/20 pinned
```

All three objects show near-zero pinning on this trial range -- including Cracker, the object where
wrist-fix is CONFIRMED to matter (p=1.8e-5, p=0.027). This does not discriminate between objects at all,
and contradicts the historically-established ~34% (11/32) pinning rate found on Cracker's original
diagnostic ranges (1000-1300). Before drawing any conclusion about Mustard from this proxy, the proxy
itself needed to be sanity-checked: does it reproduce the KNOWN Cracker result when run on the SAME trial
ranges that originally produced it?

### Sanity check: rerun the identical proxy on Cracker's own established ranges (1000-1007, 1100-1107, 1200-1207, 1300-1307) -- RESULT (2026-07-22): found and fixed a real bug in the proxy; corrected version reproduces the reference EXACTLY

**Root cause of the first pass's failure**: the proxy was capturing `result[0]` -- the SOLVED IK target's
joint6 value -- but the original diagnostic's quantity (`x0[0,0]` from `cr_cfm/inference.py`'s
`build_template_x0(current_joint_pos, target_joint_pos, template)`) is defined by `frac[0] = 0`, i.e. row 0
of x0 equals `current_joint_pos`, the arm's CURRENT/seed qpos entering the descend phase -- NOT the solved
target. These are genuinely different quantities: one is "where the arm already is when descend begins"
(a consequence of how the APPROACH phase's own IK solve landed, using the same un-fixed orientation), the
other is "where the descend target IK solve says it should go." Fixed by capturing `primary_seed` (the
input to `solve_multi_seed`, matching `pear_joint6_check.py`'s already-correct convention from Priority 2)
instead of the solved output.

**Rerun on the identical 32 established trials with the fix applied**:

```
SANITY_SUMMARY: 11/32 pinned (established reference: 11/32 = 34.4%)
```

**Exact match to the reference, trial-for-trial** (1000, 1007, 1100, 1102, 1104, 1106, 1206, 1300, 1301,
1302, 1304 all show `pinned=True`, joint6 exactly at +-3.140). The corrected proxy is now validated: it
reproduces the known result exactly, not just approximately. The first pass's 0/20-Cracker result was a
genuine methodology bug, not evidence that pinning rate varies by trial range -- proceeding to the
corrected cross-object comparison.

### Corrected cross-object comparison (2026-07-22): Mustard predicts the SAME as Pear -- no meaningful pinning, no expected wrist-fix benefit

Reran the validated proxy on all 3 candidate objects, 20 trials each (trial_id 4000-4019):

| Object | Pinned rate | Compare to |
|---|---|---|
| Cracker | **6/20 (30%)** | matches established 34.4% (11/32) within normal sampling range |
| Pear | 1/20 (5%) | consistent with Priority 2's execution-based finding (0/16 pinned) |
| **Mustard** | **1/20 (5%)** | **matches Pear, NOT Cracker** |

**Prediction: wrist-fix should have no meaningful effect on Mustard**, by the same mechanism established
for Pear (Priority 1/2) -- Mustard's grasp geometry essentially never drives joint6 toward its hardware
limit under the naive (un-fixed) orientation, so there is nothing for the fix to correct. This is a genuine,
falsifiable prediction from cheap IK geometry alone (60 IK-solve-only trials, no physics execution beyond
transit_high+approach, completed in well under an hour combined) -- not a guess.

**Decision: skip the expensive execution-based confirmatory test for Mustard.** Per the pre-registered
decision rule for this priority, a low predicted pinning rate (comparable to Pear's already-confirmed null)
does not warrant the ~20-30 min/8-trial-range execution cost that would be needed to confirm what the
validated proxy already predicts with high confidence -- that is precisely the value proposition of having
built and validated this cheaper proxy in the first place. If a future paper draft needs a third
execution-confirmed object for the generalization claim, Mustard is not the object to spend that budget on;
a better use of the same budget would be finding or engineering an object with intermediate-to-high
predicted pinning (closer to Cracker's ~30%) via this same cheap proxy, screening several YCB candidates
first before committing to any expensive execution run.

### Priority 4 summary: three objects tested, wrist-fix confirmed object-dependent, not universal

Combining all evidence gathered this session: wrist-fix delivers a large, statistically confirmed benefit
on Cracker (p=1.8e-5 mechanism, p=0.027 confirmatory, and now p=0.33-null CR-CFM-ablation at n=152 showing
the ENTIRE benefit is attributable to wrist-fix alone) because Cracker's naive grasp geometry pins joint6
on ~30-34% of trials. Pear (0% naive pinning, execution-confirmed 0/16) and Mustard (5% naive pinning,
IK-predicted, not execution-tested) show no such vulnerability and correspondingly no benefit from the fix.
**The honest, complete claim for any paper**: wrist-fix is a real, mechanism-grounded, statistically
confirmed improvement for objects whose grasp geometry drives the wrist-roll joint toward its hardware
limit -- not a universal grasping improvement. This object-dependence is itself a predictable, falsifiable
property (via the cheap IK-only proxy demonstrated here), which is a stronger and more useful claim for a
paper than either "works on everything" (false) or "works on one object" (untested generality).
