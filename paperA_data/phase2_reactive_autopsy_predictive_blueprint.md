# Phase 2: The "Reactive" Autopsy & The "Predictive" Blueprint

**Date**: 2026-07-10
**Status**: analysis complete, including a physical test that substantially revised
Part C's original conclusion. Pending decision on Phase 3/4 scope (see roadmap plan file).
**Scope**: this document analyzes Phase 1 of the year-long follow-on roadmap
(`/home/lina/.claude/plans/floating-crunching-yeti.md`), pursued after `paper_final.tex`
was finalized for RA-L submission. It is not part of `paper_final.tex` itself.

## Summary

Phase 1 built and physically tested an MPC-style real-time correction mechanism for
6-DoF grasp execution: right before closing the gripper, use a learned model to search
a small set of local pose corrections and apply the best one. Three consecutive
physical pilots (n=25/object, 3 objects) were all net negative, despite each of two
intermediate fixes substantially improving the model's offline validation metric.
This document formalizes four findings from that process, the last of which revised
the document's own initial conclusion after a physical test:

- **(A) Benchmark critique**: the offline validation metric never once predicted the
  physical result's direction across three independently-motivated model variants.
- **(B) Root-cause analysis**: two real, previously-undiagnosed bugs, plus a third,
  newly-confirmed mechanistic issue (settle non-idempotency, directly measured) that
  plausibly explains why the physical results were negative even when the underlying
  model had real signal.
- **(C1) Retrospective evidence (misleading)**: a zero-new-simulation reanalysis of
  already-collected data suggested the model's predictions nearly *double* the
  pre-close contact rate when evaluated open-loop, implying a pivot to predictive,
  world-model-style evaluation would fix the problem.
- **(C2) Physical test of that prediction (the actual finding)**: implementing and
  physically testing open-loop candidate selection was **worse than every reactive
  round tested** (−49.3pp pooled, vs. −9.3 to −18.7pp for reactive correction) --
  because the "open-loop" implementation still evaluated candidates by repeatedly
  settling the *same shared, live object*, which drifts cumulatively even without the
  gripper ever closing. The retrospective analysis (C1) was itself a second instance
  of Part A's core claim: it did not predict the physical result either. The
  corrected takeaway is sharper than "reactive vs. predictive" -- see the revised
  argument at the end of Part C.

## Part A: Benchmark Critique -- offline validation did not predict physical outcome

Three model variants were trained and offline-validated, each with a substantially
improved offline metric over the last, and each physically tested on the same
3-object (Pear/TomatoSoupCan/CrackerBox), n=25/object pilot against the same locked
baseline data:

| Round | Training target | Fix applied | Offline top-1 selection accuracy (chance ≈ 11%, 9 options/group) | Physical pooled Δ vs. baseline |
|---|---|---|---|---|
| 1 | Regress `jaw_obj_xy_gap` | added directional offset feature | 37.5% | **−9.3pp** |
| 2 | Classify `bilateral_contacts` | retargeted to predict success directly (wrong spawn range) | 72.7% | **−18.7pp** |
| 3 | Classify `bilateral_contacts` | spawn-range mismatch fixed | 50.0% | **−13.3pp** |

Per-object numbers, statistical tests, and raw data are in `paperA_data/README.md`'s
"❌ CONCLUDED" entry and `paperA_data/worldmodel_trajs/pilot_*.jsonl`.

**The offline metric is not even directionally informative here.** Round 2 had the
*best* offline accuracy (72.7%) of the three and the *worst* physical result
(−18.7pp). Fixing a genuine bug (spawn-range mismatch, round 2→3) *lowered* the
offline metric (72.7%→50.0%) while *improving* the physical result (−18.7pp→−13.3pp).
There is no monotonic, or even directionally consistent, relationship between the two
across three independent trials.

We also formally note the diagnostic instrument used throughout: `get_grasp_debug_metrics()`
in `tango_robot/env_soarm.py` returns `jaw_obj_xy_gap` (XY distance between the jaw
midpoint and the object centre), `ori_err_norm` (orientation error from ideal top-down),
`bilateral_contacts`/`left_contacts`/`right_contacts` (per-jaw contact counts), and
`symmetry_score`. These same fields correctly diagnosed EBM v1's catastrophic failure
earlier this session (a training-time-only, no-physical-testing diagnosis that held up).
The instrument itself is not the problem -- the problem is treating an *offline proxy
validation built on it* as sufficient evidence before a physical pilot.

**This is a specific, reproducible instance of a problem the literature already
identifies at the level of infrastructure.** Wang et al., "Vision-Language-Action in
Robotics: A Survey of Datasets, Benchmarks, and Data Engines" (arXiv:2604.23001),
argue that future VLA progress depends less on model architecture and more on the
co-design of data engines and *structured evaluation protocols* -- i.e., that the
field's benchmarks and evaluation pipelines are themselves an under-scrutinized
bottleneck, not a solved, trustworthy foundation to build on. Our three-round result
is a small, controlled, mechanistically-traced demonstration of exactly that claim:
an evaluation protocol (held-out top-1 delta-selection accuracy on a proxy label) that
looks like a legitimate gate, and improves substantially round over round, while
carrying no reliable relationship to the metric that actually matters (physical task
success).

## Part B: Root-Cause Analysis -- three failure mechanisms, three literature anchors

### B1: Directional information loss (representation problem)

The first model variant used the scalar `base_jaw_gap` (a distance) as its only
context feature about the candidate's current state. A scalar magnitude cannot
express *which direction* to correct, only *how far off* the current settle is.
Top-1 delta-selection accuracy was 16.7%, barely above the ~11% chance level. Adding
the directional offset vector (`base_off_x`/`base_off_y`, the jaw-midpoint-minus-
object-centre vector, not just its norm) as a feature raised accuracy to 37.5%.

This is a representation-fidelity problem: the *label* (`jaw_obj_xy_gap`) computed by
`get_grasp_debug_metrics()` was always geometrically complete, but the *training
feature set* built from it discarded direction. Ye et al., "3D Generation for
Embodied AI and Robotic Simulation: A Survey" (arXiv:2604.26509), identify exactly
this class of bottleneck at survey scale: a persistent gap between geometric quality
and physical validity in the representations embodied learning pipelines are built
on. Our instance is a small, concrete case of that same gap -- not in mesh generation
(the survey's usual scope) but in how a downstream training pipeline chose to encode
an otherwise-complete geometric signal.

### B2: Spawn-range train/deployment mismatch (an intra-simulation Sim2Real gap)

`paperA_data/scripts/collect_mpc_correction_data.py` spawned objects at
`y ∈ [-0.46, -0.34]` (constants copied from `scripts/record_trajectory.py` without
verification), while `tango_robot/ui.py`'s actual evaluation path
(`env.load_isolated_obj`, used by every physical pilot and every number in
`paper_final.tex`) spawns at `y ∈ [-0.35, -0.10]`. These ranges barely overlap. The
correction model was trained almost entirely on object positions from a different
part of the workspace than the one it was asked to correct at deployment time. This
bug was found only because the user directly asked to sanity-check the simulation's
object and arm positions -- worth recording as a general lesson: *when an ML fix
improves an offline metric but does not transfer to physical results, checking for a
data-generation/deployment distribution mismatch is a productive next step before
concluding the method itself is unsound.*

The Ye et al. survey (arXiv:2604.26509) frames 3D generation's role in embodied AI
around three pillars: Data Generator, Simulation Environment, and **Sim2Real
Bridge** -- the last explicitly concerned with distribution gaps between where
training data comes from and where a system is deployed. The conventional framing of
that gap is simulation-vs-real-world. Our bug is a **recursive, intra-simulation**
instance of the identical structural problem: the data-generating distribution and
the deployment distribution diverged *while both remained entirely in simulation*,
which is more insidious precisely because the surface assumption ("it's all sim, so
there's no domain gap") is false. This is, to our knowledge, an underexplored
variant of the Sim2Real framing worth naming explicitly: **Sim2Sim gap**, or more
precisely, *data-generation/deployment distribution mismatch within a single
simulation stack* -- a failure mode standard Sim2Real tooling does not check for,
because it assumes the "sim" side is monolithic.

### B3: Settle non-idempotency (newly confirmed this session)

`_execute_grasp_physics_topdown`'s trust-but-verify correction path calls
`_settle_at_pose` up to three times without an intervening `reset_robot()`: once for
the original candidate, once for the proposed correction, and once more to "revert"
if the correction proved worse. We tested whether this revert is a true no-op.

**Method**: for 15 (object, seed) combinations across Pear/TomatoSoupCan/CrackerBox,
called `_settle_at_pose` at the same target (a) twice back-to-back with no
intervening call, and (b) with a different-target settle inserted in between (matching
the real correction-then-revert sequence), and measured drift in `jaw_obj_xy_gap`.

**Result** (`paperA_data/scripts/check_settle_idempotency.py`,
`paperA_data/worldmodel_trajs/settle_idempotency_check.jsonl`):

| Test | Mean \|drift\| | Max \|drift\| |
|---|---|---|
| A: immediate repeat, same target, no intervening call | 0.0146 m | 0.1445 m |
| B: intervening different-target settle, then revert | 0.0316 m | 0.2133 m |

**`_settle_at_pose` is not idempotent even in Test A** -- two calls to the identical
target, with nothing else happening in between, can produce meaningfully different
settled geometry (one sample showed a 7.5× change in `jaw_obj_xy_gap`, 0.022→0.167m).
Test B (matching the real revert flow) shows roughly double the average drift and a
larger worst case.

**Root cause, confirmed by reading the code**: `reset_robot()` (`tango_robot/env_soarm.py`,
zeros `qvel` and teleports the arm to `HOME_QPOS`) is called exactly **once**, at the
top of `_execute_grasp_physics_topdown`, before the *first* settle. `_settle_at_pose`
itself never calls it. Every subsequent call -- including the correction attempt and
the revert -- starts its hover approach (`move_ee`, physics-based interpolation, not
teleportation) from whatever arm configuration and residual velocity the *previous*
settle call left behind, not from a canonical reset state. The "same" nominal target
therefore is not physically the same command across repeated calls within one episode.

This means "revert to baseline" in the current implementation does not actually
recover the baseline's original outcome -- it re-executes a *different* physical
process that happens to target the same coordinates. The framing in
"Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and
Mitigation Strategies" (arXiv:2603.16952) is directly relevant here, even though
their scope is compute/memory/latency rather than physics state: their central claim
is that real-time embodied deployment is a *systems* problem, where correctness
depends on coupling between timing, state, and the executing process, not just model
accuracy. Our finding is a physics-simulation instance of that same class of problem:
a reactive, closed-loop correction mechanism couples decision quality to a live,
evolving physical process whose state is not fully reset between decision points --
exactly the kind of state-coupling fragility that survey's "Deployment Gauntlet"
framing warns reactive systems are prone to, independent of how accurate the
underlying model is.

## Part C: Forward-Looking Argument -- predictive vs. reactive, and what "predictive" actually requires

**Original claim (C1, retrospective)**: abandon reactive, high-frequency,
closed-loop-verified correction; pivot toward predictive, world-model-style
evaluation -- assess several candidate outcomes once, independently, before
committing, rather than committing and then verifying against a live, state-coupled
physical process that Part B3 shows is not cleanly reversible. This is the central
thesis of "World Model for Robot Learning: A Comprehensive Survey" (arXiv:2605.00080;
authors include Pieter Abbeel, Jitendra Malik, Yilun Du, Jiajun Wu among others),
which surveys world models across policy learning, planning, simulation, and
evaluation from exactly this "assess before acting" premise. As shown below, this
claim survived a *retrospective* data check but not a *physical* one -- the revised,
more precise version is at the end of this section.

**C1: retrospective, zero-new-simulation evidence** (`paperA_data/scripts/analyze_openloop_vs_closedloop.py`):
reusing the already-collected round-3 dataset (`paperA_data/worldmodel_trajs/mpc_correction_{pear,can,cracker}.jsonl`,
120 candidate groups, each with 1 base + 8 delta rows and real recorded
`bilateral_contacts` outcomes from the *original, single, clean* data-collection
settle -- no repeated in-episode calls), we retrospectively compared:

- **Open-loop / predictive**: for each group, commit to the model's top-predicted row
  (single evaluation, no verification), use its real recorded outcome.
- **Closed-loop / reactive (idealized)**: commit to the top-predicted row only if its
  real outcome is at least as good as the group's own recorded base-row outcome,
  otherwise "revert" to that recorded base value. This is the *best possible case*
  for the reactive protocol -- it assumes a perfectly idempotent revert, which Part B3
  shows the real system does not have.

| Protocol | Success rate (bilateral_contacts=1) |
|---|---|
| Baseline only (no model, delta=0 always) | 15.0% (18/120) |
| Open-loop / predictive | **28.3% (34/120)** |
| Closed-loop / reactive (idealized, perfect revert) | 29.2% (35/120) |

**The model's predictions carry real signal** -- nearly double the baseline
pre-close-contact rate, whether used open-loop or in an idealized closed-loop (the two
are nearly identical here; revert was only triggered in 1/120 groups). This is the
missing link between Parts A/B and the physical pilot failures: **the correction
mechanism was never primarily a modeling problem.** The signal exists and is
substantial when evaluated the way the training data itself was generated -- one
clean, independent settle per candidate. It is the *reactive deployment protocol* --
multiple dependent, non-idempotent `_settle_at_pose` calls compounding state drift
within a single episode (Part B3) -- that destroys this signal before it reaches the
physical pilot.

**Caveat (flagged before the physical test, confirmed correct by it)**: this
retrospective analysis operates on `bilateral_contacts` at the pre-close settle
stage, the same proxy label the model was trained on -- not final grasp success. The
~2x improvement shown here demonstrated the *proxy signal* survives open-loop
*retrospective replay*; whether it carries through to a genuinely live open-loop
*execution* was an untested, falsifiable prediction. It was tested. It failed, badly.

### C2: physical test of the open-loop prediction -- catastrophically negative, and why

We ran the predicted protocol for real (`paperA_data/scripts/run_openloop_select_pilot.py`):
per episode, generate 5 candidates (identical formula and seeding to `ui.py`'s own
Baseline candidate generator), evaluate each with one `_settle_at_pose` call (never
closing the gripper), commit to and physically execute only the model's top choice.
Same 3-object, n=25 pilot, same locked baseline comparison.

| Object | Baseline | Open-loop select | Δ | McNemar p |
|---|---|---|---|---|
| Pear | 64.0% | 12.0% | −52.0pp | 0.0002 |
| TomatoSoupCan | 88.0% | 24.0% | −64.0pp | 0.0001 |
| CrackerBox | 40.0% | 8.0% | −32.0pp | 0.0386 |
| **Pooled** | **64.0%** | **14.7%** | **−49.3pp** | — |

**This is far worse than any reactive-correction round tested (−9.3pp to −18.7pp),
not better.** The retrospective analysis's optimism did not survive contact with a
live pilot -- which is itself a second, independent instance of Part A's central
claim (offline/retrospective validation does not reliably predict physical
deployment), now demonstrated on the *predictive* protocol too, not just the
reactive one.

**Root cause, confirmed directly** (`_settle_at_pose` called 5 times in sequence,
object position logged before and after each): the object's live position drifts
*monotonically* across the evaluation phase even though the gripper never closes --
one traced example showed cumulative XY drift of 0.3cm / 1.0cm / 1.7cm / 2.2cm / 2.7cm
across the five evaluation calls. Candidates are generated within a ±6cm window
around the object's *original* position; by the fifth evaluation the object has
moved far enough that even a correctly-identified "best" candidate's coordinates no
longer align with where the object actually is by the time of the sixth, real,
committing settle.

**This sharpens, and partially revises, the argument in Part B3.** The mechanism is
the same one identified there (repeated live settle calls on a shared, persistent
object without a full reset compound state drift), but it is not specific to
reactive, verify-and-revert protocols. **Any protocol that evaluates multiple
candidates via repeated live physics queries against the one object it is about to
manipulate inherits this fragility, regardless of whether the decision logic on top
is reactive or "predictive."** Calling a protocol open-loop at the level of decision
logic (commit once, no verification) does not make the underlying physical process
open-loop if evaluation itself still requires repeatedly touching the same live
object -- and in this implementation, evaluating 5 candidates before committing is
*worse* than reactive correction's 2-3 touches, not better, because there are more
opportunities for cumulative drift before the one commitment that matters.

### What This Argues For, Concretely (revised after the physical test)

The original claim -- "evaluate several candidates independently, then commit" -- is
correct in spirit but was implemented wrong: "evaluate via `_settle_at_pose`" is not
actually independent evaluation, because it still perturbs the one real object shared
across all candidates. A genuinely independent, world-model-style evaluation needs to
score candidates **without touching the live/simulated object at all** during
evaluation -- e.g., a learned model that predicts `bilateral_contacts`/`jaw_obj_xy_gap`
purely from the candidate's geometric features and the object's *originally observed*
pose (exactly the static-feature scoring EBM v2 already does, and exactly what this
session's retrospective analysis implicitly assumed when it reused each row's
independently-collected ground truth rather than re-querying a shared object) --
committing to execute only once no matter which candidate wins. Concretely: this is
closer to EBM v2's original design (already tested this session, reached parity with
baseline: 74.0% vs. 77.7%, not a clean win either, but nowhere near this catastrophic)
than to anything requiring a live settle-based evaluation loop, open- or closed-loop.
The lesson from this whole Phase 1+2 arc is now sharper than originally framed: **the
failure mode is not "reactive vs. predictive," it is "physically-grounded, live,
repeated-query evaluation vs. genuinely offline, model-only evaluation" -- only the
latter avoids compounding simulation state drift, and EBM v2's already-tested,
already-parity-reaching design is the closer existing example of it, not the new
mechanisms built in Phase 1/2.**

## References

1. World Model for Robot Learning: A Comprehensive Survey (authors include Pieter Abbeel, Jitendra Malik, Yilun Du, Jiajun Wu among others; exact author ordering not verified). arXiv:2605.00080, 2026.
2. Wang, Z. et al. Vision-Language-Action in Robotics: A Survey of Datasets, Benchmarks, and Data Engines. arXiv:2604.23001, 2026.
3. Ye, T. et al. 3D Generation for Embodied AI and Robotic Simulation: A Survey. arXiv:2604.26509, 2026.
4. Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies. arXiv:2603.16952, 2026.
