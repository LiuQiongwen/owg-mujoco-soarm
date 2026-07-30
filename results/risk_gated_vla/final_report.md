# Final report: object-centric counterfactual world critic and policy risk gate

Date: 2026-07-30. Independently re-verified (scene-key disjointness, checkpoint
train/val isolation, and headline statistics recomputed from raw `scenes.jsonl`
rather than trusted from summary files) before this revision.

## Executive conclusion

The old pre-execution critic fails, but the idea does not. After correcting
causal labels, workspace geometry, paired candidate pools, and the bilateral
grasp criterion, an object-relative critic generalizes to two fully untouched
test batches. On the frozen 150-scene confirmatory batch, using the actually
**live-executed** paired outcomes (not an offline re-scoring — see the
methodology note below), it improves top-1 grasp success from **36.0%
(54/150) to 50.0% (75/150), +14.0pp, 27 paired wins vs 6 losses, exact
McNemar p=3.24e-4**. This is the current positive paper result.

The ensemble-uncertainty gate does not add value: it accepts 98.7% of critic
choices and scores about the same as the ungated critic. The action-policy
integration is functional, but the present 15-demo ACT pilot fails online and
is not a positive VLA result. See the negative/incomplete-results archive
below for the full, itemized list — nothing in this study was hidden or
reframed as positive after the fact.

## Methodology note: two numbers per comparison, and why they differ slightly

Every scene records both (a) the outcome that was **actually, live executed**
for the method selected at collection time (`outcomes.<method>.success`), and
(b) an offline re-scoring against the same scene's fully-swept 10-candidate
ground truth (`oracle_per_candidate`), used to evaluate models that were never
the live "world_critic" for that collection run (`global_bce`, `object_bce` —
only `object_counterfactual` was ever live-selected). These two numbers should
agree for `geometry` and `object_counterfactual` (the two methods that WERE
live-executed) but do not have to agree exactly, because **MuJoCo's contact
solver is not perfectly reproducible on marginal grasps**: re-executing the
literal same candidate pose in the literal same scene occasionally flips the
boolean success outcome. Independently verified: across the confirmatory
batch's 150×2=300 live-vs-reswept comparisons, exactly 2 flipped (drill scene
9's geometry pick: live=True, resweep=False; mustard scene 47's critic pick:
live=False, resweep=True) — a ~0.67% marginal-grasp non-determinism rate. The
dev-test batch shows the same pattern at the same rate (1 flip per method out
of 90). This is a genuine physical-reproducibility limitation, not a script
bug, and is the same class of finding already documented independently in
`paper_advanced_robotics.tex` (~12.5% single-run instability in a different
pipeline) — cite it as a stated limitation, not something to paper over.

**Reporting convention used below**: for `geometry` and `object_counterfactual`
(live-executed, paired A/B), report the live-executed `outcomes`-based number
as primary — it is the actual online paired trial, not a redundant re-run.
For `global_bce`/`object_bce` (never live-selected in this collection run),
the offline re-scored number is the only number that exists and is reported
as such, clearly labeled offline.

## Why apparently conflicting results exist

There are two separate critic experiments and they must not be combined.

### A. Phase-1 exploratory retraining on base-seed 42

- Source: `phase1/scenes.jsonl`, 150 scenes.
- Checkpoints: `phase1/critic_ckpts/`.
- The 80/20 per-seed validation metrics are promising but optimistic because
  the same small validation split selects the best of roughly 40 epochs.
- Evaluation on all 150 scenes is contaminated: every ensemble member trained
  on approximately 80% of that population.
- The reported 61.3% versus 34.0% full-population result is exploratory only
  and must not be cited as generalization evidence.

The methodological warning in
`phase1/EXPLORATORY_counterfactual_critic.md` is correct for this experiment.

## Complete experiment inventory and evidence status

The following inventory prevents pilot, contaminated, and confirmatory numbers
from being mixed in the manuscript.

| Batch | Scenes | Purpose | Main outcome | Manuscript status |
|---|---:|---|---|---|
| `smoke/` | 6 | import, determinism, pool identity | harness checks only | diagnostic, do not cite as performance |
| `pilot_prod_n10_20260730/` | 30 | early production-primitive sanity check | geometry 80%, critic 56.7% | diagnostic; superseded by strict protocol |
| `pilot_bilateral_bias01_n10_20260730/` | 30 | strict bilateral/contact sanity check | geometry 53.3%, critic 33.3% | diagnostic; small pilot |
| `phase1/` base-42 | 150 | pre-registered stale-critic gate | stale critic 20.7%, AUROC 0.4996 | citable only as the negative stale-checkpoint result |
| `counterfactual_train_n40_20260730/` base-100 | 120 | critic training + scene-grouped validation | training source | never an evaluation result |
| `counterfactual_test_n30_20260730/` base-200 | 90 | independent development test | live critic 48.9% vs geometry 33.3%, +15.6pp, p=.00258 | citable independent result |
| `confirmatory_n50_seed300_20260730/` base-300 | 150 | frozen confirmatory test | live critic 50.0% vs geometry 36.0%, +14.0pp, p=3.24e-4 | primary citable result |
| `phase1/critic_ckpts/` full-population eval | 150 | exploratory retraining on phase1 population | 61.3% vs 34.0% | contaminated; never cite |
| `risk_gate.json` | 150 | frozen uncertainty threshold test | coverage 98.7%, no gain | negative ablation |
| ACT invalid pilot | 30 demos | recorder audit | all five arm actions were zero | invalid; exclude |
| ACT v2 pilot | 15 demos | corrected recorder + ACT | offline MAE .087, online 0/1 | incomplete pilot; negative online result |
| Phase 3 hardware protocol | 0 trials | safety and transfer design | no physical execution | design only; no hardware claim |

The primary quantitative claim therefore uses only the base-200 and base-300
rows. All scene populations are disjoint by `(object, scene_seed)`, and the
base-100 checkpoint train/validation keys have zero overlap with either test
population.

## Integrated research story

The study has four logically ordered claims:

1. **Audit diagnosis:** the predecessor result cannot be trusted because its
   method-dependent seed breaks pairing and its contact-at-close label does not
   require lifting; the stale critic consequently has AUROC 0.4996 under the
   corrected harness.
2. **Prospective reconstruction:** using only pre-execution, object-relative
   features and strict live execution, the rebuilt critic selects better grasp
   candidates than geometry on two independent batches (+15.6pp and +14.0pp).
3. **Boundary characterization:** pairwise BPR is not independently separated
   from object-relative BCE; ensemble uncertainty does not improve selection;
   marginal MuJoCo outcomes have a small ~0.6--1% flip rate.
4. **Deployment readiness boundary:** the corrected ACT pipeline is runnable
   but 15 demonstrations/object are insufficient for closed-loop recovery, and
   real hardware has only been specified, not executed.

The resulting T-RO framing is therefore an audit-and-reconstruction paper with
one prospective world-critic case study, not a completed VLA or sim-to-real
paper. The critic improvement is the main result; risk gating and ACT define
the honest limits of the current evidence.

### B. Strict independent train/test/confirmation chain

The later experiment uses disjoint scene populations:

| Role | Base seed | Scenes | Used for model selection? |
|---|---:|---:|---|
| Train/validation | 100 | 120 | Yes |
| Independent development test | 200 | 90 | No |
| Frozen confirmatory test | 300 | 150 | No |

All pairwise scene-key intersections among phase1/base-42, train/base-100,
test/base-200, and confirmation/base-300 are exactly zero (independently
recomputed, not just asserted). Inspection of the saved base-100 checkpoints'
`train_keys`/`val_keys` confirms zero overlap with both test batches, and that
both sets are a subset of the base-100 population only.

## Independent results

### Development test: 90 untouched scenes (base-seed 200)

| Method | Success | Scoring | Delta vs geometry | Exact McNemar |
|---|---:|---|---:|---:|
| Geometry | 30/90 (33.3%) | live-executed | -- | -- |
| Global BCE | 38/90 (42.2%) | offline re-scored | +10.0pp (vs offline geo 32.2%) | p=0.108 |
| Object-relative BCE | 42/90 (46.7%) | offline re-scored | +14.4pp (vs offline geo 32.2%) | p=0.0072 |
| **Object-relative counterfactual** | **44/90 (48.9%)** | **live-executed** | **+15.6pp** | **p=0.00258** |

Per-object (live-executed, geometry / counterfactual): cracker 4/30 vs 4/30
(tie), drill 7/30 vs 18/30, mustard 19/30 vs 22/30.

### Frozen confirmation: 150 new untouched scenes (base-seed 300)

| Method | Cracker | Mustard | Drill | Pooled | Scoring |
|---|---:|---:|---:|---:|---|
| Geometry | 9/50 | 29/50 | 16/50 | 54/150 (36.0%) | live-executed |
| Global BCE | 9/50 | 35/50 | 17/50 | 61/150 (40.7%) | offline re-scored |
| Object-relative BCE | 9/50 | 45/50 | 21/50 | 75/150 (50.0%) | offline re-scored |
| **Object-relative counterfactual** | **9/50** | **42/50** | **24/50** | **75/150 (50.0%)** | **live-executed** |

Object-relative counterfactual versus geometry (both live-executed, the
strongest evidence in this study): **27 paired wins, 6 losses, exact McNemar
p=3.24e-4, +14.0pp**. Object-relative BCE and counterfactual are not
significantly different from each other (offline comparison, 2 wins/3 losses,
p=1.0), so **the defensible novelty claim is object-centric learned
counterfactual scoring, not that the BPR pairwise loss is independently
proven to be the load-bearing component** — this remains open, not resolved
positively or negatively.

## Risk gate result

The uncertainty threshold was calibrated on the base-200 development set and
frozen before reading base-300 confirmation outcomes. On confirmation:

- critic coverage: 98.7%;
- gated policy: 75/150 (50.0%);
- ungated critic: 75/150 (50.0%, live-executed) / 76/150 (offline re-scored,
  see methodology note);
- geometry fallback: 54/150 (36.0%, live-executed).

The current ensemble standard deviation is not a useful failure detector —
gating barely moves the number in either direction. Report the critic
improvement; do not claim an uncertainty-gating gain.

## ACT/action-policy integration

The MuJoCo-to-LeRobot pipeline now supports 20 Hz recording, multiple objects,
critic-guided demonstrations, checkpoint loading, offline diagnostics, and
closed-loop joint control with the same target-specific bilateral attachment
rule as the scripted grasp primitive.

One audit found that an earlier recorder silently stored zero for all five arm
actions due to nonexistent `*_act` actuator names. Its 30-demo dataset and
500-step checkpoint are invalid and excluded. The corrected v2 pilot contains
15 episodes/1,005 frames with nonzero variance in all six action dimensions.
A 13M-parameter ACT trained for 1,000 steps reached loss 0.568 and offline
one-step MAE 0.087, but failed its first online mustard rollout by knocking the
object down 4.5cm (independently verified against
`results/risk_gated_vla/act_v2_online_smoke_n1.json`: dz=-0.0451, success=false).
Five demos per object are insufficient for closed-loop recovery.

## Negative and incomplete results — full archive

Per this study's own rule (do not report only positive results; distinguish
evidence from hypothesis from refuted conclusions), every negative or
inconclusive finding produced in this investigation, in one place:

1. **Original `wm_reranking_results.md` +15.6pp claim: invalid, doubly
   discredited.** `audit.md` Section 4: the predecessor eval script encoded
   `method` into its RNG seed, so "geometry" and "world_model" never shared a
   candidate pool or scene — the comparison was never actually paired. `audit.md`
   Addendum: even independent of pairing, the underlying success criterion
   (`contact or grasped or lifted`, checked before any lift) saturated to
   150/150 regardless of candidate quality. Do not cite this file's numbers.

2. **Stale critic (`world_model/mlp_predictor.pkl`, 2026-05-15) formally
   fails the pre-registered Phase 1 gate.** `phase1/RESULT.md`: pooled
   delta -14.0pp (wrong sign vs the required +8pp), only 1/3 objects in the
   pooled direction, and — the clearest single number — **AUROC=0.4996 on
   1500 real per-candidate ground-truth labels: exactly chance**. This
   checkpoint was trained on labels from the same broken success criterion
   and pre-rotation workspace center documented in item 1; the failure is
   fully explained, not mysterious.

3. **Phase-1 exploratory critic's full-population number (61.3% vs 34.0%,
   +27.3pp) is contaminated and must not be cited.** Every ensemble member
   trained on ~80% of the exact population it was "evaluated" against. See
   `phase1/EXPLORATORY_counterfactual_critic.md`. (The properly held-out,
   disjoint-seed re-run in this same report's main results section is what
   should be cited instead.)

4. **BPR pairwise loss's independent contribution is not established.**
   Object-relative BCE (no pairwise term) and object-relative counterfactual
   (BCE + within-scene BPR) are statistically indistinguishable from each
   other on the frozen confirmatory batch (p=1.0). The paper's novelty claim
   must rest on "object-centric learned scoring beats geometry," not on the
   pairwise-loss mechanism specifically.

5. **Ensemble-uncertainty risk gating adds no measurable benefit.** Frozen
   threshold from base-200, evaluated blind on base-300: gated success is
   statistically indistinguishable from the ungated critic's own success rate
   (coverage 98.7%, i.e. the gate almost never actually fires). Do not claim
   a gating contribution in the paper.

6. **ACT/VLA integration is a working pilot, not a result.** First real
   online rollout knocked the target object 4.5cm off position and failed.
   Five demonstrations/object is not enough data for any closed-loop recovery
   behavior to have been learned. A first dataset (30 demos, 500-step
   checkpoint) was found and discarded after an audit revealed the recorder
   silently logged all-zero actions for all five arm joints (nonexistent
   `*_act` actuator names) — that checkpoint is invalid, not merely weak.

7. **Physics-level non-determinism on marginal grasps (~0.6-1% flip rate).**
   Re-executing the identical candidate pose in the identical scene
   occasionally changes the boolean success outcome (2/300 in the
   confirmatory batch, 2/180 in the dev-test batch — both methods affected
   roughly equally, not biased toward one). Small enough not to change any
   conclusion in this report, but real, and should be stated as a limitation
   rather than implied away by treating any single execution as ground truth.

8. **Two project-wide infrastructure bugs found during this study, out of
   this study's own scope to fully re-verify.** `audit.md` Addendum B: the
   production `physics_weld_after_bilateral` grasp mode's weld-attach gate
   accepted single-jaw contact, not the bilateral contact its name and
   documented protocol require — fixed in `tango_robot/env_soarm.py`, but
   whether this affected prior `paperA_data/`/RA-L-submission results using
   that mode is a separate, unresolved question, explicitly flagged as out of
   scope here. `audit.md` Addendum C: the pre-rotation `_CENTRE_Y=-0.40`
   workspace constant, hardcoded in several May/June scripts, produces ~4.6cm
   of descend-IK error under the current (July) rotated mount — any other
   script still using that constant should be treated as suspect until
   checked.

## Recommended paper scope and next experiment

The shortest defensible paper path is:

1. Lead with the frozen critic result (live-executed: +14.0pp, p=3.24e-4;
   consistent, same-direction result on the independent dev-test batch:
   +15.6pp, p=0.00258).
2. Present causal-admissibility and paired shared-pool evaluation as core
   methodology — this is what let the original invalid +15.6pp claim be
   caught and replaced with a defensible one.
3. Treat risk-gate uncertainty and ACT as ablations/integration pilots, not
   positive claims (item 5 and item 6 above).
4. State the BPR-loss and marginal-grasp-nondeterminism limitations plainly
   (items 4 and 7) rather than letting a reviewer find them first.
5. For a stronger VLA section, collect at least 50 valid v2 demonstrations per
   object plus perturbed/DAgger recovery trajectories, then evaluate a frozen
   ACT checkpoint on new scenes.
6. Generate multiple ACT/VLA proposals (stochastic latent samples or explicit
   candidate chunks), FK-project their terminal grasp poses, and pass those
   proposals through `ActionPolicyRiskGate`. A single deterministic action
   chunk cannot demonstrate proposal-level risk gating.

## Key artifacts

- `counterfactual_train_n40_20260730/scenes.jsonl` (base-100, train)
- `counterfactual_models_20260730/` (checkpoints + `validation_summary.json`)
- `counterfactual_test_n30_20260730/scenes.jsonl` + `critic_comparison.json` (base-200, dev-test)
- `confirmatory_n50_seed300_20260730/scenes.jsonl` + `critic_comparison.json` + `risk_gate.json` (base-300, frozen confirmation)
- `world_model/risk_gate.py`
- `scripts/eval_act_mujoco.py`, `scripts/eval_act_offline.py`
- `results/risk_gated_vla/act_v2_online_smoke_n1.json`
- `audit.md` (Phase 0 audit + Addendum: the two project-wide bugs, item 8 above)
- `phase1/RESULT.md`, `phase1/EXPLORATORY_counterfactual_critic.md` (superseded exploratory line, item 2-3 above)
