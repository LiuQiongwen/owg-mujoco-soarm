# Research Proposal: Affordance-Auxiliary Small-VLA Fine-Tuning for 6-DoF Grasp Candidate Generation

**Date**: 2026-07-12
**Status**: draft, offline-validated at toy scale, not yet novelty-checked externally (Codex/GPT-5.4 returned 401, unavailable this session), not yet piloted at real VLA scale.

## Problem Anchor

- **Bottom-line problem**: Paper A's original headline method (OT-CFM candidate generation) is empirically dead (pooled -10.0pp vs baseline, p=0.0025, 7/7 objects negative). We need a genuinely new, learned, methodologically defensible core method for 6-DoF grasp candidate generation, at T-RO/IJRR bar, under small-data (~400 verified trials/object x 7 objects) and single-GPU (RTX 3060 6GB) constraints.
- **Must-solve bottleneck**: every previous attempt to *generate or correct raw world-frame pose (x,y,z,roll,pitch,yaw)* directly has failed or shown no advantage over trivial baselines (see "Ruled-out routes" below). We need a method that targets a genuinely informative representation.
- **Non-goals**: replacing the existing LGGSN pairwise reranker or the consensus candidate-selection heuristic (both keep working, both stay in the pipeline as-is); full VLA foundation-model pretraining (no compute/data budget for that); real-time reactive correction during physical execution (already tried, killed).
- **Constraints**: single RTX 3060 6GB GPU, ~400 sim-verified trials/object x 7 objects, ~41 (extensible) real SO-ARM101 teleop demonstrations of one object (Pear) via LeRobot, no access to external reviewer (Codex 401).
- **Success condition**: a documented, honest offline validation chain (as done for every ruled-out route below) showing the new mechanism provides a real, non-trivial, non-artifactual improvement over the strongest trivial control, *before* committing to physical pilots.

## Ruled-Out Routes (do not re-propose; each has a decisive offline or physical falsification already on record, this session and earlier)

1. **Minibatch-OT-coupled CFM** — physical pilot net -10.0pp pooled, p=0.0025, 7/7 objects negative.
2. **C²OT condition-aware OT fix** — partial/inconsistent, Pear (worst object) unchanged.
3. **MPC-style real-time settle correction** (learned state-transition model, online search before gripper close) — 3 physical pilot rounds, all net negative (-9.3/-18.7/-13.3pp).
4. **Retrieval-Anchored Residual Flow Matching (RARFM)** — single real-object anchor bank: -18.0pp vs naive mean-shrinkage control. Oracle same-object single-NN: still -17.4pp vs control (proves it's an estimator-noise problem, not a data-scale problem). k-NN(10)-averaged oracle: ties control (0.0pp advantage) — fixes the noise but adds no value. Object-centroid-normalized coordinates: does not help (-14.0pp, confirms it's not a coordinate-frame problem).
5. **Geometric-feature-space generation + nearest-neighbour back-projection** (generate a target descriptor, retrieve nearest real candidate) — naive "typical successful descriptor" nearest-centroid selection: +4.6pp mean lift over random pick, but this is *weaker* than what the already-existing, already-trained LGGSN pairwise reranker plausibly achieves on the same features — no clear incremental value over existing infra unless reframed around candidate-sampling density (which needs new infra to test, not yet done).

All five were killed or shown non-viable using **existing sim data only**, no new physical trials burned.

## Key Diagnostic Finding That Motivates This Proposal

Logistic-regression AUC for predicting true physical grasp success (`label`), per object, comparing raw world-frame pose (x,y,z,roll,pitch,yaw) vs. the existing 12-dim LGGSN-style object-relative geometric/affordance features (`width, score, dz, dz_lift, need_dz, H, dist_to_centroid, z_rel, local_point_density, normal_consistency, contact_width_ratio, pe_ik`):

| Object | pose AUC | geom AUC |
|---|---|---|
| pear | **0.483 (chance)** | **0.903** |
| mustard | 0.542 (chance) | 0.916 |
| can | 0.504 (chance) | 0.846 |
| banana | 0.735 | 0.914 |
| cracker | 0.777 | 0.725 |
| cylinder | 0.617 | 0.772 |
| drill | 0.658 | 0.604 |
| **pooled (joint model)** | **0.581** | **0.725** |

For exactly the objects hit hardest by every previous generative failure (Pear, Mustard, Can), raw pose carries **no success signal at all**. Object-relative geometric/affordance features carry strong signal. This explains, in one clean number, why every method that generates or corrects raw pose has struggled, and why LGGSN (which scores using these geometric features) and consensus (which implicitly benefits from the same structure) keep working.

**A further, additional diagnostic (toy-scale, this session)**: a 32-dim bottleneck representation of the per-candidate SAM visual feature (`visual_feat`, 256-dim, confirmed to vary *per-candidate*, not just per-scene), when trained *only* to regress pose, achieves a downstream success-probe AUC of 0.775 ± 0.004 (5 seeds). Adding an auxiliary task — forcing the same bottleneck to *also* regress the 12-dim geometric descriptor — raises this to 0.791 ± 0.012 (4/5 seeds improved, one tied). Small but consistent and non-artifactual (unlike routes 1-5 above, this is a genuine positive control, not a trivial-baseline tie). Caveat: even the best bottleneck (0.791) underperforms a direct classifier on the raw uncompressed `visual_feat` (0.850) — the compression itself is lossy, and the auxiliary task only partially recovers what pose-only training throws away. This means the effect is real but the current toy architecture (2-layer MLP, 32-dim bottleneck, naive equal-weighted multi-task loss) is almost certainly not tuned well — a real prototype needs to do better than partial recovery.

## Method Thesis

**One-sentence thesis**: fine-tune a small, single-GPU-feasible pretrained VLA (SmolVLA) on the existing sim data + growing real teleop data, with an auxiliary decoder head that reconstructs the 12-dim object-relative geometric/affordance descriptor from the shared representation — forcing the policy's internal representation to retain the success-informative signal that pose-only supervision discards — and additionally pretrain/regularize the shared trunk with a cheap, *offline* (not online-reactive) auxiliary dynamics-prediction signal reusing the already-built `_settle_at_pose` sub-process data (small-perturbation → resulting geometric-descriptor-change), giving the representation a lightweight "world-model" sense of local grasp geometry without any real-time correction loop.

**Why this is the smallest adequate intervention**: no new candidate generation, no new retrieval infrastructure, no real-time correction loop (the three components that already failed). Just two auxiliary losses bolted onto an existing, off-the-shelf, small VLA backbone, using data (sim geometric labels, settle-subprocess rollouts, real teleop demos) that already exists or is cheap to extend.

**Why this is timely / frontier-native**: directly analogous to recently published mechanisms — auxiliary affordance decoders in **AffordVLA** (arXiv:2605.17517) and **SG-VLA** (arXiv:2603.22760), and offline world-model-rollout-as-training-signal in **RISE** — but no existing paper combines (a) a *diagnosed*, *quantified* pose-vs-affordance information asymmetry specific to small-data single-object-category grasp learning, (b) affordance-auxiliary supervision reusing an existing hand-engineered feature set (LGGSN's 12-dim descriptor) rather than a learned affordance map, and (c) a cheap settle-subprocess-based dynamics auxiliary signal recycled from a previously-failed real-time-correction project (repurposed as an offline representation-learning signal, not an online loop).

## Contribution Focus

- **Dominant contribution**: affordance-auxiliary multi-task fine-tuning of a small VLA for 6-DoF grasp candidate generation in the small-data, single-GPU regime, motivated and validated by an explicit, quantified pose-vs-affordance information-asymmetry diagnosis (the AUC table above) — itself a reusable diagnostic methodology, not just a one-off finding.
- **Optional supporting contribution**: the settle-subprocess-derived offline dynamics auxiliary signal (repurposing Phase-1-MPC's cheap data-generation infrastructure as a representation-learning signal instead of an online correction loop) — a concrete illustration of the "predictive/offline vs reactive/online" methodological argument already drafted in `paperA_data/phase2_reactive_autopsy_predictive_blueprint.md`, tying this new method back to that existing analysis.
- **Explicit non-contributions**: not proposing a new VLA architecture (SmolVLA used as-is); not proposing a new retrieval or generative-sampling mechanism (routes 4/5 already explored and not pursued further); not claiming real-time/reactive capability.

## Proposed Method

### Complexity Budget

- **Frozen/reused**: SmolVLA pretrained backbone (LoRA-adapted, not fully retrained); existing LGGSN 12-dim geometric feature extractor (used as auxiliary-label generator, not retrained); existing `_settle_at_pose` sub-process (used as a cheap auxiliary data source, not modified); existing LeRobot recording/training pipeline.
- **New trainable components** (2, within `MAX_NEW_TRAINABLE_COMPONENTS`): (1) a small affordance-auxiliary decoder head (linear or 2-layer MLP) attached to SmolVLA's shared representation; (2) a small dynamics-auxiliary decoder head predicting settle-subprocess geometric-descriptor deltas from (representation, perturbation) pairs.
- **Tempting additions intentionally excluded**: no new candidate retrieval bank (route 4, killed); no world-model-guided online action search (route 3, killed); no direct raw-pose generative redesign (route 5, weak evidence only).

### System Overview

```
[SAM visual feat / RGB obs] -> SmolVLA shared trunk (LoRA-tuned)
                                   |-- main head: grasp pose (x,y,z,roll,pitch,yaw)
                                   |-- aux head A: 12-dim geometric/affordance descriptor
                                   |                (label source: existing LGGSN candidate DB)
                                   |-- aux head B: predicted Δ(geometric descriptor) under a
                                                    small pose perturbation
                                                    (label source: cheap settle-subprocess replay,
                                                     reused from the killed Phase-1 MPC project,
                                                     used OFFLINE only)
```

### Training Plan

1. **Stage 0 (already done, this session)**: toy-scale validation that the affordance-auxiliary mechanism gives a real (if modest) representation-quality improvement (AUC 0.775 -> 0.791) — decisive enough to proceed, not decisive enough to skip a proper-scale check.
2. **Stage 1 (next, before any real-hardware time)**: reproduce the same auxiliary-vs-single-task comparison using the actual SmolVLA architecture + LoRA adapters (not a 2-layer MLP proxy) on the existing 7-object sim dataset, with proper bottleneck sizing and multi-task loss weighting swept (not naive equal-weighting as in the toy check). Go/no-go: does the gap widen meaningfully beyond the toy check's +1.6pp, and does the multi-task representation close more of the gap to the raw-`visual_feat`-classifier ceiling (0.850)?
3. **Stage 2**: add the settle-subprocess dynamics-auxiliary head, same offline comparison, check for further improvement (or at least no regression).
4. **Stage 3**: fine-tune on real Pear teleop demonstrations (LoRA), extend recording to 2-3 more objects if Stage 1-2 signal justifies the operator time.
5. **Stage 4**: sim physical pilot (n=25, 3 objects) comparing this method's candidate quality against Baseline/consensus, before any further real-hardware investment.

### Failure Modes and Diagnostics

- **Failure mode 1**: Stage 1 shows the gap does not widen at real-VLA scale (the toy MLP's +1.6pp was itself close to noise). Diagnostic: rerun toy check with more seeds/bootstrap CI around the +1.6pp estimate before investing in Stage 1 infrastructure.
- **Failure mode 2**: auxiliary supervision helps representation-probe AUC but does not translate into better *generated pose* quality (the actual thing that matters). Diagnostic: Stage 1/2 must evaluate on generated-pose quality (e.g., re-run the pose-AUC-style analysis on the fine-tuned model's own outputs), not just probe AUC on the frozen representation.
- **Failure mode 3**: settle-subprocess dynamics auxiliary signal, reused from a killed project, turns out to encode the same non-informative pose-centric structure and does not help. Diagnostic: Stage 2 must show incremental improvement over Stage 1 alone before being kept in the final design.

## Claim-Driven Validation Sketch

### Claim 1: affordance-auxiliary supervision yields a more success-informative shared representation than pose-only supervision, at real VLA scale (not just toy MLP scale).
- Minimal experiment: SmolVLA + LoRA, single-task vs multi-task (pose + geom aux), frozen-representation linear probe AUC, 5 seeds, bootstrap CI.
- Baseline/ablation: pose-only SmolVLA fine-tune (no aux head).
- Metric: linear-probe AUC on held-out sim candidates.
- Expected evidence: multi-task AUC significantly (CI-non-overlapping) above single-task, and gap to raw-feature ceiling (0.850) narrower than the toy check's residual gap.

### Claim 2: the improved representation translates into better *generated* candidate poses, not just better probe AUC.
- Minimal experiment: sample candidate poses from the fine-tuned model (both variants), re-score with the existing evaluation harness (physics execution or the offline pose-AUC proxy), compare success-rate-relevant metrics.
- Baseline: Baseline (random CoM) and current LGGSN+consensus pipeline.
- Metric: physical success rate (sim, n=25, 3 objects) — same protocol as all prior Paper A comparisons.
- Expected evidence: not required to beat consensus outright (that is not the point) — required to show a real, non-trivial improvement over a pose-only-supervised SmolVLA fine-tune, establishing the auxiliary mechanism's value in isolation.

## Novelty Argument (self-reviewed; Codex external review unavailable, 401)

**Closest work**: AffordVLA (implicit affordance-feature alignment injected into VLA) and SG-VLA (auxiliary decoders for affordance/pose reconstruction) already do "affordance-auxiliary VLA training" as a general mechanism. **This proposal is not novel at the mechanism-class level** — it is a specific, motivated instantiation for a regime (single-object-category, ~400 sim trials + tens of real demos, single 6GB GPU) that those papers do not target, combined with (a) a quantified, paper-specific diagnostic argument for *why* this mechanism should help here (the AUC table), and (b) the offline-repurposed settle-subprocess dynamics signal, which — as far as this session's literature search found — has no direct precedent (world-model-rollout-as-offline-representation-auxiliary, specifically recycled from a documented failed *online* mechanism, is a distinct framing from RISE's world-model-rollout-as-training-data).

**Honest verdict**: this is a **NOVEL-BUT-INCREMENTAL** combination, not a fundamentally new mechanism. Its strongest defensible framing for T-RO/IJRR is likely: "a diagnosed information-asymmetry (novel, quantified, method-agnostic finding) + a targeted, small-data instantiation of an existing affordance-auxiliary VLA mechanism to fix it (engineering contribution) + an honest accounting of the same failure-diagnosis tradition already established in this project's other work." This is weaker than a from-scratch new mechanism but stronger than incremental foundation-model theater, because the "why" is load-bearing and quantified, not asserted.

## Next Steps

- [x] Stage 1 (real SmolVLA rerun, 2026-07-12) — **KILLED**. See "Stage 1 Result" below.
- [ ] Re-attempt external novelty check once Codex/GPT-5.4 auth is restored — moot for this proposal now, kept for process record.
- [ ] ~~If Stage 1 confirms, proceed to Stage 2-4~~ — not applicable, Stage 1 did not confirm.
- [x] Update `paperA_data/README.md` with a consolidated entry (this route + routes 1-5) so future sessions do not re-propose it.

## Stage 1 Result (2026-07-12): KILLED — toy-MLP signal did not survive real-architecture scale

Reran the exact single-task-vs-multi-task probe-AUC comparison using the
**real SmolVLA action-expert** (not the toy 2-layer MLP), with real rendered
images (regenerated via `scripts/regen_lggsn_scene_images.py`, since the
original data collection never saved raw images — object spawn position is
deterministic from the first two draws of `np.random.default_rng(seed)`,
before any candidate sampling, so exact scene reproduction was possible).
PEFT/LoRA could not be used as originally planned — lerobot's `wrap_with_peft`
refuses to wrap a freshly-initialized (never-pretrained) action-expert
("training from scratch using PEFT is unlikely to yield good results"); used
direct fine-tuning of the expert instead (VLM backbone still frozen via
existing config defaults `freeze_vision_encoder=True`/`train_expert_only=True`)
— a fair substitute for this specific single-task-vs-multi-task comparison,
since it doesn't depend on LoRA specifically.

3 seeds, 1500 rows, 300 steps, batch_size=6, `paperA_data/scripts/stage1_smolvla_affordance_check.py`:

| seed | single-task (pose-only) AUC | multi-task (pose+geom-aux) AUC | advantage |
|---|---|---|---|
| 0 | 0.681 | 0.698 | +0.017 |
| 1 | 0.718 | 0.687 | -0.032 |
| 2 | 0.680 | 0.681 | +0.001 |
| **mean** | | | **-0.005** |

Sign flips across seeds, mean advantage is noise-level. This directly matches
"Failure mode 1" anticipated in this doc's own risk section: *"the toy MLP's
+1.6pp was itself close to noise"* — confirmed. The auxiliary-affordance
mechanism does **not** survive the jump from a 2-layer-MLP proxy to the real
SmolVLA action-expert architecture at this data scale (1500 rows, 300 steps).

**Decision**: do not proceed to Stage 2-4 (settle-subprocess dynamics head,
real teleop fine-tuning, physical pilot) or the cross-object few-shot-transfer
follow-up test that was about to be designed on top of this — there is no
baseline in-distribution effect to test transfer of. This route is killed,
joining routes 1-5. **Reusable assets from this attempt, kept for future
work**: 1400 real rendered scene images
(`grasp_6dof/dataset/lggsn_scene_images/`, paired with existing
`lggsn_candidates_v9.jsonl` rows by `scene_id`) — the first real-image
dataset for this sim benchmark, useful for any future method that needs
actual images rather than precomputed SAM feature vectors.
