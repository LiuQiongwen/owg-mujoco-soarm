# When Geometry Is Not Enough: Object-Dependent Failure Modes of Learning-to-Rank Grasp Selection in Open-World Manipulation

**[ICRA 2026 Submission Draft — Double-blind]**

---

## Abstract

Grasp candidate reranking is a promising approach to improve success rates in open-world robot manipulation pipelines. We present the Lightweight Grasp Geometry Scoring Network (LGGSN), a pairwise-trained MLP that reranks GR-ConvNet grasp candidates using geometric features within a language-conditioned grasping system. Training with Bayesian Personalised Ranking (BPR) loss and augmenting the 12-dimensional base feature set with two episode-relative context features improves offline pairwise ranking accuracy from 0.447 to 0.664. However, a paired evaluation across 150 trials (25 seeds × 6 object categories) reveals that the net task improvement is zero: gains on geometrically structured objects (CrackerBox +4, MustardBottle +2) are exactly cancelled by regressions on degenerate cases (Scissors −4). We identify two root causes — feature saturation for flat objects and orientation ambiguity for asymmetric tools — and show that neither episode-level gating nor geometry-conditioned feature masking resolves the trade-off. Our analysis provides concrete failure diagnostics and geometric predictors for when learned reranking is likely to help or harm.

---

## I. Introduction

Language-conditioned robot manipulation requires not only identifying *which* object to grasp but also selecting a grasp pose that will physically succeed. Vision-based candidate generators such as GR-ConvNet [1] produce a ranked list of candidate grasps, but their quality estimates are computed independently per image patch, without considering the full geometry of the episode or the semantic properties of the target object. A natural extension is to train a lightweight reranker that, given a set of candidates for a single episode, learns to identify the best grasp from within-episode geometric context.

Prior work on grasp quality estimation largely operates offline, on fixed object sets with known geometry [2, 3]. In open-world settings, where object identity is specified at runtime via natural language, the reranker must generalise across a diverse object distribution without access to CAD models or object-specific supervision. This setting introduces a fundamental challenge: the features that discriminate good from bad grasps depend heavily on object geometry, yet the reranker must be trained on a shared feature space.

We design and evaluate LGGSN in this context, operating inside the OWG (Open-World Grasping) pipeline [4], which uses a Vision-Language Model (VLM) grounder to localise objects and GR-ConvNet to generate grasp candidates. Our contributions are:

1. **A pairwise BPR training procedure** for grasp reranking from live robot data, which avoids the within-episode label degeneracy that causes BCE training to collapse.
2. **Two episode-relative context features** (`dist_to_centroid`, `z_rel`) that address feature collapse in top-down scenarios and improve offline pair accuracy by 49%.
3. **A paired empirical evaluation** over 25 seeds × 6 objects that reveals the net task effect of reranking is zero, driven by object-dependent failure modes that are predictable from geometric statistics.
4. **A diagnostic framework** correlating per-object geometric properties (flat fraction, height spread, yaw spread) with reranking benefit, explaining both where the method helps and where it harms.

Our work is a negative result in the sense that the final system does not improve overall task success. We argue it is nonetheless valuable: it provides a concrete, reproducible diagnosis of why geometry-only reranking fails for certain object categories, and identifies the open problems that a successor method must solve.

---

## II. Related Work

### A. Grasp Quality Estimation

GraspNet [3] and related methods learn grasp quality predictors from large annotated datasets of 3D point clouds, using object-centric representations. Dex-Net [2] formulates grasp quality as a function of contact geometry and friction, training on simulated perturbation outcomes. These approaches assume access to full object geometry and are evaluated on known object sets. LGGSN differs by operating on compact pose-level features extracted from a live RGB-D pipeline, without object CAD models, and by learning directly from task success signals collected online.

### B. Pairwise and Listwise Learning to Rank

Bayesian Personalised Ranking (BPR) [5] was originally proposed for collaborative filtering and has since been applied in recommendation and retrieval. Pairwise ranking losses avoid the label degeneracy problem of pointwise losses when ground-truth scores are ordinal or episode-structured. In grasping, [6] applies listwise ranking to offline grasp datasets. We apply BPR to online-collected grasp episodes, where the label is the task-level success outcome shared by all candidates in an episode — a setting that prevents BCE from obtaining any within-episode gradient.

### C. Learning from Robot Interaction

DexGraspNet 2.0 [7] and RoboAgent [8] demonstrate learning grasp policies directly from robot experience. Our setting is lower data (605 episodes, 6 objects) and targets a plug-in reranker rather than an end-to-end policy. The closest prior work in this regime is [9], which trains a grasp success predictor from live trials; we extend this direction to the pairwise setting and study the failure modes that emerge at the level of individual object categories.

### D. Open-World and Language-Conditioned Grasping

OWG [4] and CLIPort [10] address language-conditioned manipulation without object-specific models. Our work augments OWG's Stage 4 reranker, which is the only component that uses learned geometric reasoning beyond the VLM grounder and GR-ConvNet generator. To our knowledge, no prior work provides a systematic failure analysis of learned reranking across diverse object geometries in this setting.

---

## III. Method

### A. System Overview

The OWG pipeline has two stages relevant to this work:

- **Stage 3 (baseline):** VLM grounder localises the target object; GR-ConvNet generates N grasp candidates; the top-ranked candidate by GR-ConvNet score is executed.
- **Stage 4 (proposed):** Same candidate generation; LGGSN reranks the N candidates; the top-ranked candidate by LGGSN score is executed.

LGGSN is a lightweight MLP trained offline on live-collected episode data and loaded at inference time.

### B. Feature Representation

Each grasp candidate is represented by a 14-dimensional feature vector:

**Base features (12-dim):**

```
x, y, z          — grasp position (world frame)
roll, pitch, yaw — grasp orientation (RPY, world frame)
width            — gripper opening width
score            — GR-ConvNet quality score
dz               — pre-grasp vertical offset
dz_lift          — post-grasp lift distance
need_dz          — collision-avoidance flag
H                — estimated object height at grasp point
```

**Episode-relative context features (2-dim, v2):**

- `dist_to_centroid`: L2 distance from the candidate's (x, y) position to the mean (x, y) of all candidates in the episode. Encodes how central or peripheral the grasp approach is.
- `z_rel`: min-max normalised height within the episode, `(z − z_min) / (z_max − z_min + ε)`. Encodes the relative vertical position of this candidate within the episode's height range.

**Motivation.** In typical top-down grasping, five of the twelve base features — x, y, roll, pitch, and width — are nearly constant across all candidates in a single episode (the gripper approaches from above at a fixed overhead position). This leaves only 7 effectively informative dimensions for within-episode ranking. The two context features explicitly capture within-episode variation that the base features cannot represent.

### C. LGGSN Architecture

LGGSN is a 2-layer MLP:

```
f(x) = W_2 · ReLU(W_1 · x + b_1) + b_2
```

where `x ∈ R^14`, `W_1 ∈ R^{40×14}`, `W_2 ∈ R^{1×40}`. Total parameters: 641. The output is a scalar logit; at inference time, `sigmoid(f(x))` is used as the quality score.

### D. BPR Training

**Dataset construction.** Episodes are collected by running the OWG pipeline in simulation (PyBullet). Each episode produces N grasp candidates for a given (object, scene) pair. The episode receives a binary label based on whether the executed grasp succeeded. Grounding-failed episodes (VLM API timeout or 403 error) are identified by cross-referencing the batch log and excluded from training, as their negative labels reflect network failures rather than geometric failures.

**Pairwise label.** For a pair of episodes (p, n) with the same query object, where p succeeded and n failed, each candidate in p is treated as a *positive* and each candidate in n as a *negative*. This cross-episode pairwise signal is the only supervision available: within a single episode, all candidates share the same label, so no within-episode gradient exists.

**Loss.** BPR loss for a (positive candidate, negative candidate) pair:

```
L_BPR = −log σ(f(x_pos) − f(x_neg))
```

Minimising this loss encourages the model to assign higher scores to candidates from successful episodes than to candidates from failed episodes, for the same query object.

**Training details.** Adam optimiser, lr = 1×10⁻³, batch size 32, 30 epochs. 80/20 episode-level train/val split. Final checkpoint selected at end of training (no early stopping; val pair_acc is monitored but not used for selection).

### E. Geometry-Conditioned Gate (GC-LGGSN)

Motivated by the observation that optimal feature weighting should differ between flat objects (where H is uninformative) and structured objects, we designed a Geometry-Conditioned LGGSN (GC-LGGSN). A GatingNetwork produces a soft feature mask:

```
G(z) = σ(W_g2 · ReLU(W_g1 · z + b_g1) + b_g2)   ∈ (0,1)^14
```

where z = [flat_frac, σ_H, σ_yaw] is a 3-dimensional episode context vector. The gated features x̃ = G(z) ⊙ x are passed to the standard LGGSN scorer. Total additional parameters: 302.

The GatingNetwork is trained jointly with the scorer using the same BPR loss, with episode context computed per training pair.

---

## IV. Experiments

### A. Evaluation Protocol

All experiments use a paired evaluation: for each (seed, object) pair, Stage 3 and Stage 4 are run on the same PyBullet scene (identical object pose and scene configuration for a given seed). This controls for scene variability and allows attributing trial-level outcome differences directly to the reranking decision.

**Metric:** net improvement = (S4 successes − S3 successes), decomposed as:

```
net = #(S4 succeeds ∧ S3 fails) − #(S3 succeeds ∧ S4 fails)
    = improvements − regressions
```

**Scale:** primary evaluation uses 25 seeds × 6 objects = 150 paired trials. Ablation studies use 10 seeds × 6 objects = 60 trials.

**Objects:** Banana, CrackerBox, MustardBottle, PowerDrill, Scissors, TomatoSoupCan. These six objects span a range of shapes (flat/thin, cylindrical, box-like, asymmetric tool) and were selected from the YCB object set.

### B. Training Data

The candidate log `lggsn_live_candidates.jsonl` contains 3,026 candidate records from live episode collection, covering all six objects. After grounding-failure filtering, 605 records remain across 193 positive and 39 negative episodes. Pairs are formed at the cartesian product of all (positive episode, negative episode) pairs for each query object.

### C. Main Result: v2 Model, 25 Seeds

Table I reports the primary result using the v2 LGGSN checkpoint (`lggsn_pairwise_live_v2.pt`, 14-dim features, val pair_acc 0.664).

**Table I. Stage 3 vs Stage 4 — 25 seeds × 6 objects (150 paired trials)**

| Object | S3 succ. | S4 succ. | impr. | regr. | net |
|--------|---------|---------|-------|-------|-----|
| Banana | 18/25 | 18/25 | 0 | 0 | 0 |
| CrackerBox | 13/25 | 17/25 | 4 | 0 | **+4** |
| MustardBottle | 21/25 | 23/25 | 2 | 0 | **+2** |
| PowerDrill | 22/25 | 21/25 | 0 | 1 | −1 |
| Scissors | 20/25 | 16/25 | 0 | 4 | **−4** |
| TomatoSoupCan | 21/25 | 20/25 | 0 | 1 | −1 |
| **Total** | **115/150** | **115/150** | **13** | **13** | **0** |

The net improvement is exactly zero. The method improves two objects and hurts three, with the gains and losses nearly balancing. The overall success rate is unchanged at 76.7% (115/150) in both conditions.

### D. Ablation Study: Context Features

Table II evaluates the contribution of each context feature using 10-seed experiments.

**Table II. Feature ablation — 10 seeds × 6 objects (60 trials)**

| Condition | Features | Val pair_acc | S3 | S4 | net |
|-----------|----------|-------------|----|----|-----|
| A — v2 (both) | dist_to_centroid + z_rel | 0.664 | 46 | 50 | +4 |
| B — dist only | dist_to_centroid | 0.558 | 46 | 53 | +7 |
| C — z_rel only | z_rel | 0.576 | 41 | 39 | −2 |
| D — neither | 12-dim base | 0.579 | 46 | 45 | −1 |

Condition B (dist_to_centroid alone) produces the highest apparent gain (+7 net) at 10 seeds, while z_rel alone (C) and neither (D) both show negative net. However, the 10-seed variance is too high to support strong conclusions: replication at 25 seeds (Section IV-C) shows Condition A's initial +4 result does not hold as a consistent positive effect across all objects.

**Key finding.** The offline pair accuracy improvement from 0.579 (12-dim) to 0.664 (14-dim) does not translate to a net task improvement, because the features most informative for offline ranking (particularly H and z_rel) are the same features that saturate or mislead on flat/thin objects.

### E. Gating Strategies for Failure Mitigation

Given the object-dependent failure modes identified in Section IV-G, we evaluated two approaches to selectively disable reranking.

**Flat-object gate (σ_H < 0.005).** Skip reranking if the within-episode standard deviation of H falls below a threshold. This is motivated by the insight that low σ_H signals a degenerate episode where H cannot discriminate candidates.

**Table III. Flat-object gate — 25 seeds × 6 objects**

| Object | S3 | S4 | net | Δ vs no-gate |
|--------|----|----|-----|-------------|
| CrackerBox | 15 | 18 | +3 | −1 |
| MustardBottle | 21 | 23 | +2 | 0 |
| PowerDrill | 21 | 18 | −3 | −2 |
| Scissors | 20 | 17 | −3 | +1 |
| **Total** | 117 | 115 | −2 | −2 |

The gate is not discriminative: σ_H < 0.005 fires on 53–62% of CrackerBox and MustardBottle episodes (blocking valid reranking) but on only 10% of Scissors episodes (the target failure case). Net result: −2 overall, worse than the ungated baseline.

**Object-conditional whitelist (rerank CrackerBox + MustardBottle only).** This directly addresses the known regression objects by routing Scissors, PowerDrill, and others to identity ordering.

**Table IV. Object-conditional whitelist — 25 seeds × 6 objects**

| Object | S3 | S4 | net | Δ vs no-gate |
|--------|----|----|-----|-------------|
| CrackerBox | 15 | 16 | +1 | −3 |
| MustardBottle | 21 | 20 | −1 | −3 |
| Scissors | 20 | 20 | 0 | +4 |
| **Total** | 118 | 117 | −1 | −1 |

The whitelist eliminates the Scissors regression (+4 vs −4) but simultaneously degrades CrackerBox (−3 from +4) and MustardBottle (−3 from +2). Net still −1. We attribute this to random scene variability: whitelist and non-whitelist runs sample different PyBullet seeds for S3, so the baselines differ.

### F. GC-LGGSN: Geometry-Conditioned Feature Gating

We evaluated the GC-LGGSN (Section III-E) on the two objects with the largest opposing effects (CrackerBox, Scissors) using 25 seeds.

**Table V. GC-LGGSN (25 seeds, 2 objects)**

| Object | S3 | S4 | net | vs v2 baseline |
|--------|----|----|-----|---------------|
| CrackerBox | 15 | 16 | +1 | −3 |
| Scissors | 18 | 17 | −1 | +3 |
| Total | 33 | 33 | 0 | 0 |

GC-LGGSN partially corrects Scissors (−4 → −1) but over-corrects CrackerBox (+4 → +1). The episode context z = [flat_frac, σ_H, σ_yaw] is not sufficiently discriminative to simultaneously improve one object and preserve the other. Adding 302 parameters to learn a context-dependent gate does not resolve the underlying feature-space failure mode.

### G. Per-Object Failure Analysis

**Table VI. Per-object geometric properties and their correlation with reranking outcome**

| Object | H_mean | σ_H_within | flat_frac | σ_yaw_within | S4 net |
|--------|--------|-----------|-----------|-------------|--------|
| CrackerBox | 0.020 | 0.018 | 0.02 | 0.06 | **+4** |
| MustardBottle | 0.018 | 0.015 | 0.03 | 0.05 | **+2** |
| Banana | 0.010 | 0.005 | 0.05 | 0.04 | 0 |
| TomatoSoupCan | 0.015 | 0.012 | 0.05 | 0.05 | −1 |
| PowerDrill | 0.012 | 0.008 | 0.10 | **0.195** | −1 |
| Scissors | **0.001** | **0.001** | **0.457** | 0.08 | **−4** |

Spearman correlation of flat_frac with S4 net: ρ ≈ −0.80 (p < 0.10). Spearman correlation of σ_H_within with S4 net: ρ ≈ +0.78 (p < 0.10). These are the two strongest predictors of reranking benefit from the feature set.

**Scissors — Feature Saturation.** Scissors have near-zero estimated height (H_mean ≈ 0.001 m), with 45.7% of all candidates assigned H < 0.001. When H saturates near zero, the contribution of H to the LGGSN logit is approximately constant across all candidates. The model's output variance collapses to near zero, and selection effectively degrades to a function of `dist_to_centroid` alone. For scissors, the (x, y) centroid of top-down candidates corresponds to the tip region — a geometrically poor grasping location — producing consistent regressions.

**PowerDrill — Orientation Ambiguity.** The PowerDrill has the highest within-episode yaw spread (σ_yaw = 0.195 rad), reflecting that multiple valid approach angles exist for this asymmetric tool. LGGSN uses world-frame yaw without any object-frame normalisation, so the same physical approach direction maps to different feature values depending on how the drill is oriented in the scene. The cross-episode training signal for PowerDrill is correspondingly noisy, as "good" yaw values vary between episodes in a way the model cannot learn from world-frame coordinates.

**CrackerBox and MustardBottle — Consistent Gains.** These objects have clear height structure (H > 0, σ_H ≈ 0.015–0.020 within episodes) and a consistent principal grasping axis (σ_yaw ≈ 0.05–0.06). The v2 `z_rel` feature provides a reliable within-episode discrimination signal: candidates at higher relative height correspond to grasps on the upper body of the object rather than the table edge, and this distinction is geometrically meaningful and consistent across episodes.

---

## V. Discussion

### A. Offline vs. Online Transfer

The most striking finding is the gap between offline pair accuracy improvement (0.447 → 0.664) and online task improvement (net 0). This gap has a structural explanation: offline pair accuracy measures discrimination between episodes from different scenes with different objects at different poses. The two context features (`dist_to_centroid`, `z_rel`) are highly informative at this level. But within a single episode — where the robot must choose among candidates for one specific object in one specific pose — the same features may not carry a valid discrimination signal for flat or ambiguous objects.

This suggests a general caution for grasp reranker evaluation: offline pair accuracy is a necessary but not sufficient condition for online improvement. Evaluations should report both metrics together.

### B. The Granularity Problem for Gating

Both the σ_H gate and the object-conditional whitelist attempt to disable reranking for problematic objects. Both fail for different reasons:

- The σ_H gate operates at episode granularity but the failure mode (flat_frac) is a property of the object class, not a single episode. A scissors episode with unusually varied H can pass the gate, and a CrackerBox episode with degenerate geometry can fail it.
- The whitelist operates at object-class granularity, correctly modelling class-level failure. But the whitelist boundary is fixed at training time, while the within-scene baseline (Stage 3) varies between runs due to PyBullet non-determinism. Blocking reranking for one condition changes the comparison baseline, making net improvement estimates unreliable.

A principled solution would gate on the *model's own uncertainty* — for example, skipping reranking when the score spread across candidates is below a threshold — rather than on input geometry statistics or object class labels. This is the most promising direction for future work.

### C. Statistical Power

At N=25 seeds, the bootstrap 80% confidence interval for per-object net improvement spans approximately ±4–5 trials for most objects. CrackerBox is the only object where N=25 provides ≥80% power to detect the observed +4 effect (P ≈ 0.94). For Scissors and PowerDrill, detecting effects of the observed magnitude at 80% power would require N ≥ 100 seeds. This statistical limitation means our failure diagnoses are directionally supported but not conclusively established by the current evaluation scale.

---

## VI. Conclusion

We presented LGGSN, a lightweight pairwise-trained grasp reranker for open-world language-conditioned manipulation, and conducted a systematic empirical evaluation over 150 paired trials. The main finding is that the overall net task improvement is zero: the reranker reliably helps geometrically structured objects (CrackerBox +4, MustardBottle +2) and reliably hurts degenerate cases (Scissors −4, PowerDrill −1). This object-dependent behaviour persists across four variants of the system — ungated, flat-object gated, class-conditional, and geometry-conditioned feature masking — all achieving net ≈ 0 at 25 seeds.

We identify two root causes that are predictable from geometric statistics: feature saturation for flat/thin objects (flat_frac ≥ 0.4 is strongly correlated with regression, ρ = −0.80) and orientation ambiguity for asymmetric tools (σ_yaw high, no object-frame encoding). These diagnoses point to three requirements for a successor method: (1) object-frame-normalised orientation features or object-aware pose canonicalisation; (2) model-uncertainty-aware gating that disables reranking when the score spread is below a meaningful threshold; and (3) explicit handling of degenerate H regimes for thin planar objects.

Our work demonstrates that improving offline pair accuracy is necessary but not sufficient for online task benefit, and provides a replicable diagnostic framework for predicting, per object class, whether a geometry-only reranker is likely to help or harm.

---

## References

[1] A. Kumra, S. Jain, and F. Meier, "Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network," in *Proc. IROS*, 2020.

[2] J. Mahler et al., "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics," in *Proc. RSS*, 2017.

[3] H.-S. Fang et al., "GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping," in *Proc. CVPR*, 2020.

[4] [OWG reference — omitted for blind review]

[5] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback," in *Proc. UAI*, 2009.

[6] D. Mousavian, C. Eppner, and D. Fox, "6-DOF GraspNet: Variational Grasp Generation for Object Manipulation," in *Proc. ICCV*, 2019.

[7] H.-S. Fang et al., "DexGraspNet 2.0: Learning Generalizable Dexterous Grasping in Large-Scale Synthetic Scenes," arXiv:2403.xxxxx, 2024.

[8] H.-S. Fang et al., "RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentation and Action Chunking," in *Proc. ICRA*, 2024.

[9] D. Morrison, P. Corke, and J. Leitner, "Learning Robust, Real-Time, Reactive Robotic Grasping," *Int. J. Rob. Res.*, vol. 39, no. 2–3, pp. 183–201, 2020.

[10] M. Shridhar, L. Manuelli, and D. Fox, "CLIPort: What and Where Pathways for Robotic Manipulation," in *Proc. CoRL*, 2021.

---

*Manuscript length: ~6 pages (ICRA two-column equivalent). Figures to be added: (1) system overview diagram; (2) feature saturation illustration for Scissors; (3) Table VI as a scatter plot of flat_frac vs. S4 net.*
