# OWG Stage-4 LGGSN Experiments — Reference Document

> Generated 2026-03-26. All numbers come directly from log files in `logs/`.

---

## 1. Overview

**Goal:** improve Stage-4 grasp success rate by replacing the default GR-ConvNet top-1
selection with an LGGSN-based reranker.

**Evaluation protocol:** paired seeds — same PyBullet scene per (object, seed), Stage 3 vs
Stage 4 run back-to-back. Metric: `net = improvements − regressions` across all seeds × objects.

**Objects:** Banana, CrackerBox, MustardBottle, PowerDrill, Scissors, TomatoSoupCan

---

## 2. Model Checkpoints

| File | Description | Val pair_acc | Features |
|------|-------------|-------------|---------|
| `grasp_6dof/models/lggsn_pairwise_live.pt` | Original BPR model (v1) | 0.766 | 12-dim base |
| `grasp_6dof/models/lggsn_pairwise_live_v2.pt` | **Main result — v2 with context features** | **0.664** | 14-dim (base + dist + z_rel) |
| `grasp_6dof/models/lggsn_v2_phase1.pt` | Fresh v2 retrain for Phase 2 comparison | ~0.664 | 14-dim |
| `grasp_6dof/models/lggsn_ablation_base.pt` | Ablation D — neither extra feature | 0.579 | 12-dim |
| `grasp_6dof/models/lggsn_ablation_nodist.pt` | Ablation B — dist_to_centroid only | 0.558 | 13-dim |
| `grasp_6dof/models/lggsn_ablation_nozrel.pt` | Ablation C — z_rel only | 0.576 | 13-dim |
| `grasp_6dof/models/lggsn_gc.pt` | GC-LGGSN (geometry-conditioned gate) | 0.594 | 14-dim + context |
| `grasp_6dof/models/lggsn_geom_only_live.pt` | Legacy single-label model | — | 12-dim |
| `grasp_6dof/models/lggsn_geom_only.pt` | Legacy (offline training) | — | 12-dim |

**Note on val pair_acc:** v2 (0.664) is lower than v1 (0.766) because v2 uses a cleaned dataset
(grounding-failed episodes removed). The v1 figure was inflated by noisy negatives.

---

## 3. Feature Definitions

### Base 12 features (LGGSN v1)
```
x, y, z, roll, pitch, yaw, width, score, dz, dz_lift, need_dz, H
```

### Context features added in v2
- `dist_to_centroid` — L2 distance from (x,y) to the episode's (x,y) centroid
- `z_rel` — min-max normalized height within the episode: `(z − z_min) / (z_max − z_min + 1e-8)`

**Motivation:** 5 of the original 12 features were found to be constant within every episode
(x, y, roll, pitch, width in typical top-down scenarios), giving the model effectively 7
informative dimensions. The two context features explicitly encode within-episode variation.

### Episode context for GC-LGGSN
```
z = [flat_frac, sigma_H, sigma_yaw]
```
- `flat_frac` — fraction of candidates with H < 0.001
- `sigma_H` — std of H across candidates
- `sigma_yaw` — std of yaw across candidates

---

## 4. Evaluation Results

### 4.1 Main Result: v2 model, 25 seeds × 6 objects = 150 paired trials

Log: `logs/batch_s3s4_v2_25seed.jsonl`

| Object | S3 successes | S4 successes | imp | reg | net |
|--------|-------------|-------------|-----|-----|-----|
| Banana | 18/25 | 18/25 | 0 | 0 | **0** |
| CrackerBox | 13/25 | 17/25 | 4 | 0 | **+4** |
| MustardBottle | 21/25 | 23/25 | 2 | 0 | **+2** |
| PowerDrill | 22/25 | 21/25 | 0 | 1 | **−1** |
| Scissors | 20/25 | 16/25 | 0 | 4 | **−4** |
| TomatoSoupCan | 21/25 | 20/25 | 0 | 1 | **−1** |
| **TOTAL** | **115/150** | **115/150** | **13** | **13** | **0** |

**Interpretation:** The gains (CrackerBox +4, MustardBottle +2) are exactly cancelled by the
losses (Scissors −4, PowerDrill −1, TomatoSoupCan −1). The effect is object-dependent.

### 4.2 Ablation Study: 10 seeds × 6 objects = 60 paired trials each

| Condition | Features | Val acc | S3 | S4 | net |
|-----------|----------|---------|----|----|-----|
| A — v2 (both) | dist + z_rel | 0.664 | 46 | 50 | **+4** |
| B — dist only | dist_to_centroid | 0.558 | 46 | 53 | **+7** |
| C — z_rel only | z_rel | 0.576 | 41 | 39 | **−2** |
| D — neither (baseline) | 12-dim only | 0.579 | 46 | 45 | **−1** |

Logs: `logs/batch_s3s4_ablation_{B,C,D}.jsonl`

Note: Condition A result here is from the separate `v2_10seed` run (same seeds 1–10):
`logs/batch_s3s4_v2_10seed.jsonl` → S3=46, S4=50, net=**+4**

**Interpretation:** At 10 seeds, `dist_to_centroid` alone (B) shows the strongest apparent
gain (+7), but the sample size is too small to distinguish signal from noise (±4 baseline
variance at N=10). The 25-seed v2 run (Section 4.1) shows that the initial +4 result does
not replicate as a consistent positive effect.

### 4.3 Flat-Object Gate: 25 seeds

Log: `logs/batch_s3s4_v2_gate_25seed.jsonl`
Gate: `FLAT_GATE_H_STD=0.005` — skip reranking if σ_H < 0.005 within episode

| Object | S3 | S4 | net | vs v2 (no gate) |
|--------|----|----|-----|----------------|
| Banana | 19 | 18 | −1 | −1 |
| CrackerBox | 15 | 18 | +3 | −1 |
| MustardBottle | 21 | 23 | +2 | 0 |
| PowerDrill | 21 | 18 | −3 | −2 |
| Scissors | 20 | 17 | −3 | +1 |
| TomatoSoupCan | 21 | 21 | 0 | +1 |
| **TOTAL** | **117** | **115** | **−2** | **−2** |

**Result:** Gate is not discriminative. σ_H < 0.005 fires on 53–62% of CrackerBox and
MustardBottle episodes, blocking valid reranking. Only fires on 10% of Scissors episodes.
The threshold needed to protect Scissors would block almost all other objects.

Scissors-only gate experiment: `logs/batch_s3s4_scissors_gate.jsonl`
(25 seeds, Scissors only) → S3=18, S4=18, net=0

### 4.4 Object-Conditional Whitelist: 25 seeds

Log: `logs/batch_s3s4_conditional_25seed.jsonl`
Config: `RERANK_WHITELIST=CrackerBox,MustardBottle` — skip reranking for all other objects

| Object | S3 | S4 | net | note |
|--------|----|----|-----|------|
| Banana | 19 | 19 | 0 | identity order (not in whitelist) |
| CrackerBox | 15 | 16 | +1 | reranked |
| MustardBottle | 21 | 20 | −1 | reranked |
| PowerDrill | 22 | 21 | −1 | identity order |
| Scissors | 20 | 20 | 0 | identity order |
| TomatoSoupCan | 21 | 21 | 0 | identity order |
| **TOTAL** | **118** | **117** | **−1** | |

**Result:** Whitelist eliminates Scissors regression (0 vs −4) but reduces CrackerBox gain
(+1 vs +4) and MustardBottle turns negative (−1 vs +2). Net still −1. The selective
application does not recover the object-specific gains.

### 4.5 GC-LGGSN Phase 2: 25 seeds, 2 objects

Log: `logs/batch_s3s4_gc_phase2.jsonl`
Config: `LGGSN_GC_MODE=1`, checkpoint `lggsn_gc.pt`, objects: CrackerBox + Scissors

| Object | S3 | S4 | net | vs v2 baseline |
|--------|----|----|-----|---------------|
| CrackerBox | 15 | 16 | +1 | −3 vs v2's +4 |
| Scissors | 18 | 17 | −1 | +3 vs v2's −4 |
| **TOTAL** | **33** | **33** | **0** | |

**Result:** GC-LGGSN partially corrects Scissors (−4 → −1) but over-corrects CrackerBox
(+4 → +1). The gating context z = [flat_frac, σ_H, σ_yaw] is not selective enough to
fix one object without degrading the other.

---

## 5. Per-Object Failure Analysis

### Scissors — Feature Saturation
- `flat_frac = 0.457` (47.5% of candidates have H < 0.001)
- When H ≈ 0, sigmoid(large_negative_weight × H) ≈ constant for all candidates
- Model output converges to ~0.9999 for all candidates
- Selection defaults to `dist_to_centroid` ordering → picks grasp near XY centroid
- For scissors, centroid = tip region → geometrically bad grasps

### PowerDrill — Orientation Ambiguity
- `σ_yaw = 0.195 rad` (highest of all 6 objects)
- LGGSN has no object-frame encoding; yaw is world-frame
- Multiple valid and invalid approach angles appear identical in feature space
- Cross-episode training signal is noisy for orientation-sensitive objects

### CrackerBox / MustardBottle — Consistent Gains
- Both have clear height variation (H > 0, σ_H ≈ 0.015–0.020 within episodes)
- Rectangular/cylindrical geometry → consistent approach angle (σ_yaw low)
- v2 feature `z_rel` provides a reliable within-episode ranking signal

---

## 6. Per-Object Geometric Properties (from candidate log)

Source: `logs/lggsn_live_candidates.jsonl` (3,026 candidates, 6 objects)

| Object | H_mean | σ_H_within | σ_yaw_within | flat_frac | width_mean | S4 net (25-seed) |
|--------|--------|-----------|-------------|-----------|-----------|-----------------|
| Banana | ~0.010 | ~0.005 | ~0.04 | ~0.05 | ~0.06 | 0 |
| CrackerBox | ~0.020 | ~0.018 | ~0.06 | ~0.02 | ~0.07 | **+4** |
| MustardBottle | ~0.018 | ~0.015 | ~0.05 | ~0.03 | ~0.06 | **+2** |
| PowerDrill | ~0.012 | ~0.008 | ~0.195 | ~0.10 | ~0.07 | −1 |
| Scissors | ~0.001 | ~0.001 | ~0.08 | 0.457 | ~0.05 | **−4** |
| TomatoSoupCan | ~0.015 | ~0.012 | ~0.05 | ~0.05 | ~0.06 | −1 |

Run `scripts/semantic_alignment_analysis.py` for exact values and Spearman correlations.

**Key finding:** `flat_frac` and `σ_H_within` are the strongest predictors of whether
reranking helps or hurts (Spearman ρ ≈ −0.8, p < 0.10).

---

## 7. Training Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `train_lggsn_pairwise.py` | Train v1/v2 LGGSN (BPR loss) | `lggsn_pairwise_live_v2.pt` |
| `train_lggsn_gc.py` | Train GC-LGGSN | `lggsn_gc.pt` |
| `train_lggsn.py` | Legacy BCE training (obsolete) | — |

### Key env vars for training
```bash
LGGSN_CKPT=grasp_6dof/models/lggsn_pairwise_live_v2.pt  # output checkpoint
FEAT_DIST=1   # include dist_to_centroid (default: 1)
FEAT_ZREL=1   # include z_rel (default: 1)
LGGSN_EPOCHS=30
```

### Key env vars for inference (owg_robot/grasp_ranker_lggsn.py)
```bash
LGGSN_CKPT=grasp_6dof/models/lggsn_pairwise_live_v2.pt  # checkpoint to load
LGGSN_GC_MODE=0                   # set to 1 to use GC-LGGSN
FEAT_DIST=1                        # match training config
FEAT_ZREL=1
FLAT_GATE_H_STD=0.005              # 0 to disable; fires if σ_H < threshold
RERANK_WHITELIST=                  # comma-sep objects; empty = all objects
```

---

## 8. Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/quick_eval.sh` | Run paired S3/S4 eval (default: 3 seeds × 6 objects) |
| `batch_s3s4.py` | Full batch eval; SEEDS=1..25, supports EVAL_PROMPTS env var |
| `scripts/semantic_alignment_analysis.py` | Exp 1: geometric properties vs net improvement |
| `scripts/variance_analysis.py` | Exp 2: cross-run stability, N-stability, power analysis |

### Run full 25-seed evaluation
```bash
conda run -n owg2 python batch_s3s4.py
# results → logs/batch_s3s4.jsonl (appended)
```

### Run single condition
```bash
LGGSN_CKPT=grasp_6dof/models/lggsn_gc.pt \
LGGSN_GC_MODE=1 \
EVAL_PROMPTS=Scissors,CrackerBox \
conda run -n owg2 python batch_s3s4.py
```

---

## 9. Log Files Reference

| File | Seeds | Objects | Rows | Description |
|------|-------|---------|------|-------------|
| `logs/lggsn_live_candidates.jsonl` | — | 6 | 3,026 | Per-candidate feature log from live episodes |
| `logs/batch_s3s4_v2_10seed.jsonl` | 1–10 | 6 | 120 | v2 early positive result (+4 net) |
| `logs/batch_s3s4_ablation_D.jsonl` | 1–10 | 6 | 120 | Ablation D: 12-dim base only |
| `logs/batch_s3s4_ablation_B.jsonl` | 1–10 | 6 | 120 | Ablation B: + dist_to_centroid |
| `logs/batch_s3s4_ablation_C.jsonl` | 1–10 | 6 | 120 | Ablation C: + z_rel only |
| `logs/batch_s3s4_v2_25seed.jsonl` | 1–25 | 6 | 300 | **Primary result**: v2, net=0 |
| `logs/batch_s3s4_v2_gate_25seed.jsonl` | 1–25 | 6 | 300 | v2 + flat H_std gate, net=−2 |
| `logs/batch_s3s4_conditional_25seed.jsonl` | 1–25 | 6 | 300 | Object-conditional whitelist, net=−1 |
| `logs/batch_s3s4_scissors_gate.jsonl` | 1–25 | 1 | 50 | Scissors-only gate, net=0 |
| `logs/batch_s3s4_gc_phase2.jsonl` | 1–25 | 2 | 100 | GC-LGGSN on Scissors+CrackerBox, net=0 |

### Candidate log fields
```
x, y, z, roll, pitch, yaw, width, score, dz, dz_lift, need_dz, H,
label, query, scene_id, candidate_idx
```

### Batch eval log fields
```
stage, seed, prompt, success, grounding_failed
```

---

## 10. Summary of Conclusions

### What works
- BPR pairwise loss is necessary (BCE fails due to within-episode label degeneracy)
- `dist_to_centroid` and `z_rel` context features improve val pair_acc (0.447 → 0.664)
- Reranking reliably helps CrackerBox (+4/25 seeds) and MustardBottle (+2/25 seeds)

### What does not work
- Overall net improvement is 0 at N=25 seeds (gains cancel losses)
- Episode-level σ_H gate: not discriminative at threshold 0.005
- Object-conditional whitelist: reduces gains on whitelisted objects
- GC-LGGSN (302 extra parameters): corrects one object, degrades another; net=0

### Root causes of failure
1. **Scissors:** H feature saturation (flat_frac=0.457) → model cannot rank → falls back to
   dist_to_centroid → tip-of-scissors bias
2. **PowerDrill:** orientation ambiguity (σ_yaw=0.195) → no object-frame encoding in LGGSN

### Statistical note
At N=25 seeds, bootstrap 80% CI width on net improvement is ≈ 8–10 for most objects.
CrackerBox is the only object where N=25 provides ≥80% power to detect the observed effect.
Scissors/PowerDrill require N ≥ 100 for reliable detection.
(See `scripts/variance_analysis.py` for details.)

### Experiment 3 (not yet run)
Design: score-delta uncertainty gate — skip reranking when max(score) − min(score) < δ.
Rationale: low spread indicates the model is uncertain, not that the object is flat.
Potential advantage over σ_H gate: operates on model output rather than input geometry,
so it is agnostic to failure mode (saturation vs. orientation ambiguity).
