# Experiment Tracker

**Date**: 2026-05-15  
**Method**: ORFN + SO-ARM Transfer

---

## Status Legend
- `[ ]` Not started
- `[~]` In progress  
- `[✓]` Complete
- `[✗]` Blocked / failed

---

## Week 1 — ORFN Sim Validation

### Implementation
- [ ] Implement ORFN in `owg_robot/grasp_ranker_lggsn.py` (Option B, 14 features)
- [ ] Add `LGGSN_ORFN=1` env var toggle
- [ ] Implement `scripts/check_pca_stability.py`

### Experiment 1 Main (25 seeds × 6 objects)
- [ ] Run: `LGGSN_ORFN=1 conda run -n owg2 python batch_s3s4.py`
- [ ] Log: `logs/batch_s3s4_orfn_25seed.jsonl`
- [ ] C1 check: Scissors net ≥ −1 → `[ ]`
- [ ] C2 check: CrackerBox net ≥ +3 → `[ ]`
- [ ] C3 check: Aggregate net ≥ +3 → `[ ]`

### Experiment 2 — PCA Stability (parallel)
- [ ] Run: `scripts/check_pca_stability.py`
- [ ] C5 check: std(yaw) < 15° for box/bottle/can → `[ ]`

### Ablations (10 seeds)
- [ ] A1: yaw_obj only → `logs/batch_s3s4_orfn_a1.jsonl`
- [ ] A2: H_rel only → `logs/batch_s3s4_orfn_a2.jsonl`
- [ ] A3: Full ORFN-14 (same as main but 10 seeds) → for consistency check

**Week 1 gate decision**: `[ ] PASS / [ ] FAIL`  
**Notes**: 

---

## Week 2 — Hardware Setup + Feature Alignment

### Experiment 4A-B — SO-ARM Sim
- [ ] SO-ARM100 URDF sourced and loaded in PyBullet
- [ ] `owg_robot/env_soarm.py` implemented
- [ ] IK solver (ikpy) integrated
- [ ] Cube pick-up smoke test passing

### Experiment 4C — Hand-Eye Calibration
- [ ] AruCo board printed and mounted
- [ ] Calibration script written: `scripts/calibrate_hand_eye.py`
- [ ] T_cam_to_base computed and saved: `config/T_cam_to_base.npy`
- [ ] Reprojection error < 3mm

### Experiment 3 — Feature Alignment
- [ ] RealSense D435i connected and streaming
- [ ] `scripts/check_feature_alignment.py` implemented
- [ ] H_rel JS divergence ≤ 0.15 (CrackerBox) → `[ ]`
- [ ] H_rel JS divergence ≤ 0.15 (MustardBottle) → `[ ]`  
- [ ] H_rel JS divergence ≤ 0.15 (Scissors) → `[ ]`

**Week 2 gate decision**: `[ ] PASS / [ ] FAIL`  
**Notes**: 

---

## Week 3 — Real-Robot Evaluation

### Experiment 5 — Paired Real-Robot Eval (30 trials)
- [ ] CrackerBox: 10 seeds × S3 + S4 — log: `logs/real_robot_eval.jsonl`
- [ ] MustardBottle: 10 seeds × S3 + S4
- [ ] Scissors: 10 seeds × S3 + S4
- [ ] C4 check: net_real ≥ 0 → `[ ]`

### Experiment 6 — Ablation on Real (15 trials)
- [ ] Scissors × ORFN-yaw only (5 trials)
- [ ] Scissors × ORFN-H only (5 trials)
- [ ] Scissors × Full ORFN (5 trials)

**Week 3 gate decision**: `[ ] PASS / [ ] FAIL`  
**Notes**: 

---

## Week 4 — Analysis + Writing

- [ ] Per-object net comparison table (sim + real)
- [ ] Feature alignment figure (JS divergence heatmap)
- [ ] Update `paper_final.tex` §III.C, §IV.A-C, §V
- [ ] New figures: `fig4_orfn_per_object.pdf`, `fig5_feature_alignment.pdf`
- [ ] Compile paper: `pdflatex paper_final.tex`

---

## Key Results Log (fill as experiments complete)

| Exp | Object | S3 | S4 (ORFN) | net | Status |
|-----|--------|----|-----------|-----|--------|
| Exp1 | Banana | — | — | — | [ ] |
| Exp1 | CrackerBox | — | — | — | [ ] |
| Exp1 | MustardBottle | — | — | — | [ ] |
| Exp1 | PowerDrill | — | — | — | [ ] |
| Exp1 | Scissors | — | — | — | [ ] |
| Exp1 | TomatoSoupCan | — | — | — | [ ] |
| Exp1 | **TOTAL** | — | — | — | [ ] |
| Exp5 | CrackerBox (real) | — | — | — | [ ] |
| Exp5 | MustardBottle (real) | — | — | — | [ ] |
| Exp5 | Scissors (real) | — | — | — | [ ] |
