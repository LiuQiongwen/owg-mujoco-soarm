# Experiment Plan — ORFN + SO-ARM Transfer

**Method**: Object-Relative Feature Normalization (ORFN) for LGGSN  
**Proposal**: `refine-logs/FINAL_PROPOSAL.md`  
**Date**: 2026-05-15  
**Timeline**: 4 weeks

---

## Claims to Validate

| # | Claim | Type | Priority |
|---|-------|------|----------|
| C1 | ORFN eliminates Scissors regression (net ≥ −1, 25 seeds) | Main result | Must |
| C2 | ORFN preserves CrackerBox gain (net ≥ +3, 25 seeds) | Main result | Must |
| C3 | ORFN achieves aggregate net ≥ +3 without retraining | Main result | Must |
| C4 | LGGSN+ORFN transfers to SO-ARM zero-shot (real-robot 30 trials) | Transfer claim | Must |
| C5 | PCA principal axis estimates are stable (std < 15°, N≥4) | Mechanism check | Should |
| C6 | H_rel distribution matches sim-to-real (histogram overlap ≥ 0.7) | World model alignment | Should |

---

## Experiment 1 — ORFN Sim Validation (Week 1)

**Goal**: Validate C1, C2, C3 in PyBullet before touching real hardware.

**Setup**:
- Implement ORFN transform in `owg_robot/grasp_ranker_lggsn.py`
- Keep same BPR checkpoint (`lggsn_pairwise_live.pt`)
- Add env var `LGGSN_ORFN=1` to enable (default off for backward compat)

**ORFN implementation checklist:**
```python
# In grasp_ranker_lggsn.py, before scoring:
if ORFN_ENABLED:
    cx, cy = candidates[:, 0].mean(), candidates[:, 1].mean()  # centroid
    z_min, z_max = candidates[:, 2].min(), candidates[:, 2].max()
    obj_h = z_max - z_min + 1e-8

    # PCA for principal axis (on xy)
    xy = candidates[:, :2] - [cx, cy]
    if len(xy) >= 4:
        U, S, Vt = np.linalg.svd(xy, full_matrices=False)
        obj_principal_yaw = np.arctan2(Vt[0, 1], Vt[0, 0])
    else:
        obj_principal_yaw = 0.0  # fallback

    features[:, 0] -= cx           # dx_obj
    features[:, 1] -= cy           # dy_obj
    features[:, 2] = (candidates[:, 2] - z_min) / obj_h  # dz_obj (=z_rel)
    features[:, 5] = (candidates[:, 5] - obj_principal_yaw + np.pi) % (2*np.pi) - np.pi  # yaw_obj
    features[:, 11] /= (obj_h + 1e-8)  # H_rel (index 11 = H feature)
    # add azimuth as 15th feature
    azimuth = np.arctan2(features[:, 1], features[:, 0])
    features = np.concatenate([features, azimuth[:, None]], axis=1)
```

**Note**: Model input dimension increases from 14 → 15. Since LGGSN is an MLP, the last
weight matrix must be expanded. Two options:
- **Option A (preferred)**: Initialize new 15th weight column as zeros → ORFN-15 = ORFN-14 + azimuth ignored. Verify this matches ORFN-14 output. Then fine-tune ORFN-15 with BPR on existing 605-row dataset (10 epochs).
- **Option B (fast baseline)**: Use 14 ORFN features (drop azimuth), no retraining needed.

**Run protocol** (same as existing batch_s3s4.py):
```bash
LGGSN_ORFN=1 SEEDS=1..25 conda run -n owg2 python batch_s3s4.py
# → logs/batch_s3s4_orfn_25seed.jsonl
```

**Success criterion (decision gate)**:
- C1: Scissors net ≥ −1 (vs current −4)
- C2: CrackerBox net ≥ +3 (vs current +4, allow ≤1 degradation)
- C3: Aggregate net ≥ +3

**If gate fails**: Check per-feature distribution shifts (log feature stats before/after ORFN).
Primary suspect: if obj_principal_yaw is noisy → zero out yaw_obj contribution (set col 5 back
to world-frame yaw) and rerun as ablation.

**Ablations in Exp 1** (run in parallel, 10 seeds each):
| ID | Config | Purpose |
|----|--------|---------|
| A1 | ORFN-yaw only (obj_principal_yaw, others unchanged) | Isolate yaw fix effect |
| A2 | ORFN-H only (H_rel, others unchanged) | Isolate H saturation fix |
| A3 | Full ORFN-14 | Check combined effect |
| A4 | ORFN-14 + azimuth (retrained 10 epochs) | Check if azimuth adds value |

---

## Experiment 2 — PCA Stability Check (Week 1, parallel)

**Goal**: Validate C5 (mechanism check for obj_principal_yaw).

**Setup**: Extract candidate sets from existing `logs/lggsn_live_candidates.jsonl` (3,026 candidates, 6 objects). Compute obj_principal_yaw per episode. Report:
- Mean, std of PCA angle per object category
- % episodes with N < 4 candidates (fallback rate)
- Within-object angle consistency (do episodes with similar layouts agree?)

```bash
conda run -n owg2 python scripts/check_pca_stability.py \
  --log logs/lggsn_live_candidates.jsonl \
  --out logs/pca_stability.json
```

**Write `scripts/check_pca_stability.py`** (~30 lines): load JSONL, group by (query, scene_id),
compute PCA angle per episode, compute across-episode std per object, plot histogram.

**Pass criterion**: std(obj_principal_yaw) < 15° for CrackerBox, MustardBottle, TomatoSoupCan.
Scissors may have high std (thin object, many orientations) — acceptable, fallback triggers.

---

## Experiment 3 — World Model Alignment Check (Week 2)

**Goal**: Validate C6 (feature distribution sim-to-real overlap before real-robot eval).

**Setup**: Place 3 YCB objects (CrackerBox, MustardBottle, Scissors) on real table.
Run GR-ConvNet on real RealSense D435i depth. Extract ORFN features.
Compare feature distributions to PyBullet log using Jensen-Shannon divergence per feature.

**Metrics**:
- H_rel distribution: JS divergence < 0.15 per object
- dist_to_centroid: JS divergence < 0.15
- yaw_obj: JS divergence < 0.20 (yaw is inherently more variable)

**This is a go/no-go gate for real-robot experiments.**

If feature distributions diverge significantly:
1. Check table-top RANSAC plane estimation (most likely source of z-drift)
2. Apply per-feature z-score normalization (μ, σ from sim log) as a calibration layer
3. Re-check JS divergence after normalization

**Script**: `scripts/check_feature_alignment.py`

---

## Experiment 4 — SO-ARM Hardware Setup (Week 2)

**Goal**: Working SO-ARM + RealSense + LGGSN+ORFN pipeline. No grasp evaluation yet.

**Checklist** (sequential — hardware dependencies):

### 4A: SO-ARM URDF + PyBullet Sim
- [ ] Download SO-ARM100 URDF from lerobot repo (`lerobot/configs/robot/so100.yaml` or community URDF)
- [ ] Load in PyBullet, verify joint names, DOF, workspace envelope
- [ ] Implement `owg_robot/env_soarm.py` mirroring `env_panda.py` structure:
  - `arm_joints`: SO-ARM100 6 joints
  - `grasp_with_soarm()`: IK via ikpy or pin, descent + close + lift
  - `reset_robot()`: home pose
- [ ] Smoke-test: pick up a cube in sim

### 4B: IK Solver
- SO-ARM100 has 6 DOF → analytical IK not trivial
- Use `ikpy` (pure Python, no ROS dependency) with SO-ARM URDF
- Pre-screen grasp candidates for reachability: reject if IK fails or joint limits violated
- Reachability filter: workspace sphere centered at SO-ARM base, r=0.30m (typical SO-ARM reach)

### 4C: Hand-Eye Calibration
- Mount RealSense D435i on a fixed stand (eye-to-hand configuration)
- AruCo board (5×5, 20mm markers): print and fix to table
- Collect 15 robot poses + AruCo detections → solve PnP → T_cam_to_base
- Script: `scripts/calibrate_hand_eye.py` (OpenCV, ~80 lines)
- Save: `config/T_cam_to_base.npy`

### 4D: Pipeline Integration Test
- Run full pipeline (RealSense → GR-ConvNet → ORFN → LGGSN → IK → SO-ARM) with no object
- Verify coordinate transforms are consistent (grasp pose in robot frame lands within workspace)

---

## Experiment 5 — Real-Robot Eval (Week 3)

**Goal**: Validate C4 (zero-shot transfer to SO-ARM).

**Protocol**:
- Objects: CrackerBox, MustardBottle, Scissors (the 3 that differ most in LGGSN performance)
- Seeds: 10 per object = 30 trials total
- Baseline: SO-ARM + GR-ConvNet top-1 (Stage 3 equivalent, LGGSN disabled)
- Proposed: SO-ARM + LGGSN+ORFN (sim-trained, zero-shot)
- Paired: same object placement per seed (place once, run S3 then S4)

**Success criterion**:
- C4: LGGSN+ORFN success rate ≥ GR-ConvNet baseline, with net ≥ 0
- Minimum meaningful result: Scissors regression eliminated (net_Scissors ≥ −1 on real)
- Strong result: CrackerBox still shows gain on real hardware

**Recording**: Log per-trial in `logs/real_robot_eval.jsonl` with fields:
```json
{"seed": 1, "object": "CrackerBox", "stage": 3, "success": true, "orfn": false}
{"seed": 1, "object": "CrackerBox", "stage": 4, "success": true, "orfn": true}
```

**Statistics**: McNemar's exact test (N=10 per object), bootstrap CI. Accept p < 0.20
as "trend" given small N; require p < 0.10 for any strong claim.

---

## Experiment 6 — Ablation Verification (Week 3-4)

**Goal**: Confirm per-component contribution matches sim results.

| Ablation | Expected (from Exp 1) | Real-robot confirmation |
|----------|----------------------|------------------------|
| ORFN-yaw only | Scissors regression reduced | 5 trials Scissors |
| ORFN-H only | No effect on Scissors | 5 trials Scissors |
| Full ORFN | Scissors best | 5 trials Scissors |

Total: 15 additional real-robot trials.

---

## Run Order and Timeline

```
Week 1:
  Day 1-2: Implement ORFN in grasp_ranker_lggsn.py (Option B first)
  Day 2:   Run Exp 2 (PCA stability, ~1h compute)
  Day 3-4: Run Exp 1 main (25-seed batch, ~8h compute)
  Day 5:   Run Exp 1 ablations A1-A3 (10-seed each, ~6h)
  
  Decision gate after Week 1:
  → If C1+C2+C3 pass: proceed to hardware
  → If only C1 or C2 fail: check ablations, fix, rerun 25-seed
  → If both fail: revisit PCA stability → possible fallback to ORFN without yaw_obj

Week 2:
  Day 1-2: SO-ARM URDF + sim (Exp 4A, 4B)
  Day 3:   Hand-eye calibration (Exp 4C) — requires physical setup
  Day 4:   World model alignment check (Exp 3) — requires RealSense
  Day 5:   Full pipeline integration test (Exp 4D)

Week 3:
  Day 1-3: Real-robot eval Exp 5 (30 trials, ~6h active robot time)
  Day 4-5: Ablation confirmation Exp 6 (15 trials)

Week 4:
  Analysis, paper writing (extend paper_final.tex §IV-V)
  Figure generation (extend scripts/make_figures.py)
```

---

## Budget Estimate

| Experiment | Compute | Robot Time |
|-----------|---------|-----------|
| Exp 1 (sim 25-seed) | ~8h GPU/CPU | 0 |
| Exp 1 ablations | ~6h | 0 |
| Exp 2 (PCA check) | ~0.5h | 0 |
| Exp 3 (feature alignment) | ~1h | 0.5h setup |
| Exp 4 (SO-ARM setup) | 2h code | 2h hardware |
| Exp 5 (real eval 30 trials) | 0 | ~6h |
| Exp 6 (ablation 15 trials) | 0 | ~3h |
| **Total** | **~17h compute** | **~11h robot** |

---

## Decision Gates

```
After Week 1: Exp 1 gate
  C1 ✓ AND C2 ✓ AND C3 ✓ → proceed to hardware (Week 2+)
  C3 fails (net < +3): check if net ≥ +1 with p<0.20 → proceed with "trend" framing
  C1 fails (Scissors still −3 or worse): debug yaw_obj stability → ablation A1

After Week 2: Feature alignment gate (Exp 3)  
  JS divergence ≤ 0.15 all features → proceed to real eval
  JS divergence > 0.15 → apply z-score calibration layer → recheck → proceed

After Week 3: Real-robot gate (Exp 5)
  net_real ≥ 0 → strong transfer claim (C4)
  net_real = −1 to 0 → "feature-level alignment verified, execution gap remains" 
  net_real < −2 → investigate IK / GR-ConvNet domain gap (separate issue)
```

---

## Novelty Isolation

The ORFN transform is new; the BPR training and LGGSN architecture are unchanged.
To isolate novelty:
- Exp 1 baseline = same checkpoint with ORFN disabled (already done, 25-seed v2 result)
- Ablations A1, A2 separate yaw_obj fix from H_rel fix
- This means any improvement from ORFN is attributable to the specific geometric normalization
  and not to retraining or architecture change

---

## Simplicity / Deletion Check

Can the paper be told without ORFN, by just filtering problematic objects?
→ Already tested (conditional whitelist, §4.4 EXPERIMENTS.md): net = −1. Does not work.

Can the paper be told without real-robot validation?
→ Sim-only is the existing ICRA paper draft (negative result). Real-robot adds the
  transfer claim, which is the main new contribution. Must include.

Is any ORFN component deletable?
→ yaw_obj: needed for C1 (Scissors fix)
→ H_rel: needed for C2 (prevents H saturation from reappearing)
→ azimuth: optional (C6 ablation will determine)

---

## Paper Extension Plan

Extend `paper_final.tex` from the existing ICRA draft:
- §III.C: Add ORFN method (replaces §III.B raw feature description)
- §IV.A: ORFN sim results (Exp 1) — new Table II
- §IV.B: Feature alignment (Exp 3) — one paragraph + histogram figure
- §IV.C: Real-robot results (Exp 5+6) — new Table III
- §V: Update Discussion — ORFN fixes failure modes; sim-to-real geometry-grounded transfer

New figures needed:
- `fig4_orfn_per_object.pdf`: per-object net comparison (baseline vs ORFN)
- `fig5_feature_alignment.pdf`: JS divergence heatmap sim vs real
- `fig6_soarm_setup.pdf`: hardware photo (optional)
