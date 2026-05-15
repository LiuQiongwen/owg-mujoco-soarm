# Final Proposal — OWG World-Model Alignment & SO-ARM Real-Robot Transfer

**Version**: V3-FINAL  
**Date**: 2026-05-15  
**Venue target**: ICRA 2027 (or RA-L)

---

## Problem Anchor

LGGSN-based grasp reranking achieves **net=0** across 150 paired trials because its 14-dimensional
geometric features are computed in world frame, causing two systematic failure modes:

1. **H saturation** (Scissors, flat_frac=0.457): When all candidates have H≈0, the model
   outputs near-identical scores → ranking collapses → implicit dist_to_centroid ordering →
   blade-tip bias → regression.

2. **Yaw ambiguity** (PowerDrill, σ_yaw=0.195 rad): World-frame yaw cannot distinguish
   valid from invalid approach orientations for asymmetric objects → noisy cross-episode signal.

All post-hoc gating strategies (σ_H gate, score-delta gate, object whitelist, GC-LGGSN)
fail because they operate on the symptom (score spread), not the root cause (frame-invariance).

The opportunity: since LGGSN features are **purely geometric** (no visual embedding), they
are inherently sim-to-real transferable — the same xyz/rpy computed from real RGBD depth maps
carries the same meaning as in PyBullet.

---

## Thesis (One Sentence)

> **Object-Relative Feature Normalization (ORFN)** — transforming LGGSN's 14 raw features
> into an object-centric coordinate frame using the VLM bounding box centroid and principal
> axis estimate — eliminates both identified failure modes and enables zero-shot sim-to-real
> transfer to SO-ARM without retraining.

---

## Dominant Contribution

**A lightweight, training-free feature transformation (ORFN)** that:
1. Replaces world-frame yaw with *object-frame approach angle* (yaw relative to object
   principal horizontal axis estimated from the candidate point cloud).
2. Replaces raw H with *contact-relative height* = H / object_height_estimate,
   making flat-object features non-degenerate even when H≈0.
3. Adds *gripper-to-object-centroid azimuth* in object frame (complements dist_to_centroid).

No architectural change to LGGSN. No retraining required (features are a deterministic
remap of existing inputs). The BPR-trained model weights are reused as-is.

---

## Rejected Complexity

| Option | Reason rejected |
|--------|-----------------|
| Per-object model heads | Requires 6× more data; cannot generalize to new objects |
| GC-LGGSN context gate | Shown empirically to be non-discriminative (net=0, Phase 4.5) |
| Visual feature embedding (ViT/CLIP) | Breaks geometric transferability; sim-to-real visual gap |
| Diffusion policy for grasp selection | Over-parameterized; destroys interpretability of failure analysis |
| Domain randomization in PyBullet | Addresses wrong level (visual); features are already geometric |
| End-to-end real-robot RL fine-tuning | Requires >200 real trials per object; resource constraint |

---

## Key Claims

| # | Claim | Test |
|---|-------|------|
| C1 | ORFN eliminates Scissors regression in sim | ΔS4-S3 Scissors ≥ −1 over 25 seeds |
| C2 | ORFN preserves CrackerBox gain | ΔS4-S3 CrackerBox ≥ +3 over 25 seeds |
| C3 | ORFN achieves positive overall net (≥+3) without retraining | Aggregate 25-seed paired eval |
| C4 | LGGSN+ORFN features transfer to real SO-ARM without retraining | Real-robot 30-trial matched eval |
| C5 | SO-ARM workspace + ORFN calibration pipeline is reproducible | ≤2h setup from raw camera to first grasp |

---

## Method Detail

### ORFN Feature Transformation (11 new features replace or augment 14 old ones)

```
Input (per candidate):
  raw: x, y, z, roll, pitch, yaw, width, score, dz, dz_lift, need_dz, H
  episode context: centroid_x, centroid_y, z_min, z_max, obj_principal_yaw (estimated)

Output (ORFN 15-dim):
  dx_obj  = x - centroid_x              # object-frame lateral displacement
  dy_obj  = y - centroid_y              # object-frame lateral displacement  
  dz_obj  = (z - z_min)/(z_max-z_min+ε)  # contact-relative height (replaces z_rel)
  yaw_obj = yaw - obj_principal_yaw     # approach angle in object frame (mod π)
  H_rel   = H / (obj_height + ε)       # contact-relative flatness (replaces raw H)
  azimuth = atan2(dy_obj, dx_obj)       # gripper azimuth in object frame
  dist    = sqrt(dx_obj²+dy_obj²)      # dist_to_centroid (kept)
  roll, pitch                           # unchanged
  width, score, dz, dz_lift, need_dz   # unchanged
```

### obj_principal_yaw Estimation
From the episode candidate set: fit a PCA on (x, y) → take angle of first principal component.
Cost: O(N) per episode, N≈5 candidates. No additional sensing required.

### SO-ARM Integration

**Hardware stack:**
- SO-ARM100 (6-DOF) + default parallel gripper
- Intel RealSense D435i (RGB-D, 30 fps)
- Hand-eye calibration: AruCo marker board (OpenCV)

**Software stack:**
```
RealSense D435i → point cloud → GR-ConvNet 2D candidates
                              → ORFN feature extraction
                              → LGGSN scoring (sim-trained weights)
                              → SO-ARM IK (lerobot/so100 URDF + ikpy/pin)
                              → SO-ARM execution
```

**World model alignment pipeline:**
1. **Table-top plane estimation**: RANSAC on depth map → table_z (replaces PyBullet `get_table_top_z`)
2. **Object point cloud**: depth segment by VLM bounding box → xyz in camera frame → robot frame
3. **Feature parity check**: compute ORFN features from real RGBD; verify distribution matches
   PyBullet log statistics (z_min/z_max, dist, H_rel) via histogram overlap ≥ 0.7

No domain randomization, no visual adaptation — ORFN features are geometry-only and
calibration-correctable.

---

## Remaining Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| GR-ConvNet fails on real RGB-D (domain gap) | Medium | Use pre-trained GR-ConvNet with depth-only input; fallback to GraspNet-1B |
| SO-ARM IK infeasible for some grasp poses | Medium | Pre-filter candidates by reachability sphere (workspace AABB) |
| obj_principal_yaw PCA unstable with N<4 candidates | Low | Fallback to world-frame yaw (existing behavior) when N<4 |
| Real RGBD H estimation noisy (depth noise) | Low | Use median H over spatial patch (3×3 depth kernel) |

---

## Final Verdict

**READY** — method is minimal, failure modes are diagnosed, complexity is justified,
sim-to-real path is geometry-grounded, experiment design is feasible in 3-4 weeks.
