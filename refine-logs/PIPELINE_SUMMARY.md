# Pipeline Summary

**Problem**: LGGSN grasp reranker achieves net=0 due to H saturation (Scissors) and yaw
ambiguity (PowerDrill); all sim experiments complete; need to fix failure modes and transfer
to SO-ARM real robot.

**Final Method Thesis**: Object-Relative Feature Normalization (ORFN) — a deterministic
14-dim feature remap using VLM-estimated bounding box centroid and PCA principal axis —
eliminates both failure modes and enables zero-shot sim-to-real transfer to SO-ARM.

**Final Verdict**: READY  
**Date**: 2026-05-15

---

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

---

## Contribution Snapshot
- **Dominant contribution**: ORFN transform (object-frame yaw + contact-relative H_rel + centroid-relative xy)
- **Supporting contribution**: SO-ARM zero-shot transfer validation via geometry-grounded feature alignment
- **Explicitly rejected complexity**: per-object models, visual embedding, GC-LGGSN, score-delta gate, domain randomization

---

## Must-Prove Claims
- C1: ORFN eliminates Scissors regression (sim, N=25)
- C2: ORFN preserves CrackerBox gain (sim, N=25)
- C3: Aggregate net ≥ +3 without retraining (sim, N=25)
- C4: Zero-shot transfer to SO-ARM: net_real ≥ 0 (real, N=10/object)

---

## First Runs to Launch

1. **Exp 2 (PCA stability)** — run immediately, 0.5h:
   ```bash
   conda run -n owg2 python scripts/check_pca_stability.py \
     --log logs/lggsn_live_candidates.jsonl --out logs/pca_stability.json
   ```

2. **Exp 1 main (ORFN sim, 25 seeds)** — after ORFN implementation (~1 day code), ~8h compute:
   ```bash
   LGGSN_ORFN=1 conda run -n owg2 python batch_s3s4.py
   ```

3. **SO-ARM URDF + sim** — parallel to Exp 1:
   ```bash
   # Source: https://github.com/TheRobotStudio/SO-ARM100
   # Load in PyBullet, implement owg_robot/env_soarm.py
   ```

---

## Main Risks

- **Risk**: obj_principal_yaw PCA noisy for thin/flat objects (Scissors)  
  **Mitigation**: Exp 2 characterizes this. If std > 30°: disable yaw_obj for Scissors,
  use world-frame yaw fallback (already in code).

- **Risk**: GR-ConvNet domain gap on real RGBD → different candidate set than sim  
  **Mitigation**: Exp 3 feature alignment check before real-robot eval. If distributions
  diverge > 0.15 JS: add z-score normalization calibration layer.

- **Risk**: SO-ARM IK infeasible for some grasp poses generated in sim-like workspace  
  **Mitigation**: Pre-filter candidates by workspace sphere (r=0.30m) in `env_soarm.py`.

---

## Next Action
- Implement ORFN in `owg_robot/grasp_ranker_lggsn.py` (see EXPERIMENT_PLAN.md Exp 1 code block)
- Write `scripts/check_pca_stability.py` (30 lines)
- Use `/run-experiment` to launch Exp 1 + Exp 2 in parallel
