---
created: '2026-03-25T05:42:17+00:00'
evidence:
- stage-02/problem_tree.md
id: problem_decompose-rc-20260325-054058-b1c38b
run_id: rc-20260325-054058-b1c38b
stage: 02-problem_decompose
tags:
- problem_decompose
- stage-02
- run-rc-20260
title: 'Stage 02: Problem Decompose'
---

# Stage 02: Problem Decompose

[thinking] The user ran a `/model` command to set the model to claude-sonnet-4-6, and there's a system prompt asking me to decompose a research problem into sub-questions. This appears to be part of an autoresearch workflow that's already running. The local command output shows a research strategist system prompt and a detailed SMART research goal that was generated. The system prompt says I should decompose the research problem "pairwise grasp scoring BPR open-world robotic grasping" into at least 4 prioritized sub-questions with sections: Source, Sub-questions, Priority Ranking, Risks. Now I'm working through the decomposition of this research problem into structured sub-questions that address the core challenges around grasp scoring and open-world robotic grasping scenarios.
## Research Decomposition: Pairwise Grasp Scoring BPR for Open-World Robotic Grasping

### Source

**Project**: OWG-main (Open-World Grasping)
**Current best**: BPR pairwise + margin_0.00 (LGGSN stage 4)
**Challenge**: High evaluation noise from PyBullet/VLM non-determinism; unclear ceiling for BPR-based ranking

---

### Sub-questions

**SQ1 — Does BPR margin tuning actually improve grasp selection quality, or is noise masking the signal?**
Current experiments use `margin_0.00`. The BPR loss `log σ(score_i − score_j − margin)` requires a well-calibrated margin to discriminate good from bad grasps. Without systematic sweep, the model may converge to near-uniform scores.

**SQ2 — What geometric features are most predictive for pairwise grasp ranking on novel objects?**
LGGSN uses some combination of point cloud features. For open-world generalization, surface normals, local curvature, approach-vector alignment, and antipodal quality likely contribute differently. No ablation exists.

**SQ3 — How much does language conditioning actually contribute to grasp ranking vs. pure geometry?**
The VLM grounding step identifies the target object, but the pairwise scorer may learn mostly geometry. If language adds <2pp, the architecture complexity isn't justified and a geometry-only BPR ranker would be simpler and more robust.

**SQ4 — Can synthetic pairwise preferences (simulation-generated success/failure pairs) transfer to real or held-out object distributions?**
Current training data comes from PyBullet rollouts. Sim-to-real and sim-to-novel-object gaps in *pairwise preference labels* are unexplored — a preference pair that's easy in simulation may be indistinguishable on a real sensor.

**SQ5 — What is the minimum number of grasp candidates needed for BPR ranking to reliably outperform random selection?**
If the planner generates only 3-5 candidates, pairwise ranking has little room to improve over random. Understanding the ranking-benefit curve vs. candidate count determines whether the bottleneck is the ranker or the candidate generator.

---

### Priority Ranking

| Rank | Sub-question | Rationale |
|------|-------------|-----------|
| 1 | **SQ1** — BPR margin tuning | Directly controls whether the model trains usefully at all; should be validated first |
| 2 | **SQ5** — Candidate count vs. ranking benefit | Determines if the architecture bottleneck is upstream (generator) or downstream (ranker) |
| 3 | **SQ2** — Geometric feature ablation | Identifies which inputs drive generalization; needed before scaling |
| 4 | **SQ3** — Language conditioning contribution | Architectural question with impact on model simplicity and interpretability |
| 5 | **SQ4** — Sim-to-novel transfer | Longer-horizon question; requires SQ1-SQ3 resolved first |

---

### Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Evaluation noise too high** to detect <5pp differences between BPR variants | High | Increase eval seeds (≥30 trials/condition), use bootstrap CI; report effect sizes not just means |
| **BPR loss degeneracy**: model assigns similar scores to all candidates, margin irrelevant | High | Monitor score variance during training; add diversity regularization or contrastive pairs |
| **Language grounding failures** corrupt pairwise labels (wrong object selected) | Medium | Filter training pairs where VLM confidence < threshold; log grounding accuracy separately |
| **PyBullet physics artifacts** make success/failure labels unreliable for real-world transfer | Medium | Cross-validate on ≥2 different physics seeds; consider partial-credit metrics (grasp stability) |
| **Candidate generator ceiling**: top-1 grasp from planner already good enough, ranking adds nothing | Low | Measure oracle rank accuracy first — if oracle@1 ≈ random@1, problem is generator not ranker |