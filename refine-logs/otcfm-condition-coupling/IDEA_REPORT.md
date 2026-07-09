# Robotics Idea Discovery Report — replacing OT-CFM as the core generator

**Direction**: find a genuinely effective, novel, top-venue-worthy core candidate-generation method for 6-DoF grasp generation, replacing the empirically-failed OT-CFM, under small-data (~400 examples/object × 7 objects) and single-GPU constraints.

**Date**: 2026-07-10
**Pipeline**: research-lit (robotics framing) → synthesis (Codex/GPT-5.4 novelty-check and external review unavailable this session — 401 Unauthorized, no valid API credentials; this report is my own analysis, flagged as such, not externally verified)

## Robotics Problem Frame

- **Embodiment**: single-arm SO-ARM101 (5-DOF arm + parallel gripper), top-down-constrained (roll=π, pitch=0 fixed), so the *learned* part of the pose is effectively 3-DoF (x, y, yaw).
- **Task family**: single-object tabletop grasping, YCB objects, MuJoCo simulation.
- **Observation/action interface**: SAM 256-d visual embedding (object-identity conditioning) + live point cloud (CoM/geometric features) → 6-DoF target pose → IK → physics-executed grasp.
- **Available assets**: MuJoCo/SO-ARM101 sim, LGGSN pairwise reranker (14-dim geometric features, BPR-trained, *already validated and working* — contributes -9.7pp when removed), fixed-seeding evaluation harness, ~400 physical-execution-success examples/object × 7 objects, SAM embeddings, single GPU (RTX 3060 6GB class).
- **Constraints**: small per-condition data, single GPU, ~1 month to submission.
- **Desired contribution type**: a working method, not just a diagnosis — per explicit user request ("找别的有效方法，我希望能发顶刊").

## Landscape Matrix (key papers found)

| Paper | Venue | Method | Key Result / Claim | Relevance |
|---|---|---|---|---|
| **Implicit Behavioral Cloning** (Florence et al.) | CoRL 2021 | Energy-based model (EBM): score candidate actions instead of directly regressing/sampling them via an explicit generator | EBM/implicit policies "often outperform common explicit behavioral cloning policies," specifically on tasks with **high combinatorial complexity** and precision requirements — the paper's whole thesis is that explicit models struggle exactly where our OT-CFM struggled (discontinuous, multimodal, data-limited action distributions) | **Directly explains our negative result** and is a well-known, highly-cited precedent for the fix |
| **CGDF: Constrained Grasp Diffusion Fields** (Singh et al.) | IROS 2024 | Part-guided diffusion *guided by* a trained EBM's energy values (hybrid, not pure diffusion) | High sample efficiency in constrained grasping "without explicitly training on massive constraint-augmented datasets" | Shows EBM-guidance is the recent, validated way to get diffusion-level flexibility at low-data sample efficiency, specifically in grasp generation |
| **GraspGen** (2507.13097) | 2025 (arXiv, real-world validated) | Diffusion-Transformer generator **paired with a discriminator that scores/filters sampled grasps** | Generalizes sim→real despite training only in sim | Structurally the same "generate + discriminate" two-stage shape we already have (OT-CFM + LGGSN) — confirms the shape is right, the generator was the broken part |
| **NeuGraspNet** | recent | Fully implicit 6-DoF grasp quality prediction from global (scene) + local (grasp) neural surface features, no explicit pose generator at all | Effective even under partial observation | A more radical "skip generation, only score+search" precedent |
| C²OT (Cheng & Schwing) | ICCV 2025 | already covered this session | condition-agnostic OT coupling fails in conditional flow matching | Explains *why* OT-CFM failed, doesn't fix the small-data problem at its root |
| Joyce et al., consensus-driven grasp uncertainty | IROS 2025 | already covered this session | perception-uncertainty ensembling | Different uncertainty source than our candidate-selection finding |

## Ranked Ideas

### Idea 1 (RECOMMENDED): Unify generation and reranking into a single energy-based / implicit candidate model

- **One-sentence summary**: Replace the two-stage "flow-matching generator (broken) → LGGSN reranker (works)" pipeline with a single implicit/energy-based model, trained the same contrastive/pairwise way LGGSN already is, that directly scores candidate poses in the low-dimensional (x, y, yaw) space and is sampled at inference via a cheap derivative-free search (e.g., cross-entropy method / gradient ascent on the pose manifold) rather than an ODE/SDE trajectory that needs OT coupling or large per-condition batches to train well.
- **Target embodiment / benchmark**: unchanged — same SO-ARM101/MuJoCo/YCB setup, same fixed-seeding harness, same physical-execution success metric.
- **Core bottleneck addressed**: exactly the one we found — explicit generative models (flow matching, diffusion) need enough per-condition samples to learn a stable, well-coupled sampling trajectory; with ~400 examples/object this breaks (our OT-CFM result) or only partially works (Remove-OT/DDPM). An energy-based model only needs to learn a *scalar score* over the 3-D pose manifold from contrastive (success vs. failure) pairs — a much lower-capacity, more sample-efficient learning problem, and this is exactly the well-established thesis of IBC (CoRL 2021) and the recent grasp-specific validation in CGDF (IROS 2024).
- **Minimum sim-first pilot**: train an EBM (small MLP, same input features as LGGSN plus SAM embedding) via a contrastive loss (InfoNCE-style: real physical-execution successes as positives, randomly perturbed/failed poses as negatives — data we may already have from the existing trial logs) on the existing ~400/object dataset; at inference, sample K candidate poses via CEM (iteratively refine a Gaussian proposal toward high-energy regions) or simple gradient ascent on the learned energy w.r.t. pose, then execute the top-scoring one (optionally still through LGGSN, or absorb LGGSN's features directly into the energy function — a design choice to resolve in the pilot).
- **Mandatory metrics**: physical grasp success rate (existing criterion), same 7 objects, same fixed-seeding harness — directly comparable to every number already collected this session (Baseline 79.1%, OT-CFM 69.1%, Remove-OT/DDPM ~baseline-competitive on 2/3 objects checked).
- **Expected failure mode if it doesn't work**: CEM/gradient-ascent sampling could get stuck in local energy maxima, or the contrastive loss could be starved of good negatives (need enough *failed* physical trials as negatives — check whether the existing dataset has failure labels, not just successes, since `GraspPoseDataset` currently filters to `label==1` only).
- **Real hardware needed?**: no — stays entirely in the existing MuJoCo pilot.
- **Novelty argument**: not a new algorithm (EBMs for robot policies are CoRL-2021-established, and EBM-guided grasp diffusion exists since IROS 2024) — the contribution is (a) showing exactly *why* explicit flow/diffusion generators fail in a small-per-condition-class regime via our own rigorous negative result, and (b) demonstrating the EBM alternative closes the gap on the *same* physically-executed benchmark where the explicit approach failed, unifying what is currently two separately-trained components (generator + reranker) into one simpler, more sample-efficient model.
- **Risk**: MEDIUM. Needs a real implementation + training pass (not just a retrain of an existing script), and CEM/energy-sampling hyperparameters need tuning. But every other piece (features, data, evaluation harness, comparison baselines) already exists.

### Idea 2: Retrieval-augmented local pose adaptation (lower risk, weaker novelty ceiling)

- **One-sentence summary**: Instead of learning a global generative distribution, retrieve the k nearest successful training poses (by SAM-embedding similarity + geometric similarity) for the current object instance and locally perturb/interpolate among them, rather than sampling from a learned ODE/energy field at all.
- **Bottleneck addressed**: with only ~400 examples/object, a global generative model (explicit or implicit) still has to interpolate/extrapolate; retrieval sidesteps the "generalize from little data" problem entirely by construction.
- **Pilot**: purely offline — reuse existing training data, no retraining needed, fastest to try.
- **Novelty**: weaker — retrieval-augmented manipulation exists (RobMRAG-style work found in the search, though mostly LLM/VLM-driven, not pose-generation-specific), so this reads more as a strong baseline/ablation than a standalone top-venue contribution. Better used as a *supporting* comparison inside Idea 1's paper (does the EBM actually beat simple retrieval, not just beat the broken OT-CFM?) than as the headline method.
- **Risk**: LOW execution risk, but LOW novelty ceiling on its own.

### Idea 3 (fallback): Lean into the diagnostic contribution itself

- **One-sentence summary**: If Idea 1's EBM pilot doesn't clearly beat baseline either, the fallback is exactly what was discussed before this pivot — a rigorous, three-way empirical comparison (explicit flow/diffusion generators fail vs. implicit/EBM and/or retrieval succeed, in a real physical robot benchmark, small-data regime) as a methodology/diagnosis contribution.
- **Precedent for diagnostic robotics papers landing at top venues**: this is a legitimate, publishable pattern (the search surfaced several ML-diagnosis papers, e.g. reproducibility-crisis-style work; in robotics specifically, benchmark/diagnosis papers are a recognized, respected category at CoRL/RSS).
- **Risk**: this is the "we already know this works as a paper" safety net — lower ceiling than Idea 1 succeeding, but zero risk of ending up with nothing.

## Recommendation

Pursue **Idea 1** as the primary pilot given the ~1-month runway: it has real citable precedent (CoRL 2021 foundational + IROS 2024 grasp-specific), directly explains our own hard-won negative result rather than ignoring it, reuses almost all existing infrastructure (LGGSN's contrastive training recipe, MuJoCo harness, evaluation pipeline), and — critically — if it works, gives a clean top-venue narrative: *"we show why explicit conditional generative candidate models fail in the small-per-class-data regime common in real robot learning, and that the fix is not a training trick (OT coupling, its condition-aware variant) but a different model class (implicit/energy-based) — validated end-to-end on physically-executed grasp success, not just a generation-quality proxy."* Idea 3 is the safety net if Idea 1's pilot disappoints; Idea 2 is a cheap, useful supporting baseline either way.

## Not Yet Done (flagged, not silently skipped)

- **No external novelty-check or GPT-5.4 review was run** — Codex MCP auth is unavailable this session (401 Unauthorized). Before committing significant implementation time, a real novelty check (does an EBM-for-grasp-generation-on-physical-success-rate paper already exist covering this exact angle?) should be run once Codex access is restored, or done manually.
- No pilot has been implemented or run yet — this is a research-direction recommendation, not a validated result.

## Next Steps (if you want to proceed)

1. Check whether the existing training data (`grasp_6dof/dataset/lggsn_candidates_v9.jsonl`) already has `label==0` (failed) rows suitable as EBM contrastive negatives — this determines whether Idea 1's pilot is a pure retrain or needs new negative-mining.
2. Implement a minimal EBM: same feature inputs as LGGSN, contrastive loss, small MLP.
3. Implement CEM or gradient-ascent sampling at inference.
4. Pilot on the same 3 objects (Pear/TomatoSoupCan/CrackerBox) at n=25 first, exactly like the OT-CFM/DDPM/Remove-OT/Stratified-OT checks this session, before committing to a full 7-object campaign.
