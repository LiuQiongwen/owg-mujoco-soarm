# Robotics Idea Discovery Report

**Direction**: 为机器人抓取/操作方向的 T-RO / IJRR 投稿寻找一个真正新颖、可行的核心方法，主题是"世界模型（world model）与 VLA（vision-language-action）模型的结合"。
**Date**: 2026-07-15
**Pipeline**: robotics frame → literature survey (WebSearch, 8 queries + 3 fetches) → idea generation/filtering → feasibility/pilot design → novelty check → external review (attempted via Codex/GPT-5.2, failed with 401 auth error — self-conducted critical review instead) → this report

## Robotics Problem Frame

- **Embodiment**: single 6-DoF arm (SO-ARM101 real hardware; Piper real hardware + RoboSuite sim as the newer validation platform)
- **Task family**: 6-DoF tabletop grasping / pick-and-place, candidate generation + selection
- **Observation / action interface**: RGB wrist camera, SAM 256-dim visual embeddings, proprioception; newly available: dedicated depth camera, Meta Quest 3 (passthrough AR + hand tracking); actions are 6-DoF grasp poses executed via IK → joint position control
- **Learning regime**: hybrid — geometric heuristic reranking (LGGSN, BPR pairwise-trained, the one validated independent contribution), imitation learning (ACT via LeRobot), a reused (non-novel) VLA framework; several generative-model and world-model routes already tried
- **Available assets**: MuJoCo/SO-ARM101 sim, Piper/RoboSuite sim, YCB objects, ~400 real physical grasp-execution samples/object × 7 objects (with ground-truth outcomes already logged), real SO-ARM101 + LeRobot pipeline, new depth camera + Quest3, cloud GPU rental now under consideration (previously capped at one 6GB GPU)
- **Constraints**: small real-data regime; compute constraint loosening but not yet resolved; T-RO/IJRR bar requires genuine methodological novelty, not diagnosis-only or training-free heuristics
- **Desired contribution type**: method (primary) — ideally one that reuses the team's own validated BPR/pairwise-training paradigm (from LGGSN) rather than starting from scratch

## Already-Excluded Routes (do not re-propose under a new name)

1. OT-CFM grasp candidate generation — significantly worse than random baseline on small multi-condition data (-10pp pooled, p=0.0025).
2. Plain CFM/DDPM (no OT coupling) — on par with baseline; confirms the OT coupling specifically was the problem, not generative candidate modeling in general.
3. C²OT condition-aware OT fix — partial, inconsistent improvement.
4. **MPC-style real-time world-model correction before closing the gripper** (autonomous, no human in the loop) — three independent physical pilots, all net negative (-9.3pp/-18.7pp/-13.3pp). Any new world-model proposal whose core mechanism is still "autonomously search/correct the action in a tight real-time loop" is a variant of this and should be rejected on sight.
5. Ensemble+consensus candidate selection — strong result on SO-ARM101 (Pear 6%→68%, TomatoSoupCan 34%→64%) but a training-free geometric heuristic, not novel enough alone; **also just failed to replicate on a second embodiment (Piper) under two independently-designed noise models** — a genuine, separate cross-embodiment negative finding, useful as background/motivation but not itself the method under search here.

## Landscape Matrix (from literature survey)

| Axis | Finding |
|---|---|
| World models for manipulation | Very active, crowded (2 recent surveys: arXiv:2606.00113, arXiv:2605.00080; combination papers VLAW, MIND, 3D-VLA, World-Value-Action Model, TACO) |
| VLA uncertainty / failure prediction | Crowded, but uniformly **intrinsic** — calibrated from the policy/model's own outputs (conformal prediction on action tokens in ReconVLA; diffusion-policy denoising uncertainty in "Uncertainty Comes for Free"; perturbation-based methods) — none use an *external* world model + human-judgment calibration loop |
| World-model data augmentation for imitation learning | Strong, close prior art: **"Dream to Manipulate" (ICLR 2025)** — compositional world models + Gaussian Splatting + equivariant transforms for one-shot policy learning. Occupies the "world model generates synthetic training data" niche thoroughly; a simpler depth-camera version would be scooped, not novel. |
| AR/XR for human-robot interaction, "preview + veto" pattern | Established field (VAM-HRI survey line; "Alteration Previews"; EVE; a 2026 NYU AR system). **Confirmed via direct fetch**: every example found visualizes the robot's *known, deterministic, already-planned* kinematic path (waypoints, collision buffers) for one-way situational awareness — no learned prediction of *uncertain physical outcomes*, no feedback loop into any model's training/calibration. |
| Gap identified | The specific combination of (a) an *external* world model predicting *uncertain physical* grasp outcomes, (b) AR visualization of that prediction (not a known path) for manipulation-candidate verification, (c) a closed *active-calibration* loop turning cheap human judgments into BPR-style pairwise training signal, does not appear directly occupied by anything found. |

## Ranked Ideas

### Idea 1: AR-Mediated Active Calibration of World-Model Grasp-Outcome Predictions via Pairwise Human Judgment — RECOMMENDED (with major revision)

- **One-sentence summary**: A lightweight world model predicts each VLA grasp candidate's physical outcome (post-lift object pose, contact stability); the prediction is shown via Quest3 passthrough AR before execution; the human's fast accept/reject judgment becomes a BPR-style pairwise training signal — reusing LGGSN's exact training paradigm — that calibrates the world model's own confidence over time, closing an active-learning loop rather than autonomously correcting actions.
- **Embodiment**: SO-ARM101 / Piper, tabletop grasping
- **Benchmark / simulator**: existing MuJoCo/RoboSuite sim + the team's own ~400-sample/object real-execution logs (already contain ground-truth outcomes, no new data collection needed for the first pilot phase)
- **Bottleneck addressed**: VLA policies have no calibrated, externally-verified confidence in whether a proposed grasp will actually succeed physically — the team's own prior attempt to address this via autonomous real-time correction (MPC route) failed; this proposes a structurally different, human-in-the-loop calibration mechanism instead of an autonomous one
- **Pilot type**: sim/offline first (mandatory), real hardware only with explicit approval afterward
- **Minimum sim-first pilot**: train the outcome-predictor on existing real logs; simulate the calibration loop using the simulator's ground truth as an oracle proxy for "human judgment" (no Quest3 needed); measure whether successive calibration rounds reduce expected calibration error (ECE) vs. a no-loop baseline — this isolates the SCIENTIFIC claim (does active calibration help) before any AR engineering investment
- **Mandatory metrics**: calibration error (ECE) before/after the loop, correlation between predicted confidence and actual physical success, an explicit with-loop vs. without-loop ablation (this ablation is the load-bearing experiment — without it the paper collapses into a systems demo)
- **Expected failure mode if wrong**: the calibration loop doesn't measurably improve ECE beyond what passive logging of execution outcomes already provides — i.e., the "active"/human-judgment part turns out to add no signal beyond what the robot already observes for free when it executes and fails/succeeds
- **Whether it truly needs real hardware**: NO for the core scientific claim (sim-first pilot with an oracle proxy is sufficient to validate or kill the idea); YES for a full systems paper demonstrating the AR interface itself, but that should be a second phase, not the entry point
- **Novelty**: real but narrow — every individual component (world models, VLA uncertainty, AR-HRI preview) is separately crowded; the specific combination is not directly occupied by anything found, but the wedge is thin and depends on precise framing
- **Reviewer score (self-conducted, see below)**: PURSUE WITH MAJOR REVISION
- **Hardware risk**: LOW for the pilot (existing data + sim), MEDIUM-HIGH for the full AR system (real-time 6-DoF object tracking + AR registration/rendering is nontrivial systems engineering, not yet built by this team)
- **Next step**: run the sim-first oracle-proxy pilot before touching Quest3 hardware at all

### Idea 2 (downranked): World model as VLA action-uncertainty predictor without AR (Candidate B, original framing)
- Killed/downranked because: the field is now crowded specifically here — ReconVLA (conformal prediction on action tokens), "Perturbation-Based Uncertainty for Failure Detection in VLA Models", and "Uncertainty Comes for Free" (diffusion-policy denoising uncertainty) all address near-identical ground, all in 2025-2026, all using *intrinsic* model uncertainty. Differentiating an *external* world-model-based version without the AR/human-calibration-loop angle would be a much thinner novelty claim than Idea 1's full combination.

### Idea 3 (downranked): Depth-camera-driven real-to-sim world model for VLA data augmentation (Candidate C, original framing)
- Killed because: "Dream to Manipulate" (ICLR 2025) already occupies this niche with more sophistication (compositional/equivariant structure, Gaussian Splatting) than a simple depth-camera world model would offer. Re-attempting this without a sharp differentiator would very likely fail novelty check outright.

## Self-Conducted Critical Review (Codex/GPT-5.2 external review unavailable — 401 auth error)

Scored 1-10, weighted toward Problem Fidelity, Method Specificity, Contribution Quality, Novelty:

| Dimension | Score | Note |
|---|---|---|
| Problem Fidelity | 8 | Real, actively-cited bottleneck (VLA grasp trust/calibration) |
| Method Specificity | 6 | Core loop is concrete; AR rendering/registration and the exact pairwise-comparison formulation still hand-wavy, need sharpening before implementation |
| Contribution Quality | 6-7 | Real risk of being "3 ideas in a trenchcoat" (world model + AR + active learning) unless the with/without-loop ablation is made the explicit, disciplined centerpiece |
| Genuine Novelty vs. prior art | 6 | Real but narrow wedge; individually-crowded components, combination not directly found elsewhere but depends heavily on careful positioning against Dream to Manipulate, ReconVLA-family work, and Alteration Previews |
| Feasibility given stated infra | 6 | Sim-first phase is easy (existing data + code); real Quest3 phase is a genuine, unbudgeted systems-engineering lift |
| Risk of repackaging the failed MPC route | 9 (low risk) | The "world model never autonomously modifies the action" design principle is a genuine, verifiable architectural difference |
| Venue readiness (T-RO/IJRR specifically) | 7 | Appropriately scoped for this venue tier if evaluation is rigorous; would likely be under-scoped for CoRL/RSS without more ML novelty |

**Most damaging objection**: this may fundamentally be a systems/HCI contribution wearing robotics-ML clothing — a hostile reviewer would ask "why does this need AR/passthrough specifically, versus a 2D screen with a confirm/reject button? What does the headset modality itself contribute beyond lower interaction friction, and is 'lower friction' alone enough novelty for T-RO/IJRR?" **Fixable, not fatal**: either (a) run an explicit AR-vs-screen ablation on judgment speed/quality/accuracy, making the modality itself part of the empirical claim, or (b) honestly reframe AR as a deployment/engineering choice for a later systems paper, and run the CORE scientific claim (does active calibration improve ECE) with a simple screen-based interface first — which is also exactly what the sim-first pilot already does.

**Overall verdict: PURSUE WITH MAJOR REVISION.** The calibration-loop mechanism is genuinely novel and directly reuses validated infrastructure (LGGSN's BPR objective) rather than starting a new method family from zero. But before any Quest3/AR engineering investment: (1) run the sim-first, oracle-proxy calibration pilot on existing data to get a real signal on whether the core claim holds at all, (2) make the with/without-loop ablation the explicit centerpiece of the eventual paper, not an afterthought, (3) decide explicitly whether AR is a scientific claim (needs its own ablation) or a deployment vehicle (can be deferred) before committing engineering time to it.

## Evidence Package for the Top Idea

- **Required baselines**: no-calibration-loop world model (passive logging only), the team's existing LGGSN reranker (as a "geometric-only" baseline for comparison), and — if feasible — a 2D-screen confirm/reject interface as an AR-vs-screen ablation
- **Required metrics**: expected calibration error (ECE), predicted-confidence-vs-actual-success correlation, human judgment time/cognitive load (if the AR-vs-screen ablation is run), physical execution success rate as a downstream check
- **Required failure cases**: cases where the world model is confidently wrong (high predicted confidence, actual failure) both before and after calibration — this is where the loop's value (or lack of it) will be most visible
- **Whether real robot evidence is mandatory**: not for the initial go/no-go signal (sim + existing real logs suffice); yes for any final publishable claim about the AR interface specifically

## Next Steps

- [x] Implement the sim-first, oracle-proxy calibration pilot using existing ~400-sample/object logs (no new hardware needed)
- [ ] If ECE improves with the loop: sharpen the exact pairwise-comparison formulation and world-model architecture before touching Quest3
- [ ] If ECE does not improve: treat as a real negative result (consistent with this project's existing honest-reporting convention) and reconsider Idea 2/3 or a fresh direction, rather than forcing the AR angle regardless of outcome
- [ ] Only after a positive sim-first signal: scope the Quest3/depth-camera systems-engineering work as a distinct, separately-budgeted phase

## Sim-First Pilot Result (2026-07-15): negative — the specific active-calibration mechanism does not help, uncertainty sampling actively hurts calibration for most of the label budget

**Can support**: a real, controlled, statistically-tested answer to the report's own stated go/no-go question. Used the team's existing `logs/lggsn_live_candidates.jsonl` (4288 real physical-execution-outcome rows; Scissors excluded per project memory's confirmed ui.py CFM fallback bug, 2026-07-09; remaining 5 objects — Banana, CrackerBox, MustardBottle, PowerDrill, TomatoSoupCan — 3248 rows, 67.1% positive rate). Built a minimal logistic-regression "world model" outcome predictor (12 continuous grasp-geometry features + one-hot object class, deliberately simple so the comparison isn't confounded by deep-learning training instability) and compared two conditions with an IDENTICAL model class, IDENTICAL BPR-style pairwise update rule (same pairwise family as LGGSN's own training), and IDENTICAL total label budget at every checkpoint — the only difference being selection order:

- **Passive** (matches "no-loop" baseline): labels revealed in random order, simulating ordinary passive accumulation of execution logs.
- **Active** (the proposed mechanism): labels revealed via uncertainty sampling — at each step, query whichever remaining candidate the current model is least confident about (predicted probability closest to 0.5) — the standard operationalization of "actively decide which candidate is worth a human judgment."

20 independent repeats per condition, held-out test set built from entirely separate scenes (426 rows, 20% of scenes) to prevent leakage between candidates from the same grasp scene.

**Result: active selection is WORSE than passive for nearly the entire label-budget range**, not better. At an early checkpoint (167 labels), active's mean ECE was 0.284 vs. passive's 0.202 (worse by +0.087, paired t-test p=0.0048 — statistically real, not noise). This gap persists and only narrows late in the budget, with active finally edging passive at the very end (2822 labels, ECE 0.192 vs 0.211, diff −0.020) but not significantly (p=0.12) — consistent with both conditions simply converging once nearly all data is revealed regardless of order, not with active selection providing any genuine label-efficiency advantage.

**Cannot support**: the core calibration-loop mechanism as specifically proposed (uncertainty-sampling-driven active selection feeding a BPR pairwise update). This is a real negative result for exactly the regime — label efficiency, i.e. getting better calibration from FEWER human judgments — that would justify building the Quest3/AR interface at all. If a passive, no-effort logging pipeline calibrates just as well or better than actively soliciting judgments, the entire "cheap, low-friction active human-in-the-loop" value proposition of Idea 1 loses its justification.

**Likely explanation** (consistent with known active-learning literature, not a pilot bug): uncertainty sampling is a well-documented weak choice specifically for *calibration* objectives, even when it can help raw classification accuracy — querying only near-boundary/ambiguous cases skews the training distribution away from the natural label distribution the test set is drawn from, which is precisely what miscalibrates a model. This doesn't mean "active calibration is impossible," it means *this specific* active-selection strategy is the wrong one; better calibration-aware active learning strategies exist (e.g. querying to directly minimize expected calibration error rather than pure entropy/margin uncertainty) that were not tested here.

**Per the report's own stated protocol, this is a stop-and-reconsider signal, not a green light for Quest3 engineering.** Two honest paths forward, not a default continuation:
1. Try a calibration-aware (not uncertainty-based) active-selection strategy as one more cheap sim-only pilot before giving up on Idea 1 entirely — this is a small, well-scoped follow-up, not a new commitment.
2. Treat this as a genuine negative result for Idea 1 as designed, and move to Idea 2 or Idea 3 (both already downranked for novelty reasons in this report, so would need their own re-justification) or a fresh direction, rather than building the AR/Quest3 system anyway on the strength of the mechanism alone.

**Files**: `/tmp/claude-1000/.../scratchpad/wm_calibration_pilot.py` (pilot script), `wm_calibration_pilot_results.json` (full per-checkpoint results).

## Follow-up: calibration-aware active selection (2026-07-15) — fixes the "active is worse" problem but still shows no real advantage over passive; Idea 1's core mechanism does not survive a second, more favorable test

**Can support**: a direct test of path 1 above. Replaced pure uncertainty sampling with a calibration-gap-targeted selector: bin remaining candidates by current predicted probability (the same bins ECE itself uses), compute the empirical `|confidence - accuracy|` gap per bin from already-revealed labels, give unexplored bins an exploration bonus, and reveal a candidate from whichever bin has the largest current miscalibration gap — directly targeting the quantity ECE measures instead of generic decision-boundary uncertainty. Ran all three conditions (passive, active-uncertainty, active-calibration) head-to-head, same model, same BPR update, same budget, 20 repeats each.

**Result**: `active_calibration` is no longer significantly worse than passive at any checkpoint tested (p=0.32, 0.66, 0.73 at early/mid/final checkpoints — compare to `active_uncertainty`'s p=0.0048/0.0015/0.12, confirming that strategy really was the problem, not a pilot artifact). But it is *also* not significantly better than passive at any checkpoint — the mean ECE differences are small and sign-flip across checkpoints (e.g. −0.0206 early, −0.0081 mid, +0.0056 final), consistent with noise around zero rather than a real effect in either direction.

**Cannot support**: any version of Idea 1's core value proposition — that ACTIVELY soliciting judgments (however smartly selected) produces meaningfully better calibration than PASSIVELY logging execution outcomes for free. Two different active-selection strategies were tested under idealized conditions (a perfect, noiseless oracle standing in for human judgment, which is strictly more favorable than any real human interaction would be): one was significantly worse, the other was a statistical tie. Neither clears the bar needed to justify building a Quest3/AR interface whose entire premise is "cheap active human judgment is worth more than what the robot already logs for free."

**Verdict on Idea 1**: negative, on its own terms, per the report's own pre-registered go/no-go protocol — not from a single unlucky run, but from two independent attempts including a specifically-designed fix for the first attempt's most obvious weakness. Recommend not proceeding to Quest3/AR engineering for this specific mechanism. Idea 2 and Idea 3 (both downranked earlier in this report for novelty reasons, not feasibility) would need their own fresh justification before pursuing, or a new direction should be sought.

**Files**: `wm_calibration_pilot.py` (updated, three-way comparison), `wm_calibration_pilot_v2_results.json`.

---

# Direction 2: Cross-Embodiment Grasp Reranking (2026-07-15)

World model + VLA (Direction 1) was abandoned after two independent negative pilots. Fresh literature survey conducted per user request ("换方向，做个调研"), explicitly stepping outside the world-model/VLA/generative-candidate/training-free-heuristic frames already exhausted.

## Literature Survey Summary

Searched: pairwise preference learning for grasp reranking, failure recovery/retry, tactile/force-feedback grasp quality, cross-embodiment grasp candidate selection (4 WebSearch queries + 2 WebFetch deep-reads).

**Top finding: GraspGen-X** (arXiv:2606.00998, 2026) — cross-embodiment 6-DoF diffusion-based grasping with a generator + discriminator, the discriminator conditioned on a 12-dim gripper "swept volume" representation (open + half-closed cube dimensions + translation, through a 3-layer MLP to a 512-dim embedding). Zero-shot tested on 10 held-out real grippers never seen in training (mAUC 0.506 vs. 0.398 for retargeting baselines), and validated on real **Robotiq-2F140 and AgileX Piper hardware** (79.0% success) — Piper is a platform this team already owns. Critically, GraspGen-X's discriminator is trained **pointwise** (binary positive/negative classification), not pairwise — a real, checkable difference from LGGSN's already-validated BPR pairwise paradigm.

Other findings: Freeform Preference Learning (arXiv:2606.32027) — natural-language preference axes with per-axis pairwise comparisons, relevant to extending LGGSN's comparison dimensions beyond geometry but not grasp-specific and data requirements unclear; FAR (arXiv:2607.01111) — test-time failure-aware retry using policy confidence, doesn't need a learned failure classifier, but requires infrastructure (policy confidence signal) this team doesn't currently have and isn't LGGSN-adjacent; tactile-grasp-quality literature is rich (ManiSkill-ViTac 2025 benchmark, TacRefineNet, etc.) but requires new hardware (no tactile sensors currently owned), deprioritized.

**Checked for direct novelty collision**: no paper found combining "pairwise BPR training + embodiment/gripper conditioning + small real-data regime + validated on the team's own two owned real platforms" — nearby work (GraspGen-X, EAGG, 𝒯(ℛ,𝒪) Grasp, HRDexDB) all differs in training objective (pointwise), data scale (large-scale procedural/synthetic training), or scope (dexterous multi-finger hands, not this team's 2-finger grippers).

## Idea: extend LGGSN with embodiment conditioning, tested on SO-ARM101 + Piper

**Motivation**: this team's own recent negative finding — the training-free consensus-selection heuristic failed to replicate when ported from SO-ARM101 to Piper — reframed as a research question rather than a dead end: does a *learned*, embodiment-conditioned pairwise reranker succeed where the training-free heuristic failed?

## Pilot 1: minimal pointwise feasibility check (2026-07-15)

**Data reality found while designing the pilot** (not assumed in advance): SO-ARM101 has dense pairwise-labeled data (`logs/lggsn_live_candidates.jsonl`, 3248 rows across 5 objects, Scissors excluded per project memory's confirmed data-quality issue). Piper only has **pointwise** per-trial outcomes (one executed+labeled candidate per trial; the other N-1 candidates in each trial's sampled pool were never physically executed, so have no ground truth) — a genuine density mismatch between platforms, not a detail to gloss over. Existing Piper result files on disk were also found to be stale, generated under a since-reverted, worse-performing code variant (35% Cracker instead of the validated 45%) — **regenerated clean data** (trial_id 700-719 × {cracker, mustard, pear}, 60 rows total, under the final validated pipeline code) before running anything.

Built a minimal, honestly-coarse shared feature space — 3 conceptually-aligned features (height above table, an IK-convergence-quality proxy, a positioning-correction-magnitude proxy) — standardized **within each platform separately** before pooling (the principled way to align genuinely different units/frames/scales across embodiments, not a hack) — plus an embodiment one-hot. Pointwise logistic regression throughout (the richest objective both platforms can currently supply; full BPR-pairwise-on-both-platforms is a follow-up, not this pilot).

**First-pass result (single 5-fold CV)**: zero-shot transfer (train SO-ARM101 only, test Piper) AUC=0.437 (worse than chance); pooled training (SO-ARM101+Piper, no embodiment feature) AUC=0.662; pooled + additive embodiment one-hot AUC=0.662 — **bit-for-bit identical** to the no-embodiment version (the optimizer drove the embodiment feature's weight to exactly zero).

**Strengthened result (repeated 5-fold CV × 10 repeats = 50 evaluations, plus an embodiment × feature interaction-term condition to give the model capacity for genuinely different per-platform feature-outcome relationships, not just a different baseline rate)**:

| Condition | Mean AUC | vs. pooled baseline |
|---|---|---|
| A: zero-shot transfer (SO-ARM101 → Piper) | 0.446 | — |
| B: pooled, no embodiment feature | 0.654 | — |
| C: pooled + additive embodiment one-hot | 0.658 | +0.004, p=0.057 |
| D: pooled + embodiment × feature interaction | 0.673 | +0.019, p=0.357 |

**Can support**: a robust, highly significant finding that **B beats A** (pooled cross-platform training vs. naive zero-shot transfer): +0.208 AUC, paired t-test **p<0.0001**. This is the headline result — a simple pointwise model trained jointly on both platforms transfers dramatically better than naive zero-shot, in sharp, quantified contrast to the training-free consensus heuristic's total failure to transfer at all.

**Cannot support**: that explicit embodiment conditioning (the original GraspGen-X-inspired mechanism) adds value beyond plain pooling — neither the additive one-hot (p=0.057) nor the richer interaction-term version (p=0.36, larger point estimate but not significant with n=60 Piper trials) reached significance. This is genuinely inconclusive, not a clean negative — D's positive point estimate (+0.019) with a plausible mechanism (interaction terms let the model learn different feature-outcome relationships per platform) suggests conditioning might help with more data, but the current sample (60 Piper trials) is underpowered to tell.

**Reframed core finding**: the defensible, well-supported story here is not "embodiment-conditioned reranker" as originally scoped from GraspGen-X, but **"training-free geometric heuristics (consensus selection) fail completely to transfer across embodiments, while even a minimal learned model trained jointly across platforms transfers robustly — with the further, more nuanced, currently-open question of whether explicit conditioning adds anything on top of that."** This directly extends the team's own prior negative finding rather than requiring an unrelated new setup, uses only infrastructure already owned (both real platforms, existing LGGSN codebase to extend), and has a credible, checked-for-collision related-work anchor (GraspGen-X) to position against.

**Honest limitations**: n=60 Piper trials is small; the 3-feature shared space is a coarse, hand-picked proxy, not LGGSN's real 14-dim feature set; pointwise-only (not yet using SO-ARM101's real pairwise/BPR richness); embodiment representation is a bare one-hot (2 platforms), not GraspGen-X's rich continuous gripper encoding. All of these are addressable next steps, not fatal flaws.

**Files**: `xembod_pilot.py` (pilot script, 4-condition repeated-CV comparison), fresh Piper data at `pre_close_refresh_{cracker,mustard,pear}_700-720.json`.

## Pilot 2: scaled-up data resolves the underpowered interaction-term question — conditioning DOES help, but only as interaction terms, not additive (2026-07-15)

**Can support**: per Next Steps below, scaled Piper data from 60 → 150 trials (added trial_id 720-749 × {cracker, mustard, pear}, 30 more per object, same final validated pipeline code) and reran all 4 conditions with the same repeated 5-fold CV × 10 protocol.

| Condition | Mean AUC | vs. B |
|---|---|---|
| A: zero-shot transfer | 0.448 | — |
| B: pooled, no embodiment feature | 0.679 | — |
| C: pooled + additive embodiment one-hot | 0.679 | +0.0001, p=0.807 |
| D: pooled + embodiment × feature interaction | **0.720** | **+0.0408, p=0.0043** |

The A vs. B gap held and strengthened (+0.231, p<0.0001 — the headline "pooling beats zero-shot" finding is robust across both sample sizes). The additive embodiment feature (C) remained exactly as useless as before (p=0.81, consistent with the earlier bit-for-bit-identical result — this is now confirmed at two different sample sizes, not a fluke). **The interaction-term condition (D), previously inconclusive at n=60 (p=0.36), is now clearly significant at n=150 (p=0.0043)** — the earlier result was genuinely underpowered, not a null effect.

**Cannot support**: that this is now a finished result — n=150 is still modest, the feature space is still a coarse 3-proxy stand-in for LGGSN's real 14 dimensions, and this is still pointwise-only supervision on the Piper side (see remaining limitations below).

**Upgraded core finding, now with a real mechanism**: explicit embodiment conditioning does provide a genuine, statistically significant benefit over naive pooling — but *only* when expressed as **interaction terms that let the feature-outcome relationship itself differ by platform**, not as a naive additive embodiment embedding/bias (which was tested twice, at two sample sizes, and added nothing both times, p=0.81 and p=0.057). This is a more specific and more interesting methodological claim than "add embodiment conditioning" in the abstract: **the mechanism by which embodiment conditioning is expressed matters as much as whether it's present at all** — a plausible, checkable, and to our knowledge not directly tested-elsewhere claim (GraspGen-X's pointwise discriminator conditions the whole architecture on gripper representation from the start, rather than isolating and comparing additive-vs-interaction conditioning against a shared-backbone baseline the way this pilot does).

**Files**: same `xembod_pilot.py` (now loading 4 data chunks per object), new Piper data at `pre_close_refresh_{cracker,mustard,pear}_{720-730,730-740,740-750}.json`.

## Pilot 3: enriched feature set (5-dim, adding real yaw + object height) — the interaction-term conditioning result does NOT replicate; only "pooling beats zero-shot" survives (2026-07-15)

**Can support**: per Next Steps below, instrumented `run_pick_and_place()` (`piper_pick_and_place.py`) to log `grasp_yaw` (extracted from the final grasp orientation matrix) and `object_H` (per-object top-surface offset, `OBJECT_TOP_OFFSET`) — the two LGGSN-14-dim fields identified as carrying real per-trial signal (roll/pitch are fixed constants on both platforms; `dz`/`dz_lift`/`need_dz` were found to be degenerate zero-valued in the live SO-ARM101 data itself, so faking Piper equivalents for those would add no information; `dist_to_centroid`/`z_rel` are episode-level features requiring multiple labeled candidates per scene, which Piper's pointwise data structurally cannot supply without new data collection). Regenerated Piper data twice under this new schema: first a 60-row batch (trial_id 800-819), then scaled to 150 (adding 820-849) to match the earlier pilot's statistical power, using a genuinely richer 5-feature shared space: `[z, yaw, H, quality_score, correction_proxy]`.

| Condition | Mean AUC (n=150, 5 features) | vs. B |
|---|---|---|
| A: zero-shot transfer | 0.402 | — |
| B: pooled, no embodiment feature | 0.756 | — |
| C: pooled + additive embodiment one-hot | 0.756 | +0.0002, p=0.162 |
| D: pooled + embodiment × feature interaction | 0.725 | **−0.030, p=0.048** |

**Cannot support**: Pilot 2's "interaction-term conditioning helps" finding (+0.041, p=0.004 with the 3-feature set). With the richer, more legitimate 5-feature set at the same sample size (n=150), condition D is now **significantly WORSE** than plain pooling, not better — the opposite direction. This is the third independent test of embodiment conditioning in this report (additive at n=60, additive+interaction at n=150 with 3 features, additive+interaction at n=150 with 5 features), and the interaction-term result flipped sign across feature-set choices, while the additive-embodiment result has now been null three times running (p=0.81, 0.057, 0.16). Likely explanation: interaction terms double the added parameter count, and with Piper contributing only 150 of the ~3400 pooled training rows, the extra capacity is likely overfitting to sample-specific noise in whichever direction the particular feature set/fold split happens to favor — not learning a real, transferable platform-specific relationship.

**What actually survived three independent replications, robustly and by a wide, consistent margin**: pooled cross-embodiment training beats naive zero-shot transfer. B vs. A: p<0.0001 in the original 3-feature/n=60 pilot, p<0.0001 at n=150, and p<0.0001 again with the enriched 5-feature set at n=150 (effect size, if anything, grew larger with better features: +0.21 → +0.23 → +0.35 AUC). **Explicit embodiment conditioning (additive or interaction-based) has not shown a reproducible benefit across any of the three configurations tested** — this should now be reported as an open/negative finding, not a positive one, correcting the more optimistic framing in Pilot 2 above.

**Revised core finding for this direction**: the defensible, three-times-replicated claim is **"training-free geometric heuristics (consensus selection) fail completely to transfer across embodiments, while even simple pooled/joint training of a learned reranker transfers robustly — with no evidence so far that explicit embodiment conditioning adds anything beyond that."** This is a narrower, more honest claim than Pilot 2's "conditioning mechanism matters" framing, but it is the one actually supported by the accumulated evidence, and it is still a genuine, useful, checkable methodological finding directly extending the team's own prior negative result.

**Files**: `xembod_pilot_v2.py`, Piper data at `pre_close_refresh_{cracker,mustard,pear}_{800-820,820-830,830-840,840-850}.json`, code instrumentation in `piper_pick_and_place.py` (`grasp_yaw`/`object_H` fields added to the result dict).

## Pilot 4: genuine pairwise BPR training (2026-07-15) — 4th confirmation of the core finding, but the conditioning question stays unresolved with this small a sample

**Can support**: instrumented `run_pick_and_place()` with a `candidate_selection=<int>` mode (forces execution of one specific pool index instead of "best"/"consensus") so every candidate in a sampled pool can be individually executed and labeled — the structure LGGSN's real BPR pairwise objective needs, which all prior pilots in this report lacked on the Piper side (pointwise-only: one executed+labeled candidate per trial). Collected 3 genuinely mixed-label Cracker scenes (trial_id 900/901/902, same object pose held fixed per scene via `np.random.seed`, same candidate pool held fixed per scene via a freshly-reseeded `rng` before each of the 10 candidate executions): 2/10, 3/10, 5/10 success respectively — all three scenes have both positive and negative labels, i.e. real pairwise training signal, not degenerate. Trained actual BPR pairwise scorers (matching `train_lggsn_pairwise.py`'s own loss) on pooled SO-ARM101 (37 mixed-label episodes, 4609 pairs) + Piper data, evaluated with leave-one-scene-out (the only defensible split at n=3 Piper scenes) using pairwise accuracy (LGGSN's own validation metric, majority baseline 0.50).

| Condition | Per-scene pairwise accuracy (3 held-out scenes) | Mean |
|---|---|---|
| A: zero-shot | 0.625, 0.143, 0.280 | 0.349 (below chance) |
| B: pooled | 0.188, 0.810, 0.640 | 0.546 |
| C: pooled + additive embodiment | 0.500, 1.000, 0.440 | 0.647 |
| D: pooled + embodiment interaction | 0.500, 1.000, 0.800 | 0.767 |

**Cannot support**: drawing any conclusion about the conditioning question (B vs. C vs. D) from this table — n=3 held-out folds is nowhere near enough for the per-scene accuracies to be trustworthy (each is itself estimated from only 16-25 pairs drawn from a single scene, not independent samples), and there is no meaningful way to compute a real significance test at this scale. The apparent monotonic B<C<D trend here is **the opposite direction of Pilot 3's well-powered (n=150, p=0.048) finding that interaction terms hurt** — this is not a replication that overturns Pilot 3; it is a much noisier, much smaller-sample result that should not be trusted over it. Reporting it transparently rather than cherry-picking the more encouraging-looking table.

**What this pilot DOES robustly reconfirm**: zero-shot cross-embodiment transfer performs at or below chance (0.349 mean, and even the best individual scene only reached 0.625) — this is now the **4th independent confirmation** (pointwise 3-feature, pointwise 5-feature at two sample sizes, and now genuine pairwise BPR) of the one finding that has held up every single time it was tested.

**Honest final state of the embodiment-conditioning question**: unresolved, not negative and not positive. Three different test configurations gave three different answers (inconclusive at n=60, significantly positive at n=150/3-feature, significantly negative at n=150/5-feature, and now a noisy/uninterpretable positive-looking trend at n=3-scenes-pairwise). The only way to actually resolve this would be substantially more Piper data (particularly more genuinely pairwise scenes, which are expensive — full-pool execution, not single-candidate trials) — not something to fake a conclusion about from underpowered pilots.

**Files**: `collect_pairwise_piper.py` (candidate-pool full-execution collector), `xembod_pilot_v3_pairwise.py`, data at `pairwise_piper_cracker_{900,901,902}.json`, code change in `piper_pick_and_place.py` (`candidate_selection` now accepts an int).

---

# EXPERIMENT_PLAN.md Stage 0-2 Results (2026-07-15/16)

Executed per `EXPERIMENT_PLAN.md`. Full plan and rationale live there; this section records what actually happened.

**Stage 0 (infrastructure)**: added `EmbodimentLGGSN` to `lggsn_model.py` (reuses the existing `GatingNetwork`/`GC_LGGSN` mechanism for the "interaction" conditioning mode — embodiment identity modulates a soft per-feature gate, exactly matching what the toy-model pilots' interaction terms were approximating). Promoted the pairwise data collector to `piper_robosuite/piper_pairwise_collector.py`.

**Stage 1 (data collection)**: collected 25 mixed-label Cracker scenes x 10 candidates = 250 trials (scenes 900-923 + 950; one more batch, 924-926, was still running in the background when Stage 2 started and was not included), via the promoted collector. Meets the plan's ~20-30 scene target.

**Stage 2 (offline validation, the real architecture this time)**: trained `EmbodimentLGGSN` with genuine BPR pairwise loss (matching `train_lggsn_pairwise.py`'s own loss, not the earlier pilots' toy logistic-regression proxy) on pooled SO-ARM101 (37 episodes, 4609 pairs) + the Stage 1 Piper data, with proper **22-fold leave-scene-out CV** (vs. Pilot 4's uninterpretable 3 folds):

| Condition | Mean pairwise accuracy | vs. `none` |
|---|---|---|
| Zero-shot (SO-ARM101 only) | 0.811 | — |
| Pooled, `none` (no embodiment feature) | 0.955 | — |
| Pooled, `additive` | 0.947 | −0.008, p=0.198 |
| Pooled, `interaction` | 0.953 | −0.002, p=0.769 |

**Can support**: the core finding, now with the real architecture and a properly-powered test — pooled training beats zero-shot transfer, **p=0.0001, +0.144 pairwise accuracy**. This is the **5th independent confirmation** across every configuration tested in this direction (toy pointwise x2 sample sizes x2 feature sets, toy pairwise, and now the real model), and by far the most trustworthy given it uses the actual production architecture and loss, not a proxy.

**Also can support — the conditioning question is now genuinely resolved, not just "still unresolved"**: with 22 folds (vs. 3) and the real model, neither `additive` (p=0.20) nor `interaction` (p=0.77) beats plain pooling. Per `EXPERIMENT_PLAN.md`'s own pre-registered Stage 2 gate ("if `interaction` mode does not beat `none`/`additive` with the larger dataset either, treat the conditioning question as settled negative... do not carry an unresolved claim into Stage 3's expensive live test") — **this gate has fired. Embodiment conditioning is dropped from the paper's claims.** The headline claim is narrower but solid: pooling, not conditioning, is what does the work.

**Files**: `lggsn_model.py` (`EmbodimentLGGSN`), `piper_robosuite/piper_pairwise_collector.py`, `piper_robosuite/pairwise_results_cracker_*.json` (25 scenes), `piper_robosuite/stage2_train_embodiment_lggsn.py`, `piper_robosuite/stage2_results.json`.

**Next**: ~~per the plan, Stage 3 proceeds...~~ **SUPERSEDED, see the correction immediately below — do not act on the table above without reading it.**

## ⚠️ CORRECTION (2026-07-16): the entire "pooling beats zero-shot" result above was very likely a data-leakage artifact. Direction 2 is closed.

**Can support**: a serious methodological catch, made before committing to Stage 3's expensive live-execution budget, not after wasting it.

While designing Stage 3 (which needs the trained reranker to actually SELECT a candidate before execution), realized the feature sets used in every pilot and in the Stage 2 table above include values that are **not knowable until after a candidate has already been executed**: SO-ARM101's `score`/`need_dz` and Piper's `quality_score` (−descend IK error, but the version logged AFTER prior phases' physical trajectory, not an isolated check) and `correction_proxy` (−pre_close_drift_cm, which by definition cannot be known before the arm has already moved there). This is confirmed by this project's own prior work: `train_geo_ebm_grasp.py`'s header comment explicitly documents that `score`/`dz`/`dz_lift`/`need_dz` are "execution-derived... cannot be computed for an arbitrary candidate pose without already having executed it" on the SO-ARM101 side. A model trained on these features can legitimately evaluate already-executed, already-labeled historical data (which is all a passive reranker validation needs) — but **cannot be used to select a not-yet-executed candidate**, because at selection time those feature values don't exist yet. Every AUC/pairwise-accuracy number in this report up to this point was computed in a setting where this distinction didn't matter (offline evaluation of historical logs) — it only becomes a problem the moment the plan tries to use the SAME model for live selection, which is exactly what Stage 3 requires.

**Corrected Stage 2 (features restricted to `[z, yaw, H]` — the only values genuinely knowable from a candidate's pose and the object's geometry before any execution or IK solve)**:

| Condition | Mean pairwise accuracy |
|---|---|
| Zero-shot | 0.8236 |
| Pooled, `none` | 0.8236 |
| Pooled, `additive` | 0.8200 |
| Pooled, `interaction` | 0.8198 |

**Zero-shot and pooled training are now IDENTICAL** (diff=0.0000, p=1.0000). The entire "pooling robustly beats zero-shot, confirmed 5+ times" narrative built up across every pilot in this report was, in hindsight, very likely driven almost entirely by the execution-derived features acting as a near-tautological success predictor (a diagnostic value like "how much did the object drift during closing" is definitionally close to "did the grasp fail") — not by the model learning genuinely transferable cross-embodiment geometric knowledge.

**Addendum (2026-07-16), narrowing WHICH side actually leaked**: while building `causal_validity_audit/` as a formal tool (see `CAUSAL_VALIDITY_METHOD.md`), traced the SO-ARM101 side's `score`/`dz`/`dz_lift`/`need_dz` against the actual LIVE inference code path (`grasp_ranker_lggsn.py`, `policy.py`, `batch_s3s4.py`) rather than trusting the `train_geo_ebm_grasp.py` comment quoted above at face value — that comment turns out to describe a different, inactive legacy dataset (`grasp_6dof/dataset/all_lggsn.csv`), not the live pipeline. Independently verified against the on-disk `logs/lggsn_live_candidates.jsonl` (4,288/4,288 rows): `dz`/`dz_lift`/`need_dz` are hardcoded constant 0.0 in the live pipeline (dead features, not leakage), and `score` is a genuine pre-execution GR-ConvNet quality proxy, computed identically at training-log-write time and live-inference time. **The leakage that collapsed this direction's effect to null was concentrated entirely on the Piper side** (`quality_score`, `correction_proxy` — this part independently confirmed directly against `piper_pick_and_place.py`, unaffected by this correction). The corrected-Stage-2 result and verdict above stand unchanged — a pooled model is only as valid as its most-contaminated input, and Piper's contamination alone was sufficient — but the attribution is now precise rather than diffuse across both platforms. Also worth noting as a small, honest irony: the very first version of the causal-validity registry built to formalize this lesson repeated a milder version of the same mistake (trusting a plausible-looking comment instead of the live code) before being corrected the same way the original bug was found.

**Second addendum (2026-07-16), a third and fourth correction, this time caught by automation rather than by hand**: while building `causal_validity_audit/auto_tagger.py` (an automated dataflow tagger, see `AUTO_TAGGER_ALGORITHM.md`) as a stronger, algorithmic follow-up to the manual registry, its static analysis of the real `run_pick_and_place` function independently flagged `grasp_yaw` as `EXECUTION_DERIVED` — disagreeing with the hand-built registry, which had it marked admissible. Tracing why confirmed the tool was right: `grasp_mat` is reassigned post-commit at the "pre-close refresh" step, and `grasp_yaw` reflects that later, post-descend value, not the original pre-execution candidate orientation. This had real consequences — `grasp_yaw` was part of the "Stage 2 CORRECTED" `[z, yaw, H]` feature set this very section cites above as the clean, trustworthy result. It wasn't fully clean. A second bug compounded this: `retrospective_audit.py`'s own historical-feature-set list used the generic string `"yaw"` for the Piper side, which silently resolved against the SO-ARM101 side's (correctly admissible) `"yaw"` entry instead of the actual Piper field `"grasp_yaw"`, masking the contamination in the retrospective demonstration meant to catch exactly this. Both fixed; re-ran Stage 2 with a genuinely clean `[z, H]` feature set (dropping `yaw`/`grasp_yaw` from both platforms). **The qualitative null finding survives** — zero-shot, pooled-none, pooled-additive, and pooled-interaction are still exactly identical (diff=0.0000) — but the reported pairwise accuracy dropped from 0.8236 to 0.1327 (now below the 0.50 majority baseline). **Verified, not assumed, why** (2026-07-16, third addendum): `H` is an exact dataset-wide constant across this Cracker-only collection (σ=0, n=250), and `z` is a near-constant *per scene*, checked directly across all 25 scenes — 14/25 show within-scene spread under 2×10⁻⁴ (floating-point/physics-settling noise), because `spawn_pos` is the object's own spawn height, read once before any candidate-specific action, structurally unable to distinguish which of the 10 pooled candidates is later attempted. `[z, H]` therefore has almost no capacity to discriminate between candidates drawn from the same scene — exactly the comparison the pairwise objective is evaluated on — which explains both the exact cross-condition equality and the specific below-chance number, not just "some feature happens to be constant." The corrected number and this precise explanation should be used in any paper material; the 0.8236 figure and the looser "H is a constant, probably why" explanation are both superseded. A genuinely predictive, still-admissible candidate feature (the original pre-commit grasp orientation, distinct from the disqualified execution-derived `grasp_yaw`) was identified but not re-collected/tested — flagged as concrete future work.

**Third addendum (2026-07-16), an unrelated infrastructure bug found while chasing the flagged future-work item, and the strongest re-verification yet**: while instrumenting the "genuinely predictive, still-admissible candidate feature" flagged above (the true pre-commit candidate orientation, logged as a new `candidate_grasp_yaw` field), found that `PiperMultiObjectScene`'s placement sampler was constructed without an explicit `rng`, silently defaulting to robosuite's own OS-entropy-seeded generator — completely independent of every `np.random.seed(...)` call this project's Piper work has relied on. A "scene's" 10 candidates were never actually facing the same object placement, despite the code's own documentation asserting they were. Confirmed via direct re-runs (unmodified script, same scene, three different outcomes across three runs) and via 7-9cm of spawn-position spread found in already-collected data that should have been constant. Fixed in `piper_multi_object_scene.py`.

This does **not** affect the causal-validity criterion or any provenance verdict — those trace code, not data statistics. It does affect specific accuracy numbers computed under the false shared-placement assumption. Re-collected all 25 Cracker scenes under the fix and re-ran `[z, H]`: **the null finding survives exactly** (0.1036, diff=0.0000 across all four conditions, still zero-shot=pooled). Then tested `candidate_grasp_yaw` directly: a naive pooled correlation looked dramatic (r=-0.55, p<0.0001, n=250 — the strongest signal found anywhere in this entire investigation), but decomposed into a between-scene confound: mean orientation and success rate correlate strongly *between* scenes (r=-0.71, p=0.0001 across 25 scenes — different placements are both systematically differently-oriented and differently difficult for unrelated geometric reasons), while the *within*-scene correlation — the only one that could inform a real live-selection decision — is indistinguishable from zero (mean +0.03 across 10 mixed-label scenes, no individual scene reaching significance). The flagged future-work item is now closed: no admissible per-candidate feature tested so far carries real selection-relevant signal for Cracker, and this is now demonstrated with the project's most rigorous single test of that claim. See `paper_tro.tex`/`paper_tro_draft.md` §IV-E, `CAUSAL_VALIDITY_METHOD.md`'s addendum.

**Follow-up check**: tested whether a genuinely valid, cheap pre-execution feature — `score_candidate_ik` (the SAME isolated, seed-from-`READY_QPOS` kinematic IK check `select_best` already legitimately uses, recomputed fresh for all 250 already-collected Piper candidates, matching the exact candidate pool via the same RNG seed) — carries any real signal at all, even within Piper's own data alone (a simpler, prerequisite question before re-attempting cross-embodiment claims). **Correlation with success: 0.06.** Mean IK quality for successes (−0.0065) vs. failures (−0.0082): nearly identical. No useful signal.

**Why**: consistent with this project's own much earlier root-cause investigation into Cracker's failures (this session's Piper README, before Direction 2 even started) — the actual determinant of success is execution-time contact dynamics during `descend` (sustained grazing contact, object drift under the approaching gripper), which a purely kinematic, pre-execution IK check cannot see by construction. It doesn't know anything about collision geometry or contact forces, only whether the target pose is kinematically reachable.

**Verdict: Direction 2 (cross-embodiment grasp candidate reranking) is closed, honestly negative.** Not because embodiment transfer is impossible in general, but because — at least for Cracker, with the features actually available before committing to execute a candidate — there is no demonstrated predictive signal for a reranker (of any kind, any training scheme, any embodiment-conditioning mechanism) to learn from. The problem this whole direction was trying to solve around (poor grasp reliability) has its real cause somewhere a candidate-selection model structurally cannot reach: the physical approach/descend trajectory's execution dynamics, not which candidate pose gets chosen from a pool. This loops back to and reinforces the Piper README's own much earlier finding that the fix likely needed is in the grasping pipeline's execution precision (approach/descend contact-awareness), not in candidate selection — a conclusion reached independently by two different investigations in this project now.

**Do not**: proceed to Stage 3 (no valid selection mechanism to test); reuse the pre-correction Stage 2 numbers or the earlier pilots' "5-6x confirmed" framing in any paper material; assume this closes the door on cross-embodiment questions in general — it closes this SPECIFIC operationalization (reranking pre-execution candidates for Cracker with these features), not the broader question.

**Files**: `recompute_valid_ik_feature.py`, `piper_valid_ik_features.json`, corrected `stage2_train_embodiment_lggsn.py` (now uses `[z, yaw, H]` only, `FEATURE_DIM=3`), corrected `stage2_results.json`.

## Next Steps

- [x] Scale up Piper trial count — done
- [x] Replace the coarse 3-feature proxy space with a richer feature set (added yaw, object_H) — done; corrected the conditioning finding rather than confirming it
- [x] Extend from pointwise-only to genuine pairwise BPR training on the Piper side — done (3 scenes, 30 trials); reconfirmed the core finding a 4th time, left the conditioning question honestly unresolved rather than force a conclusion from 3 scenes
- [ ] Run `/novelty-check` and an external review pass on the now well-established, four-times-replicated "pooled cross-embodiment training beats zero-shot transfer, training-free heuristics fail entirely" claim before committing further engineering time — this is the one claim in this direction actually ready for that step; the conditioning-mechanism question is not ready and needs substantially more Piper data (more full-pool-executed scenes) before it's worth another pilot round

---

# Direction 3: World-Model-Driven Sim-to-Real Transfer (2026-07-16) — CLOSED, does not fit team resources

**Prompted by**: after Direction 1 (world model + VLA + Quest3 active calibration) and Direction 2
(cross-embodiment reranking) both closed negative, and after the team's strategic decision to narrow the
T-RO paper scope and explicitly add real Piper hardware to it (architecture only, hardware not yet
connected — see `piper_real_backend.py`), asked what remains viable specifically at the world-model/VLA
intersection given the project now has zero real Piper data but extensive Piper simulation data.

**Proposed mechanism**: distinct from the already-failed MPC-style real-time correction (Direction 1's
predecessor, closed in an earlier session window) — instead of using a world model *during* execution to
search/correct actions in the loop, use a world model *before* deployment as a training-data source or
robustness filter, so that a policy trained mostly/entirely in simulation transfers to real hardware with
few or zero real demonstrations. Timing and role are genuinely different from the excluded route.

## Literature check

Three lines of recent work implement this general idea, at three very different scales:

1. **World model as auxiliary training signal** (WorldVLA, DreamVLA) — future-frame prediction as a
   secondary loss alongside action prediction. Doesn't specifically address sim-to-real with scarce real
   data; not a close match to the team's actual bottleneck (zero real Piper data).

2. **World model as an imagination-augmented RL/self-improvement loop** — **RISE**
   (github.com/OpenDriveLab/RISE): a compositional world model (video-diffusion dynamics model + a
   value model initialized from π₀.₅) generates imagined rollouts that get mixed with real offline data
   (best mix found: ~60% real / ~40% imagined) to fine-tune a VLA policy. Validated on a **dual 7-DoF
   AgileX bimanual arm**, reporting +50 to +60 percentage points over baseline on three contact-rich
   tasks. **Disqualifying fact for this team**: the offline data driving each of RISE's per-task runs was
   2,300–3,000 real demonstrations — the world model amplifies real data, it does not replace it. This
   project has 0 real Piper demonstrations, not thousands, so RISE's mechanism is not applicable as-is.

3. **World-action model sim-to-real with zero real demos** — a very recent (arXiv:2606.31101, explicitly
   flagged in the paper itself as "early results... official work... will be released soon") adaptation of
   Cosmos Policy (a video-diffusion visuomotor control model) trained on ~800 synthetic demonstrations per
   task with heavy domain randomization from a purpose-built "AnyTask motion planning pipeline," achieving
   35% average zero-shot success on a real Franka across lift/drawer-open/pick-place. This is the closest
   match to the team's actual situation (zero real data) — but it is disqualified for a different reason:
   it requires (a) a large video-diffusion backbone (Cosmos-scale, well beyond anything trained so far in
   this project — LGGSN is a lightweight MLP-class scorer, not a generative video model), (b) a dedicated
   synthetic-data-generation pipeline with "extensive domain randomization" that does not exist in
   `piper_robosuite/` today and would be a multi-week infrastructure build on its own, and (c) is itself an
   unreleased, unreproduced preliminary result — there is no public code, and the paper's own wording
   ("will be released soon") means there is no stable target to differentiate against or build on top of
   yet. Chasing this specific mechanism now means competing with a well-resourced team's not-yet-published
   follow-up work using a fraction of their compute and data infrastructure.

## Verdict: closed, does not survive contact with the team's actual resources

**Can support**: the mechanism is genuinely distinct from the excluded MPC real-time-correction route (no
drift/repackaging concern) — but that was never the binding constraint. The binding constraint is scale.
Every viable prior-art implementation of "world model enables sim-to-real with little/no real data" needs
either (a) thousands of real demonstrations to amplify (RISE — team has zero), or (b) a video-diffusion-
scale world model plus a purpose-built synthetic-data pipeline neither of which currently exist in this
codebase (the Cosmos-derived paper). Building either from scratch is not a T-RO-timeline-compatible pilot;
it is a separate multi-month infrastructure project, and would land the team in direct, under-resourced
competition with an unpublished result from a much larger effort.

**Cannot support**: any claim that a lightweight, LGGSN-scale world model could reproduce either result at
this team's compute/data budget — no literature evidence exists that the mechanism degrades gracefully to
a small MLP-class model or a few hundred already-collected Piper sim trials; the two working examples found
both lean on the specific thing this team lacks (real data volume, or a heavy generative backbone).

**Recommendation**: do not open this as a new pilot. This is the sixth direction closed this project
(after: OT-CFM/C²OT generative candidates, MPC real-time world-model correction, ensemble-consensus
candidate selection cross-embodiment failure, AR-mediated active calibration, cross-embodiment grasp
reranking, and now world-model-driven sim-to-real transfer). Given the team already made a strategic scope
decision — narrow the T-RO paper to validated wins (gripper-controller bug fix methodology, Pear/Mustard
honest success rates, the cross-embodiment pooling-vs-zero-shot finding from Direction 2, real-hardware
backend architecture) and accept Cracker's execution-precision limitation as documented future work rather
than a problem requiring a new solution — the honest next step is to execute that already-agreed scope
rather than continue searching for a seventh mechanism. Future work section can legitimately cite this
literature check (RISE, the Cosmos-derived sim2real paper) as the identified path *once* the team has either
substantially more real Piper data or access to video-diffusion-scale compute — neither of which is true
today.

**⚠️ Note (2026-07-22)**: the "cross-embodiment pooling-vs-zero-shot finding from Direction 2" cited above
as a validated win is STALE — Direction 2's own later corrections (see that section) found this exact
result was a data-leakage artifact and closed Direction 2 as honestly negative. Do not cite it in paper
material; this paragraph was never updated when that correction landed.

---

# Direction 4: Re-opening Direction 3 under new resources (cloud GPU + real Piper + depth camera) — NARROWED, not reopened as originally scoped (2026-07-22)

**Prompted by**: the team is now acquiring three resources that were the binding constraints on Direction
3's closure — cloud GPU rental (confirmed feasible, ~$1-3/hr for A100/H100 class via RunPod/Lambda/Vast.ai),
a physical Piper arm (previously sim-only; `piper_real_backend.py`'s API is now verified against the real
`piper_sdk`, 2026-07-22, though never hardware-tested), and a depth camera (previously unused). Asked
whether Direction 3's core mechanism — world model as a pre-deployment data amplifier for sim-to-real
transfer — is now resource-matched, specifically in the **low-real-data regime** (tens of real demos, not
RISE's thousands), since that specific regime was never tested by RISE or the Cosmos-derived paper.

## Literature check: the "low-real-data world-model amplification" niche got substantially more crowded since 2026-07-16

Read past the abstracts (not just titles) of the four closest 2026 candidates:

1. **Sim-and-Real Co-Training** (arXiv:2503.24361, Franka Panda, RGB-only) — DOES run a real-data-scarcity
   ablation (40-400 real demos on MultiTaskPnP) and finds co-training remains beneficial even at 40 demos.
   But the mechanism is **MimicGen-style trajectory splicing** (geometric transform + concatenation of
   segments from real teleop demos) — no learned dynamics/world model at all. This is genuinely different
   from RISE's mechanism, and importantly: it demonstrates that a near-zero-cost, training-free
   augmentation method already captures real benefit at exactly the data scale this team would operate at.
2. **R2RDreamer** (arXiv:2606.17040) — targets "a few source demonstrations" (low-data framing, close to
   this team's regime) but the mechanism is 3D point-cloud editing + video completion (ControlNet-style),
   focused on spatial generalization, not a forward dynamics model predicting physical consequences of
   actions. No real-data-scarcity ablation found in the abstract-level content checked.
3. **OA-WAM** (arXiv:2605.06481) — a genuine world-action model (slot-based next-frame prediction +
   flow-matching action decoding) but evaluated on LIBERO/SimplerEnv (standard sim benchmarks), no evidence
   of a real-hardware low-data ablation.
4. **Mask2Real-WM** (arXiv:2607.04546) — the closest single match found: a genuine learned dynamics model
   (predicts future segmentation masks from actions) fine-tuned on **"fewer than 2.5h of real
   demonstrations"** (a real low-data regime) explicitly for "policy evaluation, planning, and data
   augmentation" — nearly the exact framing originally proposed. **Disqualifying difference**: scoped to
   23-DoF **dexterous hands**, not simple 2-finger parallel grippers, and uses segmentation masks as the
   intermediate representation (ControlNet + Stable Video Diffusion rendering), not RGB-D/depth-conditioned
   dynamics. Different embodiment complexity class and different representation choice, but close enough
   that this niche is not open in general — only a precisely-scoped corner of it.

## Verdict: the general framing is no longer novel; a narrower, precisely-positioned wedge survives

**Cannot support**: "world model amplification helps in a low-real-data regime" as a general, open
question — Mask2Real-WM already answers a close version of it (for dexterous hands, mask-based
representation), and Sim-and-Real Co-Training already shows a training-free alternative captures real
benefit at a comparable data scale without needing any learned world model at all.

**Can support, narrowly**: none of the four papers checked run the SPECIFIC controlled comparison this
team's actual resources make cheap to run — **does a lightweight, RGB-D/depth-conditioned learned dynamics
model earn its complexity over training-free MimicGen-style trajectory splicing, specifically for simple
2-finger parallel-gripper tabletop grasping (not dexterous hands), at the real-data budget a small team can
actually collect (tens of demos, days not weeks of teleop time)?** This directly targets the "minimal
adequate mechanism" question this project has repeatedly found decisive elsewhere (Stage 12's wrist-fix
beating CR-CFM outright is the closest in-project precedent) — and Sim-and-Real Co-Training's own result
(cheap augmentation already helps at 40 demos) makes it a live, real possibility that the answer here is
also "no, the simple baseline wins," which would itself be a legitimate, honest, publishable finding
consistent with this project's pattern.

## Recommended staged pilot (cost-gated, cheapest-first — do not skip Stage 0)

- **Stage 0 (near-zero cost, no cloud GPU, no hardware wait)**: implement MimicGen-style trajectory
  splicing on the EXISTING Piper sim trajectory data (Cracker 127-152 trajectories, Pear 143) as the cheap
  baseline augmentation method. Measure whether it improves a simple policy/reranker's performance at
  simulated low-data budgets (subsample to 20-50 "real" trajectories, treat the rest as unavailable). This
  can start today with zero new spend and directly tests whether the cheap mechanism alone already captures
  the available benefit in this team's own data — if it does, that is the answer, and a learned world model
  is not justified.
- **Stage 1 (cloud GPU, still no real hardware needed)**: only if Stage 0 leaves clear headroom (the cheap
  baseline does NOT capture all the benefit), train a lightweight depth-conditioned dynamics model on
  existing sim data (a single rented A100 40GB is sufficient for a model at this scale, not H100/multi-GPU)
  and compare imagined-rollout augmentation against Stage 0's baseline, still entirely in simulation.
- **Stage 2 (real hardware, only after Stage 1 shows a positive gap)**: collect a small real Piper dataset
  (tens of demos, 1-2 objects) using the depth camera for RGB-D capture, fine-tune the Stage 1 dynamics
  model, and run a paired real-hardware comparison (baseline vs. MimicGen-splicing vs. world-model
  augmentation) following this project's own established McNemar's-test paired-trial convention.
- **Do not proceed to Stage 1 or 2 if Stage 0 shows the cheap baseline already captures the benefit** — that
  outcome is itself the (honest, low-cost) answer, and building the more expensive pipeline anyway would
  repeat exactly the "did not survive contact with team resources" mistake that closed the original
  Direction 3.

## Stage 0 result (2026-07-22): directionally promising but NOT yet confirmed — n=8 is exactly the scale this project has repeatedly shown can mislead

**Design**: fixed one reproducible random 40-trajectory Pear subset (seed=42, copied out of the 143 available
so both conditions train from the IDENTICAL data). Trained two `CRFlowNet` checkpoints via `train.py`'s
existing `train()` function, differing ONLY in `augment_subsegments` (already-implemented, already-used-by-
every-checkpoint-this-session flag: `True` resamples 240 sub-segments from the same 40 raw trajectories,
`False` uses the 40 raw trajectories directly, no other change). Evaluated both on the established held-out
Pear range (trial_id 3000-3007, wrist_friendly_orientation=True, 3-repeat majority vote — identical
methodology to every other paired comparison this session).

```
trial=3000: aug=True succeed   aug=False fail    <- discordant, favors augmentation
trial=3001: aug=True succeed   aug=False succeed
trial=3002: aug=True succeed   aug=False succeed
trial=3003: aug=True succeed   aug=False fail    <- discordant, favors augmentation
trial=3004: aug=True succeed   aug=False succeed
trial=3005: aug=True FAIL      aug=False succeed <- discordant, favors NO augmentation
trial=3006: aug=True succeed   aug=False fail    <- discordant, favors augmentation
trial=3007: aug=True fail      aug=False fail
```

**augment_subsegments=True: 6/8 (75%) vs. augment_subsegments=False: 4/8 (50%)**. 4 discordant trials, 3
favoring augmentation and 1 against. McNemar's exact test: **p=0.625 — not significant**, but this project
has repeatedly found that n=8 alone is not a reliable signal in either direction (Stage 12's own early
4-range result looked promising at small n and needed scaling to n=152 before the true, much weaker effect
was confirmed; several other stages saw n=8 "wins" evaporate at larger samples). The direction here (3:1
discordance favoring augmentation, 75% vs 50%) is a real, non-trivial effect size if it holds, and would
independently reproduce Sim-and-Real Co-Training's finding (cheap, training-free augmentation helps even at
~40 real demos) in this project's own codebase — but it has not yet cleared this project's own bar for
"confirmed," which has consistently required scaling past n=8 before trusting a direction.

**Recommendation, per the pre-registered Stage 0 gate**: this result is too promising to close the
augmentation angle here, but too small to justify Stage 1's cloud-GPU spend yet. The correct next step is
still cheap and still needs no cloud GPU: run 1-2 more 8-trial confirmatory ranges (same fixed 40-trajectory
subset, same two checkpoints already trained — no retraining needed, just more evaluation trials) to reach
at least n=16-24 before deciding whether to (a) escalate to genuine cross-trajectory MimicGen-style
splicing as Stage 0.5, or (b) treat this as a real but modest effect from within-trajectory resampling alone
and stop there. Do not skip straight to cloud GPU spending on the strength of an n=8 result — that is
precisely the mistake this project's own history (Stage 12, among others) has shown costs more than it
saves.

**Files**: `scratchpad/stage0_augment_pilot.py`, `scratchpad/stage0_pear_augTrue.pt`,
`scratchpad/stage0_pear_augFalse.pt`, `scratchpad/stage0_pear_subset40/` (the fixed 40-trajectory subset,
seed=42).

## Stage 0 CONFIRMATORY result (2026-07-22): the n=8 signal did NOT hold up — reversed at n=24, essentially a coin flip

Ran two more 8-trial ranges (3100-3107, 3200-3207) on the SAME two already-trained checkpoints, no
retraining, exactly per the plan above.

```
Range          augment=True   augment=False   discordant (favor True / favor False)
3000-3007      6/8            4/8             3 / 1
3100-3107      4/8            6/8             1 / 3
3200-3207      3/8            4/8             0 / 1
POOLED n=24    13/24 (54.2%)  14/24 (58.3%)   4 / 5
```

**McNemar's exact test on the pooled n=24: p=1.0000.** The direction from the original n=8 (75% vs 50%,
3:1 discordance favoring augmentation) did not merely weaken, it **reversed** (augment=False now slightly
ahead, 58.3% vs 54.2%), and the discordant-pair split is now essentially a coin flip (4 vs 5). This is
exactly the pattern this project has hit before (Stage 12's own early promising small-n result) and the
reason the pre-registered gate required confirmatory ranges before escalating.

**Verdict: negative. This specific augmentation mechanism (within-trajectory subsegment resampling,
`augment_subsegments=True`) shows no real benefit at a 40-trajectory Pear budget.** The original n=8 result
was noise, not signal. Per the pre-registered decision rule, this closes the cheap Stage 0 test WITHOUT
escalating to genuine cross-trajectory MimicGen-style splicing (Stage 0.5) or any cloud GPU spend — there is
no confirmed positive effect here to build on top of. This is itself an honest, useful, zero-cost finding:
it means Sim-and-Real Co-Training's low-data-regime result (cheap augmentation helps at ~40 real demos) does
NOT trivially transfer to this project's own architecture/task via the weakest, cheapest form of
augmentation already available — a stronger augmentation mechanism (real cross-trajectory splicing) MIGHT
still work, but this result gives no evidence either way for that stronger version, and does not by itself
justify building it before running that specific, still-cheap test if the direction is revisited later.

**Files (confirmatory)**: `scratchpad/stage0_augment_confirm.py`, `scratchpad/stage0_confirm_3100.log`,
`scratchpad/stage0_confirm_3200.log`.
