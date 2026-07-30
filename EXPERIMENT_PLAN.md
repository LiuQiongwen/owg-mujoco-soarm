# Experiment Plan: Cross-Embodiment Grasp Candidate Reranking

Follows `IDEA_REPORT.md`'s Direction 2. Everything in that report so far was
retrospective/offline pilot work (logistic-regression proxies on already-
collected logs, evaluated by AUC or pairwise accuracy). This plan moves
from "does the signal exist in principle" to a paper-grade experiment: the
real LGGSN architecture, a properly-sized dataset, and — critically — a
**live closed-loop physical execution test**, not just offline metrics.

## What's already established (do not re-litigate, build on it)

1. **Training-free heuristic (consensus selection) does not transfer across embodiments** — validated on real physical execution data, two independently-designed noise models, both negative.
2. **Zero-shot transfer of even a learned model is bad** — pairwise accuracy 0.35-0.45 (at/below the 0.50 chance baseline), confirmed 4 independent times across different feature sets and training paradigms (pointwise logistic regression and genuine BPR pairwise).
3. **Pooled/joint training across both platforms robustly beats zero-shot** — p<0.0001 in three well-powered pointwise tests (n=60, n=150 with two feature sets), the single most reproducible finding in this whole direction.
4. **Whether explicit embodiment conditioning (beyond pooling) adds value is unresolved** — flipped sign across pilot configurations, genuinely needs more data, not another small pilot.

## Paper narrative this plan is designed to support

Not "a new algorithm" — an honest, rigorous **cross-embodiment empirical study** with a practically useful takeaway: *when deploying a validated grasp reranker to new hardware, pooling a modest amount of real execution data from the new platform into joint training recovers transfer performance that a training-free heuristic cannot recover at all, and zero-shot transfer of the learned model alone cannot recover either.* This is the kind of finding T-RO/IJRR values — grounded in two real owned platforms, statistically rigorous, with a clear practitioner-facing conclusion — without requiring a novel model architecture to be the contribution.

Secondary claim (if Stage 2 resolves it): the *right* way to express embodiment conditioning (interaction/gating vs. naive additive) matters, with additive consistently shown to add nothing (3-for-3 null so far) and interaction-based conditioning still an open question.

## Stage 0 — Infrastructure (prerequisite, ~1-2 days)

- [ ] Extend the real LGGSN model/training code (`lggsn_model.py`, `train_lggsn_pairwise.py`) to accept an optional embodiment-conditioning input, with three modes selectable at train time: `none` (current behavior), `additive` (embodiment one-hot concatenated to features), `interaction` (embodiment one-hot + embodiment × feature product terms — i.e. a FiLM-style gate). Do **not** build a new model from scratch; this should be a small, reviewable diff to the existing validated architecture.
- [ ] Finalize the shared cross-embodiment feature set. Current proxy is 5-dim (z, yaw, H, quality_score, correction_proxy); decide whether to keep it at 5 or invest in deriving `dz`/`dz_lift`/`need_dz` Piper-equivalents properly (lower priority — these were found to be near-degenerate even in the SO-ARM101 source data) or `dist_to_centroid`/`z_rel` (only possible once full-pool execution data exists per scene, which Stage 1 below produces as a side effect).
- [ ] Promote `collect_pairwise_piper.py` (candidate_selection=int mode) from a scratchpad script into the `piper_robosuite/` package proper, since Stage 1 depends on running it at scale, repeatedly, by anyone continuing this work.

## Stage 1 — Data collection (smoke-test-then-scale, matching project convention)

**Smoke test (done)**: 3 Cracker scenes × 10 candidates = 30 trials, all 3 scenes mixed-label. Confirms the collection pipeline works end to end.

**Scale-up target**: this is a real compute/time budget decision, not free — full-pool execution is ~10x the cost of a single-candidate trial. Two honest options, pick one explicitly rather than drifting into it:

- **Option A — breadth**: more objects, fewer scenes each (e.g. 5 scenes × 10 candidates × {cracker, mustard, pear} = 150 pairwise-labeled trials). Better for the "does this generalize across objects" question, worse statistical power per-object for the conditioning question.
- **Option B — depth**: fewer objects, many more scenes (e.g. 20-30 scenes × 10 candidates on Cracker alone = 200-300 trials). Better power specifically for resolving the still-open conditioning question (Stage 2's leave-scene-out CV needs more than 3 folds to mean anything), worse for generalization claims.

**Recommendation**: Option B first, on Cracker (the object with the most informative mixed-label rate observed so far, ~20-50% success — neither floor nor ceiling). Resolve the conditioning question on one object properly before spending the 10x-cost budget breadth-wise on objects where the answer might just be "same as Cracker." If Stage 2 shows a real, well-powered conditioning effect on Cracker, THEN spend the breadth budget checking it replicates on Mustard/Pear.

- [x] Collect ~20-30 Cracker scenes × full 10-candidate pools — **done, 25 scenes / 250 trials** (scenes 900-923 + 950), via the promoted `piper_pairwise_collector.py`, run in parallel batches of 3 scenes/call.
- [x] Verify every scene is mixed-label — done; all 25 used scenes had both successes and failures.

## Stage 2 — Offline/retrospective validation (cheap, do before Stage 3) — DONE (2026-07-16)

- [x] Train real `EmbodimentLGGSN` (all 3 conditioning modes) on pooled SO-ARM101 + the Stage 1 Piper dataset — `piper_robosuite/stage2_train_embodiment_lggsn.py`.
- [x] Leave-scene-out CV on the Piper side — 22 folds (2 of the 25 collected scenes were single-label after final counting and excluded, matching the "verify mixed-label" step above).
- [x] **Go/no-go gate result: FIRED, negative.** `additive` p=0.198, `interaction` p=0.769 vs. `none` — neither beats plain pooling. Conditioning is dropped from the paper's claims per the plan's own pre-registered rule. Headline result: pooled training beats zero-shot, p=0.0001 (+0.144 pairwise accuracy) — the real-architecture confirmation of the finding every pilot in `IDEA_REPORT.md` pointed to.

## Stage 3 — Live closed-loop physical execution test (the actual paper-grade experiment)

This is the step none of the pilots so far have done: **using the trained reranker to actually pick which candidate gets executed**, then measuring physical success rate — not offline AUC on historical logs. This is what would need to appear in the paper as the headline result.

**Design** (matching this project's established paired-trial convention throughout — see `piper_consensus_experiment_runner.py` for the pattern):

- Fresh, held-out set of Piper trial_ids (not used in Stage 1/2 training).
- For each trial_id: generate ONE candidate pool (`sample_candidate_pool`, same seeding convention as before), then evaluate every compared strategy on the **same pool** (paired design):
  1. Random pick (floor baseline)
  2. IK-error-best pick (the existing "ikmargin"-style baseline)
  3. Consensus pick (training-free heuristic — known-negative baseline, included to make the transfer-failure claim visually concrete in the paper, not to re-litigate it)
  4. Reranker pick, zero-shot (SO-ARM101-only trained)
  5. Reranker pick, pooled (no conditioning) — ~~6. conditioning variant~~ dropped, Stage 2's gate fired negative
- Execute the SELECTED candidate through the full physical pick-and-place pipeline (`run_pick_and_place`), record success/failure.
- n=20-30 trial_ids minimum per object (matching this project's established pilot-then-scale sizing), on Cracker first, then whichever additional objects Stage 1 covered.

**Statistics**: McNemar's exact test (`scipy.stats.binomtest`) for each pairwise strategy comparison, matching every other paired-trial comparison in this project's history. Report per-object AND pooled results; do not average across objects without first checking the effect direction is consistent (an established project practice after `paperA_data`'s conventions).

## Stage 4 — Ablations (only after Stage 3's headline result is in hand)

- **Data-efficiency curve**: retrain the pooled model with 10%, 25%, 50%, 100% of the Stage 1 Piper data, re-run Stage 2's offline eval at each size. Produces a practically useful "how much real data on a new platform do you actually need" curve — a genuine, novel, checkable empirical contribution independent of the conditioning question's outcome.
- **Feature ablation**: 3-feature vs 5-feature vs (if built) richer feature set, offline eval only (no need to repeat the expensive Stage 3 live test for this).
- **Cross-object generalization**: does a reranker pooled-trained on Cracker+SO-ARM101 transfer to Mustard/Pear on Piper without object-specific pooled data? (a natural, cheap extension question if Stage 1 collected multi-object data under Option A instead of B.)

## Explicit non-goals for this plan

- Not attempting to resolve `dist_to_centroid`/`z_rel` feature parity beyond what Stage 1's full-pool collection provides for free.
- Not repeating the world-model/VLA/AR direction (Direction 1) — closed, negative, documented.
- Not claiming a novel model architecture — the contribution is the empirical cross-embodiment study and (if it survives Stage 2's gate) the conditioning-mechanism finding, not the LGGSN extension code itself, which is a small, incremental diff.

## Status (2026-07-16): CLOSED at Stage 2 — see `IDEA_REPORT.md`'s "CORRECTION" section, do not proceed to Stage 3

Stages 0-2 were executed and initially reported a strong result (pooled training beats zero-shot, p=0.0001). While designing Stage 3 (which requires the reranker to select a not-yet-executed candidate), found that every feature used up to that point (`score`/`need_dz` on SO-ARM101, `quality_score`/`correction_proxy` on Piper) is **execution-derived** — not knowable before a candidate has already been run — confirmed by this project's own `train_geo_ebm_grasp.py` header comment, which documented the identical problem on the SO-ARM101 side previously. Rerunning Stage 2 with only genuinely pre-execution-valid features (`z`, `yaw`, `H`) made the entire effect disappear (zero-shot = pooled, p=1.0000). A follow-up check using a legitimately valid feature (`score_candidate_ik`, the same pure kinematic check `select_best` already uses) found essentially zero correlation with success (0.06) even within Piper's own data alone.

**Stage 3 will not be run.** There is no validated selection mechanism to test — building the expensive live-execution comparison on top of a model that offline analysis now shows has no real pre-execution predictive signal would not produce a meaningful result. Direction 2 (cross-embodiment grasp reranking) is closed as a research direction, honestly negative: the real determinant of grasp success (execution-time contact dynamics during descend) is not accessible to any pre-execution candidate-selection model by construction, regardless of embodiment, training scheme, or conditioning mechanism. See `IDEA_REPORT.md` for the full writeup and the connection back to this project's earlier, independent root-cause finding (Piper README) that pointed at the same underlying cause from a completely different angle.
