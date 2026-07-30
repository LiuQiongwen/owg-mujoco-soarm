# Object-Centric Counterfactual Critics for Robust Grasp Candidate Selection

**[SUPERSEDED as a standalone submission target, 2026-07-30 — kept as source material, do not
resume drafting this as its own paper.]** Decision: merge into `TRO_PAPER_OUTLINE.md` §4.5 as a
third, prospective application of the causal-validity criterion, alongside §4.2-4.3's retrospective
cross-embodiment case (avoids submitting the same core methodology as two separate T-RO papers).
The Abstract/Contributions/section skeleton below remain useful as the source draft for §4.5's
prose and are not to be deleted, but the actual writing target is now `paper_tro.tex`/
`paper_tro_draft.md`, not a LaTeX conversion of this file. See `TRO_PAPER_OUTLINE.md`'s §4.5 for
the merged, condensed version of everything below.

**[Draft — branch `paper/risk-gated-vla-draft`, frozen at tag `risk-gated-vla-frozen-20260730`]**

Working alternate title (more ambitious, not yet earned by the evidence in this draft — the risk
gate and VLA integration are negative/incomplete results, not supporting claims): *Causally Valid
Object-Centric World Critics for Risk-Aware VLA Grasp Selection*. Recommend keeping the
conservative title until/unless a future hardware or VLA extension actually demonstrates the
risk-aware/VLA claim — see Limitations and the Hardware Extension section for why.

---

## Abstract

A learned critic that reranks grasp candidates before execution is only as good as the causal
validity of its training labels and the fairness of the evaluation that measures it. We show both
can silently break, and how to catch it. A pre-execution grasp critic trained on simulated MuJoCo
outcomes initially appeared to improve top-1 grasp success by +15.6 percentage points over a
geometric heuristic baseline — a result that collapsed under scrutiny: the evaluation harness
encoded the compared method into its random seed, so "geometry" and "world-model" trials never
shared a candidate pool or scene, and the underlying success label counted mere post-close gripper
contact as success regardless of whether the object was ever lifted. We formalize a causal-validity
audit for this class of pipeline, rebuild the evaluation as a shared-candidate-pool paired design
with a production, bilaterally-gated physics grasp primitive, and find the original critic's
success probability carries no real signal at all (AUROC = 0.4996 against 1,500 real per-candidate
outcomes — chance level). We then train an **object-relative counterfactual critic** — pose
features expressed relative to the target object, plus point-cloud statistics and object identity,
trained with a within-scene Bradley-Terry-style pairwise loss on causally-admissible features only
— and evaluate it on two disjoint, held-out scene batches never touched by training or model
selection. The corrected critic significantly outperforms the geometric baseline on both: +15.6pp
(McNemar exact p=0.00258, n=90) on an independent development-test batch, and +14.0pp (27 paired
wins vs. 6 losses, exact McNemar p=3.24e-4, n=150) on a frozen confirmatory batch never inspected
before the gate was evaluated. We report negative and incomplete results with the same rigor:
an ensemble-uncertainty risk gate calibrated on held-out data adds no measurable benefit over the
ungated critic; whether a pairwise loss term is independently responsible for the gain (versus
object-relative features alone) is not established; and a small-data (15-demonstration) imitation
policy pilot integrates end-to-end but fails its first closed-loop rollout. We release a
causal-validity audit tool, the paired-evaluation harness, and a real-hardware validation protocol
as a basis for future work.

## Contributions

1. **A causal-validity audit protocol for pre-execution grasp critics**, extending and applying an
   existing formal PRE_EXECUTION-admissibility criterion (feature provenance registry +
   automated static-analysis tagger) to a new pipeline, and using it to catch — before any paper
   claim was made on top of it — a stale checkpoint whose apparent signal was chance-level
   (AUROC=0.4996) once causal and pairing defects were corrected.
2. **Diagnosis and correction of two compounding evaluation defects** in a predecessor pipeline:
   (a) a random-seed scheme that silently coupled the compared method identity into scene/candidate
   sampling, defeating the paired design its own statistics assumed; (b) a success criterion that
   counted transient gripper contact — checked before any lift — as success, saturating measured
   performance independent of grasp quality. Both are shown, not merely asserted, via direct
   reproduction against the original data.
3. **An object-relative counterfactual grasp critic** that is causally admissible by construction
   (every input feature traceable to pre-execution scene/candidate state) and significantly
   outperforms a geometric heuristic on two independent, disjoint, held-out MuJoCo test batches —
   the paper's central positive result.
4. **A fully itemized negative/incomplete-results record**, produced by the same statistical
   discipline as the positive result, not omitted or reframed: an uncertainty-based risk gate with
   no measurable benefit, an unresolved ablation (pairwise loss's independent contribution), and a
   small-data imitation-learning pilot that fails online despite offline metrics looking reasonable.
5. **A real-hardware validation protocol**, explicitly scoped as future work (Section 7): staged
   pilot-then-scale-up design, safety checklist, and a corrected object choice (the protocol's own
   design review found the originally-proposed test object was out-of-distribution for the trained
   critic and substitutes an in-distribution pair instead) — not yet executed.

## 1. Introduction

*(To draft: motivate why pre-execution grasp candidate scoring is a useful, common pattern
[Dex-Net-style analytic scorers, GraspGen, learned rerankers]; state the paper's actual thesis —
that the causal-validity and evaluation-pairing failure modes shown here are a general risk for
this whole model family, not a one-off bug in one project's code; preview the audit-then-rebuild
narrative arc that structures the paper, mirroring the honest "we found our own result was wrong,
here's how, here's the corrected version" arc already established as this project's house style
in the T-RO causal-validity work.)*

## 2. Related Work

*(To draft — sections to fill, do not fabricate citations here; run a literature pass before
writing prose)*
- Learned grasp-candidate scoring / reranking (Dex-Net, GraspNet-1Billion, GraspGen, 6-DOF
  GraspNet) — where do their pipelines sit relative to the PRE_EXECUTION-admissibility criterion?
  (GraspGen already checked as a compliant reference case in `CAUSAL_VALIDITY_METHOD.md` — reuse
  that analysis here rather than re-deriving it.)
- Data leakage / evaluation-protocol failure modes in offline RL and imitation learning
  (the arXiv:2605.11479-class prior art already identified in `CAUSAL_VALIDITY_METHOD.md`).
- Risk-aware / uncertainty-gated action selection in manipulation and VLA policies — position this
  paper's negative risk-gate result honestly against any prior work claiming a positive gating
  effect; do not imply this result generalizes beyond the ensemble-disagreement gate tested here.
- Sim-to-real transfer for grasp success prediction — motivate Section 7 as future work, not a
  claim.

## 3. Method

### 3.1 Causal-validity criterion (reused, not reinvented)

Restate Definitions 1-3 and the audit algorithm from `CAUSAL_VALIDITY_METHOD.md` at the level of
generality needed for this paper (candidate pool, selection-time information set, PRE_EXECUTION
admissibility, the labels-are-exempt corollary). Cite as this project's own prior formalization;
this paper's contribution here is *application and extension* (registering a new pipeline's fields,
Section 3.2), not the criterion itself.

### 3.2 Object-relative counterfactual critic

- Feature set: candidate pose expressed relative to the detected/simulated object position, sin/cos
  yaw, gripper opening, 9-dim point-cloud statistics, object-identity one-hot (3 classes:
  CrackerBox/MustardBottle/PowerDrill). All PRE_EXECUTION-admissible per the Section 3.1 criterion
  — state this as a checked property (registered in `causal_validity_audit/provenance.py`), not an
  assumption.
- Architecture: 2-hidden-layer MLP (64/64, SiLU, dropout 0.05), 5-seed ensemble.
- Loss: binary cross-entropy on the per-candidate ground-truth outcome, plus (for the
  `object_counterfactual` variant) a within-scene Bradley-Terry/BPR-style pairwise term comparing
  successful vs. failed candidates from the *same* scene — the "counterfactual" framing: what would
  have happened had a different candidate in the same, already-observed scene been chosen.
- Training/validation split: scene-grouped (never split a scene's candidates across train/val),
  stratified per object, reproducible per seed.

### 3.3 Paired candidate-selection evaluation harness

- Per-scene protocol: build one candidate pool (object placement + K sampled grasp poses) before
  any method is chosen; every compared method executes against a *fresh, identically re-placed*
  copy of the same scene, sharing the same pool. This directly fixes the predecessor pipeline's
  seed-coupling defect (Section 4.1).
- Production execution primitive: the same bilaterally-gated, weld-based physics grasp routine
  used elsewhere in this project's simulation stack (not a bespoke, more lenient reimplementation —
  Section 4.2 shows why that distinction matters).
- Statistics: McNemar's exact test (paired), bootstrap CIs, per-object and pooled effects; gates
  pre-registered before any result was inspected (`preregistration.yaml`).

## 4. Diagnosing the Predecessor Pipeline (methodology-as-result)

This section is itself a result, not just setup — structure it as such, matching this project's
established house style of reporting a self-correction as a first-class finding rather than
suppressing it.

### 4.1 The seed-coupling defect

Show the exact seeding formula, the analytic proof it depends on method identity, and the direct
empirical confirmation (0/250 paired trials shared even the same executed candidate position).

### 4.2 The success-criterion defect

Show the legacy `contact or grasped or lifted` criterion, its saturation (150/150 sampled
candidates "succeeded" regardless of pose, across 2 objects/15 scenes), and the corrected
criterion (bilateral contact + verified post-settle lift, via the production primitive).

### 4.3 Consequence

The originally-reported effect (+15.6pp, "world model" vs. geometry) is unsupported by either the
pairing or the underlying success labels. Reframe explicitly: this is why the paper's positive
result (Section 5) is built on a from-scratch corrected pipeline, not a patch to the original one.

## 5. Results

### 5.1 Stale-checkpoint gate (negative, reported first — matches this project's own
pre-registered-gate discipline)

The original critic checkpoint, re-evaluated under the corrected pipeline, fails the pre-registered
gate: pooled effect −14.0pp (wrong sign vs. the required ≥+8pp), AUROC=0.4996 on 1,500 real
per-candidate labels (chance level). Full numbers: `results/risk_gated_vla/phase1/RESULT.md`.
Explain *why* (Section 4.2's success-criterion defect contaminated its training labels too) —
this is not a mysterious failure.

### 5.2 Object-relative counterfactual critic — the positive result

| Batch | Base seed | n | Geometry | Critic | Δ | Exact McNemar |
|---|---:|---:|---|---|---:|---:|
| Independent development test | 200 | 90 | 30/90 (33.3%) | 44/90 (48.9%) | +15.6pp | p=0.00258 |
| Frozen confirmatory test | 300 | 150 | 54/150 (36.0%) | 75/150 (50.0%) | +14.0pp | p=3.24e-4 (27W/6L) |

Both batches: scene keys disjoint from training (base seed 100) and from each other (independently
verified, not merely asserted — pairwise key-intersection checks reported in
`final_report.md`). Per-object breakdown, the marginal-grasp physics-nondeterminism footnote
(~0.6-1% flip rate on repeated identical-pose execution), and the live-executed-vs-offline-rescore
reporting convention are documented in full in `final_report.md` and should be reproduced here
verbatim when this section is finalized, not re-derived from memory.

### 5.3 Ablation: is the pairwise loss load-bearing?

Object-relative BCE alone (no pairwise term) is statistically indistinguishable from the full
`object_counterfactual` variant on the frozen confirmatory batch (2 wins/3 losses, p=1.0). State
this as an open question, not a negative result about pairwise losses in general — n is small
for this specific ablation.

## 6. Limitations

Structure as a direct list, matching `final_report.md`'s negative-results archive (condense to
the paper-relevant subset; the full archive with all 8 items stays in the results directory as
supporting material, not all of it needs full prose treatment in the paper body):

1. Risk gate: no measurable benefit over the ungated critic (coverage 98.7% — the gate rarely
   actually fires). State plainly; do not claim a gating contribution.
2. Pairwise-loss contribution unresolved (Section 5.3).
3. Simulation-only: no real-hardware validation in this draft (Section 7 states the plan).
4. Small object set (3 YCB objects); generalization to a broader object distribution untested.
5. Physics-level non-determinism on marginal grasps (~0.6-1% outcome-flip rate on repeated
   identical execution) — a measurement-noise floor on any single-run success-rate estimate in
   this simulator, disclosed rather than hidden.
6. Imitation-learning integration (ACT) is a working pipeline, not a validated capability — first
   online rollout failed; 5 demonstrations/object is not enough data to expect closed-loop
   robustness. Report as integration status, not as a result bearing on the paper's central claim.

## 7. Hardware Extension (future work, explicitly not executed in this draft)

Summarize `results/risk_gated_vla/PHASE3_REAL_HARDWARE_PROTOCOL.md`: platform (SO-ARM101 +
RealSense D435i, already connected/verified this project cycle), the corrected object choice
(CrackerBox + MustardBottle, in-distribution for the trained critic — the protocol's own design
review caught that the originally-proposed Pear was out-of-distribution and is not used), staged
pilot-then-scale-up design with a pre-registered go/no-go gate, and the full safety checklist.
State plainly that Stage 0 (camera repositioning) is an unresolved physical-setup blocker shared
with `LINE_B_EXPERIMENT_PLAN.md`, and that no result from this section exists yet.

## 8. Conclusion

*(To draft last, once Sections 1-7 are stable — should restate the Abstract's claims at paper
length, explicitly reiterating what is and is not demonstrated, matching this project's established
discipline of not letting the conclusion overclaim relative to the results section.)*

---

## Drafting notes (remove before submission)

- Source of truth for every number in this draft: `results/risk_gated_vla/final_report.md` and
  `audit.md`. If a number here and a number there ever disagree, `final_report.md` wins — it was
  independently re-verified from raw `scenes.jsonl`, this draft was not (yet).
- Do not add real-hardware numbers to Section 5 or the Abstract until Section 7 actually produces
  them — this is the exact discipline this paper's own Section 4 is about.
- Venue/template not yet chosen. This project's existing LaTeX infrastructure
  (`interact` class, used for `paper_tro.tex`/`paper_advanced_robotics.tex`) is available and
  known-working if a T-RO/Advanced-Robotics-style venue is chosen; RA-L's template
  (`paper_final.tex`) is available if that route is preferred instead. Confirm venue before
  converting this markdown draft to LaTeX, since page limits materially affect how much of
  Section 2/4/6 survives.
