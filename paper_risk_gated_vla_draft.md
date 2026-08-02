# Object-Centric Counterfactual Critics for Robust Grasp Candidate Selection

**[ACTIVE standalone submission target, reinstated 2026-08-02 — the 2026-07-30 merge-into-T-RO
decision below is reversed.]** This paper is being drafted and converted to LaTeX
(`paper_risk_gated_vla.tex`) in its own right, not folded into `paper_tro.tex`. Rationale for the
reversal: T-RO is already at its 8-page free limit after the LGGSN ablation subsection (§IV-G) was
added, leaving no room to do this material justice as a subsection; this paper's own arc (a critic
that initially audited as chance-level, AUROC=0.4996, then was rebuilt and validated with a real,
twice-replicated effect on disjoint held-out batches) is a complete, self-contained empirical story
that reads as filler when compressed into one subsection among several; and two independently
verified 2026 papers (arXiv:2606.04233, arXiv:2605.18045 — see Related Work) place this exact
evaluation-rigor narrative inside a live, currently-active discussion worth targeting directly.
The PRE_EXECUTION-admissibility criterion itself remains T-RO's own contribution (§4.2-4.3 there);
this paper cites that as prior in-project work and contributes a second, deeper case study plus a
new critic architecture on top of it (see Related Work below for how the two are positioned to
avoid overlap). See `TRO_PAPER_OUTLINE.md`'s now-struck-through §4.5/§4.6 for the historical merge
plan and for §4.6's mechanistic-analysis content, which still needs porting into this draft's
Results section before LaTeX conversion (tracked separately).

**[Draft — branch `paper/risk-gated-vla-draft`, frozen at tag `risk-gated-vla-frozen-20260730`]**

Working alternate title (more ambitious, not earned by the evidence in this draft — decided, not
just "not yet," as of 2026-08-02): *Causally Valid Object-Centric World Critics for Risk-Aware VLA
Grasp Selection*. The risk gate and VLA integration are negative/incomplete results, not
supporting claims (Section 6), and the pairwise-loss mechanism the "world critic" framing would
lean on is quantifiably unresolvable at the current data scale (Section 5.3: ~210 discordant pairs
needed for 80% power, 5 observed). Keep the conservative title unless/until a future hardware or
VLA extension, or a genuinely larger-scale ablation, actually earns the risk-aware/VLA claim.

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
4. **A complementary, explicitly offline mechanistic analysis** (Section 5.4) decomposing the same
   critic into contact/lift/success/failure-type heads to explain *where* it works (PowerDrill is
   measurably harder than CrackerBox/MustardBottle) and *why* (a specific, actionable
   success/no_contact confusion pattern) — kept clearly separate from the live-executed primary
   result, not merged with it.
5. **A fully itemized negative/incomplete-results record**, produced by the same statistical
   discipline as the positive result, not omitted or reframed: an uncertainty-based risk gate with
   no measurable benefit, an unresolved ablation (pairwise loss's independent contribution), and a
   small-data imitation-learning pilot that fails online despite offline metrics looking reasonable.
6. **A real-hardware validation protocol**, explicitly scoped as future work (Section 7): staged
   pilot-then-scale-up design, safety checklist, and a corrected object choice (the protocol's own
   design review found the originally-proposed test object was out-of-distribution for the trained
   critic and substitutes an in-distribution pair instead) — not yet executed.

## 1. Introduction

Pre-execution grasp candidate scoring — training a model to rank a pool of not-yet-executed grasp
poses so the best one can be selected before any physical or physically-simulated commitment — is
a common pattern across analytic scorers [Mahler et al., RSS 2017], learned discriminators
[arXiv:2507.13097], and geometric rerankers, including this project's own LGGSN (companion T-RO
work). Its correctness rests on two assumptions that are easy to state and easy to violate
silently: every input feature must be computable *before* the chosen candidate executes (causal
validity), and comparing two scoring methods requires evaluating them against the same candidate
pool under the same conditions (paired evaluation). Neither violation is visible from an aggregate
accuracy or success-rate number alone — both look like a real result until traced to source.

We hit both violations in the same pipeline. A pre-execution grasp critic ("world model"), trained
on simulated MuJoCo outcomes, was originally reported to beat a geometric heuristic baseline by
+15.6 percentage points. Auditing this claim before building on it — applying this project's own
PRE_EXECUTION-admissibility criterion and provenance registry (companion T-RO work) to a new
pipeline — surfaced two compounding defects, neither itself a feature-provenance violation: the
evaluation harness's random seed encoded which method was under test, so "geometry" and "critic"
trials never executed against the same scene or candidate pool despite the paired statistics
assuming they did; and the success label counted transient post-close gripper contact, checked
before any lift, regardless of whether the grasp actually succeeded. Re-evaluated under a
from-scratch, causally-admissible, genuinely paired harness, the original checkpoint's apparent
signal is AUROC=0.4996 against 1,500 real per-candidate outcomes — exactly chance.

Rather than discard the direction, we rebuilt it. An object-relative counterfactual critic — pose
features expressed relative to the target object, point-cloud statistics, and object identity,
every field registered PRE_EXECUTION-admissible *before* training, not after — significantly
outperforms the geometric baseline on two independent, disjoint, held-out scene batches, one of
them a frozen confirmatory batch never inspected before its pre-registered gate was evaluated. We
report this result with the same statistical discipline that caught the original claim's
invalidity, and we report, with equal weight, what this investigation could not establish: an
uncertainty-based risk gate with no measurable benefit, a pairwise-loss ablation quantifiably not
resolvable at the current evaluation scale (Section 5.3), and a small-data imitation-learning
integration that has not yet demonstrated closed-loop robustness.

This paper is organized around that arc — audit, diagnose, rebuild, re-verify — mirroring this
project's companion T-RO investigation into the same class of failure in a different pipeline. We
do not claim the specific defects found here are universal; we claim the failure mode — an
aggregate metric that cannot distinguish a causally valid, fairly evaluated result from an invalid
one — is a structural risk for any pipeline in this family, and that catching it requires checking
an evaluation's construction, not only its output.

## 2. Related Work

*(Literature pass completed 2026-08-02 — every citation below was independently fetched and
verified against its arXiv/OpenReview/NeurIPS page before inclusion, not taken from a search
snippet alone. `2605.11479` in particular is NOT the leakage paper the earlier placeholder assumed
it was — see that paragraph below; the placeholder's characterization was wrong and should not be
propagated into `CAUSAL_VALIDITY_METHOD.md`'s own citation of the same arXiv ID without a similar
correction there.)*

**Learned grasp-candidate scoring and reranking.** Dex-Net~2.0 [Mahler et al., RSS 2017] and
GraspNet-1Billion [Fang et al., CVPR 2020] learn grasp-quality predictors from large
annotated/simulated datasets assuming full scene geometry at query time; 6-DOF GraspNet
[Mousavian, Eppner, and Fox, ICCV 2019] extends this to a variational proposal generator. None of
the three states an input/label admissibility distinction as a formal, checkable criterion.
GraspGen [arXiv:2507.13097] is the closest prior system to satisfy the PRE_EXECUTION-admissibility
criterion we apply here by construction: its discriminator's inference-time inputs are a
point-cloud encoding and candidate pose only, with execution-derived signal confined to training
labels. As in this project's companion T-RO work, we treat GraspGen as a confirmed-compliant
reference case, not a counterexample — external validation of the criterion, not prior art for it.

**Evaluation-protocol failures in offline policy/critic assessment.** Data leakage and unfair
offline comparison are recognized general risks in sequential-decision-making evaluation.
Concurrent work on offline policy evaluation for manipulation, "Offline Policy Evaluation for
Manipulation Policies via Discounted Liveness Formulation" [Wang, Bowden, Crosby, and Bansal,
arXiv:2605.11479], diagnoses a *different* integrity problem from ours: finite-horizon episodes
bias value estimates under classical TD/Monte Carlo evaluation (truncation bias), which they
address with a liveness-based Bellman operator, evaluated on VLA and diffusion-policy manipulation
tasks plus cloth folding. This is adjacent evidence that offline/simulated manipulation-policy
evaluation is error-prone in more than one structurally distinct way — not directly overlapping
prior art for the seed-coupling and success-criterion defects Section 4 diagnoses, and should not
be cited as a leakage paper (an earlier internal draft note mischaracterized it as one).
Independently, and closer in spirit to our diagnosis, "What Are We Actually Benchmarking in Robot
Manipulation?" [Jiang, Tan, Wheeler, Sun, Ayalew, and Walter, arXiv:2606.04233] audits LIBERO and
CALVIN and finds only 19.8% of LIBERO's reported advances are statistically significant once
evaluation noise is accounted for, alongside shortcut-solvable benchmark items and
train/test-distribution proximity effects that inflate apparent generalization. Their specific
failure modes (shortcut solvability, creeping overfitting, data-source dependence) are distinct
from the seed-coupling and success-criterion defects Section 4 diagnoses in our own pipeline, but
the broader concern is the same and, as of mid-2026, actively converging from multiple independent
directions: a robot-manipulation benchmark score is not self-certifying, and treating one as
evidence of general capability without auditing how it was produced is an increasingly recognized,
not a niche, risk. We position this paper as one concrete, fully-traced instance of that broader
concern, not as its sole discovery.

**Risk-aware and uncertainty-gated action selection in VLA policies.** ReconVLA [Chen, Lyu, and
Beksi, arXiv:2604.16677] applies conformal prediction directly to a pretrained VLA policy's action
tokens, yielding calibrated uncertainty estimates that correlate with execution quality, and
extends the same idea to state-space outlier detection as a failure-anticipation mechanism.
SafeVLA [Zhang et al., NeurIPS 2025 Spotlight, arXiv:2503.03480] frames safety alignment as a
constrained Markov decision process, optimizing against elicited unsafe behaviors and reporting an
83.58% reduction in cumulative safety-violation cost versus prior methods. Both report a positive
effect from their respective uncertainty/safety mechanism. Our own ensemble-disagreement risk
gate — calibrated on held-out data and evaluated with the same statistical rigor as our positive
result — shows no measurable benefit over the ungated critic (coverage 98.7%, Section 6). We do
not read this as evidence against uncertainty-gating in general: ReconVLA's per-token conformal
calibration and SafeVLA's constrained-optimization framing are both structurally different from a
post-hoc ensemble-disagreement filter applied to a pre-execution candidate scorer. Concurrent work
directly asking when uncertainty-gating helps, "Confidence-Gated Robot Autonomy" [Gaus, Charaja,
and Haeufle, ICRA 2026 workshop, arXiv:2605.18045], identifies a dataset-dependent *competence
regime*: below it, uncertainty rankings are weak and unstable regardless of which uncertainty
source (softmax, MC dropout, ensemble) is used; above it, simple proxies suffice and the gating
threshold matters more than the method. Our critic's ~50% pooled success rate (Section 5.2) may
simply not yet be in the regime their analysis identifies as necessary for a gate to add value —
we flag this as the most likely explanation for our null result rather than leaving it
unexplained, without claiming to have tested for the regime directly. We report our own result as
a specific negative finding for the specific gate design tested, with the same discipline as our
positive finding, not omitted or hedged into ambiguity.

**Sim-to-real transfer for grasp success prediction.** Out of scope for this paper's evidence base;
motivates Section 7's protocol as a stated future-work step, not a claim made here.

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
  (Note on terminology: this is a same-scene, within-episode alternative used as a *training
  signal* for the critic, not the post-hoc explanation or formal potential-outcomes sense of
  "counterfactual" common in the 2025-2026 causal-ML and explainability literature — the pairwise
  comparison never requires estimating an outcome under an unobserved intervention, only comparing
  two candidates both already scored against the same, already-observed scene.)
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
verified, not merely asserted — pairwise key-intersection checks reported in `final_report.md`).

**Per-object breakdown (live-executed, geometry / counterfactual).** Development test: CrackerBox
4/30 vs. 4/30 (tie), PowerDrill 7/30 vs. 18/30, MustardBottle 19/30 vs. 22/30. Frozen confirmatory:
CrackerBox 9/50 vs. 9/50 (tie, unchanged), MustardBottle 29/50 vs. 42/50, PowerDrill 16/50 vs.
24/50. The pooled gain is concentrated in PowerDrill and MustardBottle at both evaluation scales;
CrackerBox is flat at both — the critic never selects a different top-1 candidate than geometry
for this object, at either scale, not a near-miss that a larger sample might resolve.

**Live-executed vs. offline-re-scored reporting convention.** `geometry` and
`object_counterfactual` were the two methods actually live-selected during data collection, so
their reported numbers are live-executed paired outcomes — the real online trial, not a re-run.
`global_bce` and object-relative BCE (Section 5.3) were never live-selected in this collection run;
their numbers are offline re-scores against the same scene's fully-swept candidate ground truth.
This distinction matters for a subtler reason than convenience: MuJoCo's contact solver is not
perfectly reproducible on marginal grasps. Independently verified across the confirmatory batch's
300 live-vs-reswept comparisons, exactly 2 flipped (a ~0.67% marginal-grasp non-determinism rate;
the development-test batch shows the same pattern, 1 flip per method out of 90) — a genuine
physical-reproducibility floor on any single-run success-rate estimate in this simulator, not a
script bug, and the same class of finding independently documented in a companion pipeline
(~12.5% single-run instability, `paper_advanced_robotics.tex`). We report it as a stated
limitation rather than treating any single execution as ground truth.

### 5.3 Ablation: is the pairwise loss load-bearing?

Object-relative BCE alone (no pairwise term) is statistically indistinguishable from the full
`object_counterfactual` variant on the frozen confirmatory batch (2 wins/3 losses, p=1.0). This is
not merely "n is small" — it is quantifiably not resolvable by modestly scaling up the same
evaluation: exact power analysis (`research_agent_pilots/lggsn_analysis/statistics.py`'s
`mcnemar_power`/`mcnemar_required_n_for_power`, reused here rather than re-derived) on the
observed 2-vs-3 discordant split gives essentially zero post-hoc power and a required **210
discordant pairs** for 80% power at this effect size — against only 5 observed here out of 150
evaluated pools, resolving this would need on the order of several thousand additional evaluated
candidate pools, not a modest top-up. **Do not attempt to close this ablation by collecting more
data at the current scale; scope the paper's central claim to avoid depending on it** (Section 8):
the defensible claim is "object-centric, causally-admissible scoring beats geometry," not "the
within-scene pairwise loss term is what makes it work." This conclusion, and the reasoning behind
it, is already stated in `results/risk_gated_vla/final_report.md`'s "Recommended paper scope" —
restated here with the exact power numbers rather than the original's qualitative "n is small."

### 5.4 Mechanistic analysis: multi-head decomposition (offline, complementary)

Section 5.2 establishes *that* the corrected critic beats geometry, live-executed — the paper's
primary quantitative claim. This subsection asks *why and where* it works, via a second,
explicitly **offline re-scoring** experiment on the same causally-admissible features. It is
complementary evidence, not a replication, and must never be pooled with or presented as confirming
Section 5.2's live-executed numbers.

**Setup.** The same object-relative feature set, decomposed into four prediction heads instead of
one: `bilateral_contact`, `lifted`, `success` (binary), and a 3-class `failure_type`
(`success`/`no_contact`/`weld_no_lift`). Two additional classes originally specified,
`contact_no_weld` and `lifted_then_dropped`, were confirmed **structurally absent** from every
collected batch — the weld gate is currently identical to bilateral contact by construction, and the
`fell_off` threshold never triggers within the post-grasp settle window — so the 3-class scheme is
what was actually trained; the 6-class taxonomy is documented future schema, not a current claim.
Trained on the same base-100/dev-200/confirmatory-300 split chain as Section 5.2, with checkpoint
and loss-weighting choice frozen on dev-200 only and confirmatory-300 read exactly once.

**Result (offline, confirmatory-300).** The success head alone, used for top-1 selection, beats
geometry pooled +10.7pp (46.0% vs. 35.3%, McNemar p=0.0052) — directionally and in rough magnitude
consistent with Section 5.2's live-executed +14.0pp, reported as separate, complementary evidence
for the same claim, not pooled or averaged with it. All three heads reach AUROC 0.90-0.93 with
ECE≈0.04 (reasonably calibrated).

**Interpretability payoff.** The failure-type confusion matrix exposes a specific, actionable
asymmetry a single-head success critic cannot show: 92.2% recall on true `no_contact`, but 32% of
true successes are misclassified as `no_contact` — the model is conservative/under-confident on
success, not dangerously over-confident on failure.

**Limitation, load-bearing, not a footnote.** `weld_no_lift` support is 100% drill-attributed
(118/118); with PowerDrill excluded, training support for that class collapses to 1 example. This
must be reported as a within-drill measurement, not a demonstrated cross-object failure-mode
generalization capability. A genuine leave-one-object-out test was not run in this pass.

**A methodological note worth keeping, not hiding.** The `equal` vs. `success_weighted`
loss-weighting ablation produced measurably different trained weights (confirmed by direct
parameter comparison) but *identical* argmax-based top-1 accuracy at every one of 5 seeds on the
small internal validation split used during training — only the larger dev-200 batch's continuous
AUROC distinguished them. Coarse top-1 metrics can be an insufficiently sensitive selection
criterion for this kind of ablation at small validation-set scale; this was caught by
cross-checking, not silently reported as "no difference."

Neither this subsection nor Section 5.2 licenses a risk-gate or VLA-policy improvement claim
(Section 6's negative results on both stand unchanged) — the AUROC/interpretability gains here are
about the critic's own outputs, not about downstream gating or imitation-policy performance.

## 6. Limitations

Per this study's own rule — do not report only positive results; distinguish evidence from
hypothesis from refuted conclusion — every negative or unresolved finding this investigation
produced, condensed from `results/risk_gated_vla/final_report.md`'s full archive to the subset
that bears directly on this paper's central claim:

1. Risk gate: no measurable benefit over the ungated critic (coverage 98.7% — the gate rarely
   actually fires). We do not claim a gating contribution. The most likely explanation, per
   concurrent work on when uncertainty-gating helps at all [Gaus, Charaja, and Haeufle,
   arXiv:2605.18045] (Section 2), is that our critic's ~50% pooled success rate is not yet in the
   competence regime that analysis identifies as necessary — stated as the most likely
   explanation, not a claim we separately tested for it.
2. Pairwise-loss contribution unresolved, and not cheaply resolvable at this evaluation scale
   (Section 5.3 — quantified: ~210 discordant pairs needed for 80% power, against 5 observed).
   The paper's central claim does not depend on resolving this.
3. Simulation-only: no real-hardware validation in this draft (Section 7 states the plan).
4. Small object set (3 YCB objects); generalization to a broader object distribution untested.
5. Physics-level non-determinism on marginal grasps (~0.6-1% outcome-flip rate on repeated
   identical execution) — a measurement-noise floor on any single-run success-rate estimate in
   this simulator, disclosed rather than hidden.
6. Imitation-learning integration (ACT) is a working pipeline, not a validated capability — first
   online rollout failed; 5 demonstrations/object is not enough data to expect closed-loop
   robustness. Report as integration status, not as a result bearing on the paper's central claim.

## 7. Hardware Extension (future work, explicitly not executed in this draft)

We designed, but did not execute, a protocol to test whether Section 5.2's simulated advantage
transfers to physical SO-ARM101 hardware — a sim-to-real gap check, not a re-run of Section 5.2's
statistical claims; a physical pilot that fails to replicate the simulated effect would not
invalidate Section 5.2, and is the same class of outcome this project's own prior real-hardware
work has already hit and reported honestly (Section 4's companion investigation; this project's
independent SO-ARM101 execution-fidelity pilots).

**Object choice, corrected before any data collection.** The critic's object one-hot only covers
{CrackerBox, MustardBottle, PowerDrill} (Section 3.2); an earlier proposal to pilot on Pear was
caught, during protocol design, as an out-of-distribution input the critic was never trained to
handle — not a like-for-like generalization test. The protocol instead specifies CrackerBox and
MustardBottle, both in-distribution and both objects this project already handles on real
SO-ARM101 hardware. This correction is recorded here, before any trial, per this study's own
no-post-hoc-selection discipline (Section 3.3).

**Platform.** SO-ARM101 with a relative-delta-clamped real backend
(`robots/soarm_real_backend.py`), a RealSense D435i for pose estimation, and the same rotated-mount
geometry and top-down IK bias already used in this study's simulation. Connectivity, calibration,
and low-level motor control are already verified working from this project's independent
real-hardware track; a thin real-hardware analogue of the simulation's candidate-pool-build/score/
execute pipeline is new code to write, not new infrastructure to design.

**Design.** A matched-block paired trial (or, if placement restoration proves unreliable, an
explicitly-declared unpaired two-proportion design instead — chosen before running, not after
seeing results) with a small pre-registered pilot (12-20 total grasp attempts across both objects
and methods) gating a larger scale-up (20-30 trials/object/condition): proceed only if the pilot's
direction is consistent with simulation on both objects and no safety incident occurred; otherwise
stop and report the sim-to-real gap honestly rather than scaling up in search of significance. A
seven-item safety checklist (continuous human supervision, an active relative-motion clamp,
pre-execution joint-range verification, workspace clearance, a confirmed e-stop path, reduced
speed through the pilot stage, and an explicit collision check against the camera mount and tray)
must be re-confirmed live at execution time, not assumed from this document.

**What this would, and would not, settle.** A completed pilot/scale-up would establish whether the
simulated +14-16pp advantage survives the sim-to-real gap on CrackerBox and MustardBottle
specifically, at the tested sample size. It would not establish generalization to PowerDrill or
Pear, and does not test the risk gate at all (Section 6) — only top-1 critic selection, to keep
the physical pilot's scope minimal and interpretable.

**Status, stated plainly.** Camera repositioning (Stage 0 of the protocol) is an unresolved
physical-setup blocker, shared with an unrelated, independent line of work in this project, and is
a precondition for every later stage. No result from this section exists in this paper. If a
future revision adds one, only then would the more ambitious title (this file's header note) be
earned.

## 8. Conclusion

We set out to validate a pre-execution grasp critic and instead found that our own strongest
reported result for it was invalid — not through a single bug, but two independent evaluation
defects that each, alone, would have been enough to invalidate the comparison. Rather than treat
that as a dead end, we applied and extended this project's causal-validity criterion to catch it,
rebuilt the evaluation as a genuinely paired, causally-admissible design, and found a real,
replicated, live-executed effect: an object-relative counterfactual critic beats a geometric
baseline by +15.6pp on an independent development batch (p=0.00258, n=90) and +14.0pp on a frozen,
pre-registered confirmatory batch never inspected before evaluation (p=3.24e-4, n=150). We report
this as the paper's central claim, and only this claim — not a claim about the specific pairwise
loss term, which exact power analysis shows is not resolvable at the current data scale without
roughly two orders of magnitude more evaluated candidate pools (Section 5.3), and not a claim
about risk-aware gating, which measurably added nothing here (Section 6). A real-hardware
validation protocol is designed and the platform is connected and verified, but no hardware result
exists in this paper; Section 7 states the plan, not a result.

The discipline we believe generalizes beyond this specific pipeline is not the critic architecture
or the loss function — it is the practice of auditing an evaluation's construction before trusting
its output, reporting a negative gate result with the same rigor as a positive one, and stating
plainly, at every step, what the evidence does and does not support.

---

## Drafting notes (remove before submission)

- Source of truth for every number in this draft: `results/risk_gated_vla/final_report.md` and
  `audit.md`. If a number here and a number there ever disagree, `final_report.md` wins — it was
  independently re-verified from raw `scenes.jsonl`, this draft was not (yet).
- Do not add real-hardware numbers to Section 5 or the Abstract until Section 7 actually produces
  them — this is the exact discipline this paper's own Section 4 is about.
- Venue/template: decided 2026-08-02 — reuse this project's IEEEtran-based LaTeX infrastructure (as
  `paper_tro.tex`/`paper_final.tex` already do — corrected 2026-08-02: `interact` is
  `paper_advanced_robotics.tex`'s class, not theirs) rather than RA-L's exact `paper_final.tex`
  content, targeting a workshop or a modest-tier evaluation-rigor-focused venue rather than
  T-RO/RA-L tier (per the standalone-vs-merged analysis this decision followed from). No hard page
  limit assumed for the first LaTeX pass; trim Section 2/4/6 to fit once an actual venue and its
  page limit are chosen.
