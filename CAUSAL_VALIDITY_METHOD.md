# Causal-Validity Auditing for Learned Grasp-Candidate Scorers — T-RO Core Method

## The gap this fills

Data leakage from features that are only knowable after the fact is a well-known general ML
problem, and offline-RL has recent work on label leakage in policy evaluation
(e.g. arXiv:2605.11479). But no prior work formalizes this specifically for **learned
grasp-candidate scorers/rerankers** as an explicit, automatically-checkable criterion. The
correct practice already exists as an *implicit* norm in careful published work — confirmed
here against GraspGen (arXiv:2507.13097): its discriminator's inference-time inputs are point
cloud + candidate pose only (pre-execution admissible), while its training labels come from
simulated shaking-motion outcomes (execution-derived labels are fine; execution-derived *input
features* are not). Nobody has stated this as a formal criterion or built a tool that checks a
feature set against it before a model gets trained.

## The formal criterion (paper-ready, codebase-independent)

**Definition 1 (Candidate pool, execution trace).** A scene induces a candidate pool
`C = (c_1, ..., c_N)`, each candidate specified by a pose `p_i ∈ SE(3)` and any generator-time
attributes `a_i` (e.g. predicted gripper width). Let `G` denote static, execution-independent
scene/object information (geometry, calibration constants — the same for every candidate in the
scene). If `c_i` is physically executed, let `T_i` denote its execution trace: the full state
history of that trial (contact forces, intermediate poses, net displacement) and its terminal
success label `y_i ∈ {0,1}`.

**Definition 2 (Selection-time information set).** At the moment a scorer must choose which
candidate(s) in `C` to execute, the selection-time information set is

```
I(C) = {p_1,...,p_N} ∪ {a_1,...,a_N} ∪ G ∪ K(C, G)
```

where `K(C, G)` is the set of values obtainable from isolated, execution-free queries against a
candidate pose and `G` (e.g. a kinematic IK residual or a collision check evaluated at `p_i`
without moving the arm). By construction `I(C)` contains no `T_j` for any `j` — nothing in the
pool has been executed yet.

**Definition 3 (PRE_EXECUTION-admissible feature).** A feature function `φ` is
**PRE_EXECUTION-admissible** if there exists `g` such that, for every candidate `c_i` in every
pool `C`, `φ(c_i) = g(I(C))` — i.e. `φ(c_i)` is `I(C)`-measurable. Note `I(C)` includes the poses
of *every* candidate in the pool, not just `c_i` — a feature may legitimately depend on sibling,
not-yet-executed candidates (e.g. distance to the pool's centroid) without being execution-derived;
"depends on more than one candidate" and "depends on execution" are independent axes, and
conflating them is itself a mistake this project's reference implementation made and corrected
(§4.3). Otherwise `φ` is **EXECUTION_DERIVED**.

**Proposition (the audit rule).** A scorer `S` used to select among not-yet-executed candidates
is causally valid iff `S(c_i) = h(φ_1(c_i),...,φ_k(c_i))` for admissible `φ_1,...,φ_k`. Any
feature set containing an execution-derived `φ_j` is invalid for live selection *regardless of
how well `S` performs when evaluated offline* on historical `(c_i, T_i)` pairs — offline
evaluation implicitly supplies `T_i` for every `i` in the evaluation set, silently satisfying
`φ_j` even though `φ_j` would be undefined for a genuinely new, not-yet-run candidate. This is
exactly why every AUC/pairwise-accuracy number computed before this project's Stage 3 design
review looked fine: the contamination is invisible under offline evaluation by construction.

**Corollary (labels are exempt).** Execution-derived quantities remain valid as supervision
targets `y_i`, or as auxiliary values used only to construct `y_i` (e.g. a continuous quality
score used to rank pairs for BPR training) — the constraint binds the *input* side of `S`, not
the *label* side. This is why GraspGen's design is correct (execution-derived training labels,
pre-execution-only inference inputs), and why the mistake is easy to make: the same raw field
name can legitimately live in a dataset's label construction and illegitimately leak into its
feature columns, and nothing about the field's name signals which role it's playing.

**Algorithm (the audit).**
1. For every raw field `f` used anywhere in the data pipeline, tag it once at its point of
   computation with provenance `τ(f) ∈ {PRE, EXEC}`, determined by applying Definition 3 to the
   *actual code path that computes `f`* — not a comment about it, not an assumption about a
   similarly-named field elsewhere, not a description of a different dataset. Trace the call site.
2. Before training or invoking any live-selection scorer `S` with input feature set
   `F = {f_1,...,f_k}`: reject if `∃ f_j ∈ F` with `τ(f_j) = EXEC` or `τ(f_j)` unregistered
   (fail closed — an unaudited field is not assumed safe).
3. Re-run step 2 whenever `F` changes *or whenever the code path computing any `f_j` changes* —
   `τ(f_j)` is a property of the code that produces the value, not of the field's name, so a
   refactor can silently change `τ(f_j)` without changing `F`. This step is not hypothetical:
   it is exactly the check that would have caught both of the reference implementation's own
   mislabelings (§4.3), which arose from trusting a stale description instead of re-tracing code.

`causal_validity_audit/provenance.py` implements steps 1-2 as `FieldSpec`/`audit_feature_set()`;
step 3 is currently a process discipline (re-run the audit after any feature-pipeline change),
not yet automated — flagged as future work below.

This distinction is silent by construction: every offline AUC/pairwise-accuracy number computed
on historical logs looks identical regardless of which side of the line a feature falls on — the
violation only becomes visible the moment the same feature set gets reused for live selection.

## The tool

`causal_validity_audit/provenance.py` — a registry (`FieldSpec(provenance, reason)`) tagging
every raw logged field in this project's two grasping pipelines (SO-ARM101's 14-dim LGGSN
fields, Piper's `run_pick_and_place` result dict) as `PRE_EXECUTION` or `EXECUTION_DERIVED`,
with the specific reason cited per field (e.g. `train_geo_ebm_grasp.py`'s own header comment for
`score`/`dz`/`need_dz`; "logged after prior phases' physical trajectory already ran" for Piper's
`quality_score`). `audit_feature_set(feature_names)` raises `CausalValidityViolation` listing
every non-admissible or unregistered field — **fail closed**: an undeclared field is treated as
a violation, not assumed safe.

## Retrospective demonstration (working code, run and verified)

`causal_validity_audit/retrospective_audit.py` runs the registry against every feature set this
project's cross-embodiment reranking pilots actually used, in chronological order. Output:

```
PASS  Pilot 1-2, SO-ARM101 side (3-feat: [z, score, need_dz])
FAIL  Pilot 1-2, Piper side (3-feat: [z, quality_score, correction_proxy])
PASS  Pilot 3, SO-ARM101 side (5-feat: [z, yaw, H, score, need_dz])
FAIL  Pilot 3, Piper side (5-feat: [z, yaw, H, quality_score, correction_proxy])
PASS  Pilot 4 / Stage 2 pre-correction, SO-ARM101 side (reused Pilot 3's 5-feat)
FAIL  Pilot 4 / Stage 2 pre-correction, Piper side (reused Pilot 3's 5-feat)
PASS  Stage 2 CORRECTED, SO-ARM101 side (3-feat: [z, yaw, H])
PASS  Stage 2 CORRECTED, Piper side (3-feat: [z, yaw, H])
PASS  Follow-up check, Piper side (score_candidate_ik alone)

6 admissible, 3 would have been flagged before ever training a model.
```

**The leakage was concentrated entirely on the Piper side** (`quality_score`, `correction_proxy`
— confirmed directly against `piper_pick_and_place.py::run_pick_and_place`). Every pooled pilot
that produced the apparent "pooling beats zero-shot" effect (p<0.0001, replicated 5 times across
Pilots 1-2, 3, and 4/pre-correction Stage 2) is still flagged FAIL overall — a pooled model is
only as valid as its most-contaminated input, and Piper's contamination alone was sufficient to
invalidate the pooled result. The audit reproduces, in milliseconds, a verdict that took this
project weeks of pilots, a full `EmbodimentLGGSN` architecture build, a 22-fold CV run, and a
near-miss on committing to an expensive live-execution Stage 3 test to discover manually.

**A correction made to this registry itself, worth including in the paper** (it strengthens
rather than weakens the argument): the first version of `provenance.py` labeled the SO-ARM101
side's `score`/`dz`/`dz_lift`/`need_dz` as `EXECUTION_DERIVED` by quoting a comment in
`train_geo_ebm_grasp.py` without checking whether that comment described the code path actually
feeding the deployed model. It didn't — it described a separate, inactive legacy dataset
(`grasp_6dof/dataset/all_lggsn.csv`). Tracing the *actual* live inference path
(`grasp_ranker_lggsn.py`'s `_featurize_one`, `policy.py`'s live `rank()` call site,
`batch_s3s4.py`'s log writer) and independently verifying against the on-disk training log
(4,288/4,288 rows have `dz=dz_lift=need_dz=0.0`) showed `score` is actually a harmless
pre-execution GR-ConvNet proxy and the other three are dead constant-zero features — not
leakage, just uninformative. The registry was corrected accordingly. This self-correction is
itself a small demonstration of the paper's core argument: even a tool built specifically to
enforce rigor can silently inherit an unverified claim (a comment describing the wrong code
path) unless it's checked against ground truth rather than trusted at face value — exactly the
discipline the audit exists to formalize.

## Why this is the paper's core contribution, not a footnote

1. **Formal + checkable, not just a diagnostic anecdote**: `provenance.py` is a reusable
   artifact — any new feature added to either pipeline gets tagged once, and every downstream
   model-training script can call `audit_feature_set()` as a gate before training starts.
2. **Externally grounded, not self-referential**: the criterion matches what a careful published
   system (GraspGen) already does implicitly. The contribution is naming and formalizing the
   norm, not inventing a requirement nobody follows.
3. **Empirically dramatic**: a 5×-replicated, p<0.0001 effect reduced to p=1.0000 is about as
   large a demonstration of the failure mode's real-world magnitude as this kind of methodology
   paper can offer — most leakage papers argue from smaller or synthetic examples.
4. **Honest positioning**: this reframes the paper from "we tried many things and most failed"
   into "we built and validated a tool that would have told us three of our own pilots were
   invalid before we ran them, and precisely which platform's features caused it" — a genuine,
   generalizable rigor contribution, with the negative-results
   ledger (`RULED_OUT_METHODS.md`) and the gripper-bug fix (`GRIPPER_BUG_METHODOLOGY.md`) as
   supporting case studies of the same "verify before trusting a plausible offline number"
   discipline, rather than being the paper's main claim.

## The automated version: a real new algorithm, not just a bigger registry

`AUTO_TAGGER_ALGORITHM.md` documents `causal_validity_audit/auto_tagger.py`: a static dataflow
analyzer (execution-touching call-graph + marker-gated forward taint propagation) that infers
provenance automatically instead of requiring per-field manual registration. Run against the real,
live `run_pick_and_place` function, it independently caught a real error the (already twice-
corrected) hand-built registry still had: `grasp_yaw` was mislabeled `PRE_EXECUTION`, because
`grasp_mat` is reassigned post-commit at the "pre-close refresh" step and a human reading the
function once missed the later reassignment. This is the project's **third** self-correction to
its own causal-validity claims — and unlike the first two, this one had a real downstream
consequence: `grasp_yaw` was part of the "Stage 2 CORRECTED" feature set already cited elsewhere in
this project's documentation as clean. It wasn't. Re-verified with a genuinely clean `[z, H]`
feature set: the qualitative null finding survives, but the reported accuracy number changed
(0.8236 → 0.1327) and needed correcting in every document that cited it. See
`AUTO_TAGGER_ALGORITHM.md` for the full algorithm, validation, and a fourth correction (a field-name
collision bug in the retrospective demonstration itself) caught while re-verifying this one.

## External boundary case: Freeform Preference Learning (arXiv:2606.32027)

Checked as a second published system, deliberately chosen because it's the case the criterion
should NOT flag. FPL's reward model takes a full trajectory + a preference label and predicts an
axis-specific reward — used for post-hoc reward shaping in RL, not for scoring not-yet-executed
candidates before selection. Its input is legitimately trajectory-level/execution-derived *by
design*, because its role in the pipeline is different (reward model for already-completed
rollouts, not a pre-execution candidate scorer). The criterion correctly does not apply to it —
useful for the paper as a worked example distinguishing "execution-derived input because that's
what the model's role requires" from "execution-derived input leaked into a model whose role
requires pre-execution admissibility," which is the actual failure mode this project hit.

## Remaining work before this is submission-ready

- [x] Wired `audit_feature_set()` as an enforced import-time gate in
      `stage2_train_embodiment_lggsn.py` (fails fast if `soarm_feat`/`piper_feat` are ever edited
      to reintroduce an inadmissible field) — verified working.
- [x] Wired the same gate into `train_lggsn_pairwise.py`'s `FEATURE_COLS` — this is the script
      that trains the checkpoint actually deployed live (`lggsn_pairwise_live_v2.pt`). Verified:
      gate passes (all 14 columns confirmed admissible, including the corrected
      `dist_to_centroid`/`z_rel`/`score`/`dz`/`dz_lift`/`need_dz` entries above) — this is a
      completeness/enforcement addition, not a fix for a live bug; the deployed RA-L checkpoint's
      training pipeline was never actually contaminated, per the trace above.
- [x] Formal criterion written as Definitions 1-3 + Proposition + Corollary + Algorithm (see
      above), codebase-independent, drop-in ready for the paper's Method section. Algorithm step
      3 (re-audit whenever a feature's computing code changes, not just when the feature list
      changes) is stated but **not yet automated** — currently a process discipline, not a CI
      check. Automating it (e.g. hashing each field's source function and invalidating its
      provenance tag on change) is real, scoped future work, not done in this pass.
- [x] Decided: include both self-corrections briefly, as §4.3 in `TRO_PAPER_OUTLINE.md` — one
      short paragraph each (not a full incident report), framed as evidence the discipline works
      even against its own author, not as a caveat weakening the tool's credibility.

## Addendum (2026-07-16): an unrelated infrastructure bug, and the strongest re-verification yet

While testing the previously-flagged "genuine per-candidate admissible feature" future-work item
(the original pre-commit candidate orientation), found that `PiperMultiObjectScene`'s placement
sampler (`UniformRandomSampler`) was constructed without an explicit `rng`, silently defaulting to
robosuite's own OS-entropy-seeded generator — completely independent of every `np.random.seed(...)`
call this project's Piper experiments have relied on since the platform was built. Confirmed via
direct re-runs (identical, unmodified collection script, same scene, three different outcome
patterns across three runs) and via 7-9cm of spawn-position spread found within already-collected
data that should have been constant within a scene. Fixed in `piper_multi_object_scene.py` by
explicitly deriving a seeded generator from the legacy global state at scene-construction time.

**This does not affect the causal-validity criterion, registry, or auto-tagger findings above** —
those are established by tracing code, not by statistical properties of collected data. It does
affect specific accuracy numbers computed on data grouped under the (previously false) assumption
of shared placement within a scene. Re-collected the 25-scene Piper dataset under the fix and
re-ran the corrected `[z, H]` evaluation: **the null finding survives exactly** — all four pooling
conditions remain identical (0.1036, diff=0.0000). Also tested the flagged feature directly: a naive
pooled correlation looked dramatic (r=-0.55, p<0.0001, the strongest signal found in this entire
project) but decomposed into a between-scene confound (r=-0.71 between scenes — different
placements are both systematically differently-oriented and differently difficult, for unrelated
geometric reasons) with essentially zero real within-scene signal (mean r=+0.03, no scene reaching
significance). See `paper_tro.tex`/`paper_tro_draft.md` §IV-E for the full writeup — this is now
the paper's most rigorous single demonstration of the causal-validity thesis, since it shows a
feature that looks like the strongest predictor in the whole investigation providing zero real
selection-relevant signal once properly decomposed.
