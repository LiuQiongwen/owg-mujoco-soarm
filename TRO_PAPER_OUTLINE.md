# T-RO Paper Outline — Narrowed Scope (2026-07-16)

**Distinct from `paper_final.tex`** (RA-L, SO-ARM101, already submitted 2026-07-11 — do not
touch). This is a separate paper: Piper 6-DoF grasping, targeting T-RO, per the strategic
narrowing decision (accept Cracker's execution-precision limitation as documented future work
rather than a problem requiring a solution; explicitly include real hardware, architecture-only
given the arm isn't connected).

## 1. Introduction
- Motivation: cross-embodiment grasp reliability, SO-ARM101 → Piper as a second, independently-
  implemented 6-DoF platform.
- Contribution list (headline first, reordered — causal-validity audit is now the core method):
  1. **A formal, checkable causal-validity criterion for learned grasp-candidate scorers**
     (pre-execution admissibility, including the pool-relative clause), a reference-implementation
     audit tool (`causal_validity_audit/`), enforced as an import-time gate in both training
     scripts, externally grounded against GraspGen (arXiv:2507.13097, confirmed doing this
     correctly already, implicitly) and Freeform Preference Learning (arXiv:2606.32027, confirmed
     as a correct non-flagged boundary case). See `CAUSAL_VALIDITY_METHOD.md`.
  1b. **An automated version of the above**: `causal_validity_audit/auto_tagger.py`, a static
      dataflow analyzer (execution-touching call-graph + marker-gated forward taint propagation)
      that infers provenance automatically from a single per-function marker instead of requiring
      per-field manual registration. Validated against real, live code, it independently caught a
      genuine contamination bug (`grasp_yaw`) the hand-built registry had missed and had already
      reported as clean — with a real downstream empirical consequence (a previously-published
      "clean" accuracy number changed from 0.8236 to 0.1327 on re-verification). See
      `AUTO_TAGGER_ALGORITHM.md`. This is the genuine new-algorithm contribution.
  2. Diagnosis and fix of a silent gripper-controller double-scaling bug in a widely-used
     composite-controller framework (robosuite) — a generalizable methodology, not just a
     one-off patch. See `GRIPPER_BUG_METHODOLOGY.md`.
  3. Validated success-rate improvements on Pear/Mustard/Cracker from that fix, confirmed at
     paper-scale n (Section 4 — done, see `GRIPPER_BUG_METHODOLOGY.md`'s final table).
  4. A hardware backend architecture + explicit safety verification checklist for physical
     deployment (Section 5). Physical validation is future work — stated plainly, not implied.
  5. An honest, well-instrumented negative-results ledger (Section 6 / Future Work): eleven
     independently tested mechanisms across candidate generation, candidate selection,
     execution-control, active human-in-the-loop calibration, and world-model sim-to-real
     transfer, each ruled out with a paired-trial significance test. See `RULED_OUT_METHODS.md`.
  6. **A third, prospective application of the causal-validity criterion** (§4.5, added
     2026-07-30, merged in from the standalone `paper_risk_gated_vla_draft.md` line rather than
     submitted separately — see that file's header note): an object-relative counterfactual
     grasp-candidate critic, built causally-admissible from the start rather than audited after
     the fact, that significantly outperforms a geometric baseline on two disjoint, fully
     held-out MuJoCo test batches (dev-test +15.6pp, McNemar p=0.00258, n=90; frozen confirmatory
     +14.0pp, p=3.24e-4, n=150) — while a chance-level stale checkpoint (AUROC=0.4996) evaluated
     under the SAME corrected harness demonstrates why the discipline matters even when nothing
     about the *criterion* itself is violated (that checkpoint's failure traces to two separate,
     non-causal-validity bugs: a seed-coupling defect that broke the paired evaluation design, and
     a success criterion that counted transient contact as success). Complements §4.2-4.3's
     retrospective mistake-catching with a case where applying the discipline prospectively
     produces a working, validated result, not just a caught mistake. Full negative/incomplete
     results (uncertainty risk gate: no benefit; small-data ACT pilot: fails online) reported
     alongside it, not omitted. See `results/risk_gated_vla/final_report.md` and `audit.md`.
     A complementary, explicitly offline mechanistic analysis (§4.6, added 2026-07-30) decomposes
     the same critic into contact/lift/success/failure-type heads to explain *where* it works
     (drill is measurably harder than cracker/mustard) and *why* (a specific, actionable
     success/no_contact confusion pattern) — kept clearly separate from the live-executed primary
     result, not merged with it.

## 2. Related Work
- Composite/hybrid grasp controllers; robosuite/MuJoCo-based manipulation benchmarks.
- Cross-embodiment transfer in grasping (GraspGen-X, Freeform Preference Learning, FAR — from
  Direction 2's literature survey).
- Contact-aware / compliant execution (CoorGrasp, AutoDex) — cited as the identified but
  unreachable-at-current-resources mechanism for Cracker's remaining failure mode.
- World-model-driven sim-to-real transfer (RISE, Cosmos-derived world-action models) — cited in
  Future Work as the identified path once real-data volume or compute scale changes.

## 3. System
- Piper + RoboSuite simulation environment (`PiperMultiObjectScene`, `ArmIK`,
  `run_pick_and_place`).
- LGGSN pairwise reranker (SO-ARM101 side, already the RA-L paper's contribution — here it's both
  reused infrastructure AND the system under audit in Section 4, since it's the deployed live
  scorer the causal-validity criterion applies to).

## 4. Causal-Validity Auditing for Learned Grasp-Candidate Scorers (core method)
- Pull directly from `CAUSAL_VALIDITY_METHOD.md`: the formal PRE_EXECUTION-admissibility
  criterion (including the pool-relative clause), the `causal_validity_audit/provenance.py`
  registry + `audit_feature_set()` gate, wired as an enforced import-time check in both
  `train_lggsn_pairwise.py` (the live-deployed checkpoint's training script) and
  `stage2_train_embodiment_lggsn.py`.
- **4.1 — External grounding**: GraspGen (arXiv:2507.13097) confirmed already following this
  criterion implicitly (discriminator inputs are point cloud + pose only; labels are the
  execution-derived signal); Freeform Preference Learning (arXiv:2606.32027) as a correct
  non-flagged boundary case (trajectory-level input is legitimate given its different role —
  post-hoc reward shaping, not pre-execution selection).
- **4.2 — Retrospective demonstration**: `causal_validity_audit/retrospective_audit.py` run
  against every feature set this project's own cross-embodiment reranking pilots used
  (IDEA_REPORT.md Direction 2). Result: correctly flags every pilot that produced the illusory
  "pooling beats zero-shot" effect (p<0.0001 ×5, collapsed to p=1.0000 once corrected) and passes
  every pilot the project ultimately trusted. Precisely attributes the leakage to the Piper side
  (`quality_score`, `correction_proxy`) — the SO-ARM101 side's flagged fields turned out, on
  tracing the actual live-inference code path, to be a harmless proxy plus dead constants.
- **4.3 — The tool's own four self-corrections**, worth including plainly rather than omitting —
  and worth the full four, not a curated two, because the escalation across them is itself the
  argument: (1) `score`/`dz`/`dz_lift`/`need_dz` mislabeled by quoting a comment describing a
  different, inactive dataset instead of tracing the live path; (2) `dist_to_centroid`/`z_rel`
  mislabeled by conflating "depends on sibling pool candidates" with "depends on execution"; (3)
  `grasp_yaw` mislabeled by a human missing a later variable reassignment — caught not by more
  careful manual review, but by the automated tagger (§4.4); (4) the retrospective demonstration's
  own test cases mislabeled by a field-name collision (`"yaw"` vs. `"grasp_yaw"`) that silently
  masked correction (3)'s finding in the one row meant to prove the pipeline was clean. Corrections
  1-2 were caught by a human re-checking harder. Corrections 3-4 were only caught once the process
  was automated — a real argument for why 4.4's algorithm matters beyond convenience.
- **4.4 — Automated tagging** (`AUTO_TAGGER_ALGORITHM.md`): the algorithm itself, its validation
  against real code, and the re-verified empirical result after removing `grasp_yaw`
  (§4.3-corrected: accuracy 0.8236 → 0.1327, qualitative null finding unchanged).
- **4.5 — Prospective application: an object-relative counterfactual grasp critic** (merged
  2026-07-30 from the standalone `paper_risk_gated_vla_draft.md` draft — see that file's header
  note; source data/results in `results/risk_gated_vla/`, not re-derived here). Where §4.2-4.3
  show the criterion catching mistakes *after* they were made, this subsection shows applying it
  *from the start*:
  - A predecessor critic pipeline (`world_model/`, pre-dating `causal_validity_audit/` entirely)
    reported +15.6pp over geometry. Diagnosed as invalid on two independent grounds, neither of
    which is itself a causal-validity violation: (a) its evaluation harness's RNG seed encoded
    the compared method identity, so paired trials never shared a scene or candidate pool
    (`audit.md` §4, directly reproduced: 0/250 paired trials shared even the executed candidate
    position); (b) its success criterion counted transient post-close gripper contact as success
    regardless of whether the object was ever lifted (saturates to 150/150 regardless of pose).
  - Rebuilt as a causally-admissible-by-construction critic (object-relative pose + point-cloud
    stats + object identity — every field registered `PRE_EXECUTION` in `provenance.py`, verified
    by the same gate as §4's other pipelines) plus a from-scratch paired evaluation harness fixing
    both defects above.
  - The corrected pipeline also surfaces a **fourth self-correction-adjacent finding, this time
    in shared project infrastructure rather than in the audit tooling itself**: the production
    `physics_weld_after_bilateral` grasp primitive's weld-attach gate accepted single-jaw contact,
    not the bilateral contact its name and documented protocol require — found only because this
    study executes the primitive far more densely (13x/scene) than prior usage. Fixed in
    `tango_robot/env_soarm.py`; flagged as possibly affecting other `physics_weld_after_bilateral`
    results project-wide, explicitly out of scope to re-verify here.
  - **Result**: re-evaluated under the corrected harness, the stale checkpoint fails a
    pre-registered gate outright (AUROC=0.4996 on 1,500 real per-candidate labels — chance level).
    A freshly-trained, causally-admissible critic, evaluated on two scene batches disjoint from
    training/model-selection and from each other (independently key-intersection-verified, not
    asserted), beats geometry: dev-test +15.6pp (McNemar p=0.00258, n=90), frozen confirmatory
    +14.0pp (27 wins/6 losses, p=3.24e-4, n=150) — live-executed paired outcomes, not an offline
    re-scoring (see `final_report.md`'s methodology note on a small, physically-real
    non-determinism footnote: ~0.6-1% of repeated identical-pose executions flip their boolean
    outcome, MuJoCo's contact solver is not bit-reproducible on marginal grasps).
  - **Negative/incomplete results, reported with equal rigor, not omitted**: an ensemble-
    uncertainty risk gate calibrated on held-out data adds no measurable benefit over the ungated
    critic (coverage 98.7% — it almost never fires); whether the within-scene pairwise loss term
    is independently responsible for the gain vs. object-relative features alone is unresolved
    (object-relative BCE vs. the full counterfactual variant: p=1.0, not distinguishable at this
    n); a small-data (15-demo) ACT imitation-policy pilot integrates end-to-end but fails its
    first closed-loop rollout. Real-hardware validation is designed
    (`results/risk_gated_vla/PHASE3_REAL_HARDWARE_PROTOCOL.md`) but not yet executed — folds into
    Section 6 rather than duplicating it.
- **4.6 — Mechanistic analysis: multi-head contact/lift/success decomposition** (added
  2026-07-30, source: `results/risk_gated_vla/phase1/multitask_outcome_critic/C3_RESULT.md`).
  §4.5 establishes *that* the corrected critic beats geometry (live-executed, the paper's primary
  quantitative claim); this subsection asks *why and where it works*, via a second, complementary,
  explicitly **offline re-scoring** experiment — must never be merged with or presented as
  confirming §4.5's live-executed numbers.
  - Same causally-admissible features as §4.5, decomposed into four prediction heads instead of
    one: `bilateral_contact`, `lifted`, `success` (binary), and a 3-class `failure_type`
    (`success`/`no_contact`/`weld_no_lift` — the two additional classes originally specified,
    `contact_no_weld` and `lifted_then_dropped`, were confirmed **structurally absent** from every
    collected batch — the weld gate is currently identical to bilateral contact by construction,
    and the `fell_off` threshold never triggers within the post-grasp settle window — so the
    3-class scheme is what was actually trained, with the 6-class taxonomy kept only as documented
    future schema, not a current claim).
  - Trained on the same base-100/dev-200/confirmatory-300 split chain as §4.5, checkpoint and
    loss-weighting choice frozen on dev-200 only, confirmatory-300 read exactly once (audit trail:
    `confirmatory_run_log.jsonl` confirms a single run).
  - **Result (offline, confirmatory-300)**: the success head alone, used for top-1 selection, beats
    geometry pooled +10.7pp (46.0% vs. 35.3%, McNemar p=0.0052) — directionally and in rough
    magnitude consistent with §4.5's live-executed +14.0pp, reported as separate, complementary
    evidence for the same claim, not pooled or averaged with it. All three heads reach AUROC
    0.90–0.93 with ECE ≈0.04 (reasonably calibrated).
  - **The interpretability payoff**: the failure-type confusion matrix exposes a specific,
    actionable asymmetry a single-head success critic cannot show — 92.2% recall on true
    `no_contact`, but 32% of true successes are mis-classified as `no_contact` (the model is
    conservative/under-confident on success, not dangerously over-confident on failure).
  - **Explicit limitation, load-bearing, not a footnote**: `weld_no_lift` support is 100%
    drill-attributed (118/118) — with drill excluded, training support for that class collapses to
    1 example. This must be reported as a within-drill measurement, not a demonstrated
    cross-object failure-mode generalization capability. A genuine leave-one-object-out test (to
    become §C.4-style future work) was not run in this pass.
  - **A methodological note worth keeping, not hiding**: the `equal` vs. `success_weighted`
    loss-weighting ablation produced measurably different trained weights (confirmed by direct
    parameter comparison) but *identical* argmax-based top-1 accuracy at every one of 5 seeds on
    the small internal validation split used during training — only the larger dev-200 batch's
    continuous AUROC distinguished them. Worth a sentence in the paper's own methods discussion:
    coarse top-1 metrics can be an insufficiently sensitive selection criterion for this kind of
    ablation at small validation-set scale; this was caught by cross-checking, not silently
    reported as "no difference."
  - Neither this subsection nor §4.5 licenses a risk-gate or VLA-policy improvement claim (§4.5's
    own negative results on both stand unchanged) — AUROC/interpretability gains here are about
    the critic's own outputs, not about downstream gating or imitation-policy performance.

## 5. The Gripper-Controller Bug (supporting case study)
- Pull directly from `GRIPPER_BUG_METHODOLOGY.md`: the bug, how found, two fix attempts (one
  dead end kept for methodological value, one that worked), final confirmatory results.
- **Final confirmed rates (n=40-60/object, frozen final baseline config, clean trial range
  5000-5059)**: Cracker 50% (n=40), Mustard 70% (n=40), Pear 43.3% (n=60 — the original
  single-batch 65% figure did not replicate across two independent tie-breaker batches and
  should not be used; reported transparently in `GRIPPER_BUG_METHODOLOGY.md`).
- Frame explicitly as a second, independent demonstration of the same "verify against ground
  truth, don't trust a plausible surface explanation" discipline as Section 4 — not a second,
  unrelated headline claim.

## 6. Toward Real-Hardware Deployment
- Pull from `REAL_HARDWARE_ARCHITECTURE.md`: design pattern, safety-first refusal to guess
  units, explicit VERIFY checklist. Frame as "architecture + checklist, physical validation
  pending" — see the suggested framing sentence in that doc.

## 7. What Doesn't Work (honest negative results / Future Work)
- Table from `RULED_OUT_METHODS.md`, condensed to fit page budget (likely: full table in
  appendix, 3-4 sentence summary + the 4-row execution-control sub-story in main text, since
  those four are the most directly relevant to Cracker's remaining limitation and tell a
  complete, coherent mini-narrative on their own).
- Explicit statement: Cracker's remaining execution-precision gap is diagnosed (contact dynamics
  during descend, not candidate selection — corroborated by two independent investigations) but
  not solved; stated as future work requiring either tactile sensing or a fundamentally different
  contact-aware control mechanism than the four tested here.

## 8. Conclusion
- Reframe the paper's actual contribution honestly: not "we solved cross-embodiment grasping,"
  but "we formalized and tooled a causal-validity criterion that a careful published system
  already follows implicitly, demonstrated its practical importance on our own costly mistake
  (including catching two mistakes in the tool itself), found and fixed a real
  previously-undiagnosed execution bug along the way, characterized what remains broken with
  unusual rigor, and built the architecture for the next real-hardware phase."

## Open items before this can be written up formally
- [x] Section 5's confirmatory-results table filled in (see above).
- [x] Causal-validity method built, verified, and given its own section (4) rather than folded
      into the negative-results table — resolves the earlier open item about how prominently to
      feature Direction 2's leakage lesson.
- [x] Formal PRE_EXECUTION-admissibility criterion written as Definitions 1-3 + Proposition +
      Corollary + Algorithm in `CAUSAL_VALIDITY_METHOD.md`, codebase-independent, drop-in ready
      for the Method section. One piece intentionally left as future work: Algorithm step 3
      (re-audit when a field's *computing code* changes, not just when the feature list changes)
      is specified but not yet automated.
- [ ] Page-budget pass once a full draft exists (T-RO does not have RA-L's 8-page hard limit, but
      should still be checked against T-RO's own submission length norms) — Section 4 is now the
      largest section and may need trimming (4.3's self-correction narrative is valuable but
      could compress to a paragraph rather than three subsections). §4.5+§4.6 (added 2026-07-30)
      make this more pressing, not less — together they are now the longest part of the outline;
      §4.6 in particular may compress to a short paragraph + one figure (the confusion matrix) in
      the actual submission rather than the full itemized list kept here for working reference.
- [x] §4.5's literature pass done: `results/risk_gated_vla/LITERATURE_AND_NOVELTY_PLAN.md`, 19
      independently verified papers (6 with author lists confirmed via direct WebFetch) across
      world models, counterfactual/critic evaluation, VLA verifiers, uncertainty/conformal
      methods, DAgger/recovery data, adaptive chunking, and grasp success prediction — that
      document's own §B.5 is also where the "object-centric executable action verifier" naming
      (used in §4.5/§4.6 above) was decided, externally grounded rather than asserted. Section 2's
      actual prose still needs writing from this material, but the literature itself is no longer
      an open gap.
- [ ] Real-hardware validation for §4.5/§4.6 (`PHASE3_REAL_HARDWARE_PROTOCOL.md`) is designed, not
      executed — decide before submission whether Section 6 needs it to be at least a pilot-scale
      result, or whether "designed, explicitly future work" (this outline's existing framing for
      the Piper hardware backend) is an acceptable posture for §4.5/§4.6's claims too.
- [ ] §4.6's own natural follow-up (a real leave-one-object-out retraining ablation, not just the
      lighter "support without drill" reporting done in the current pass) is scoped in
      `LITERATURE_AND_NOVELTY_PLAN.md` §C.4 — not started, needs a decision on whether it's worth
      the new-data-collection cost before submission or stays future work.
