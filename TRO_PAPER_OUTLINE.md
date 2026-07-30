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
      could compress to a paragraph rather than three subsections).
