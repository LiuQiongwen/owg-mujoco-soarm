Migration v1 charter — OWG → Piper feasibility → static critic

Phase: Piper base audit CLOSED (`docs/PIPER_AUDIT_CLOSURE_20260808.md`).
Execution baseline: `PIPER_BASELINE_V1` (`calib/piper_baseline_v1.json`).
Finding the P2 mediator is NOT a precondition for anything below.

## Architecture (two layers, deliberately separate)

    OWG grounding -> 6-DoF candidates -> Piper feasibility -> critic -> execution
                                                                 |
    pre-execution critic : reject clearly infeasible, rank the rest
    execution monitor    : carry the state divergence that only becomes
                           visible from descend onward

The critic's job is **pre-execution discrimination**, not explaining every
execution outcome. Evidence for the split: P1.2 (first separation appears
at descend, not before) and P2/Phase-A (audited static features do not
mediate the one known causal effect). Anything only visible during
execution belongs to the monitor, not the critic.

## Hard rules (apply to every critic/monitor experiment)

R1–R5 in `docs/EXPERIMENT_HYGIENE_RULES.md` are binding, not advisory.
Restated where they bite hardest here:

1. **Instrumentation before interpretation.** Any feature, score, AUC or
   calibration metric that lands suspiciously close to an input constant,
   zero, or a natural bound is an instrumentation fault until the
   measurement chain is checked. This caught three wrong conclusions in the
   audit; it is an acceptance condition for reported critic numbers, not a
   post-hoc sanity check.
2. **No outcome-derived features.** Anything the label is defined from is
   refused entry. Enforce in code (see `OUTCOME_DERIVED` in
   `scripts/piper_outcome_dataset.py`), not by review.
3. **No pooled-only claims.** Report within-object and
   leave-one-object-out alongside any pooled metric. Object-level
   confounding faked three strong-looking separators in the audit.
4. **Ordered paired designs.** n=12 cells are not confirmatory here
   (P1.1). Small samples are acceptable only as ordered paired sweeps.

## Data constraints inherited from the audit

- **`rel_dist` must be object-normalised or object-conditioned.** Within
  object ρ ≈ −0.50…−0.66 (p < 1e-4); across object cells ρ = −0.34,
  p = 0.21. A pooled feature here would silently encode object identity.
- **Pear is suspect and sensitivity-heavy.** It supplies 18 of 32 failures
  in the P1 dataset and its targeting is uncharacterised (the P2 effect is
  strongest there and unexplained). It must not dominate training; report
  per-object and worst-object performance, not just macro average.
- **Object set under this baseline:** cracker, can, mustard, pear. `banana`
  and `drill` cannot be spawned; `clamp` is mechanically infeasible
  (150mm vs 100mm opening). Historical banana/drill results are not
  reproducible here and are not valid comparisons.
- **`can` is near-ceiling** (39/40) — a control, not a source of signal.
- **Pre-2026-07-15 data is not a valid baseline** (gripper double-scaling
  fix).

## Useful positives to encode as features

- Measured max opening 100.0mm — gate candidates on required width.
- Objects sit 15–21mm beyond the finger envelope tip; Piper grasps with
  the edge of its contact region, not centred. Descriptive, not a P2
  explanation.

## Optional, non-blocking

Wider pear sweep (±30–45mm, ordered paired) to test whether +15mm is an
interior optimum or a rising edge.
