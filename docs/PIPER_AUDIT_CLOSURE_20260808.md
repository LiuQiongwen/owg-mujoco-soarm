Piper execution-semantics audit — closure (2026-08-08)

Closes the audit phase. Baseline: `PIPER_BASELINE_V1`
(`calib/piper_baseline_v1.json`), frozen at `a961e9d`.

## Established (positive)

| item | status |
|---|---|
| capture / TCP reference | `robot0_eef_site` is a sound grasp reference; the "65.6mm offset" was a measurement artifact (retracted) |
| gripper opening | measured 100.0mm inner-face at the ctrlrange floor; supersedes the stale ~7.6cm in code comments |
| contact / penetration | healthy; blocked-closure symmetric, no solver warnings |
| success semantics | `Lift._check_success` inherited unmodified, height-based, no weld/attach shortcut |
| determinism | verified (8/8 same-seed reproduction) |
| `can` graspability | NOT ungraspable — 6/6; historical 0/25 predates the 2026-07-15 double-scaling fix |
| grasp engagement geometry | object centre sits 15–21mm *beyond* the finger envelope tip end; Piper grasps with the edge of the contact region, not centred |

## Established (negative / withdrawn)

Four claims were withdrawn after direct testing: `transit_high` as dominant
failure mode (tautological label, AUC 0.50); the 65.6mm TCP offset
(measurement artifact); grasp-height-above-object as a separator (falsified
by intervention, `can` 6/6 at that height); penetration depth as a
separator (AUC 0.43 at n=160, CI spanning chance).

## The one robust causal result

The P2 aim-offset intervention moves outcomes: pear 3/12 → 12/12, **50/0
concordant seed-pairs, p=8.9e-16**; mustard p=1.3e-3; cracker null.
Replicated at n=180.

Six mediator families excluded by direct measurement — contact-local
geometry, IK solvability, joint-limit margin, joint tracking error,
frame/phase-timing semantics, finger-envelope placement + first-contact
timing. Each failed the same test: it does not move when the intervention
moves.

**Precise statement of the open question:** under the currently audited set
of observables, the effect is unexplained. This bounds *these* features,
not all possible static features.

## Why this closes rather than blocks

Two independent lines converge: P1.2 (no promoted pre-execution separator;
first separation at descend) and P2/Phase-A (audited static features do not
mediate the one known causal effect). Both point at the same design
conclusion, which is actionable now:

    pre-execution critic    -> reject clearly infeasible, rank the rest
    execution-time monitor  -> capture the state divergence that only
                               becomes visible from descend onward

That is a two-layer design motivated by measurement rather than by
assumption, and it does not require the P2 mediator to be identified first.

## Carried forward into migration

- Sample-size rule (R5): n=12 cells are not confirmatory here; use ordered
  paired designs. See `docs/EXPERIMENT_HYGIENE_RULES.md` for R1–R5, all
  promoted from concrete failures in this investigation.
- `rel_dist` is a *within-object* criterion only (ρ ≈ −0.5…−0.66 within
  object; ρ = −0.34, p = 0.21 across object cells) — any critic feature on
  it needs per-object normalisation.
- Object set under this baseline: cracker, can, mustard, pear. `banana` and
  `drill` cannot be spawned (placement sampler rejects their radius), so
  their historical results are not reproducible here. `clamp` is
  mechanically infeasible (150mm vs 100mm opening).
- Pear's failures dominate the P1 dataset (18 of 32); pear targeting is
  suspect but uncharacterised, so pear-heavy training data should be
  treated accordingly.

## Optional, not a blocker

Wider pear sweep (±30–45mm, ordered paired) to determine whether +15mm is
an interior optimum or still on a rising edge. Scientifically informative,
constrains the space of remaining explanations, and independent of
migration.
