Milestone 2 pre-registration — robustness-labelled critic benchmark

Written BEFORE any Milestone 2 data is collected. Recorded so the design
cannot be tuned to results, per the audit's own history of hypotheses that
looked strong until tested.

Status when written: Milestone 1 delivered (`f46d58c`, `ece7af5`) —
binary-label benchmark, pooled AUC 0.72 shown to be object identity,
within-object AUC at chance for logistic regression on all four objects.

## Hypothesis (falsifiable, stated before the test)

**H1.** Milestone 1's within-object chance performance is caused primarily
by *label noise*, not by feature inadequacy. A single 0/1 outcome is a
high-variance label for perturbation-sensitive candidates: 17/36 candidates
in the P2 grid are partially robust (1/5–4/5), so the label is near a coin
flip for roughly half the data.

**Prediction if H1 is true:** the *same 8 pre-execution features*, trained
against robustness labels, give materially better within-object performance
than against binary labels.

**Prediction if H1 is false:** within-object performance stays near chance
under robustness labels too, which is stronger evidence that the
representation — not the supervision — is the limit, and justifies
investing in C2 (candidate-local visual / point-cloud + embodiment).

Either outcome is informative. The comparison that matters is
**same features, different label**, so the two must not change together.

## Pre-registered perturbation set (fixed now, not tuned later)

Chosen before collection and NOT to be adjusted after seeing results. Note
the ±15mm range in P2 was itself selected during exploration, so reusing it
unmodified would risk fitting the label definition to known outcomes; the
set below is therefore specified independently and symmetrically.

- Axes: local Y (along-object) and local X (closing axis), applied
  separately, descend phase only.
- Magnitudes: {0, ±5, ±10} mm on each axis → 9 perturbations per candidate.
- Seeds: ≥ 30 candidates per object (vs 12 in the exploratory grid).
- Objects: cracker, mustard, pear. `can` excluded from scoring (39/40,
  near-ceiling; retained as a control only).

## Label format

Store **k and N**, never a bare float. `robust_success = k/N` discards the
sample size, and 7/10 is not 70/100. Persisting k/N permits
binomial-aware evaluation later (Wilson intervals, weighted losses,
uncertainty-aware calibration) without recollection.

Each candidate row records: `k`, `N`, `robust_rate`, Wilson 95% CI, and the
per-perturbation outcomes.

## Models

- **C0** — the same 8 geometric pre-execution features as Milestone 1.
- **C1** — C0 + embodiment feasibility (opening margin, IK feasibility,
  joint-margin minimum, workspace radius). Feasibility terms are *gates and
  features here*, not treated as grasp quality, per the audit's finding
  that IK residual and joint margin do not explain outcome differences.
- **C2** — candidate-local visual / point-cloud representation +
  embodiment descriptors. Only run after C0/C1 pass all checks.

Simple auditable models first (logistic / ridge regression, random forest).

## Mandated reports (all four, every model)

overall · per-object · worst-object · object-held-out.
Plus, for the regression form: Brier score, calibration (ECE / reliability
curve), and ranking quality after binning robustness — correlation alone is
not sufficient.

## Exploratory vs confirmatory

- The existing n=36 P2-derived robustness estimates are **exploratory
  only** and may not be used as confirmatory evidence.
- Confirmatory claims require the pre-registered collection above.
- R1–R5 (`docs/EXPERIMENT_HYGIENE_RULES.md`) apply throughout. In
  particular R1: any AUC, Brier or calibration figure landing suspiciously
  on chance, on a bound, or on the base rate is an instrumentation fault
  until the measurement chain is checked.

## Decision rule (recorded in advance)

| result | conclusion |
|---|---|
| C0 within-object improves materially under robustness labels | supervision was the limit; keep the feature set, fix the labels |
| C0 stays at chance, C1 improves | embodiment feasibility carries signal binary labels obscured |
| C0 and C1 both stay at chance | representation is the limit; C2 is justified |
| any pooled figure improves while per-object does not | object identity again — reject, do not report |
