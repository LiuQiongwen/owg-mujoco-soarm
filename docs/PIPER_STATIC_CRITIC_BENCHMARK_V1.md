Piper static-critic benchmark v1 — the pooled number is object identity (2026-08-08)

Migration v1, milestone 1. 160 samples (128 success / 32 failure), 8
pre-execution features, simple auditable models only.

```bash
conda run -n tango python scripts/piper_static_critic_benchmark.py
```

## Headline: the pooled AUC is not measuring grasp quality

```
overall (stratified 5-fold CV)
  logreg  AUC=0.724   Brier=0.1416   (base-rate Brier 0.1600)
  rf      AUC=0.715   Brier=0.1531

per-object AUC (out-of-fold)
  logreg  can=0.231  cracker=0.489  mustard=0.500  pear=0.545   WORST=0.231
  rf      can=0.231  cracker=0.732  mustard=0.583  pear=0.221   WORST=0.221
```

**Pooled AUC 0.72, but within-object logreg is at chance on every object
(0.489 / 0.500 / 0.545) and below chance on `can`.** The pooled score is
almost entirely the model learning *which object this is* — features like
`object_height_mm` and `support_width_mm` are near-perfect object
identifiers, and the objects have very different base rates (can 39/40,
pear 22/40). Reported alone, "static critic achieves AUC 0.72" would have
been a clean, plausible, and completely misleading result.

This is the failure mode the charter's four-report rule exists to catch,
and it fired on the first run.

## Object-held-out, read with care

```
  logreg  can=0.385  cracker=0.736  mustard=0.642  pear=0.573
  rf      can=0.103  cracker=0.840  mustard=0.902  pear=0.630
```

The high RF numbers (cracker 0.840, mustard 0.902) are *not* safe to quote
either. `can` contributes 1 failure in 40, so its AUC is meaningless in
both directions (0.103 is not "inverted signal", it is one sample). And
with 4 objects, held-out-object AUC is computed on a single object's ~40
trials at a time.

## What can honestly be claimed

- **Within-object discrimination is at or near chance for the linear
  model** on all four objects. On the current feature set, a static critic
  does not rank candidates within an object better than random.
- **RF finds some within-object signal on cracker only** (0.732), not
  reproduced on mustard (0.583) or pear (0.221). One object out of four is
  not a result.
- **Calibration is barely better than the base rate** (Brier 0.1416 vs
  0.1600); the model is close to predicting the prior.

## Why this is the right milestone anyway

This is the first Piper static-critic number that is trustworthy, and its
value is precisely that it is negative and correctly scoped. It gives every
later model — local RGB-D, point-cloud encoder, GraspGen-style
discriminator — a baseline that cannot be beaten by accidentally encoding
object identity, because per-object and worst-object are now mandatory
reporting.

It is also consistent with the audit: P1.2 found no promoted pre-execution
separator, and P2/Phase-A found the audited static features do not mediate
the one known causal effect. A third independent method now agrees. The
claim remains scoped as before — *this* feature set is insufficient, which
does not establish that no static representation could succeed.

## Limits (do not over-read the negative either)

- 32 failures total, 18 of them pear. Thin.
- 8 hand-built geometric features, no visual or point-cloud input — the
  representation most likely to carry the signal has not been tried.
- `can` is near-ceiling and should be excluded or treated as a control in
  future runs rather than scored.

## Next

Per the charter, the useful next step is more failure-balanced data before
richer models — a stronger encoder on 32 failures will overfit and the
per-object reports will not be able to detect it.

## Addendum: the P2 grid is already a robustness dataset

The proposed shift from binary labels (`candidate → one rollout → 0/1`) to
robustness labels (`candidate → N perturbed rollouts → success rate`) can be
evaluated on data already collected: the P2 aim-offset grid is exactly
nominal ± local perturbation, 5 offsets × 12 seeds × 3 objects.

Computing `robust_success_rate` per (object, seed) over its 5 perturbations:

```
cracker  mean=0.80 std=0.36   distribution over 0..5/5: [1,1,1,0,0,9]
mustard  mean=0.67 std=0.37   distribution over 0..5/5: [2,1,0,1,4,4]
pear     mean=0.65 std=0.25   distribution over 0..5/5: [0,1,2,5,1,3]

all-or-nothing (0/5 or 5/5):  19/36 candidates
partially robust (1/5..4/5):  17/36 candidates
```

**Nearly half of candidates (17/36) have graded robustness that a binary
label discards entirely.** Pear is the clearest case — its candidates
cluster at 2/5–3/5 rather than at the extremes (std 0.25, the lowest of the
three), i.e. pear candidates are genuinely perturbation-sensitive rather
than simply good or bad.

This is a direct argument for robustness labels over binary ones on this
platform, and it reframes the P2 finding: perturbation sensitivity is not
an anomaly to be explained away by a single mediator, it is **signal about
the candidate** that the current label format throws away.

It also suggests why the benchmark above is at chance within object. The
label it was trained on ("did this exact pose succeed") is partly a coin
flip for 17/36 candidates, so the ceiling on within-object AUC is bounded
by label noise before any feature is considered. A robustness-labelled
benchmark is the correct comparison, and it is the recommended next
milestone.
