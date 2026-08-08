160-rollout separation analysis (P1.1), PIPER_BASELINE_V1 — 2026-08-08

Supersedes `docs/PIPER_EXECUTION_TRACE_20260807.md`'s headline finding.

```bash
conda run -n tango python scripts/piper_outcome_dataset.py 40
conda run -n tango python scripts/analyze_piper_outcome_dataset.py
```

160 rollouts, 4 objects × 40 seeds: can 39/40, cracker 33/40, mustard
34/40, pear 22/40 → 128 successes / 32 failures.

## RETRACTION: contact penetration depth does not replicate

`docs/PIPER_EXECUTION_TRACE_20260807.md` promoted
`min_contact_dist_at_close_mm` as "the first surviving separator", on
within-object AUCs of 0.06 / 0.04 / 0.00 from a 36-rollout sample.

On 4× more data it is not a separator at all:

```
min_contact_dist_at_close_mm   pooled AUC 0.43   95% CI [0.30, 0.56]
  per-object: cracker 0.34, mustard 0.39, pear 0.08
  means: success -1.24mm vs failure -1.07mm   (was -1.68 vs -0.79 at n=36)
```

The CI spans 0.50. The effect shrank by roughly 5×. **Withdrawn.**

This one was caught by the promotion criteria before anything was built on
it, which is the intended behaviour — but the underlying cause is worth
recording, because it invalidates more than one variable.

## Why the small sample misled: spawn position dominates, and 12 seeds is not enough

The 36-rollout trace and this dataset were produced by identical code, yet
per-object rates differ wildly (cracker 33% → 82.5%, pear 75% → 55%).

Determinism was verified first: re-running cracker seeds 3001–3008 today
reproduces the earlier outcomes **8/8 exactly**. So the pipeline is
deterministic and both datasets are individually valid.

That leaves sampling. Under an i.i.d. rate of 0.825, observing 4/12 has
probability ≈1e-4, so seeds 3001–3012 were not merely unlucky — outcome is
strongly determined by the spawn draw, and a 12-seed window can sit in a
systematically hard region of that distribution.

**Consequence: n=12/object is not a usable sample size for this pipeline,
and every conclusion previously drawn at that scale is suspect.** This
applies retroactively to several earlier passes in this investigation
(the 10-seed A/B, the 13-pair orientation A/B/C, the 3-seed contact
traces). It does not un-withdraw anything — those were withdrawn on other
grounds — but it means none of them should be revived as evidence either.

## The one variable that survives all guards: `gripper_q_at_close`

```
gripper_q_at_close     pooled AUC 0.15   95% CI [0.07, 0.24]
  per-object:  cracker 0.28   mustard 0.03   pear 0.13     (all < 0.5)
  leave-one-object-out: can 0.18, cracker 0.10, mustard 0.14, pear 0.17
  means: success -0.0305   failure -0.0194
```

Passes every promotion gate: CI excludes 0.50, all objects agree in
direction, weakest per-object |AUC−0.50| ≥ 0.10, and LOO is stable.

**Interpretation.** The finger joint runs −0.05 (open) → −0.004 (closed),
so *more negative = held further open*. Successful grasps end closing with
the jaws ~30.5mm from closed; failures close ~19.4mm further. The fingers
are physically blocked by whatever is between them, so this variable is
essentially **how much object is actually between the jaws at closing
time**.

That makes it mechanically interpretable rather than merely correlated:
failures close further because they caught a narrower cross-section — an
edge, a corner, or nothing — while successes closed on the object's full
width.

**It is still downstream of the grasp** (a mediator, not a root cause), so
the same caveat as before applies: it diagnoses, it does not yet explain.
But unlike penetration depth, it replicates, and it points somewhere
specific upstream.

## Revision: bilateral contact is weakly informative after all

The previous doc reported `bilateral_at_close` as carrying no signal (AUC
0.54, n=36). On 160 rollouts:

```
bilateral_at_close   pooled AUC 0.61   95% CI [0.54, 0.68]
  per-object: cracker 0.57, mustard 0.58, pear 0.62  (consistent direction)
```

The CI now excludes 0.50 and the direction is consistent, so the
association is real — but weak, and below the promotion threshold
(weakest per-object deviation 0.07 < 0.10). Correct statement: **a weak but
consistent positive association**, not "no association" as previously
written, and not a mechanism.

## Also not promoted, and why (the guards doing work)

- `lift_height_gain_mm` — pooled AUC 0.80, but per-object 0.60 / 0.42 /
  0.98: inconsistent in direction. Also outcome-adjacent (lifting is most
  of succeeding).
- `transit_above_tray_converged` — pooled 0.59, per-object 0.46 / 0.50 /
  0.77. Object-confounded, exactly as in the previous pass.
- `pre_close_drift_mm`, `pre_close_rotation_deg` — pooled 0.25 / 0.30 but
  per-object inconsistent.
- `finger_obj_overlap_at_close_mm` — pooled 0.58, CI spans 0.50.

## Where this leaves P1 and what follows

- `gripper_q_at_close` is the first replicating, object-independent,
  mechanically interpretable separator found in this investigation.
- It is a mediator. The upstream question it poses is concrete and
  testable: **what determines whether the fingers close on the object's
  full width versus a narrow cross-section?** That is a property of *where
  the candidate aims*, which connects directly to P3 (candidate critic)
  rather than to platform physics.
- P2's planned grip-force intervention is now less well-motivated than it
  was: it was designed around penetration depth, which has just been
  withdrawn. A cross-section / aim-point intervention targets the variable
  that actually replicates.

Also worth noting for P1.2: `first_contact_step` scored pooled AUC 0.59
[0.50, 0.68] — borderline, and the trajectory data
(`outputs/piper_outcome_trajectories.jsonl`, 160 rollouts) is collected but
not yet analysed. The prefix analysis is the remaining P1 deliverable.
