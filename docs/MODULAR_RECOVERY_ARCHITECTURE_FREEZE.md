# Modular recovery controller — frozen architecture

**Status: FROZEN 2026-08-06.** This document fixes the retained system's exact
component stack, each component's measured contribution, what was tried and
rejected, and the boundary of what the evidence supports. It is the reference
for any downstream write-up; it is not a proposal for further work.

Placed under `docs/` rather than `results/` deliberately: `results/` is
gitignored, and this project has already lost a cited result's evidence chain
to exactly that (see the `sec:pcfix` incident, 2026-08-06).

---

## 1. Retained architecture

ACT proposes nominal actions; classical components handle distribution shift
and unsafe actions. Monolithic learned recovery is **not** part of the system
(§3).

```
ACT policy (clean, 10k steps)
  └─ nominal action proposal
displacement-threshold failure trigger   (0.02 m)
  └─ R0: RF-ranked regrasp from a refreshed candidate pool
        └─ R1: attached-state lift            (offset 0.08 m)
              └─ current-state replanning     (≤3 attempts, 15 candidates each)
                    └─ IK-validity gate       (abort before execution if 0 valid)
success criterion: lift ≥ 0.07 m
```

Frozen parameters (from
`replanned_regrasp_confirmation_seed1071_n150_spec.json`, `frozen_before_results`,
2026-08-05):

| Parameter | Value |
|---|---|
| `max_policy_steps` | 80 |
| `displacement_threshold_m` | 0.02 |
| `candidate_count_per_attempt` | 15 |
| `maximum_replanned_attempts` | 3 |
| `attached_lift_offset_m` | 0.08 |
| `success_lift_threshold_m` | 0.07 |

---

## 2. Measured contribution of each component

**Batch A — seed 1041, n=150** (`ATTACHED_RECOVERY_CONFIRMATION_SEED1041.md`),
50 paired trials/object, cracker + mustard + drill:

| Stage | Cracker | Mustard | Drill | Pooled |
|---|---:|---:|---:|---:|
| ACT alone | 7/50 (14%) | 2/50 (4%) | 24/50 (48%) | **33/150 (22.0%)** |
| + R0 unattached regrasp | 17/50 (34%) | 24/50 (48%) | 37/50 (74%) | **78/150 (52.0%)** |
| + R1 attached lift | 22/50 (44%) | 36/50 (72%) | 39/50 (78%) | **97/150 (64.7%)** |

- R1 vs R0: 19 R1-only successes, **0** R0-only — exact two-sided McNemar
  **p = 3.81e-6**
- R1 vs ACT: 64 vs 0 — **p = 1.08e-19**
- R0 vs ACT: 45 vs 0 — **p = 5.68e-14**
- The rule is nested: R1 converts no ACT or R0 success into a failure.
- R1 recovered 19 of 34 attached-insufficient-lift cases.

**Batch B — seed 1071, n=150** (`REPLANNED_REGRASP_CONFIRMATION_SEED1071.md`):

| Object | R1 top-1 | R1 + current-state replanning | Gain |
|---|---:|---:|---:|
| Cracker | 15/50 (30%) | 19/50 (38%) | +4 |
| Mustard | 33/50 (66%) | 33/50 (66%) | 0 |
| Drill | 41/50 (82%) | 46/50 (92%) | +5 |
| **Pooled** | **89/150 (59.3%)** | **98/150 (65.3%)** | **+9 (+6 pp)** |

- 9 replanning-only successes, **0** top-1-only — exact McNemar **p = 0.0039**
- All 9 gains followed a refreshed state estimate: 8 on attempt 2, 1 on attempt 3.
- Contrast: a static top-3 ablation (more candidates, no refreshed state) added
  **0** successes in 60 paired trials. The effect is replanning, not candidate count.

**IK-validity gate** (`ik_safety_gate_counterfactual_seed1071_n150.json`):
rule = "stop before grasp execution when IK-valid candidate count is zero".
Counterfactual audit over the same 150 trials: gate events are rare and cost
nothing — `counterfactual_lost_successes: 0`. Decision: `safe_to_enable`.

> **Do not compare Batch A and Batch B row-for-row.** R1 pooled is 97/150
> (64.7%) at seed 1041 but 89/150 (59.3%) at seed 1071. These are independent
> confirmatory batches, not a single ablation ladder. Each batch's *internal*
> paired comparison is the valid claim; the cross-batch difference is
> seed/scene variation.

---

## 3. Explicitly excluded (tried, measured, not retained)

| Component | Evidence | Decision |
|---|---|---|
| Monolithic ACT recovery fine-tune (clean + confirmed recovery, 4k) | 7/30 vs 7/30, McNemar p=1 | stop |
| Recovery-only specialist (2k) | 0/15 | stop |
| RaC Round-1 specialist (4k) | 0/15 | **do_not_promote** — root cause recorded: the collector labeled all 8394 frames `correction`, so retreat/regrasp/lift were never separable supervision |
| Phase-pilot augmentation (2k) | 1/15 vs 2/15, p=1 | do not promote |
| One-shot 3 mm contact servo | 19/30 vs 19/30, p=1.0, zero discordant; 23 of 24 corrections still ended without bilateral contact | reject (this heuristic, not contact feedback in general) |
| Adaptive lift extension | dev (seed 1081, 20/object): gain only on mustard, +2 (0 before-only, p=0.5) → `object_specific_candidate`; mustard confirmation (seed 1091, n=50): 39→40, +1 (0 before-only), p=1.0 | **not_confirmed** |
| Static top-3 candidates | 0 added successes / 60 paired trials | reject |

**Failure-detector MLP — trained but NOT in the system.**
`failure_detector_mlp_seed0.pt` + calibration exist (threshold 0.32; test
tp=17 fn=1 fp=1 tn=17, balanced accuracy 0.944, but **n=36 only**;
validation balanced accuracy 0.806). Verified by grep: no evaluation script
consumes it — only `scripts/calibrate_failure_detector.py` references it. The
deployed trigger remains the 0.02 m displacement threshold. This is a
newly-grown component held **outside** the freeze pending an explicit
decision; integrating it would be a new experiment, not a characterization of
the retained system.

---

## 4. What this evidence does and does not support

**Supports:** on 3 YCB objects in MuJoCo, a modular controller — ACT for
nominal action proposal plus classical failure handling — raises closed-loop
grasp success from 22.0% to ~65%, with each retained component's contribution
independently confirmed on a fresh seed under a pre-registered, frozen spec,
using exact paired McNemar tests.

**Does not support:**
- **Any real-hardware claim.** All results are simulation. There is zero
  real-hardware paired data. `real_robot_preflight_20260804.json` reports
  `ready_for_motion: false` with 5 blockers (serial device, motor calibration,
  RealSense device, hand-eye independent validation, hand-eye RMSE ≤ 10 mm).
- **A uniform mechanism story.** Both retained gains are object-concentrated:
  R1's gain is mustard-dominated (+24 pp mustard vs +4 pp drill); replanning's
  gain is cracker+drill only (0 on mustard). Pooled p-values are strong; the
  per-object mechanism is not uniform.
- **Generalization beyond cracker/mustard/drill**, or beyond this ACT
  checkpoint and this candidate generator.

**Next-experiment gate (carried forward from
`ACT_RECOVERY_DECISION_SUMMARY.md`, unchanged):** do not spend another
training run on monolithic ACT recovery unless a new intervention changes the
control formulation. Any next ACT experiment must add an explicit deployable
signal or objective and be evaluated against this retained controller with
strict paired seeds. Otherwise prioritize real-robot validation.
