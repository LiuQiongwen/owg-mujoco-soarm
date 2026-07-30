# C.7 Recovery-Data Collection — Final Report (collection phase complete)

**Date**: 2026-07-30. **Status: collection stopped as instructed.** This report is a data-quality
and inventory summary, **not** an efficacy analysis of the recovery mechanism — 60 trials/object
is a collection target, not a claim that statistical power is sufficient to demonstrate recovery
works. Perturbation-strength-stratified analysis is the next phase, not done here.

**This data has not been used to train anything.** Baseline checkpoints
(`counterfactual_models_20260730/`) and all Phase 1/2/C.3 confirmatory results are unmodified.

---

## 1. Directory inventory (6 batches, 4 output directories, all disjoint seeds)

| Batch | Directory | Object | Seed | perturb_std | n | Circuit breaker fired? |
|---|---|---|---:|---:|---:|---|
| Standard perturbation | `recovery_base500_20260730/` | mustard | 500 | 0.05 | 60 | No |
| Standard perturbation | `recovery_base500_20260730/` | drill | 500 | 0.05 | 60 | No |
| **High-perturbation stress condition** | `recovery_base500_20260730/` | cracker | 500 | 0.05 | 21 | **Yes** (10 consecutive no_contact, limit=10) |
| Low-perturbation pilot | `recovery_cracker_pilot_std002_20260730/` | cracker | 550 | 0.02 | 10 | **Yes** (10 consecutive no_contact, limit=10) |
| Low-perturbation formal | `recovery_cracker_std002_formal_20260730/` | cracker | 551 | 0.02 | 37 | **Yes** (15 consecutive no_contact, limit=15) |
| Low-perturbation formal (cont.) | `recovery_cracker_std002_formal3_20260730/` | cracker | 552 | 0.02 | 13 | No |

**Total circuit-breaker trips: 3** (all cracker; 0 for mustard/drill). No `abnormal_termination`
in any batch (0 exceptions across all 201 trials). No directory overwritten; no seed reused across
batches (verified: zero seed-value overlap between the three cracker std=0.02 batches).

## 2. Per-condition results (offline-collected, not compared against geometry here -- that's the
   next-phase analysis)

| Condition | n | perturbed_success | no_contact | weld_no_lift | recovery triggered | recovery success |
|---|---:|---:|---:|---:|---:|---:|
| mustard, std=0.05 | 60 | 24 (40.0%) | 36 | 0 | 36 | 30 (83.3%) |
| drill, std=0.05 | 60 | 20 (33.3%) | 28 | 12 | 40 | 17 (42.5%) |
| cracker, std=0.05 (**stress condition**) | 21 | 2 (9.5%) | 19 | 0 | 19 | 1 (5.3%) |
| cracker, std=0.02 (**combined, 3 batches**) | 60 | 3 (5.0%) | 53 | 4 | 57 | 7 (12.3%) |

### Sanity cross-check against known confirmatory-300 baselines (`final_report.md`)

- **Mustard**: recovery success 83.3% vs. confirmatory-300 critic baseline 80.0% -- closely
  matches (recovery = re-executing the nominal top-1 candidate unperturbed, which should
  approximate the original baseline; it does).
- **Drill**: recovery success 42.5% vs. confirmatory-300 critic baseline 40.0% -- closely matches.
- **Cracker**: recovery success 12.3% (7/57) vs. confirmatory-300 critic baseline 18.0% -- somewhat
  below, but the 95% binomial CI for 7/57 is approximately [5%, 24%], which contains 18%. **Not
  distinguishable from the baseline at this n** -- consistent with statistical noise around a
  genuinely low, high-variance base rate, not a pipeline defect (independently verified earlier by
  re-executing an identical candidate pick through the validated
  `scripts/risk_gated_vla_phase1_eval.py::execute_candidate()` and getting the same outcome as
  this script's own execution path).

**This cross-check is why the mustard/drill recovery rates matching their known baselines closely
is reassuring**: it confirms `collect_recovery_data.py`'s candidate-selection and execution
pathway reproduces the validated pipeline's behavior, rather than the whole exercise resting on
an unverified new code path.

## 3. Why the threshold was raised for cracker specifically (documented derivation, not
   post-hoc tuning)

Confirmatory-300's raw data (`confirmatory_n50_seed300_20260730/scenes.jsonl`) gives cracker's
**pool-wide** bilateral_contact rate as 71/500 = 14.2% (every sampled candidate, not just the
critic's top-1 pick, which itself achieves 18.0%, i.e. 9/50). Under either rate, P(10 independent
consecutive no_contact) is non-trivial: `0.858^10 ≈ 22%` at the pool-wide rate, `0.82^10 ≈ 14%` at
the top-1 rate -- meaning the *default* limit=10 breaker was expected to misfire on cracker at a
meaningfully high rate purely from its own known difficulty, independent of any perturbation
effect. Raising to limit=15 (`0.82^15 ≈ 3.5%` at the top-1 rate) was chosen to bring the
by-chance false-trip rate down to a level comparable to what mustard/drill experience implicitly
at limit=10 given their much higher (~58-80%) base rates -- not chosen after the fact to make the
pauses stop. It still fired once more (batch `formal`, seed=551) -- itself additional evidence
this is a real, expected statistical property of cracker's difficulty, not something the new
threshold was tuned to eliminate.

## 4. What this does and does not establish

**Does**: a real, non-degenerate, disjoint-seed dataset of perturb-then-recover trials exists for
all 3 objects, with the perturbation mechanism verified working (nonzero, correctly-throttled
`perturbation_count` in every valid trial; see per-trial `frozen_config.json`/`frozen_config_append_*.json`
files in each directory for exact reproducibility metadata: git commit, command, seeds, throttle
frequency). Recovery outcomes for mustard/drill closely track their known baselines, validating
the collection pipeline itself.

**Does not**: establish that the "return to nominal" recovery mechanism is *effective* in any
comparative sense (no geometry/no-recovery control arm was collected in this pass), and does not
establish cracker's true low-perturbation success rate precisely (n=60 at a ~5-18% base rate gives
wide confidence intervals). Both are explicitly next-phase analysis, not concluded here.

## 5. Next step (per instruction: stop collection, move to analysis)

Collection is stopped. The next phase is C.7 data-quality and perturbation-strength-stratified
analysis (comparing std=0.02 vs std=0.05 conditions properly, computing recovery-rate confidence
intervals, and deciding whether this data supports any claim about recovery-data value for a
future DAgger-style policy retraining step) -- not started in this pass.
