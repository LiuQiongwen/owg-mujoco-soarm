# Status: cracker rows in this directory are a HIGH-PERTURBATION STRESS-CONDITION partial set

**Do not treat as the standard-perturbation recovery dataset for cracker.**

`recovery_trials.jsonl`'s first 21 rows (all `object=cracker`, `perturb_std=0.05`,
`base_seed=500`) triggered the collection script's consecutive-no_contact circuit breaker
(10 consecutive `no_contact` perturbed outcomes, trials 11-20) and were stopped for manual review,
per this run's own pre-registered safety rule. Decision: freeze these 21 rows as-is (do not
delete, do not resume collecting more cracker trials at `perturb_std=0.05` into this file).

- Cracker's own unperturbed baseline (confirmatory-300, `final_report.md`) is already low: 18%
  for both geometry and the trained critic.
- At `perturb_std=0.05` (same value used for mustard/drill in this same run), cracker's perturbed
  success rate in these 21 rows is 2/21 (~9.5%), recovery success 1/19 triggered (~5.3%) --
  both below the unperturbed baseline, consistent with the perturbation being disproportionately
  harsh for this specific object rather than a collection bug (the throttle/circuit-breaker
  mechanism itself was independently verified working correctly via a separate n=10 sanity check
  and a smoke re-check of this exact script version before this run started).
- `frozen_config.json` in this directory documents the exact configuration that produced these 21
  rows -- preserved unmodified.

**Disposition**: `perturb_std=0.05` is being treated as a *high-perturbation stress condition* for
cracker specifically, reported separately from the standard-perturbation condition, not mixed into
it. A separate, lower-perturbation pilot for cracker (`perturb_std=0.02`, new seed, new directory)
is the next step -- see `../recovery_cracker_pilot_std002_20260730/` if that pilot has run.

Mustard and drill continue in this SAME directory at the original `perturb_std=0.05` (their own
circuit-breaker behavior, if any, is independent and reported separately per object) -- see
`DATA_COLLECTION_REPORT.md` once their collection completes.
