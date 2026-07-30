# C.7 Recovery-Data Collection Report

**base_seed=552**, objects=['cracker'], target=13/object. git commit: `be61883dad64ec19552107a72bf36be075ec3d67`. Started: 1785417363.897049.

| Object | Valid trials | Perturbed successes | Recovery triggered | Recovery successes | Abnormal terminations |
|---|---:|---:|---:|---:|---:|
| cracker | 13/13 | 0 | 13 | 0 | 0 |

## Failure-type distribution (perturbed attempts)

| Object | success | no_contact | weld_no_lift | abnormal_termination |
|---|---:|---:|---:|---:|
| cracker | 0 | 11 | 2 | 0 |

## Notes
- Recovery = re-executing the SAME nominal candidate with perturbation turned off, only attempted when the perturbed attempt failed.
- `recovery_success=null` means no recovery was triggered (perturbed attempt already succeeded), not that recovery failed.
- This data has NOT been used to train anything -- collection only, per this round's scope.