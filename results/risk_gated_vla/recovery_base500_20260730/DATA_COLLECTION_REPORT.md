# C.7 Recovery-Data Collection Report

**base_seed=500**, objects=['mustard', 'drill'], target=60/object. git commit: `be61883dad64ec19552107a72bf36be075ec3d67`. Started: 1785416683.3908594.

| Object | Valid trials | Perturbed successes | Recovery triggered | Recovery successes | Abnormal terminations |
|---|---:|---:|---:|---:|---:|
| cracker | 21/21 | 2 | 19 | 1 | 0 |
| drill | 60/60 | 20 | 40 | 17 | 0 |
| mustard | 60/60 | 24 | 36 | 30 | 0 |

## Failure-type distribution (perturbed attempts)

| Object | success | no_contact | weld_no_lift | abnormal_termination |
|---|---:|---:|---:|---:|
| cracker | 2 | 19 | 0 | 0 |
| drill | 20 | 28 | 12 | 0 |
| mustard | 24 | 36 | 0 | 0 |

## Notes
- Recovery = re-executing the SAME nominal candidate with perturbation turned off, only attempted when the perturbed attempt failed.
- `recovery_success=null` means no recovery was triggered (perturbed attempt already succeeded), not that recovery failed.
- This data has NOT been used to train anything -- collection only, per this round's scope.