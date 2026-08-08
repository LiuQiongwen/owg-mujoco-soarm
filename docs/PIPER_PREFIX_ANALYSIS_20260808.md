P1.2 trajectory-prefix analysis — when do outcomes become separable? (2026-08-08)

Decides the architecture question: pre-execution critic, temporal monitor,
or failure-detection + recovery.

```bash
conda run -n tango python scripts/analyze_piper_trajectory_prefix.py
```

160 trajectories from PIPER_BASELINE_V1 (128 success / 32 failure).

## Leakage check first

Every rollout is **exactly 121 trajectory samples** (range 121–121, length
AUC 0.50). `run_pick_and_place` executes all phases regardless of outcome,
so no prefix feature can be reading "the rollout ended early". This had to
be checked before anything else — if failures produced shorter rollouts,
every fraction-based prefix feature would partly encode the outcome.

## Result: separation emerges at descend, not before

Same promotion gates as P1.1 (bootstrap CI excluding 0.5, direction
consistency across objects, weakest within-object |AUC−0.5| ≥ 0.10).

| observed through | best \|AUC−0.5\| | promoted? |
|---|---|---|
| pre-execution (candidate/scene only) | 0.11 | borderline — see below |
| `transit_high` | 0.15 | no |
| `approach` | 0.15 | no |
| **`descend`** | **0.21** | **yes** — `last_rel_dist_mm` 0.29, `min_rel_dist_mm` 0.30 |
| `descend_refresh` | 0.35 | yes — `last_grip_q` 0.15 |
| `lift` | 0.44 | yes — `last_rel_dist_mm` 0.06 |
| `transit_above_tray` | 0.43 | yes |
| `lower_into_tray` | 0.48 | yes |

Separability is **monotonically increasing** and first clears the bar at
**descend** — after the arm has moved down to the object but *before the
gripper closes*.

The earliest promoted feature is `last_rel_dist_mm` / `min_rel_dist_mm`:
the object-to-eef relative distance at the end of descend. AUC 0.29 means
**successes have the object closer to the gripper at the end of descend**;
failures have descended to a point further from the object. That is an
aim/landing-accuracy signal, and it is exactly the upstream quantity the
cross-section hypothesis predicts.

## Pre-execution: weak but not absent

Purely pre-execution features (known before any motion):

```
spawn_radius   AUC 0.39  95% CI [0.30, 0.49]   per-object 0.26 / 0.35 / 0.40
spawn_x        AUC 0.56  95% CI [0.46, 0.66]   (spans 0.5 — no evidence)
spawn_y        AUC 0.42  95% CI [0.32, 0.53]   (spans 0.5 — no evidence)
```

`spawn_radius` (distance from robot base) is **borderline**: its CI excludes
0.50 only barely (upper bound 0.49), direction is consistent across all
three objects, and the weakest per-object effect is exactly at the 0.10
threshold. Read as: **objects further from the base fail more**, a real but
weak workspace/reach effect — suggestive, not established.

## What this means for the architecture

The pattern is the "B/C" case rather than "A":

- **A pre-execution critic has a real but limited ceiling here.** The only
  pre-motion signal found is a weak workspace-distance effect. Candidate
  geometry as currently represented does not separate outcomes before
  execution.
- **The decisive signal appears during descend and consolidates through
  close.** Between `approach` (not promoted) and `descend` (promoted), the
  trials diverge — and the variable is landing accuracy relative to the
  object.
- This supports a **two-stage design**: a modest workspace/reachability
  prior at candidate time, plus a **pre-close checkpoint or temporal
  monitor** around descend, where the separation actually becomes visible
  and where there is still time to act (the gripper has not yet closed).

The causal chain now has a consistent, measured shape end to end:

```
spawn/workspace position   (pre-exec, weak: AUC 0.39 borderline)
   -> landing accuracy at descend   (rel_dist, AUC 0.29, PROMOTED, pre-close)
   -> captured width at close       (gripper_q_at_close, AUC 0.15, PROMOTED)
   -> success
```

Every link except the first is promoted under the full gates, and the two
middle links are measurable *before* the outcome is determined. None of
this is yet causal — the P2 cross-section/aim-point intervention is what
would test the middle links.

## Caveats

- 32 failures total, and `can` contributes only 1, so all within-object
  statistics rest on cracker / mustard / pear.
- Prefix features are execution-derived summaries; a richer pre-execution
  representation (local cross-section geometry at the candidate, which is
  *not* in this dataset) could raise the pre-execution ceiling. The
  conclusion is "the features currently available before motion don't
  separate", not "no pre-execution feature could".
- `rel_dist` is the distance between the object body origin and the eef
  site, so part of its variance is object-shape-dependent; the per-object
  AUCs (0.36 / 0.10 / 0.07) do vary considerably even though the direction
  is consistent.
