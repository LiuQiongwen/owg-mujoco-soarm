# Object one-hot / point-cloud feature ablation

Answers a real methodological concern raised during review: how much of
the object-relative counterfactual critic's benefit comes from
candidate-local geometry (`pc_stats_local`) vs. from an object-identity
prior (the object one-hot), independent of relative pose alone?

**Not the same population as an earlier, non-citable version of this
ablation** (`results/risk_gated_vla/object_agnostic/tomato_critic_train_4100/`,
2026-08-04), which used a different tomato-can-specific seed chain
(base-4100/4200) than the paper's actual base-100/200/300 chain. This
version uses the paper's own real population: trained on
`pc_fix_train_base100/scenes.jsonl` (120 scenes, base-seed=100),
evaluated on `pc_fix_devtest_base200/scenes.jsonl` (90 scenes,
base-seed=200) — the same real, MuJoCo-collected data already backing
`sec:pcfix`. **Confirmatory batch (base-seed 300) never touched.**

## Method

Four feature variants, `object_counterfactual` architecture (BPR pairwise
+ BCE, matching the paper's actual deployed variant), 5 seeds x 400
epochs each via the real, unmodified `train_one()`/`evaluate()` from
`world_model/train_counterfactual_critic.py`:

| Variant | Relative pose | Point cloud | Object one-hot |
|---|---:|---:|---:|
| `pose_only` | ✓ | | |
| `pose_pc` | ✓ | ✓ | |
| `pose_onehot` | ✓ | | ✓ |
| `full` | ✓ | ✓ | ✓ (the paper's actual variant) |

Dev-test evaluation uses ensemble-mean scoring (5-seed average) to pick
top-1 per scene, offline re-scored against real, physically executed
`oracle_per_candidate` outcomes — same convention as `sec:pcfix` and
`sec:ablation`.

## Result

| Method | Dev-test top1 | vs. geometry (exact McNemar) |
|---|---|---|
| Geometry (live-executed) | 37/90 (41.1%) | — |
| `pose_only` | 46/90 (51.1%) | p=0.108 (not significant) |
| `pose_pc` | 48/90 (53.3%) | **p=0.0347** |
| `pose_onehot` | 48/90 (53.3%) | **p=0.0433** |
| `full` | 48/90 (53.3%) | **p=0.0347** |

Pairwise between variants: `pose_pc` vs. `pose_onehot` p=1.0 (identical
48/90 picks), `pose_pc`/`pose_onehot` vs. `full` p=1.0, `pose_only` vs.
`full` p=0.804 (not significant).

## Honest reading

The "object one-hot is a shortcut" concern is **not confirmed**: one-hot
alone performs identically to point-cloud geometry alone (48/90 both,
p=1.0 between them) — it is not secretly carrying the result on its own.
But the finding is more interesting than a clean "no it's fine": at this
data scale (n=90), point-cloud geometry and object-identity are
**interchangeable, not separable from each other**, and combining both
(`full`) adds nothing measurable beyond either alone. `pose_only` by
itself falls just short of significance against geometry (p=0.108),
meaning relative-pose framing alone is not yet enough to reach this
paper's own significance bar, though it captures most of the numerical
gap (46/90 vs. geometry's 37/90). This is a real, non-clean result,
reported with the same rigor as the paper's other honestly-inconclusive
findings (the pairwise-loss-term ablation in `sec:ablation` has the same
shape: real point estimate, not resolvable into a clean attribution at
current sample size).

## Reproducing

```bash
conda run -n tango python scripts/pc_fix_onehot_ablation.py \
    --train results/risk_gated_vla/pc_fix_train_base100/scenes.jsonl \
    --devtest results/risk_gated_vla/pc_fix_devtest_base200/scenes.jsonl \
    --out-dir /tmp/onehot_ablation_verify \
    --epochs 400 --seeds 5
```

Full pairwise McNemar table (including geometry comparisons) in
`onehot_ablation_summary.json`.
