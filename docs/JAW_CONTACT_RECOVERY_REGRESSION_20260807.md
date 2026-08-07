# Step 3D — does the recovery result survive the corrected jaw geometry?

Follows `JAW_METROLOGY_FINDINGS_20260807.md` (step 4) and
`JAW_CONTACT_MODEL_AB_20260807.md` (step 3). Paired re-run of the attached-state
recovery protocol under both jaw contact models.

Reproduce:

```bash
N=15 SEED=1041 bash scripts/../  # see the runner recorded in each result's "config"
conda run -n tango python scripts/eval_attached_recovery_mujoco.py \
  --checkpoint paperA_data/lerobot_datasets/owg_3obj_act_clean_n50_deployable/checkpoints/010000/pretrained_model \
  --object cracker --scenes 15 --base-seed 1041 \
  --templates results/risk_gated_vla/act_recovery/recovery_success_templates_cracker_seed961_n15.json \
  --ranker-train-results results/risk_gated_vla/act_recovery/recovery_candidate_oracle_cracker_templates15_safehome_train961_n30.json \
  --jaw-contact-model measured_pads_aimed --quiet --out outputs/step3d/...json
conda run -n tango python scripts/analyze_jaw_contact_recovery_regression.py --dir outputs/step3d
```

Same checkpoint, seeds, spawns, templates, RF ranker, thresholds and variant
definitions as the frozen seed-1041 confirmation spec; `--jaw-contact-model` is
the only thing that differs. 3 objects × 15 scenes = 45 paired trials per arm
(the frozen run used 50/object; this is a regression check, not a replacement).

## Read this caveat before the numbers

**The ACT policy, the recovery templates and the RF candidate ranker were all fit
under legacy geometry.** Running them under corrected geometry is a train/test
mismatch, so the corrected arm's absolute success rates are a lower bound on what
re-fit components would achieve. Nothing below shows that recovery does not work.
What it shows is which evidence transfers and which does not.

## The ordering holds; the headline does not

Your criterion was whether `baseline < r0 < r1` survives. It does, in both arms:

| arm | baseline | r0_regrasp_only | r1_plus_attached_lift | ordering |
|---|---|---|---|---|
| `proxy_spheres` (legacy) | 3/45 | 19/45 | 27/45 | holds |
| `measured_pads_aimed` | 2/45 | 10/45 | 11/45 | holds |

But the frozen spec's **primary comparison** is r1 vs r0 — what the attached-lift
rule adds over regrasp alone, which is the selected variant's whole claim.
Recomputed within each arm, paired, exact two-sided McNemar:

| arm | r0 | r1 | gain | discordant | p |
|---|---|---|---|---|---|
| `proxy_spheres` | 19/45 | 27/45 | **+17.8 pp** | b=8, c=0 | **0.0078** |
| `measured_pads_aimed` | 10/45 | 11/45 | **+2.2 pp** | b=1, c=0 | **1.0000** |

The result is significant under the legacy collider and disappears under the
corrected one.

## Why, mechanically

`r1_plus_attached_lift` only ever fires on one failure mode: the policy ends with
the object **welded to the gripper** but not lifted far enough. That mode is
largely an artefact of the buried spheres.

| | legacy | corrected |
|---|---|---|
| attached insufficient-lift failures | **10/45** | **2/45** |
| unattached recoverable failures | 32/45 | 41/45 |
| attached-lift attempts / successes | 10 / 8 | 2 / 1 |

Under legacy geometry the weld fired on spheres buried inside objects the jaw was
not really holding, producing 10 "attached but stuck" scenes. With contact on real
pads the weld gate is much harder to satisfy, those scenes mostly do not arise,
and the rule built to rescue them has almost nothing to rescue. Its +17.8 pp is
substantially the repair of an artefact.

Per-scene label flips, paired by (object, scene index):

| label | flip rate |
|---|---|
| baseline_act_success | 5/45 = 11.1% |
| intervention_reason | 22/45 = 48.9% |
| r0_regrasp_only | 21/45 = 46.7% |
| r1_plus_attached_lift | 20/45 = 44.4% |

Intervention-reason transitions (legacy → corrected): 11 `policy_horizon_exhausted
→ collision_displacement`, 6 the reverse, 3 `policy_success → collision_displacement`.
Roughly half the scenes reach a different state, which is consistent with both a
real geometry change and the train/test mismatch above — these two cannot be
separated without re-fit components.

## Against the thresholds you set

Success flips 44% on the selected variant, and the method's own primary
comparison loses significance. That is past ">15% or ordering changed", so:

- **Regenerate, do not re-table.** The recovery training data (templates, RF
  ranker training results) and the ACT episodes were produced under geometry now
  known to be wrong in two independent ways.
- **Re-establish r1 from scratch** if it is to be kept. The honest statement today
  is that its evidence rests on a failure mode the corrected collider mostly
  removes — not that the rule is useless, which this run cannot show given the
  mismatch.
- **The ordering claim is the salvageable part.** `baseline < r0 < r1` holds in
  both arms; recovery still helps. It is the *size and significance* of the
  attached-lift increment that does not transfer.

## Suggested order for the redo

1. Re-collect ACT episodes under `measured_pads_aimed`, retrain the policy.
2. Re-fit recovery templates and the RF ranker on corrected-geometry outcomes.
3. Re-run the frozen protocol at full n (50/object) and re-test r1 vs r0.
4. Only then decide whether the attached-lift rule stays in the paper.

Step 1 has to come first: everything downstream is conditioned on the policy's
failure distribution, and that is what shifted most (48.9% of scenes changed
intervention reason).

## Still outstanding from step 3

- The visual critic has not been re-evaluated. Its outcome labels come from the
  same contact signal, so it needs the freeze-checkpoint / re-execute / re-score
  pass before any claim about it stands.
- Steps 1 and 2 (opening calibration, closing floor) are untouched by design.
