# Jaw contact model A/B — Step 3 results (2026-08-07)

Follows `JAW_METROLOGY_FINDINGS_20260807.md`. Step 3 of the agreed 4 → 3 → 1 → 2
order: fix the collider, hold control mapping fixed, measure paired label flips.

Reproduce:

```bash
conda run -n tango python scripts/derive_jaw_pads.py
conda run -n tango python scripts/compare_jaw_contact_models.py \
  --objects ScissorsC HammerC MediumClampC BananaC TomatoSoupCanC \
  --seeds 0 1 2 3 4 --out outputs/jaw_contact_ab.jsonl
```

## The two-arm design was confounded; there are three arms

The plan was to vary contact geometry alone. That is implemented and is exact —
`measured_pads` keeps the proxy spheres in place and merely stops them colliding,
so the jaw-midpoint IK that reads their `geom_xpos` solves to the same joint
angles (verified equal to 0.000000 mm, pinned by a test).

It is still not interpretable on its own, because **the legacy IK target is
itself defined by the wrong geometry.**

`_get_jaw_geom_midpoint()` drives the midpoint of the two finger meshes' frame
**origins** onto the grasp point. Its docstring called those "the true gripping
surfaces". Measured: they are **52–57 mm from the fingers' actual gripping
faces**. Aiming that point at an object parks the finger *roots* on the object
while the fingers extend past it — which is also exactly where
`_simplify_jaw_collision` puts the proxy spheres.

So the approach error and the contact error are anchored to the same wrong
location. Each makes the other invisible. Rendered at the moment the legacy
pipeline calls a grasp successful, the fingers lie across and past the banana
rather than straddling it.

A third arm was therefore added:

| arm | contact geometry | IK aims at |
|---|---|---|
| `proxy_spheres` | 6 mm spheres at the mesh frame origins | sphere midpoint |
| `measured_pads` | box pads measured off the finger meshes | sphere midpoint (unchanged) |
| `measured_pads_aimed` | same pads | **pad midpoint** |

`proxy_spheres` vs `measured_pads` isolates the collider.
`measured_pads_aimed` vs `proxy_spheres` is "corrected geometry end to end".

## Results, 25 paired trials (5 objects × 5 seeds)

Straight-down grasp at the settled centroid, requested opening 0.065 m, so
candidate selection contributes no variance. `GRIP_CLOSED`, `GRIP_OPEN` and
`move_gripper`'s map are untouched — opening calibration is step 1.

| arm | success |
|---|---|
| `proxy_spheres` (legacy) | 16/25 (64%) |
| `measured_pads` (collider only) | 8/25 (32%) |
| `measured_pads_aimed` | **19/25 (76%)** |

Per object:

| object | proxy | pads | pads_aimed |
|---|---|---|---|
| BananaC | 5/5 | 4/5 | 5/5 |
| HammerC | 5/5 | 4/5 | 5/5 |
| MediumClampC | 4/5 | 0/5 | 5/5 |
| ScissorsC | 2/5 | 0/5 | **0/5** |
| TomatoSoupCanC | 0/5 | 0/5 | **4/5** |

Label flip rates vs legacy:

| label | pads (collider only) | pads_aimed |
|---|---|---|
| bilateral_contact | 4/25 = 16% | 7/25 = 28% |
| weld_triggered | 6/25 = 24% | 7/25 = 28% |
| lifted | 8/25 = 32% | 7/25 = 28% |
| success | 8/25 = 32% | **7/25 = 28%** |

### Reading this

**The collider-only arm scoring worse is an artefact of the confound, not
evidence against the pads.** It moves contact onto a surface the arm was never
aiming at. Had the two-arm design been run alone, it would have supported the
false conclusion that the measured pads are worse than the buried spheres.

**Corrected geometry beats legacy** (76% vs 64%) while flipping 28% of success
labels. The gain is not uniform, and the two objects that move most are the
informative ones:

- **TomatoSoupCanC 0/5 → 4/5.** The can was never graspable under legacy
  geometry. With the jaw actually aimed at it and contact on real faces, it is.
- **ScissorsC 2/5 → 0/5.** The two legacy "successes" were welds on buried
  spheres. With real pads, the pads never both reach: the far pad stays 18.6 and
  19.0 mm clear. Scissors at the centroid is genuinely beyond this jaw — which is
  consistent with the 19.4 mm closing floor from step 4 and with the legacy
  manifest entry that scales Scissors 2× in Z to work around exactly that.

## Against the thresholds you set

Success flip is **28%**, which is in the ">15%" band: affected data should be
regenerated and dependent models retrained or re-confirmed, not merely re-tabled.

That said, one qualifier is load-bearing: this is 25 trials on a fixed
straight-down candidate, chosen to remove ranker variance. It bounds the size of
the effect but does not tell you how it lands on the recovery or critic
protocols, which use ranked candidates and different objects. Step 3D (the
critic/recovery regression set) is what decides that.

## What changed in code

- `tango_robot/jaw_pads.py` — derives both pads from mesh geometry in the hinge's
  polar frame. Two earlier approaches are documented there as rejected against
  measurement: closest-to-opposing-mesh averaged over angles (drifts, picks up
  the hinge) and plain SVD over those points (returns rms [43, 4, 3] mm — a
  ridge, not a plane, so its third axis is not a face normal).
- `tango_robot/env_soarm.py` — `jaw_contact_model` with three values;
  `_so101_fragment` injects the pad geoms pre-compilation (MuJoCo has no runtime
  geom-creation API); `<contact><exclude>` on the finger pair, whose convex hulls
  overlap 13.6 mm at the hinge at *every* joint angle — that overlap is the
  reason the sphere hack existed.
- `scripts/derive_jaw_pads.py`, `scripts/compare_jaw_contact_models.py`
- `tests/test_jaw_contact_models.py` — 13 tests pinning pad validity, mode
  isolation, and the measured IK-target offset.

Legacy remains the default and is byte-identical to a clean worktree at
`352e177` on four deterministic grasps — re-verified after each change in this
step.

## Next

- **3D** — run the critic/recovery regression set across `proxy_spheres` and
  `measured_pads_aimed`; check whether `baseline < one-shot < replanning` still
  holds.
- **3E** — decide what to regenerate from the 3D flip rates.
- **1** — calibrate `move_gripper`; `JawMetrology.true_opening_m` is already the
  LUT. Note this now interacts with the pads: the pad-to-pad gap is a different
  quantity from the fingertip-tip gap on a tapered scissor jaw, and the
  calibration should target whichever one the planner is meant to reason about.
- **2** — the closing floor, last.
