# Pad-contact fidelity diagnostic — results (2026-08-07)

Read-only, opt-in diagnostic. Does not modify success rules, weld triggering,
`GRIP_CLOSED`/`GRIP_OPEN`, `move_gripper()`, contact/actuator parameters, or any
historical result file. Verified byte-identical to a clean worktree at `352e177`
on the standard 4-trial regression probe, re-checked after this work.

Code: `tango_robot/pad_fidelity.py` (pure classifier, no MuJoCo dependency),
`tango_robot/env_soarm.py`'s `enable_pad_fidelity_trace` (opt-in recording hook),
`scripts/collect_pad_fidelity_diagnostic.py`, `scripts/analyze_pad_fidelity.py`.
17 pure unit tests + 7 env-level tests in `tests/test_pad_fidelity.py`, all passing.

Reproduce:

```bash
conda run -n tango python scripts/collect_pad_fidelity_diagnostic.py \
  --objects ScissorsC HammerC MediumClampC BananaC TomatoSoupCanC \
  --seeds 0 1 2 3 4 --jaw-contact-model measured_pads_aimed \
  --out outputs/pad_fidelity.jsonl
conda run -n tango python scripts/analyze_pad_fidelity.py --in outputs/pad_fidelity.jsonl
```

## Why this exists

Step 3's A/B (`docs/JAW_CONTACT_MODEL_AB_20260807.md`) showed `bilateral_contact`
can be driven by interpenetration a real pad could not produce. A prior quick
check (using an ad hoc close-probe proxy built from already-collected data)
found something more specific: on Hammer/MediumClamp/Banana, the jaw's settled
joint angle barely deviates from its free-closing target even while pads
penetrate 12–14 mm into the object — i.e., the position actuator (±3.35 force
limit) can compress substantially into these CoACD multi-part meshes without
being mechanically stalled. That observation was informal; this diagnostic
formalizes it: classify every sampled timestep of the close+settle window from
pad-to-object signed distance alone, independent of the contact solver's own
bilateral/weld decision, and report where they agree and disagree.

## What the classifier does

Four states, computed from `pad_obj_dist_fixed_m` / `pad_obj_dist_moving_m`
(exact `mj_geomDistance`, previously verified equal to `contact.dist` to within
0.03 mm):

- `NO_BILATERAL` — both pads clearly clear of the object
- `PLAUSIBLE_BILATERAL` — both pads within a small window around zero distance
- `EXCESSIVE_PENETRATION` — either pad sunk deeper than the plausible window;
  **takes priority over everything else**, so a single deeply-buried pad can
  never be diluted into a "plausible" verdict by a good-looking opposite side
- `AMBIGUOUS` — anything else (chiefly: one pad touching, one clear — a
  unilateral near-contact, which by construction cannot be bilateral)

Per-step states are aggregated into **persistent runs** (default: 8 consecutive
sub-sampled steps, ≈64 ms) before counting toward engagement/excessive
durations or the trial-level `geometric_verdict`, so single-sample solver noise
cannot flip a trial's classification.

Default thresholds (`PadFidelityConfig`): 1 mm contact tolerance, 6 mm plausible
penetration ceiling (= 2× the derived pad's own 3 mm thickness,
`tango_robot/jaw_pads.py`'s `PAD_HALF_THICK`). **Chosen from pad geometry before
this collection ran, not fit to its results** — see the sensitivity check below
for why that distinction matters here.

## Results, 25 trials (5 objects × 5 seeds, `measured_pads_aimed`)

Same deterministic spawn/candidate protocol as `compare_jaw_contact_models.py`.

### Item 4 — legacy vs geometric confusion (trial level)

| | NO_ENGAGEMENT | PLAUSIBLE_ENGAGEMENT | EXCESSIVE_PENETRATION_DOMINANT | AMBIGUOUS |
|---|---|---|---|---|
| legacy_bilateral=False | 0 | 0 | 1 | 5 |
| legacy_bilateral=True | 0 | 0 | **19** | 0 |

**Every legacy-success trial (19/19) carries a persistent excessive-penetration
run.** Not one success in this sample has a clean `PLAUSIBLE_ENGAGEMENT`
verdict.

### Items 2–3 — engagement vs excessive-penetration duration, pooled

| | samples | share |
|---|---|---|
| plausible bilateral engagement | 113 | 8.2% |
| excessive penetration | 607 | 44.1% |
| ambiguous | 503 | 36.6% |
| no bilateral | 0 | 0.0% |
| total | 1375 | |

### Item 5 — per object

| object | n | legacy_succ | verdict | median min-dist (fixed/moving, mm) |
|---|---|---|---|---|
| BananaC | 5 | 5 | 5× EXCESSIVE | −15.4 / −13.9 |
| HammerC | 5 | 5 | 5× EXCESSIVE | −14.3 / −13.5 |
| MediumClampC | 5 | 5 | 5× EXCESSIVE | −8.3 / −7.4 |
| ScissorsC | 5 | 0 | 5× AMBIGUOUS | −3.9 / 0.0 |
| TomatoSoupCanC | 5 | 4 | 5× EXCESSIVE | −9.1 / −14.2 |

Scissors never reaches a persistent bilateral run at all (consistent with
step 3's finding that the far pad stays 18.6–19.0 mm clear on this object at
this grasp point) — its 5 trials are `AMBIGUOUS`, not `EXCESSIVE`, and its
legacy success count is correctly 0.

### Item 6 — reclassification (report only, nothing overwritten)

19/25 legacy-success trials print with `geometric_verdict =
EXCESSIVE_PENETRATION_DOMINANT` alongside their unmodified `legacy_success =
True`. The full table is in the script's stdout; `outputs/pad_fidelity.jsonl`
holds every trial's `legacy_success` field byte-for-byte as the grasp function
returned it — this diagnostic reads it once for the confusion table and writes
it to a *new* file, never back into any existing result.

## Is 100% real, or a threshold artifact?

Worth stress-testing before trusting a number this extreme. Using each trial's
already-recorded **minimum** pad distance (a looser proxy than the full
persistence-based verdict — it just asks "did either pad ever cross this
threshold," no persistence requirement) across a range of plausible-penetration
ceilings, without rerunning any simulation:

| threshold | trials ever exceeding | of 19 legacy successes |
|---|---|---|
| 3 mm | 23/25 | 19/19 |
| 6 mm (default) | 20/25 | 19/19 |
| 9 mm | 15/25 | 14/19 |
| 12 mm | 14/25 | 13/19 |
| 15 mm | 5/25 | 5/19 |
| 20 mm | 0/25 | 0/19 |

The result is not a knife-edge artifact of picking exactly 6 mm: it holds at
every threshold up to 12 mm (nearly all successes) and only fully disappears
past 15–20 mm — 5–7× the pad's own thickness, well past any plausible reading
of "compliant contact." The finding is robust to reasonable threshold choice,
not an artifact of one number.

## What this does and doesn't show

**Does show**: on this sample, the geometric evidence for "the pads are
touching the object the way real rigid pads would" is absent for the large
majority of trials the legacy pipeline calls successful. The weld gate
(`bilateral_contacts`) is satisfied by penetration depths (up to −19 mm) that a
non-deformable 3 mm-thick pad cannot physically produce against a rigid object.

**Does not show**: that these grasps "aren't real" in an outcome sense — `dz`/
`lifted`/`final_z` still reflect the object moving with the gripper, because the
kinematic weld (triggered by this same contact signal) holds it there
regardless of how the contact arose. This diagnostic cannot and does not
distinguish "weld papered over a bad contact but the object still ends up
usably positioned" from "weld papered over a bad contact and the downstream
behavior is wrong" — that requires looking at what depends on the *contact
geometry itself* (e.g., recovery/critic labels keyed to `bilateral_contact`),
not just the outcome the weld already guarantees.

## Relationship to prior findings

This corroborates, with a controlled instrument, what an earlier ad hoc check
suggested informally, and gives it a name and thresholds that don't move
depending on who's looking: the position actuator's limited force
(`forcerange="-3.35 3.35"`) combined with MuJoCo's default contact compliance
lets these CoACD-decomposed objects be compressed well past where a rigid pad
would stop, without the joint angle itself visibly stalling. Step 3D's
"attached-lift" finding (recovery's headline result resting on a failure mode
that's mostly collider artifact) and this diagnostic's "most legacy successes
show excessive penetration" are two views of the same underlying issue.

## Next

- This diagnostic doesn't fix anything — by design. The two live options from
  the earlier discussion stand: (1) harden `solref`/`solimp` (elliptic friction
  cone, Newton + MultiCCD) so genuine rigid-pad stalling becomes visible in the
  joint angle, which is the deeper fix but touches contact parameters this work
  was scoped to leave alone; or (2) build an actual (not just diagnostic)
  contact-validity gate on `pad_obj_dist` with a documented, geometry-derived
  threshold, replacing `bilateral_contacts` in the weld/success decision. Both
  are new work, not something this diagnostic pass should decide unilaterally.
- If (2) is pursued, this module's `PadState`/`GeometricVerdict` machinery is
  already the right shape for it — the only change would be using its verdict
  to gate `weld_triggered` instead of only reporting it.
