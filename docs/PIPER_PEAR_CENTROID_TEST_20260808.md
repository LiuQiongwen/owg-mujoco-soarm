Pear centroid-offset test — inconclusive, and underpowered by construction (2026-08-08)

Three arms, pear, same 12 seeds as the P2 sweep, everything else fixed.

```bash
conda run -n tango python scripts/piper_pear_centroid_offset_test.py 12
```

| arm | `OBJECT_CENTROID_OFFSET_LOCAL["pear"]` | success | aim→obj XY | rel_dist | grip_q | lift |
|---|---|---|---|---|---|---|
| A current | `[-0.0014, +0.0155]` | 9/12 | 14.64mm | 33.0mm | −0.0262 | 68.2mm |
| B zero | `[0, 0]` | 11/12 | 0.00mm | 29.5mm | −0.0279 | 83.7mm |
| C negated | `[+0.0014, −0.0155]` | 11/12 | 14.64mm | 34.6mm | −0.0285 | 82.8mm |

Paired vs A: **B improved 3 / worsened 1**, **C improved 3 / worsened 1**.

## The test cannot answer the question, and I should have seen that first

With 12 seeds the paired comparison yields only 4 discordant pairs. A
3-vs-1 split is p ≈ 0.6 two-sided — nowhere near significance. **The
hypothesis is neither confirmed nor refuted.**

This is a design error on my part, and specifically one this investigation
had already paid for: P1.1 established that n=12 per cell is not a usable
sample size for this pipeline, and I then built an n=12 test to
adjudicate the strongest finding in the study. The P2 pear result was
convincing *because* it was a 5-level monotone paired design (50 concordant
/ 0 discordant); collapsing that to a 3-arm n=12 comparison threw away the
statistical structure that made it convincing.

To resolve this properly: ~40–60 seeds per arm, which is ~2–3× the runtime
of this test.

## One thing the data does say, and it complicates the hypothesis

**B and C perform identically (11/12 each, identical 3/1 paired splits)
despite having very different aim errors** — B aims exactly at the object's
body origin (0.00mm), C aims 14.64mm away from it, the same magnitude as
production but in the opposite direction.

If success were driven by *how far the aim lands from the object centre*,
B should clearly beat C. It does not. So "the centroid correction points
the wrong way, and the aim error is what hurts" is not a sufficient
account of the P2 effect, even setting the power problem aside.

What is at least consistent across A/B/C is `gripper_q_at_close`
(−0.0262 → −0.0279 → −0.0285) and `lift_height_gain` (68.2 → 83.7 →
82.8mm), both tracking success in the same direction as everywhere else in
this investigation — but with 12 seeds those differences are not
distinguishable from noise either.

## Useful consistency check that did pass

A_current gives 9/12, exactly reproducing the P2 sweep's offset-0 cell
(9/12) at the same seeds. The two experiments agree where they overlap, so
the harnesses are consistent even though the conclusion is unresolved.

## Status

- **Pear centroid-offset bug: still an open hypothesis**, now with weak
  and internally-inconsistent evidence rather than none.
- The P2 aim-offset effect itself is unaffected — it stands on its own
  much stronger design (50/0, p=8.9e-16).
- Before P3, either run this at 40–60 seeds per arm, or treat pear's
  candidate targeting as *suspect but uncharacterised* and account for
  that when interpreting anything trained on pear-heavy failure data
  (pear supplies 18 of the 32 failures in the P1 dataset).
