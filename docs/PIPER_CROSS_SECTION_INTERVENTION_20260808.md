P2 cross-section / aim-point intervention — 2026-08-08

180 trials, 3 failure-bearing objects × 5 aim offsets × 12 seeds, paired by
seed. Offsets applied along the grasp frame's local Y (perpendicular to the
jaw-closing axis), descend phases only, zero production diff.

```bash
conda run -n tango python scripts/piper_cross_section_intervention.py 12
```

## The intervention has a large, unambiguous causal effect — on two of three objects

Paired sign test over concordant/discordant (offset↑ → success↑) seed-pairs:

| object | success by offset (−15 … +15mm) | concordant/discordant | p |
|---|---|---|---|
| **pear** | 3, 4, 9, 11, **12** /12 | **50 / 0** | **8.9e-16** |
| **mustard** | 6, 7, 9, 9, 9 /12 | 21 / 5 | 1.3e-3 |
| cracker | 10, 9, 9, 11, 9 /12 | 5 / 5 | 0.62 |

Pear is perfectly monotonic in every seed that varies at all (mean per-seed
correlation +0.83, zero discordant pairs out of 50). This is the first
intervention in this investigation to move outcomes decisively, and unlike
the earlier n=12 results it is a paired 5-level design, which is why the
sample size objection from P1.1 does not apply the same way.

## But the hypothesized chain is NOT supported

The predicted chain was:

```
candidate-local geometry -> landing accuracy -> captured width -> success
```

Mustard is the only object where the width feature works correctly (see the
pear bug below), and there the first link fails:

```
offset    local_width   rel_dist   gripper_q    success
-15.0mm      49.5mm      52.9mm     -0.0196      6/12
 -7.5mm      50.4mm      51.2mm     -0.0262      7/12
  0.0mm      51.6mm      49.5mm     -0.0268      9/12
 +7.5mm      50.1mm      48.5mm     -0.0301      9/12
+15.0mm      48.8mm      48.0mm     -0.0313      9/12
```

`local_width` **peaks at offset 0 and falls off on both sides**, while
`rel_dist`, `gripper_q` and success all move **monotonically**. So success
does not track the local cross-section width at all — the widest aim point
is not the best aim point.

What does track success monotonically is `rel_dist_at_descend` (down 52.9 →
48.0mm) and `gripper_q_at_close` (more open, −0.0196 → −0.0313), i.e. the
two variables P1.1/P1.2 already promoted. **The intervention moves the
promoted mediators and the outcome together, but the proposed upstream
geometric cause is not what is driving it.**

## Bug: the width feature is invalid for pear

Pear's `jaw_band_width_mm` and `local_width_mm` are both **0.0** in every
trial. Cause: the width is measured in a ±15mm z-band around the aim point,
but pear's mesh spans +14 to +80mm *above* the grasp reference, so the band
sits almost entirely below the object and captures fewer than 5 vertices.

This matters twice over. It means the first link is **untested on the
object with the strongest effect**, and it is a reminder that the aim point
(an eef-site target) is not where the fingers are — the fingers extend
upward from roughly the eef site, so a contact-region band should be
centred well above it. Any future version of this feature must be defined
on the finger contact region, not on the eef target.

## The more likely explanation: a systematic aim error, specific to pear

The pear effect is **asymmetric**, not peaked at zero: −15mm gives 3/12 and
+15mm gives 12/12, monotonically. A "aim at the thicker cross-section"
mechanism would be symmetric about the object's centre. A systematic offset
in the nominal aim point would look exactly like this.

`OBJECT_CENTROID_OFFSET_LOCAL["pear"] = [-0.0014, 0.0155]` — a 15.5mm
correction along the object's local y, very close to the +15mm offset that
takes pear to 12/12. And `rel_dist` falls by 21mm across the sweep, meaning
the +15mm aim lands the eef *closer* to the pear's body origin, not further
— which is what you would expect if the nominal aim were displaced by
roughly that amount in the opposite direction.

**Hypothesis (untested): pear's centroid offset is applied with the wrong
sign, or in the wrong frame, so the pipeline systematically aims ~15–30mm
off, and the +15mm intervention partially cancels it.**

This is cheap to test decisively and is the obvious next step: re-run pear
at offset 0 with `OBJECT_CENTROID_OFFSET_LOCAL["pear"]` zeroed and negated,
and compare against the +15mm condition. If negating it reproduces 11–12/12,
the effect is a production bug rather than a cross-section phenomenon.

Note the file's own history is consistent with this being fragile: the
comment above that constant records two separate attempts to re-measure
these offsets, both reverted after making results worse, and notes the
values were derived from a z-slice at `GRASP_HEIGHT_OFFSET` — a height that
this investigation has since shown is not where the fingers actually close.

## Status of the chain after P2

- `rel_dist_at_descend` and `gripper_q_at_close`: promoted in P1.1/P1.2 and
  now shown to **move together with the outcome under intervention**. That
  is a substantial upgrade from correlation, though it does not establish
  which of them is causal versus mediating.
- `local cross-section width`: **not supported** as the upstream cause
  (non-monotonic in mustard, invalid in pear). The P1.2 hope that it would
  raise the pre-execution ceiling is not yet realised — and cannot be
  judged on pear until the feature is fixed.
- Effect is **object-specific**: strong in pear, moderate in mustard,
  absent in cracker. Not a general mechanism on this evidence.

## Caveats

- 12 seeds per cell. The paired 5-level monotonic design makes the pear and
  mustard results robust to that, but the cracker null is not strong
  evidence of absence.
- Only three objects, all of which happen to bear failures; `can` excluded
  by design for having ~1.
