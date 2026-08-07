# Bilateral engagement mechanism: trajectory instrumentation + aiming offset sweep (2026-08-07)

Follows `docs/BLOCKED_CLOSURE_CODAC_EXTENSION_20260807.md`, which found the
moving-side pad achieves near-ideal contact on Hammer/Banana under S1/S1b
while the fixed-side pad never engages, identically across S0/S1/S1b — ruled
out as a contact-stiffness effect and left as an open mechanism question.

Zero production-code diff, confirmed by `git status`/`git diff` on
`tango_robot/` before and after. Both scripts import (do not modify)
`scripts/microbenchmark_blocked_closure_codac.py`'s frozen-IK/deterministic-
placement machinery.

Reproduce:

```bash
conda run -n tango python scripts/instrument_closure_trajectory.py
conda run -n tango python scripts/sweep_aiming_offset.py
```

## Part 1: closure trajectory instrumentation

Full per-step trace (400 steps, S1_stiff_pads, HammerC and BananaC) of pad
distances, contact flags, object lateral drift (perpendicular to the closing
axis), and object rotation relative to its frozen start pose.

### Headline numbers

| object | first moving contact | fixed dist at that instant | lateral drift (perp) | rotation |
|---|---|---|---|---|
| HammerC | step 5 (~10ms) | 31.8 mm | +7.72 mm | +5.65° |
| BananaC | step 2 (~4ms) | 25.9 mm | +8.63 mm | +9.96° |

**Moving-side contact happens almost immediately** — within 2-5 simulation
steps of closing starting, well before any meaningful object motion could
have occurred (the earlier freeze-time check confirmed both pads start
~44mm from the object centroid, symmetric). The fixed side already being
26-32mm away at that instant means the object's actual near-surface
geometry (not its centroid) was asymmetric relative to the two pads from the
start — the centroid-only aiming reference does not put the object's surface
equidistant from both pads.

**Lateral drift and rotation continue growing for hundreds of steps after
first contact** (Hammer's `perp` timeline: 5.2mm at step 20 → 8.8mm at step
80 → oscillating flat/slowly-growing to ~7.9mm by step 380) — a pattern that
a purely static aiming error would not produce on its own; the object keeps
moving under sustained one-sided contact rather than immediately settling.
Both mechanisms — an initial geometric offset AND a continuing dynamic
drift — appear to be present.

## Part 2: aiming offset sweep — the decisive check

Per the diagnostic question this was designed to answer: does shifting the
one-time IK aim point along the closing axis (before freezing, nothing else
changed) fix bilateral engagement?

| offset | Hammer final (fixed/moving) | Banana final (fixed/moving) |
|---|---|---|
| −15mm | +35.20 / −0.07mm | +27.20 / −0.07mm |
| −10mm | +35.08 / −0.07mm | +28.02 / −0.07mm |
| −5mm | +35.00 / −0.07mm | +25.06 / −0.07mm |
| 0mm (baseline) | +34.96 / −0.07mm | +24.23 / −0.07mm |
| +5mm | +34.86 / −0.07mm | +24.11 / −0.07mm |
| **+10mm** | **−0.06 / −0.06mm ✓ BILATERAL** | +71.57 / +67.48mm |
| +15mm | +7.73 / −0.00mm | +8.74 / +9.86mm |

### Hammer: mechanism confirmed as aiming/reference geometry

**A +10mm offset (toward the fixed side) flips Hammer from persistent
one-sided contact to clean bilateral engagement** (−0.06mm both sides,
matching S1's near-ideal fixture result). Everything from −15mm to +5mm sits
flat around 35mm — the fixed side is essentially insensitive to small
offsets — then drops sharply to near-zero right at +10mm, and overshoots
past it by +15mm. This is a narrow but real sweet spot, not noise.

**This directly confirms this thread's Hypothesis 1**: the static pad
midpoint at a fixed joint angle is not the right aiming reference for a
single-hinge (non-parallel) jaw closing on an asymmetric object — the
correct "capture centre" depends on the object's specific geometry relative
to the closure kinematics, not a universal constant. On Hammer, correcting
for this with nothing but an aim-point shift — no solver change, no friction
change — is sufficient.

### Banana: no offset in this range works — a different or compounding mechanism

No tested offset (−15mm to +15mm) achieves bilateral contact on Banana.
0mm/+5mm are the closest (~24mm, barely better than baseline);  +10mm/+15mm
make it sharply *worse* (67-72mm — the aim overshoots past the object
almost entirely). This does not confirm the offset-only story for every
object: either Banana's correct offset lies outside the tested range, a 1D
offset along a single axis isn't sufficient for its geometry/orientation,
or a second mechanism (continuing slip under sustained one-sided contact,
consistent with Part 1's persistent post-contact drift) is dominant here in
a way aiming alone can't fix.

## What this establishes, and what remains open

**Established**: bilateral engagement failure on Hammer is substantially an
aiming-reference problem, fixable by an object-specific closure-aware
offset, cleanly separable from the contact-stiffness question this thread's
S0/S1/S1b work already answered. The mechanism is not a mystery on at least
one of the two objects tested.

**Not established**: a general rule for computing that offset per object
(what worked was found by 1D grid search on one frozen scene, not derived);
whether Banana needs a different axis, a larger range, or genuinely has a
second failure mode; and — per every doc in this thread — real-hardware
agreement, still untested.

## Relation to the rest of this thread

This is now a third, independent axis in the physics-v2 picture, alongside
opening calibration and contact-stiffness:

```
opening semantics (steps ①②, deferred)
contact penetration fidelity (S0→S1/S1b, established on fixtures + Hammer/Banana moving-side)
bilateral engagement fidelity (THIS DOC — mechanism identified, not yet a general fix)
real-hardware anchor (not started)
```

S1b remains a reasonable contact-parameter candidate for the first axis. It
is not, and was never claimed to be, a fix for the second — this doc is the
evidence for why those are genuinely separate problems requiring separate
fixes, exactly as this thread's prior message anticipated before the data
came in.

## Not done here

- A closure-kinematics-derived formula for the aiming offset (would need to
  understand HOW the +10mm sweet spot relates to Hammer's specific geometry
  and the jaw's hinge kinematics — the sweep found a working NUMBER, not a
  general RULE).
- Extending the offset sweep to Banana beyond ±15mm, or to a second axis.
- The continuing-drift mechanism after first contact (Part 1's second
  finding) is noted, not explained — would need contact-force/friction
  instrumentation this pass didn't build.
- TomatoSoupCanC, deferred since the CoACD extension already flagged it as
  noisier and likely compounded by its unusually large size.
