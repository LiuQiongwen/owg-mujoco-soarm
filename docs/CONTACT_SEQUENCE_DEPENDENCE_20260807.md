# Contact-sequence dependence confirmed; mechanical feasibility gate added (2026-08-07)

Executes items 1 and 4 of the follow-up plan after
`docs/CAPTURE_REFERENCE_DERIVATION_20260807.md`'s negative result (static
geometric capture reference insufficient). Zero production-code diff,
confirmed by `git status`/`git diff` on `tango_robot/` before and after.

Reproduce:

```bash
conda run -n tango python scripts/compare_offset_first_contact.py
```

## Item 1: offset=0 vs +10mm, direct first-contact/force/rotation comparison

Same Hammer scene, same S1_stiff_pads config, only the one-time IK aim point
differs. Per-step contact scanning identifies exactly when/where each pad
first touches, its contact normal (world frame), and the normal/tangential
contact force at that instant (`mj_contactForce`).

| | offset = 0mm (fails) | offset = +10mm (succeeds) |
|---|---|---|
| **moving** first touch | step 5, pos [0.0106, −0.4142, 0.8081] | step 42, pos [−0.0061, −0.4225, 0.9405] |
| **fixed** first touch | never | **step 1** (essentially instant) |
| rotation @ step 50 | 2.44° | **40.17°** |
| rotation @ step 100 | 2.91° | 29.57° |
| rotation, final | 5.65° | 28.07° |

**Moving pad's contact points differ by 133.68mm between the two runs** — not
a small shift, a different part of the object entirely. **Which side touches
first is reversed**: at 0mm, moving touches first and fixed never engages
(matching every prior measurement in this thread); at +10mm, **fixed touches
first, almost instantly** (step 1), and moving only catches up 41 steps
later, after the object has already been perturbed by the fixed-side impact.

## Conclusion: closure outcome is contact-sequence dependent, not aim-precision dependent

This directly confirms the hypothesis the capture-reference derivation's
negative result pointed to: **the +10mm offset does not locate a more precise
static centre — it changes which surface gets touched first, entirely.** The
large early rotation spike at +10mm (40° by step 50, settling back down to
28°) is consistent with the fixed pad "catching" the object almost
immediately and inducing a substantial reorientation as the object settles
against it — a qualitatively different dynamical event, not a refinement of
the 0mm trajectory.

This rules out (further, on top of the capture-reference result) the
"correct the static aim point" framing as sufficient, and supports the
user's redirected question: for a candidate pose, does there exist *some*
closure behavior (a function of exactly where first contact lands and in
what sequence) that leads into bilateral capture — a capture-basin question,
not a single-point aiming question. Building a full swept-surface/sequential-
contact model is the natural next step this result argues for, but is
substantially larger scope and is not started here without further explicit
direction.

One number worth flagging without over-interpreting: the reported
normal forces (85–427 N) are large relative to a ~0.4kg hammer's weight
(~4 N), consistent with a stiff (5ms solref) contact model's peak transient
impact force at the instant of first contact, not a steady-state load — not
chased further here, but a candidate detail for whoever next examines contact
force fidelity directly.

## Item 4: mechanical feasibility gate

Cheap, standalone check using data already established in this thread:

- `TomatoSoupCanC`'s local collision support width (measured at its frozen
  grasp pose, `docs/BLOCKED_CLOSURE_CODAC_EXTENSION_20260807.md`): **106.7mm**
- SO-101 jaw's full mechanical opening range (`calib/jaw_opening_lut.json`,
  step ①): **2.1mm to 95.7mm**

`106.7mm > 95.7mm` — **TomatoSoupCanC's local grasp width exceeds the jaw's
maximum possible opening at this candidate pose.** No contact configuration,
no aiming correction, and no closure-dynamics model can make this candidate
bilaterally capturable; it fails a purely mechanical, embodiment-level
necessary condition before any contact physics is even relevant. This is
exactly the kind of check the user proposed as an embodiment-aware candidate
feature: `local_support_width_m < max_mechanical_opening_m`, computed from
already-available production and thread-derived tooling
(`object_local_thickness_m`-style slab measurement + the opening LUT), no new
infrastructure required.

**Not implemented as a production gate here** (would touch candidate-scoring
code, out of scope for this thread's constraints) — recorded as a validated,
ready-to-use feature definition for whoever next touches candidate
generation or critic scoring.

## Updated status

| axis | status |
|---|---|
| Collision pad geometry | resolved |
| Penetration fidelity | resolved (S0→S1/S1b) |
| Grasp-reference 52–57mm error | resolved |
| `q → opening` geometric LUT | resolved |
| Formal metric opening API | not wired |
| Bilateral engagement | **confirmed contact-sequence dependent**, not an aiming-precision problem; swept/sequential model is the indicated next step, not yet built |
| Mechanical width feasibility | validated as a usable candidate feature, not yet wired into any candidate pipeline |
| Contact-induced drift (Banana) | not yet instrumented |
| Real-hardware anchor | not started |
