Phase 2Y pre-registration — virtual lateral finger-shift causal test

Written before any Phase 2Y data exists.

## Why this replaces the earlier 2A design

The first 2A draft (arm C = "match B's placement without the +15mm
heuristic") was withdrawn: if C reaches B's placement by translating the
EEF, C **is** B — identical physics, different code path — so the
comparison could not have distinguished anything.

The second draft (`do(longitudinal placement)`) was also wrong on three
counts, all corrected here:

1. **Axis.** P2's treatment was applied along local **Y** (lateral,
   `target_mat[:, 1]`). Longitudinal (local Z) placement is a *response*
   to it, not the treatment. Manipulating Z tests a different variable.
2. **Magnitude.** P2's ±15mm lateral sweep induced only ~6mm of
   longitudinal change on pear (24.6 → 26.7 → 23.0 → 21.5 → 20.3). A
   longitudinal intervention matched to P2 would be ±6mm; ±15mm would
   probe a regime P2 never visited.
3. **New confound.** Shifting finger geometry along Z moves the fingers
   vertically in world space, changing table clearance and effective grasp
   height — so "EEF unchanged" would not mean "everything but placement
   unchanged".

## The test

**Question:** with the EEF trajectory *completely unchanged*, does altering
only the finger collision geometry's lateral registration to the object
reproduce P2's ordered success effect?

**Treatment:** `δY ∈ {−15, −7.5, 0, +7.5, +15} mm`, applied to both finger
collision geoms along the gripper's local Y, in a **diagnostic model copy
only**. Levels match P2's actual treatment, so the two are directly
comparable.

**Held fixed:** candidate, EEF target, IK, controller, object initial pose,
gripper closing command.

## Naming (treatment vs response must not be conflated)

- `finger_object_lateral_registration_mm` — the **treatment** (new).
- `finger_longitudinal_placement_mm`, `envelope_fraction` — **responses**
  (existing names retained).

## Smoke gates — all four must pass before the full sweep

3–5 pear seeds, levels {−15, 0, +15} only:

1. **δY = 0 reproduces baseline byte-identically** (trajectory, contacts,
   success). If the null level does not reproduce the unmodified system,
   the instrument is invalid and nothing downstream is interpretable.
2. **Measured finger displacement relative to EEF = commanded δY**, within
   0.1mm.
3. **EEF trajectory unchanged**: `max ‖EEF_shifted(t) − EEF_baseline(t)‖`
   at numerical zero.
4. **No new non-target contacts**: table clearance unchanged; no new
   palm/object contact; no new self-collision. A horizontal shift does not
   change table height but can still alter finger↔palm and finger↔scene
   relationships.

Gate 1 is the one that matters most, per R1 — an instrument that cannot
reproduce the unmodified system is measuring itself.

## Full run

**Pear only** (strongest P2 effect, best-suited to mechanism
identification). Mustard and cracker are deliberately *not* included yet —
adding them before the mechanism is identified spends trials on
replication of an unestablished result.

`5 levels × 30–40 paired seeds = 150–200 trials`, same-seed paired.

**Recorded:** δY; lateral registration; longitudinal placement;
envelope_fraction; first_contact_side; left/right first_contact_step;
contact_step_gap; contact position along finger; object translation and
rotation before bilateral contact; gripper_q at first contact and at close;
lift height; success.

**Negative-control diagnostics, expected to stay flat:** IK residual, joint
margin, `q_cmd − q_achieved`, EEF trajectory, contact-local width,
antipodal score. If any of these move, the instrument is contaminated.

## Decision rule (recorded in advance)

| outcome | conclusion |
|---|---|
| ordered success response in P2's direction, EEF provably unchanged | lateral finger–object registration is a causal mediator |
| δY verified applied, EEF unchanged, success flat | registration is not the mediator — it merely co-varied with the P2 offset |
| success responds but with a different curve than P2 | registration is *one* causal path; P2 also acts through something else |
| δY = 0 fails to reproduce baseline | instrument invalid — discard the run, do not interpret |

## The interesting partial case

If virtual-Y reproduces the success effect while longitudinal placement
moves differently than under P2, the operative quantity is not a scalar
placement value but the **object pose expressed in the gripper contact
frame** (`T_gripper_contact⁻¹ · T_object`). That is the level at which a
swept-volume descriptor (GraspGen-X-style) or local SE(3) refinement
(DiPGrasp-style) becomes the right tool — and only then.

## If negative

Stop working on placement entirely. Move to the full dynamic-contact
trajectory: approach path → first-contact topology → object motion before
bilateral capture → retention. Swept-volume representations may still
matter there, but no claim that lateral placement is the mediator.

---

## AMENDMENT 1 (2026-08-08, after smoke, before any sweep)

Recorded transparently: this changes an **instrument** criterion, justified
by a treatment-independent measurement. No outcome criterion is touched.

### Smoke result

```
GATE 2 (measured shift == commanded):     PASS  (error 0.000mm at ±15mm)
GATE 1/3 (dY=0 == baseline, byte-ident):  FAIL  (max|ΔEEF| 1.6–4.3 ×10⁻⁴ m)
```

### Diagnosis: the platform, not the instrument

Baseline vs baseline — identical code path, no patching whatsoever —
differs by the same order:

```
baseline vs baseline   max|ΔEEF| = 1.06, 0.87, 2.65  ×10⁻⁴ m
dY=0     vs baseline   max|ΔEEF| = 1.62, 1.66, 4.33  ×10⁻⁴ m
```

**The Piper sim is not bit-reproducible run-to-run at trajectory level.**
The EEF noise floor is ~0.1–0.3mm, consistent with the ~1e-13 multithreaded
BLAS noise characterised earlier in the DLS solver, amplified through a
full PD-controlled rollout. Success outcomes remain stable (all matched
here; 8/8 in the earlier same-seed reproduction test) — trajectories do not.

A "byte-identical" gate therefore cannot be met by *any* pair of runs,
including a run against itself. The original threshold was unmeetable, not
merely strict.

### Amended Gates 1 and 3

- **Gate 1:** `dY=0` must match baseline in **success outcome**, and its
  `max|ΔEEF|` must fall within the measured baseline-vs-baseline noise
  distribution — not below 1e-12.
- **Gate 3:** `max|ΔEEF|` for a *shifted* arm must likewise be compared
  against that noise floor, not against zero.
- The noise floor must be re-measured with ≥10 baseline pairs before the
  full sweep, so "within the distribution" is a quantified claim rather
  than an eyeball comparison at n=3.

### Consequence for sensitivity (unchanged conclusion)

The treatment (±15mm, ±7.5mm) is 20–150× the 0.1–0.3mm noise floor, so the
experiment remains well-powered. This also retroactively supports P2's
levels: its smallest step (7.5mm) is ~25× the floor.

### Status

Gate 2 passed exactly. Gates 1/3 are unresolved pending the amended
noise-floor measurement. Gate 4 (no new table/palm/self collisions) is not
yet implemented. **The full sweep must not run until amended Gates 1, 3
and 4 pass.**
