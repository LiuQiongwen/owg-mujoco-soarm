P2Y-3/4 handoff smoke — 4A and same-instance 4B pass exactly (2026-08-08)

Action-replay design: rather than reimplementing close/lift (blocked by
`run_pick_and_place`'s stack-frame state), the action sequence from
`descend_refresh` onward is recorded during a normal run and replayed from
the restored MuJoCo state. 900 actions captured, state dim 281.

```
4A zero-step identity (restore, no step)
   max|dqpos| = 0.000e+00   OK
   max|dqvel| = 0.000e+00   OK
   max|dtime| = 0.000e+00   OK                      PASS (exact)

4B rollout identity (dY=0 vs dY=0, 900 replayed steps)
   max|d(eef,obj)| = 1.924e-02                      FAIL
   final object z: 0.790743 vs 0.790928
```

## 4A: MuJoCo state restore is exact

`mj_getState`/`mj_setState` with `mjSTATE_INTEGRATION` reproduces qpos, qvel
and time to **exactly zero difference**. The MuJoCo half of the handoff is
sound.

## P2Y-4B first-divergence diagnosis

The initial fresh-env replay was ambiguous. The follow-up compared original
against five zero-treatment replay conditions and added a same-env/model
replay. The handoff was moved to the actual action boundary immediately before
the first recorded `env.step`, rather than the earlier phase notification.

```
same env/model instance:  max|d(eef,obj)| = 0.000e+00 over 900 steps
fresh env:                first divergence at step 0
fresh + controller reset: first divergence at step 0
fresh + gripper state:    first divergence at step 0
```

This localises the failure to cross-instance reconstruction. Contact-heavy
floating-point amplification cannot be the primary cause because restoring and
replaying in the original env/model instance is bit-exact through the same
contact-active close/lift segment. `PiperGripper.current_action` is confirmed
missing Python state and materially reduces cross-instance error, but it is not
sufficient. Explicit composite-controller reset does not repair the mismatch.

## Next diagnostic boundary

- Audit mutable MuJoCo model arrays and robosuite env/robot state across the
  original and freshly constructed instances before adding more snapshot
  fields.
- Replace raw `ncon` changes with explicit finger-object contact onset; the
  handoff currently starts with 11 contacts already active.
- Do not run the Phase 2Y treatment sweep until cross-instance common-state
  construction passes or the treatment driver is redesigned to branch within
  one model instance with a validated model-parameter intervention.

## Consequence

Gate 3 remains suspended for the planned cross-instance treatment driver, so
the five-level sweep stays blocked. Exact same-instance replay is attainable;
there is no evidence here for replacing Gate 3 with a stochastic close-segment
floor.

Outcome field naming (`conditional_lift_success`, never `success`) is
implemented in the driver but not yet exercised, since no treatment
comparison has run.

## TEST 1 result: cause localised to missing Python-side state

```
orig vs replayA :  first |Δ|>1e-9 at step 0/900,  |Δ|=6.48e-04 m  -> final 2.26e-02
orig vs replayB :  first |Δ|>1e-9 at step 0/900,  |Δ|=6.65e-04 m  -> final 2.25e-02
replayA vs B    :  first |Δ|>1e-9 at step 0/900,  |Δ|=1.65e-05 m  -> final 2.16e-04
```

Divergence is **immediate (step 0)**, and the magnitudes separate cleanly:
original-vs-replay starts 40× larger than replay-vs-replay. Pure FP
amplification would start both pairs at the same noise level and separate
later; instead the two replays agree with each other far better than either
agrees with the original. That is a systematic offset introduced by the
restore — **state the original had that a freshly-constructed env does
not.**

**Conclusion: (a) missing Python-side state, not (b) contact amplification.**
Do NOT build a close-segment noise floor — that would absorb a real
missing-state defect as noise, the same trap as widening the episode-level
envelope earlier.

## Open problem: the handoff point is NOT contact-free

`ncon = 11` already at step 0, with the first contact-count *increase* at
step 3. So `descend_refresh` is a **common active-contact handoff state**,
not a pre-contact one. Two consequences, the second potentially serious:

1. **Naming/claims.** Phase 2Y tests "same active-contact state → different
   finger treatment → close/lift outcome". It is not "divergence begins
   from a contact-free state".
2. **Gate 3's validity condition may be untestable at this point.** The
   condition was: first divergence must occur *after* finger/object
   interaction begins. If finger/object contact has already begun at the
   handoff, that condition is trivially satisfied and discriminates
   nothing.

**Unresolved and required before P2Y-4C is worth building:** of the 11
contacts at handoff, how many are finger↔object versus object↔table /
object↔object? The scene holds three objects on a table, so most may be
unrelated to the gripper. If **zero** are finger↔object, the handoff is
effectively pre-contact for treatment purposes and Gate 3's condition
survives intact. If some are, the snapshot point must move earlier — before
any finger↔object contact — or Gate 3 needs a different discriminator.

This is one cheap query against the existing capture (classify `data.contact`
pairs at the handoff step) and should be answered first, since it decides
whether the handoff point is usable at all.

## Handoff contact classification — passes literally, fails in spirit

```
contacts at descend_refresh: 12
  finger <-> TARGET_OBJECT     0
  finger <-> table             2      <-- treatment-sensitive
  object <-> table            10

  table_collision <-> finger7_collision   dist = -0.00048 m
  table_collision <-> finger8_collision   dist = -0.00214 m
```

**Zero finger↔target-object contacts**, so by the frozen rule the handoff is
effectively pre-contact for treatment purposes and Gate 3's validity
condition survives.

**But both fingers are in contact with — and penetrating — the table**
(0.48mm and 2.14mm). This is the third case in the decision rule: a
non-object contact that the dY treatment will move. The finger-shift
treatment displaces exactly the geoms that are currently pressed into the
table, so there is a causal path from treatment to dynamics that does not
pass through the object at all.

Two consequences:

1. **This is the likely cause of the earlier Gate 3 failure** in the
   runtime-mutation instrument, which was never explained: Gate 4 found no
   *new* contacts and ~zero distance deltas, yet the EEF diverged
   pre-object-contact. An existing finger↔table contact being perturbed by
   the geometry shift produces precisely that — no new contact category, no
   large distance change, but altered constraint forces from step 0.
2. **The handoff point is still not clean**, despite passing the literal
   test. Whether the leakage is material depends on whether a *lateral*
   (local-Y) shift changes table penetration: the table is planar, so depth
   should be unchanged, but contact point locations, contact count and
   constraint conditioning can still shift.

## Required before P2Y-4C

Measure it rather than reason about it: for the compile-time dY=0 and
dY=+15 variants, compare at the handoff state the finger↔table contact
count, penetration depths, and contact positions. If those are identical,
the leakage is benign and Phase 2Y can proceed. If they differ, the
snapshot must move to a point where the fingers are clear of the table —
which likely means **earlier in descend, not later**.

Note also that fingers penetrating the table by up to 2.1mm at
`descend_refresh` is a finding about the pipeline in its own right,
independent of Phase 2Y: the gripper is being pressed into the table
surface during descend.

## dY=0 vs dY=+15 at the handoff: LEAKAGE CONFIRMED

Same handoff state (captured in the dY=0 world) restored into both
compile-time variants, compared before any step:

```
dY=0    finger7 dist=-0.000478  pos=[-0.04306, 0.09061, 0.79976]
        finger8 dist=-0.002139  pos=[-0.12387, 0.02331, 0.79893]
dY=+15  finger7 dist=-0.000613  pos=[-0.05253, 0.10224, 0.79969]
        finger8 dist=-0.002273  pos=[-0.13334, 0.03494, 0.79886]

contact count   2 vs 2   (matches)
max |Δdist|     1.343e-04 m
max |Δpos|      1.163e-02 m
```

Count matches, but **penetration depth and contact position both differ
systematically at step 0** — 11.6mm of contact-position shift and 0.13mm of
extra penetration, before a single step. The treatment perturbs table
constraint forces immediately, via a path that never touches the object.
**The `descend_refresh` handoff is invalid as a branch root.**

Note this also corrects the earlier reasoning that a planar table would
leave penetration depth unchanged under a lateral shift. Depth moved too.
Measured, not argued.

It further demotes the missing-Python-state diagnosis: the step-0
original-vs-replay divergence (~6.5e-4 m) is the same order as this
leakage, so both effects were present and the earlier attribution of the
whole discrepancy to missing state was too confident.

## Structural problem: moving the root earlier does not fix isolation

An earlier, contact-free root makes the *root* clean but does not make the
*branch* isolated. The event ordering in this scenario is:

```
root  ->  finger↔table contact  ->  finger↔object contact  ->  close/lift
```

Finger↔table contact is **treatment-relevant** (just shown) and occurs
**before** any object contact. So on any root, the treatment acquires a
table-mediated path before the object-mediated path it is meant to test.
Gate 3's condition ("no divergence before the first treatment-relevant
contact") would then only be checkable in the short window between root and
table contact, and any downstream success difference remains attributable
to either path.

This appears intrinsic to the scenario rather than to the root choice:
`pear` spans +14…+80mm above the grasp reference and sits on the table, so
a gripper reaching its CoM height necessarily brings the fingers to the
table surface. The fingers penetrating the table by 0.5–2.1mm at
`descend_refresh` is the same phenomenon.

**Options, none yet chosen:**
- fix the table-penetration issue in the pipeline first (a production change,
  and the honest prerequisite) — this is on the `piper-execution-v1` freeze
  checklist regardless;
- run Phase 2Y on a *taller* object where the grasp height keeps fingers
  clear of the table, accepting that pear (the strongest P2 effect) cannot
  be tested this way;
- accept a table-mediated path and redefine what Phase 2Y measures — but
  then it no longer isolates the finger↔object relationship, which was the
  entire purpose.

## Independent pipeline finding (do not fix mid-investigation)

`descend_refresh` has both fingers penetrating the table by 0.5–2.1mm. The
current approach/descend semantics permit the gripper to enter table contact
before closing. Belongs on the `piper-execution-v1` freeze checklist:
re-examine table clearance, candidate grasp height, and approach
termination, since it affects contact sequence and sim-to-real.

## Decision: option 1 (fix execution semantics first). Confound to pre-register

Chosen route: diagnostics-only table-clearance-corrected execution branch,
then legacy (L0) vs corrected (L1) under the original P2 five-level design,
to ask whether the Pear offset effect survives removal of the
finger↔table path.

**Confound that must be pre-registered, not discovered later:** achieving
clearance requires raising the grasp/descend pose along the approach axis.
That changes *where on the object the fingers close* — and grasp height is
already known to matter on this platform (the `GRASP_HEIGHT_OFFSET` /
capture-frame work). So L1 differs from L0 in **two** respects, not one:

```
L0 -> L1 :  table contact removed  AND  grasp height raised
```

The magnitude is not negligible. Measured penetration is 0.48mm (finger7)
and **2.14mm** (finger8), so the minimum correction is ~2.1mm+ — the same
scale as the contact-level effects under study. A null result in L1 could
then mean "the table path was the mechanism" **or** "the grasp height moved
off its validated value", and the design as stated cannot separate them.

Options to keep them separable, to be decided before running:

- **Report the correction magnitude per trial** as a covariate, and check
  whether the L0→L1 effect size correlates with it.
- **Add a third arm** — grasp height raised by the same amount *without*
  removing the table path (e.g. table lowered, or a taller object) — so
  height and table-contact are crossed rather than confounded.
- **Verify the correction is uniform** across the five dY levels. If the
  required lift differs per treatment, L1 silently applies a
  treatment-dependent height change, which would be worse than the original
  leakage.

The third check is cheap and should come first: compute required clearance
correction at each dY level before building anything.
