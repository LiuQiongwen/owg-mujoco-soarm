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

## P2Y-4D resolution: cross-instance reconstruction contract

The forensic audit found no mutable `mjModel` differences. Immediately after
restore, selected integration inputs were equal. The first difference appeared
after controller evaluation:

```text
before controller:       0 differing selected data fields
after controller:        ctrl differs
after force evaluation:  qfrc_actuator / qacc differ
after integration:       qpos / qvel differ
```

The fresh controller retained reset-state `joint_pos` / `joint_vel` caches with
`new_update=False`. Consequently its first `set_goal()` skipped cache refresh
and generated a different torque command. `composite_controller.reset()` did
not fix this because it resets goals from the same stale cache.

The validated reconstruction sequence is:

```text
mj_setState(mjSTATE_INTEGRATION)
restore PiperGripper.current_action
composite_controller.update_state()
each part controller.update(force=True)
replay actions
```

With this sequence, a fresh dY=0 compiled instance reproduced the original for
all 900 steps with `max|d(eef,obj)| = 0.000e+00`.

## Next Gate 3 boundary

- Replace raw `ncon` changes with explicit finger-object contact onset; the
  handoff currently starts with 11 contacts already active.
- Add a regression assertion for the complete cross-instance reconstruction
  sequence before enabling any treatment comparison.
- Keep compile-time treatment variants; do not replace them with runtime
  `model.geom_pos` mutation.

## Consequence

Cross-instance dY=0 identity is restored, but Gate 3 remains suspended until
the explicit finger-object contact definition and reconstruction regression are
part of the formal driver. The treatment sweep therefore remains blocked in
this step.

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

## Five-level clearance-correction uniformity probe

Using the same reconstructed seed-5001 handoff root in five separately
compiled legal variants, the minimum vertical correction needed to place both
finger collision meshes at or above the table was:

```text
dY=-15.0mm   2.0042mm
dY= -7.5mm   2.0714mm
dY=  0.0mm   2.1385mm
dY= +7.5mm   2.2057mm
dY=+15.0mm   2.2729mm
spread       0.2687mm (12.6% of the mean correction)
```

The required minima are therefore not exactly uniform; applying each level's
own minimum would introduce a monotone treatment-dependent grasp-height change.
However, a single common correction of **2.2729mm** (plus any separately
pre-registered safety margin) clears all five roots while keeping the applied
height correction identical across treatment levels. This is a geometry-only
result: no close/lift action or treatment outcome was run.

## P2Y-5A fixed +2.50mm qualification

The diagnostics-only correction is frozen in
`configs/piper/phase2y_clearance_corrected.yaml` as a common **+2.50mm**
world-Z EEF target shift. Its 0.2271mm design margin is not a hardware safety
margin; Hardware Gate 1 remains responsible for that value.

Across all five legal compiled dY variants from the same reconstructed root:

- IK converged with identical 0.066887mm residual;
- the requested +2.50mm target and realized EEF delta were identical across
  dY (realized Z delta 2.43598mm due to the common IK residual);
- finger-table contact count was zero for every level;
- minimum physical signed distance stayed positive, with the worst case
  +0.1624mm at dY=+15mm;
- candidate target XY/orientation, object pose, gripper state, and controller
  semantics remained fixed.

This qualifies the static corrected bundle geometry only. No close/lift action
or treatment outcome was executed, and no claim separates height from table
contact causally.

## Gate 3 endpoint dynamic qualification — FAIL

The minimal corrected `dY=0` versus `dY=+15mm` close/lift test was run from the
same reconstructed root. Reconstruction remained exact and both branches
generated `conditional_lift_success=true`, but the causal gate failed:

```text
reconstruction replica first divergence   none (exact)
endpoint first dynamics divergence         step 0
first finger-target contact                step 88 (dY=0), step 89 (+15)
finger-table contact at corrected root     zero
finger-table contact after first step      present
```

At step 0, finger8 penetrated the table by approximately 0.614mm (`dY=0`) and
0.720mm (`dY=+15`). Thus the static +2.50mm root clearance is consumed by the
first controller transition, reopening the treatment-dependent table path well
before target contact. The endpoint test therefore does **not** qualify Gate 3,
and the five-level sweep remains blocked.

This failure does not reopen reconstruction: the duplicate reconstructed dY=0
branch was exact. It distinguishes static bundle qualification from dynamic
path qualification. No larger margin is selected here; the transient must be
diagnosed before changing the preregistered correction.

## P2Y-5B first-step Z-loss localisation

The loss is in waypoint semantics, not restored velocity or an unexplained
controller transient:

```text
corrected root clearance             +0.2969mm
corrected first-command clearance   -19.7442mm
root EEF Z                            0.806231m
first commanded target EEF Z         0.785349m
commanded descent                   -20.8820mm
root EEF vertical velocity           +0.0000147m/s
```

One-control-step ablations confirmed the classification:

```text
first action, restored qvel          -0.6141mm clearance
first action, arm qvel zero          -0.6127mm
first action, all qvel zero          -0.6127mm
hold corrected root, restored qvel   +0.3110mm
hold corrected root, all qvel zero   +0.3114mm
```

Zeroing velocity does not rescue clearance, while holding the root does. The
first recorded action is still the legacy `descend_refresh` target roughly
20.9mm below the root. Adding +2.50mm uniformly to that target cannot make its
path table-clear. Therefore clearance correction must be specified at the
pre-contact path level; a corrected root plus a constant target offset is not a
qualified execution semantic. No planner, margin, or production behavior is
changed by this audit.

## Superseded / invalid instrumentation notes

Retained for traceability. Neither entry is part of the authoritative state
above; do not cite them as current evidence.

**Δz_clear = 2.50mm requalification — SUPERSEDED.** An ad-hoc query reported
a FAIL verdict from a signed-distance column that was invalid: it selected
table geoms by name without filtering `contype`/`conaffinity`, so a
non-colliding `table_visual` geom returned exactly `0.000000` and won the
`min()`. That is an R7 violation and the reason R7 is now enforced in code
(`scripts/piper_collision_geoms.py`, R9). The question it was attempting to
answer had already been settled correctly by the R7-compliant P2Y-5A
qualification above, which reports the same five levels with valid margins
(worst case +0.16244mm at dY=+15mm). The only part of the ad-hoc result that
was ever sound — zero finger-table contacts at all five levels, read from
`data.contact` — is subsumed by 5A.

**Commit 719ebed provenance caveat.** That commit contains Phase 2Y handoff
documentation beyond the scope its message describes, because the document
was staged wholesale from a dirty working tree that already held another
instance's P2Y-4D content. History is deliberately NOT rewritten: the
content is correct and the ancestry is traceable. The lesson is procedural —
treat commit messages, and file snapshots alone, as insufficient for result
attribution; verify against the recorded outputs.
