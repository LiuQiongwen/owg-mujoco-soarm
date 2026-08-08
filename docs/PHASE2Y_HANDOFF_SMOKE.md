P2Y-3/4 handoff smoke — 4A passes exactly, 4B ambiguous (2026-08-08)

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

## 4B: fails, but the cause is NOT established

The script initially concluded "missing state is outside MuJoCo". **That
overclaims and has been corrected in place.** Two causes are consistent with
this result and the test cannot separate them:

- **(a)** state missing outside MuJoCo — robosuite controller, interpolator,
  phase counters, RNG;
- **(b)** inherent floating-point nondeterminism amplified through contact.

(b) is a live explanation, not a formality: both replays are constructed
identically (same seed, same fresh env) and restore identically (4A exact),
so *no* state difference is required to produce divergence. The measured
1.92e-2 m is ~15× the whole-episode baseline-vs-baseline floor (1.31e-3),
which is consistent with amplification through the contact-heavy close
segment — this platform was already characterised as bifurcating rather
than smoothly noisy.

## How to distinguish (next step)

- Compare each replay against the **original** trajectory, not only against
  each other. If replay tracks the original for a while and then departs,
  that is amplification; if it departs immediately, that is missing state.
- Locate **divergence onset** per step (`Δqpos/Δqvel/Δctrl/ncon`). Missing
  controller state should manifest at or near step 1.
- Replay with the controller explicitly re-initialised vs not, and compare.

## Consequence

Gate 3 cannot be restored on this evidence, and the five-level sweep stays
blocked. But the branching design is not refuted either — if (b) dominates,
exact rollout identity may be unattainable on this platform and Gate 3's
criterion would need re-deriving in terms of the close-segment noise floor
specifically, measured the same way as the episode-level floor was.

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
