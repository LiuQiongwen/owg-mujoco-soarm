# Solref stability sweep — corrected results, and a walk-back (2026-08-07)

Follows `SOLVER_CONTACT_ATTRIBUTION_20260807.md`. Zero production-code diff,
confirmed by `git status`/`git diff` on `tango_robot/` before and after —
same pattern as every experiment in this thread: patches an already-compiled
`MjModel`'s numeric fields from a throwaway script.

**This doc corrects an overclaim made earlier in this investigation.** Read
that correction before the results — it changes what the results mean.

> **Further correction (same day, later):** `docs/CONTACT_ONSET_AUDIT_20260807.md`
> found that any table here reading "settled opening vs known thickness" for
> a fixture is contaminated by a reference-surface mismatch between
> `true_opening_m` and the actual pad collision geometry, unrelated to
> solver/stability physics — see that doc for the mechanism and corrected
> numbers on the one case it re-checked (FixtureBox30mm). The velocity/
> warning-based stability findings below (the actual point of this sweep)
> are unaffected, since they never depended on `true_opening_m`.

## The overclaim, and what was actually wrong with it

Checking HammerC/seed=1 under S1 (5ms pad solref) directly, an earlier probe
in this thread found `max object speed = 1876 m/s` and a
`Nan, Inf or huge value in QACC` warning, and reported this as **a genuine,
reproducible instability worse than the "harmless" QACC warning noted in the
prior attribution doc**.

That characterization was wrong, and wrong in a specifically instructive way:
the 1876 m/s reading came from wrapping `step_simulation` across the **whole**
`_execute_grasp` call — which is exactly the mistake `env_soarm.py`'s own
`enable_close_window_diagnostics` docstring already warns about, in words
written before this thread started:

> "measuring raw object velocity across a WHOLE grasp attempt is contaminated
> by legitimate internal teleport/park/restore cycles
> `_execute_grasp_physics_topdown` performs while positioning the arm ... an
> external ad hoc probe caught this only after producing two false
> 'explosion' readings."

The tell, in hindsight, was sitting in the first sweep's own output: the
"~1876-1888 m/s" reading was **identical to 3+ significant figures across
every one of 5 very different solref values tested, including the stock 20ms
default that every prior experiment in this thread has run under without
incident.** A genuine contact-driven instability could not plausibly be
insensitive to a 4x change in contact stiffness; a deterministic kinematic
teleport artifact would be exactly this insensitive.

Re-checked with `enable_close_window_diagnostics=True` (the production,
correctly-scoped tool) on the exact same trial: `close_window_max_speed_mps =
1.02e-14` — machine zero — with an entirely ordinary `-1mm` contact depth.
**No instability in the actual close-window physics for that trial.**

## Corrected sweep: 5 solref values × 6 scenes (30 trials)

Same design as before (2 fixtures + HammerC×3 seeds + TomatoSoupCan, solimp/
priority/cone/impratio held fixed at S1's values, only solref[0] varied
5/7.5/10/15/20 ms), rerun with `close_window_max_speed_mps` in place of the
flawed whole-attempt probe.

### Close-window speed: no instability found at any tested value

| timeconst | max close-window speed (m/s) across 6 scenes |
|---|---|
| 5.0 ms | 0.79 |
| 7.5 ms | 0.61 |
| 10.0 ms | 0.43 |
| 15.0 ms | 0.44 |
| 20.0 ms | 0.43 |

All sub-1 m/s, all physically unremarkable. No genuine contact-driven
blow-up observed at any solref value in this sample.

### The BADQACC warning: fired once, reproducibly, but not attributable to the close window

Same single trial as before (HammerC/seed=1, 5ms) reproduced the warning
again. Its own `close_window_max_speed_mps` for that exact trial:
`1.02e-14`. The warning is real and reproducible, but the close-window
physics it supposedly indicates is clean — consistent with the warning
originating in the teleport/park/restore machinery (same mechanism the
codebase's docstring already names), not in pad contact. With only one
occurrence across 30 trials, this can't be confirmed as solref-independent
either — noted as unresolved, not as ruled out.

### Success rate: too noisy to read as a stability signal

| scene | succeeds at (ms) | fails at (ms) |
|---|---|---|
| FixtureBox30mm/s0 | 5, 7.5, 15, 20 | 10 |
| FixtureCyl40mm/s0 | (none) | 5, 7.5, 10, 15, 20 |
| HammerC/s0 | 5 | 7.5, 10, 15, 20 |
| HammerC/s1 | 15, 20 | 5, 7.5, 10 |
| HammerC/s2 | 5, 15, 20 | 7.5, 10 |
| TomatoSoupCanC/s0 | 5, 7.5, 10, 15, 20 | (none) |

**Non-monotonic and scene-dependent.** FixtureBox30mm fails at 10ms while
succeeding at BOTH the stiffer 5ms and the softer 15/20ms neighbors — not
consistent with a simple "stiffer → more/fewer failures" story in either
direction. This pattern looks like IK/approach convergence sensitivity
(tiny floating-point differences from different contact parameters,
propagated through ~240 settle steps and iterative IK, occasionally landing
on a different local solution) rather than a genuine physical effect of
contact stiffness on success. **At n=1 seed per (scene, config) cell, this
noise floor is comparable to or larger than any solref-driven signal**, so
this sweep cannot responsibly be read as identifying a "stability plateau"
for success rate.

### An unresolved cross-run discrepancy, worth flagging rather than hiding

FixtureCyl40mm/seed=0 at the nominally identical config (5ms, priority=1,
same solimp/cone/impratio) gave **−7.6mm** penetration in the first
attribution experiment (`outputs/solver_contact_attribution.jsonl`) and
**−19.9mm** in this sweep (`outputs/solref_stability_sweep.jsonl`) — same
scene, same seed, same nominal contact configuration, different processes.
Not root-caused. Possible explanations not distinguished here: genuine
run-to-run non-determinism in MuJoCo's solver, or a subtle difference between
the two scripts' setup paths that isn't actually identical despite matching
config values on paper. This bears directly on the "is the rollout
deterministic" question — the honest answer from this data is **not
confirmed deterministic across separate process runs**, which is itself
useful information: any future comparison between configs needs to control
for this (same process, same seed, back-to-back) rather than trusting
separately-run experiments to be bitwise comparable.

## What stands, what doesn't

**Stands**: the first attribution experiment's headline number
(`EXCESSIVE_PENETRATION_DOMINANT` 6/10 → 0/10 under S1) is unaffected by any
of this — that metric is close-window-scoped by construction (built from
`pad_obj_dist_fixed/moving_m` sampled during the measured window, via the
pad-fidelity diagnostic's own classifier) and was never contaminated by the
flawed whole-attempt velocity probe that this doc corrects. Pad-only contact
stiffening genuinely reduces persistent excessive penetration on this sample.

**Does not stand**: any claim about which specific solref value is a
"minimal sufficient" or "stability plateau" choice. This sweep's attempt to
answer that with grasp success rate failed to produce a signal above the
apparent IK-convergence/non-determinism noise floor. Selecting a specific
value (5ms, or anything else) as a real default is not supported by the
evidence collected so far.

## Recommendation

Stop here rather than continue chasing this specific thread further. Two
honest options for whoever picks this up next:

1. **If a specific solref value is needed soon**: don't use success rate.
   Use a metric that doesn't depend on IK/approach convergence at all — e.g.
   penetration depth conditional on bilateral contact having been achieved,
   averaged over many repeats of the SAME deterministic close-only motion
   (skip the approach entirely: teleport the jaw to a fixed pre-contact pose
   and just command the close), which removes the chaos source this sweep
   couldn't control for.
2. **If not urgent**: leave pad contact at the legacy defaults for now. The
   attribution finding (contact softness is a real, fixable contributor) is
   established; the specific fix is not yet validated enough to adopt.

Either way, this is a decision for whoever resumes this thread — not decided
here.
