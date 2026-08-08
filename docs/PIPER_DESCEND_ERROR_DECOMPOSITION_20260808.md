Descend error decomposition — the "25mm tracking error" was my own instrumentation bug (2026-08-08)

The three-layer decomposition was requested precisely to stop "25mm eef
error" being named a mechanism before its cause was separated. It was the
right guard: the number is an artifact.

```bash
conda run -n tango python scripts/piper_decompose_descend_error.py
```

| object | offset | IK layer | control layer | geometry layer | "total" | qvel at settle |
|---|---|---|---|---|---|---|
| pear | −15mm | **14.98mm** | 0.098 rad | 0.001mm | 25.86mm | 0.001 |
| pear | +15mm | **15.02mm** | 0.099 rad | 0.001mm | 24.00mm | 0.001 |
| cracker | −15mm | **14.98mm** | 0.115 rad | 0.005mm | 37.83mm | 0.004 |
| cracker | +15mm | **15.02mm** | 0.115 rad | 0.002mm | 35.38mm | 0.002 |

## The artifact

The IK layer reads **exactly the offset magnitude** (14.95–15.05mm at a
±15mm offset). That is the tell.

`_SolveRecorder.solve()` logs the `target_pos` it was *called* with, but
`AimOffsetArmIK` applies the offset *inside* `_solve_impl`, downstream of
that logging. So `c["target_pos"]` is the **unoffset** target while the arm
was commanded to the **offset** one. Differencing the achieved eef against
it measures the deliberate offset, not tracking error.

**Consequences:**
- The "24–26mm tracking error" from the mediator smoke test is not tracking
  error, and the hypothesis built on it ("the offset compensates for a
  systematic undershoot") has no support.
- The same term in `scripts/piper_mediator_chain.py` is contaminated. The
  full 180-trial run currently in flight will produce a spurious monotone
  trend in that column purely because it encodes the offset. The field has
  been renamed `descend_tracking_error_mm_INVALID` so it cannot be quietly
  reused; every other column in that run is unaffected.

This also retroactively explains the earlier note that 25mm was 3–5× the
~4–7mm PD steady-state error documented in the file's own history. The
documented figure was right; my measurement was wrong.

## What the decomposition does establish

With the artifact removed, the three layers say something clean:

- **Geometry layer ≈ 0.001–0.005mm.** `FK(q_achieved)` matches the recorded
  eef pose to within microns. Frame and measurement semantics are correct —
  cause B is ruled out. This is worth having: it was a live possibility.
- **Arm has settled.** Joint velocity at the phase boundary is
  0.001–0.004 rad/s, and eef drift over the following steps is 0.02–0.66mm.
  Descend is not being cut short, so **phase termination semantics are not
  the problem** — which is where the next investigation would otherwise
  have gone.
- **Control layer is real but flat.** Joint error `‖q_cmd − q_achieved‖` is
  0.077–0.115 rad, genuinely nonzero, but it does **not** vary
  systematically with the offset (pear: 0.098 / 0.082 / 0.099 / 0.077;
  cracker: 0.115 / 0.091 / 0.115 / 0.090). Like the geometry features
  before it, a term that is constant across the intervention cannot mediate
  an effect that is monotone in it.

## Where this leaves the P2 mediator

Three candidate mediator families have now been eliminated by the same
test — does the variable actually move when the intervention moves?

| candidate | verdict |
|---|---|
| contact-local geometry (width, antipodal, centring) | constant across offsets → ruled out |
| IK solvability / joint-limit margin | residual ~0.05mm, margin ~flat → ruled out |
| joint tracking error | 0.08–0.12 rad, flat across offsets → ruled out |
| frame / phase-timing semantics | geometry layer ~0 µm, arm settled → ruled out |

What still moves with the offset is `rel_dist_at_descend` and
`gripper_q_at_close` — but `rel_dist` moving is now partly trivial: shifting
the aim by ±15mm moves the eef relative to a stationary object, so some of
that variation is the intervention by construction rather than a mediated
effect. The honest statement is that **the P2 effect is robustly real
(50/0 concordant, p=8.9e-16) and its mediator is still unidentified**, with
the four most plausible mechanical explanations now excluded.

The remaining candidate worth testing is the one this decomposition
implies but does not measure: that the offset changes where the *fingers*
end up relative to the object, independent of geometry and of arm accuracy
— i.e. the eef-to-finger relationship, which this investigation has already
had to correct once (the retracted 65.6mm "TCP offset", and the
finger-contact region spanning [−71.5, +5.0]mm along eef-local Z, not
centred on it).

## Method note

The pattern that caught this — a measured quantity coming out *suspiciously
equal to a parameter I set* — is the same one that caught the pear width
zeros and the innermost-surface collapse. Checking whether a metric equals
its own input is cheap and has now paid off three times in this
investigation.
