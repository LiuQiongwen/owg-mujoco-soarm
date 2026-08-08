Mediator chain, full 180-trial run — IK and joint margin definitively excluded (2026-08-08)

```bash
conda run -n tango python scripts/piper_mediator_chain.py 12
```

## Spread across the five offsets (max−min of cell means)

| object | success | IK pos resid | IK ori resid | joint margin | rel_dist | gripper_q |
|---|---|---|---|---|---|---|
| cracker | 10,9,11,11,9 | **0.005mm** | **0.011°** | **0.011 rad** | 7.19mm | 0.0041 |
| mustard | 6,7,9,9,10 | **0.002mm** | **0.002°** | **0.027 rad** | 4.90mm | 0.0117 |
| pear | 3,4,9,11,12 | **0.002mm** | **0.005°** | **0.007 rad** | **20.88mm** | 0.0134 |

The smoke test's reading holds at n=180 and on all three objects: **IK
position residual, IK orientation residual and joint-limit margin are flat
to within microns / hundredths of a degree** across the offsets that move
pear from 3/12 to 12/12. By R3 (a constant cannot mediate a varying
effect), all three are excluded — not "weak", excluded.

Only `rel_dist_at_descend` and `gripper_q_at_close` move.

## rel_dist tracks success within every object

Spearman ρ over all 60 trials per object:

| object | ρ | p |
|---|---|---|
| cracker | −0.496 | 5.6e-05 |
| mustard | −0.659 | 1.0e-08 |
| pear | −0.663 | 8.0e-09 |

Consistent sign, significant in all three: the closer the eef lands to the
object at descend, the more likely the grasp succeeds. Cracker is a useful
internal check — its success peaks at 0/+7.5mm where its rel_dist is
lowest, and falls at +15mm where rel_dist rises again, so the relationship
holds even where the offset effect itself is null.

**But the same relation does NOT hold across objects**: over the 15
(object, offset) cells, cell-mean rel_dist vs cell success rate gives
ρ=−0.342, p=0.21. Each object has its own baseline eef-to-body-origin
distance, so rel_dist is not an absolute criterion — only a within-object
one. Any critic feature built on it would need per-object normalisation.

## Honest limit on what this shows

`rel_dist` is partly determined by the intervention by construction —
shifting the aim ±15mm moves the eef relative to a stationary object. So
its correlation with success is not independent evidence of mediation; it
is substantially the intervention re-expressed. The same caveat applies
less strongly to `gripper_q_at_close`, which is measured after closure.

What the run does establish cleanly is negative and useful: the effect is
**not** routed through IK quality or joint-limit proximity, which were the
two most plausible remaining kinematic explanations.

## Standing state

Excluded as mediators of the P2 effect, each by direct measurement:
contact-local geometry; IK solvability; joint-limit margin; joint tracking
error; frame/phase-timing semantics.

Still open: where the object sits along the finger's finite, asymmetric
contact envelope, and the contact sequence — measured by
`scripts/piper_finger_envelope_probe.py`, run in flight.
