# Piper outcome-conditioned execution trace: first surviving separator (2026-08-07)

Built after three consecutive proposed failure mechanisms were withdrawn
(`docs/PIPER_TCP_PREMISE_RETRACTION_20260807.md`). Deliberately inverts the
order that produced those retractions: record continuous per-phase
quantities over matched success/failure rollouts first, look for separators
second. **No "first failed phase" label anywhere** — that construct is what
manufactured the `transit_high` tautology.

36 rollouts under current production code (post-revert), 12 seeds × 3
objects. Per-object: cracker 4/12, pear 9/12, mustard 11/12 → 24 successes
vs 12 failures.

```bash
conda run -n tango python scripts/piper_execution_trace.py
```

## Two controls applied before reading anything

**1. Tautology exclusion.** `dist_to_tray` scored a perfect AUC 0.00. It is
the quantity success is *defined* by (`success == dist_to_tray < tray
threshold`), so it predicts the outcome by construction and carries no
mechanistic information. Excluded. This is the same error class as the
`transit_high` label; the check is now explicit in the script.

**2. Within-object control.** Success rate varies 4/12 → 11/12 across
objects, so any variable that merely differs *by object* will show strong
pooled separation without predicting anything within a fixed object. Every
candidate was therefore re-scored inside each object separately.

Most of the pooled ranking did not survive:

| variable | pooled AUC | cracker / pear / mustard | verdict |
|---|---|---|---|
| `transit_above_tray_converged` / `_min_joint_margin` | 0.71 | inconsistent | object-confounded |
| `pre_close_rotation_deg` | 0.68 | 0.34 / 0.63 / 0.73 | object-confounded |
| `pre_close_drift_mm` | 0.63 | 0.38 / 0.26 / 0.91 | object-confounded |
| `bilateral_at_close` | 0.54 | — | **no separation** |
| `finger_obj_overlap_at_close_mm` | 0.48 | inconsistent | no separation |
| `transit_high_converged` | 0.50 | 0.50 (both means 0.000) | **zero information — independently re-confirms the earlier retraction** |

## The one surviving separator: contact penetration depth at close

```
min_contact_dist_at_close_mm
  within-object AUC:  cracker 0.06   pear 0.04   mustard 0.00
  pooled means:       success -1.679 mm    failure -0.785 mm
```

Same direction, near-perfect separation, in all three objects
independently. MuJoCo's contact `dist` is negative under interpenetration,
so **successful grasps interpenetrate roughly twice as deeply at the moment
of closing; failures make only shallow contact.** Mechanically plausible:
deeper penetration implies higher normal force, hence more friction to
resist slip during lift and transport.

This is the first variable in this investigation to survive both a
tautology check and object-level confounding.

The only other consistent variable, `lift_min_joint_margin_rad` (AUC 0.34 /
0.26 / 0.36), is weaker and points counter-intuitively (successes have
*less* joint margin at lift). Treated as a probable arm-configuration
correlate, not pursued.

## Important caveat: this is likely a mediator, not yet a cause

Penetration depth is measured at the start of `lift` — i.e. *after* the
gripper has closed. A grasp that has already gone wrong (object nudged
away, closed on an edge) would itself produce shallow contact. So this
variable may be a **symptom sitting close to the outcome** rather than an
upstream cause, and it is diagnostic before it is actionable.

Per the standard this investigation has now paid for three times, **it is a
candidate separator only until a directed intervention moves it and the
outcome follows.** No causal claim is made here.

## Next step (not run)

The natural intervention: penetration depth at a fixed commanded closure is
governed by grip force (`kp=1000`, `forcerange="-20 20"` on the finger
actuators in `piper_gripper.xml`). Raising it should deepen penetration
directly. If success rate rises with it, the mechanism is causal and
actionable; if penetration deepens but success does not follow, it is
confirmed a mediator and the real cause lies further upstream (i.e. in
whatever determines *where* the fingers land before closing).

That is a diagnostic-only physics change and must not be applied to
production without a separate validation pass — the existing contact
parameters were deliberately frozen.

## Status

- Cause of Piper grasp failures: **still not established**, but for the
  first time there is a surviving, object-independent candidate rather than
  a withdrawn one.
- `bilateral_at_close` does **not** predict success (AUC 0.54) — the
  bilateral-engagement framing carried through several earlier passes has
  no outcome support in this data.
- `transit_high` non-convergence re-confirmed as carrying zero outcome
  information, independently of the earlier analysis.
