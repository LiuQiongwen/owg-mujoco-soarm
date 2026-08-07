# Piper T_eef_capture: calibrated and confirmed rigid (2026-08-07)

Follows `docs/PIPER_GRIPPER_AUDIT_20260807.md`'s finding of a 65.6mm gap
between `robot0_eef_site` and the true fingertip midpoint. Answers the
follow-up question directly: is this a genuine rigid transform (safe to
apply regardless of arm/gripper orientation), or does it drift?

Zero diff on `tango_robot/piper_robosuite/` and `tango_robot/piper_assets/`,
confirmed by `git status`/`git diff` before and after. Constructs a standard
`PiperLiftYCB` env (the same class every existing Piper script in this repo
already uses) and reads state only.

Reproduce:

```bash
conda run -n tango python scripts/calibrate_piper_capture_frame.py
```

## Method

Per the explicit warning against a naive world-space patch ("target_pos +=
[0, 0, 0.0656]" breaks the moment the arm rotates): measured the offset in
the `eef_site`'s own local frame, `local_offset = R_eef^T @ (capture_pos_world
- eef_pos_world)`, across 5 arm joint configurations (spanning a real range
around `piper_pick_and_place.py`'s own `READY_QPOS`, not just one pose) ×
3 gripper openings (open/mid/closed) = 15 samples.

## Result: exactly rigid

```
local_offset (every one of 15 samples): [0.0000, -0.0000, -0.0656]  metres
std across all samples:  0.000 mm
max deviation from mean: 0.000 mm
```

**A pure translation of −65.6mm along the eef_site's own local Z axis, with
zero rotational component, identical to floating-point precision across
every arm pose and every gripper opening tested.** This is the cleanest
possible confirmation: `T_eef_capture` is a genuine fixed rigid-body
transform, not an approximation that happens to hold near one pose.

This is mechanically expected in hindsight (both fingers are rigidly
mounted to the same gripper module with no joint between the module root and
the finger attachment points other than the symmetric opening slide, which
by construction doesn't shift the AVERAGE midpoint), but per this thread's
standing practice, verified rather than assumed.

## The transform, ready to use

```
T_eef_capture:
  translation (eef-local frame): [0, 0, -0.0656] m
  rotation: identity

capture_pose_world = eef_pose_world ∘ T_eef_capture
  capture_pos_world = eef_pos_world + R_eef @ [0, 0, -0.0656]

Inverting for IK targeting (given a DESIRED capture-frame pose):
  eef_target_pos = desired_capture_pos - desired_capture_R @ [0, 0, -0.0656]
                  = desired_capture_pos + desired_capture_R @ [0, 0, 0.0656]
  eef_target_R    = desired_capture_R        (rotation is identity, unchanged)
```

## Next

Per the agreed plan, the paired A/B (legacy `robot0_eef_site`-targeted IK vs.
corrected capture-frame-targeted IK, same candidate/object/seed/physics/
success rule, run on `PIPER_FINDINGS_SUMMARY.md`'s representative
objects/failure scenarios) is the next step — not done in this pass. This
calibration is what makes that A/B implementable correctly (via the inverse
transform above) rather than as a fragile world-space patch.
