# SO-ARM101 Physics Grasp Calibration Findings

**Date**: 2026-05-16  
**Branch**: feature/vis-aabb

---

## Root Cause: Oblique Approach Geometry

The SO-ARM101 uses a sweeping (oblique) arm trajectory — not a pure top-down descent.
As a result, when the EEF (gripperframe site) is positioned at the object's (x, y, z),
the **scissor jaw bodies are 4-9 cm displaced in +Y** (toward the robot base) from the object:

| Approach EEF y | Jaw-body y | Object y | Gap |
|---|---|---|---|
| −0.404 m | −0.311 m | −0.400 m | 8.9 cm |
| −0.253 m | −0.191 m | −0.250 m | 5.9 cm |

The jaw **contact meshes** extend ~6-7 cm beyond the jaw body in −Y, so the effective
reach is ~2-3 cm short of the object at the standard y = −0.40 position, and the
mesh barely touches at y = −0.25.

---

## Measured Results

### At y = −0.40 (benchmark spawn position)
- Jaw-object contacts: **0** regardless of z_offset, open_length, close_fraction,
  or finger_friction.
- Root cause: jaw mesh tips are 2+ cm short of the object in Y; arm workspace
  limit (−0.4 m) prevents Y overshoot.

### At y = −0.25 (closer to robot)
- Jaw-object contacts: **5/5** (contact always detected).
- Lift success: **0/5** across 81 parameter combinations sweeping:
  - `z_offset` ∈ {−0.02, 0.0, +0.02}
  - `open_length` ∈ {0.06, 0.07, 0.09}
  - `close_steps` ∈ {150, 200, 300}
  - `finger_friction` ∈ {1.0, 2.0, 3.0}

Contact is **one-sided** (gripper palm → cube +Y face only). The moving jaw
does not reach the cube's −Y face, so no bilateral clamping force is generated.
Cube slides in −Y when the arm lifts; friction alone cannot support the weight
without a bilateral normal force.

### Why 12% success for MustardBottle in benchmark?
The mustard bottle (≈19 cm tall) has CoM at z ≈ 0.880. At that approach height,
the arm is in a less-extended configuration, marginally reducing the jaw Y-offset.
Combined with the bottle's large height (jaws sweep through the bottle's Z range),
contact is barely established on ~4% of single attempts.  With `n_grasp_attempts=3`,
the compound rate reaches ≈ 12%.  This is **accidental contact**, not a reliable grasp.

---

## Arm Workspace Test

Maximum reachable EEF y from object at y = −0.40:  **y ≈ −0.426 m** (arm workspace limit).
At that position, moving-jaw mesh front edge reaches y ≈ −0.406 m — just barely touching
the cube's +Y face but with no −Y-face reaction.

---

## Proposed Fixes

### Fix A: 6-DOF Orientation IK (long-term, proper fix)
Implement the `TODO(6dof-ik)` in `_execute_grasp_physics`.  With the gripper oriented
vertically (local Z → world −Z), the jaws open horizontally around the object.
The oblique approach geometry disappears.

### Fix B: Spawn objects at y ≥ −0.35 for physics benchmarks
Objects at y = −0.20 to −0.30 get bilateral jaw contact more reliably.
The benchmark spawn config `centre_y: −0.40` should be changed to `centre_y: −0.25`
if using physics mode for evaluation.

### Fix C: Pre-computed Y-approach offset
In `_execute_grasp_physics`, add `y_approach_offset = −0.05` to the arm's approach y.
This overshoots the object in Y, partially compensating for the jaw-body offset.
Requires extending the EE workspace limit from −0.4 to −0.5.
**Caveat**: arm X-coupling at y < −0.43 shifts geoms by 6+ cm in +X, pushing the
cube sideways before closing.

---

## Recommendation

For the research project's current goals:
- **Physics mode = accurate 0% baseline** for table-level objects.  This is CORRECT.
- Use `--grasp-mode demo_attach` for visual demos and human eval.
- Apply Fix A (orientation IK) before attempting physics-mode performance improvement.
- The calibration data confirms the world-model failure hypothesis: LGGSN/geometry
  ranking makes no difference because the grasp execution itself is the bottleneck.

---

## Files Modified / Created

- `owg_robot/env_soarm.py`: `_park_pos()` fix, primitive pool, `load_primitive()`
- `scripts/calibrate_grasp_physics.py`: calibration sweep infrastructure
- `benchmark/runner.py`: YcbMediumClamp → YcbTomatoSoupCan
- `configs/benchmark/default.yaml`: `objects: cylinder → can`
- `calib_logs/CALIBRATION_FINDINGS.md`: this document
