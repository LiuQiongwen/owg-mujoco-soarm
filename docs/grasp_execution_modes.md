# Grasp Execution Modes

This document describes the three grasp execution modes available in `owg_robot/env_soarm.py`
and explains which mode to use for evaluation, training, and demos.

---

## Summary

| Mode constant | Value | Contact gating | Lift mechanism | Use for |
|---|---|---|---|---|
| `GRASP_MODE_PHYSICS_WELD` | `"physics_weld_after_bilateral"` | physics bilateral check | kinematic weld if gate passes | **all benchmarks, training labels** |
| `GRASP_MODE_PHYSICS` | `"physics"` | physics bilateral check | kinematic weld if gate passes | legacy alias for `PHYSICS_WELD` |
| `GRASP_MODE_DEMO_ATTACH` | `"demo_attach"` | proximity (no contact req.) | unconditional kinematic weld | semantic demo recordings only |

---

## Mode 1: `physics_weld_after_bilateral` (recommended)

**Constant**: `GRASP_MODE_PHYSICS_WELD = "physics_weld_after_bilateral"`  
**Function**: `_execute_grasp_physics_topdown()`

### Execution sequence

```
1. reset_robot()
2. move_gripper(open)
3. move_ee([x, y, HOVER_Z], IK_MODE_JAW_POS)      # jaw-midpoint IK to hover
4. _solve_ik_jaw_pos_only([x, y, grasp_z])         # IK at grasp height
5. park objects → teleport arm → restore objects   # avoids penetration impulse
6. _steps(settle)
7. auto_close_gripper()  +  _steps(settle)
   ↓
   CHECK: bilateral_contacts (both jaw spheres touch object)?
     YES → _attach_obj(obj)       # kinematic weld: object follows EEF each step
     NO  → no weld
8. move_ee([x, y, HOVER_Z], IK_MODE_XYZ_ONLY)     # lift
9. _steps(100)
   ↓
   CHECK: obj_z > TABLE_TOP_Z + 0.07?
     YES → success=True, keep weld (object is "held")
     NO  → _detach_obj(), success=False
```

### Result fields in `last_grasp_metrics`

| Field | Type | Meaning |
|---|---|---|
| `exec_mode` | str | `"physics_weld_after_bilateral"` |
| `bilateral_contact` | bool | Both jaw spheres contacted the object post-close |
| `weld_triggered` | bool | `True` iff kinematic weld was activated (= `bilateral_contact`) |
| `table_contact` | bool | Any jaw sphere contacted the table surface post-close |
| `final_z` | float | Object z-coordinate after lift attempt (metres) |
| `lifted` | bool | `obj_z > TABLE_TOP_Z + 0.07` (7 cm clearance) |
| `success` | bool | `lifted and weld_triggered` |
| `bilateral_contacts` | int | 0 or 1 (legacy field, same as `bilateral_contact`) |
| `left_contacts` | int | Contacts on fixed jaw body |
| `right_contacts` | int | Contacts on moving jaw body |
| `jaw_obj_xy_gap` | float | XY distance jaw-midpoint → object CoM at close time |

### Design rationale

The SO-ARM101 scissor jaw uses 6 mm sphere collision geoms at each jaw tip (see
`_simplify_jaw_collision()`). These geoms generate point contacts with small normal
forces — insufficient to overcome the ~0.15 kg object weight through friction alone
(required: ≥0.41 N per jaw at μ=2.0; actual: < 0.1 N due to contact softness).

Pure-physics lift therefore fails with ~0% success regardless of grasp quality,
making the success signal uninformative for training or evaluation.

The bilateral-contact gate preserves scientific integrity:

- **Success requires actual jaw–object contact** on both sides.  A weld that never
  triggers means no grasp contact → failure, correctly reported.
- **The success signal reflects grasp placement quality**, not friction coefficient
  tuning: a well-positioned grasp (both jaws straddle object) yields `weld_triggered=True`;
  a misaligned grasp does not.
- **Table-contact is separately tracked**.  If `table_contact=True`, the fixed jaw
  sphere contacted the table surface during close, indicating the grasp was too low.
  This was the root cause of 0% success before the `GRASP_Z_TABLE_MARGIN=0.020`
  offset was introduced (see git log for `fix: bilateral_contacts=0`).

### Grasp z targeting

```python
# In ui.py _setup_grasps_mujoco and debug scripts:
from owg_robot.env_soarm import GRASP_Z_TABLE_MARGIN  # 0.020 m

grasp_z = float(com[2]) + GRASP_Z_TABLE_MARGIN
```

`GRASP_Z_TABLE_MARGIN = 0.020` is derived from jaw geometry:
- Half jaw z-span: ~10 mm (distance from jaw midpoint to lower jaw tip)
- Sphere radius: 6 mm
- Safety clearance: 4 mm
- Total: **20 mm above object CoM z**

This ensures the fixed (lower) jaw sphere clears the table surface for all YCB objects.

---

## Mode 2: `physics` (legacy alias)

**Constant**: `GRASP_MODE_PHYSICS = "physics"`  
**Dispatches to**: same `_execute_grasp_physics_topdown()` as `GRASP_MODE_PHYSICS_WELD`

This constant is kept for backward compatibility with existing scripts and configs.
All new code should use `GRASP_MODE_PHYSICS_WELD` explicitly.

---

## Mode 3: `demo_attach`

**Constant**: `GRASP_MODE_DEMO_ATTACH = "demo_attach"`  
**Function**: `_execute_grasp_demo_attach()`

### Execution sequence

```
1. Pre-select nearest object by XY within _GRASP_XY_PRESEL radius (10 cm)
2. move_gripper(open)
3. move_ee([x, y, HOVER_Z])    # EEF-site IK (no jaw targeting)
4. move_ee([x, y, z])
5. auto_close_gripper()  +  _steps(80)
6. _attach_obj(pre_target, offset=[0,0,-0.03])   # snap 3 cm below EEF, no contact check
   OR fallback: _attach_obj(nearest within 12 cm)
7. move_ee([x, y, HOVER_Z])    # lift with object kinematically following
8. success if obj_z > TABLE_TOP_Z + 0.07
```

### When to use

- **Semantic demo recordings** where visual plausibility matters but metrics do not.
- **Do NOT use for**:
  - Benchmark evaluation
  - LGGSN training label generation
  - Any comparison reported in a paper

The success signal is unconditional on jaw contact: if any object is within 12 cm
of the EEF at close time, the grasp "succeeds" regardless of jaw alignment.

---

## Result fields comparison

```
                        physics_weld_after_bilateral    demo_attach
bilateral_contact       ✓ real physics contact          ✗ not checked
weld_triggered          ✓ = bilateral_contact           always True if obj in range
table_contact           ✓ detected                      ✗ not checked
final_z                 ✓ real object z                 ✓ real object z
lifted                  ✓ obj_z > table+0.07            ✓ same
success                 ✓ lifted AND weld_triggered     ✓ lifted (no contact gate)
```

---

## Choosing grasp_z

Always compute `grasp_z` from the object's **body CoM** (not free-joint qpos):

```python
com     = env.get_obj_com_pos(obj_id)    # data.xpos[body_id] — actual world CoM
grasp_z = float(com[2]) + GRASP_Z_TABLE_MARGIN
```

`get_obj_pos()` returns the free-joint origin, which may be offset from the CoM by
mesh alignment. For the YCB banana, this offset was 4 mm — enough to place the jaw
above the banana entirely when combined with the now-removed GRASP_Z_OFFSET=0.06.

---

## Benchmark runner integration

`benchmark/runner.py` reads new fields from `env.last_grasp_metrics` automatically:

```python
weld_triggered = m.get("weld_triggered", None)
table_contact  = m.get("table_contact",  None)
final_z        = m.get("final_z",        None)
lifted         = m.get("lifted",         None)
```

These are stored in `TrialResult` and written to the trial log CSV.
The default `BenchmarkConfig.execution_mode` is now `GRASP_MODE_PHYSICS_WELD`.

---

## Validation results

See `scripts/validate_grasp_mode.py` for the standalone validation script.

Run:
```bash
MUJOCO_GL=egl python scripts/validate_grasp_mode.py \
    --objects Banana MustardBottle TomatoSoupCan \
    --n-trials 20 --seed 0
```

See `docs/grasp_mode_validation_results.md` for the most recent benchmark output.
