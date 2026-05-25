# SO-ARM101 Official Alignment Audit

**Date:** 2026-05-25 (updated from 2026-05-19)
**Branch:** main
**Official source:** `TheRobotStudio/SO-ARM100` → `Simulation/SO101/`
**Local robot model:** `owg_robot/assets/so101/so101.xml`
**Primary env:** `owg_robot/env_soarm.py`

---

## 1. Summary Verdict

| Component | Status | Notes |
|-----------|--------|-------|
| Robot XML (`so101.xml`) | ✅ Identical to upstream `so101_new_calib.xml` | Same model name, gains, mesh refs |
| All 13 STL mesh files | ✅ Present in `owg_robot/assets/so101/assets/` | Verbatim from upstream |
| Joint names | ✅ Exact match | All 5 arm + gripper |
| Joint limits | ✅ Exact match (new-calib ranges) | See §2.2 |
| Actuator names | ✅ Exact match | |
| Actuator gains `kp`/`kv` | ✅ kp=998.22, kv=2.731 (new-calib) | Do NOT mix with `joints_properties.xml` |
| `forcerange` layering | ✅ Class: ±2.94 Nm; per-actuator override: ±3.35 Nm | See §2.3 |
| EEF site name & position | ✅ Exact match (`gripperframe`) | See §4 |
| Gripper body / jaw names | ✅ Exact match | |
| Backlash default class | ⚠️ Defined in XML but no joints use it | Class exists; zero active backlash joints |
| Cameras | ⚠️ Custom (official defines none) | OWG overhead camera |
| Scene / table geometry | ⚠️ Custom OWG additions | Dynamic XML assembly |
| Home pose | ⚠️ Custom (official defines none) | Empirically tuned |
| IK solver | ✅ Custom — official defines none | DLS, jaw-midpoint, 6-DoF variants |
| Collision simplification | ✅ Custom — required, keep | See §3.2 |
| `physics_weld_after_bilateral` | ✅ Custom — core contribution, keep | See §6.1 |

---

## 2. Joint and Actuator Alignment

### 2.1 Joint Names

| OWG (`env_soarm.py`) | Official (`so101_new_calib.xml`) | Match |
|---|---|---|
| `shoulder_pan` | `shoulder_pan` | ✅ |
| `shoulder_lift` | `shoulder_lift` | ✅ |
| `elbow_flex` | `elbow_flex` | ✅ |
| `wrist_flex` | `wrist_flex` | ✅ |
| `wrist_roll` | `wrist_roll` | ✅ |
| `gripper` | `gripper` | ✅ |

### 2.2 Joint Limits

| Joint | Official range (rad) | OWG ctrlrange | Match |
|---|---|---|---|
| `shoulder_pan` | −1.9198621... / +1.9198621... | ±1.91986 | ✅ |
| `shoulder_lift` | −1.7453292... / +1.7453292... | ±1.74533 | ✅ |
| `elbow_flex` | ±1.69 | ±1.69 | ✅ |
| `wrist_flex` | −1.6580628... / +1.6580627... | ±1.65806 | ✅ |
| `wrist_roll` | −2.7438472... / +2.841206... | −2.74385 / +2.84121 | ✅ |
| `gripper` | −0.17453297... / +1.7453291... | −0.17453 / +1.74533 | ✅ |

Note: `wrist_roll` and `gripper` are asymmetric. The actuator `ctrlrange` rounds the full-precision XML values to 5 decimal places — functionally identical.

### 2.3 Actuator Gains and forcerange Layering

There are two layers of force limits in the XML, which can be confusing:

| Layer | kp | kv | forcerange |
|---|---|---|---|
| `<default class="sts3215">` (class default) | 998.22 | 2.731 | −2.94 / +2.94 Nm |
| Per-actuator `<position ... forcerange=...>` (override) | — | — | −3.35 / +3.35 Nm |

The per-actuator `forcerange` **overrides** the class default. All six actuators use ±3.35 Nm. The class-level ±2.94 Nm is never active. Both values appear in the official `so101_new_calib.xml`; our XML is identical.

**The kp=17.8 trap:** The official repo also distributes a `joints_properties.xml` file (used in the OLD calibration context) with `kp=17.8` — 56× lower than new-calib. This file is **incompatible** with our model. Never import or merge it. Our `so101.xml` corresponds exclusively to `so101_new_calib.xml`.

### 2.4 Joint Dynamics (class `sts3215`)

| Parameter | Official `so101_new_calib.xml` | OWG `so101.xml` | Match |
|---|---|---|---|
| `damping` | 0.60 | 0.60 | ✅ |
| `frictionloss` | 0.052 | 0.052 | ✅ |
| `armature` | 0.028 | 0.028 | ✅ |

### 2.5 Backlash Default Class

The XML defines a `backlash` default class (±0.5° = ±0.00873 rad, `damping=0.01`), matching the comment in the official source. However, **no joints in the body tree use `class="backlash"`** — it is a preparatory definition that is never instantiated. Active joint dynamics are entirely determined by the `sts3215` class. This is identical behaviour in both official and our copy.

---

## 3. Gripper Geometry

### 3.1 Body and Joint Structure

| Element | Official | OWG | Match |
|---|---|---|---|
| Fixed jaw body | `gripper` | `"gripper"` in `_cache_ids()` | ✅ |
| Moving jaw body | `moving_jaw_so101_v1` | `"moving_jaw_so101_v1"` | ✅ |
| Fixed jaw tip geom | `wrist_roll_follower_so101_v1` | `_find_collision_geom("gripper", "wrist_roll_follower_so101_v1")` | ✅ |
| Moving jaw tip geom | `moving_jaw_so101_v1` | `_find_collision_geom("moving_jaw_so101_v1", "moving_jaw_so101_v1")` | ✅ |
| Gripper joint axis | Z-axis hinge | inherited from XML | ✅ |
| Gripper open range | 0 to +1.74533 rad | `GRIP_OPEN=1.0`, `GRIP_CLOSED=0.05` | ⚠️ Custom subset |

Note on open range: `GRIP_OPEN=1.0` rad is 57% of the physical maximum (1.74533). This is a deliberate tuning choice for grasping clearance. Wider objects may need `GRIP_OPEN` increased to ~1.4–1.5.

### 3.2 Collision Geometry — OWG Custom Modification

**Official:** Jaw collision geometry uses full STL mesh convex hulls spanning ~10 cm along the jaw arm.

**OWG (`_simplify_jaw_collision()`):** Replaces the two jaw tip mesh collision geoms with 6 mm radius spheres at the same local positions, and disables the sts3215 motor collision geom on the gripper body:

```python
_R = 0.006  # 6 mm radius
self.model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_SPHERE
self.model.geom_size[gid, 0] = _R
```

**Why required:** The mesh convex hulls create 2.8–3.9 cm penetrations when the arm is teleported to a grasp configuration, generating explosive contact impulses. The 6 mm spheres produce controlled 4–5 mm penetrations that preserve bilateral contact detection without destabilising the scene.

**Consequence:** Grip friction with spheres is too low to lift ~0.15 kg objects. This is why `physics_weld_after_bilateral` was developed — the bilateral contact gate provides the success criterion; the kinematic weld provides the actual lift.

---

## 4. End-Effector Site

| Property | Official | OWG | Match |
|---|---|---|---|
| Site name | `gripperframe` | `EEF_SITE = "gripperframe"` | ✅ |
| Position in gripper body | `(-0.0079, -0.000218121, -0.0981274)` | inherited from `so101.xml` | ✅ |
| Orientation quat | `0.707107 0 0.707107 0` (Z→X forward) | used via `site_xmat` and `site_xpos` | ✅ |

**OWG site orientation convention:**
The `gripperframe` site Z-axis is the approach/closing direction. `make_topdown_rotation(yaw)` constructs a rotation so that site Z → world −Z (straight down) with jaw yaw:

```python
def make_topdown_rotation(yaw):
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, sy, 0], [sy, -cy, 0], [0, 0, -1]])
```

This convention must be maintained when integrating any external IK or control.

**Note on CoM vs. site for grasp targeting:**
`get_obj_pos()` returns the free-joint qpos origin (body reference frame). `get_obj_com_pos()` returns `data.xpos[body_id]` — the actual world-frame CoM. Grasp z-targeting must use `get_obj_com_pos()` because YCB mesh origins are often offset from the CoM.

---

## 5. Mesh Files

All 13 STL meshes in `owg_robot/assets/so101/assets/` are verbatim copies from the official `Simulation/SO101/assets/` directory:

```
base_motor_holder_so101_v1.stl    motor_holder_so101_base_v1.stl
base_so101_v2.stl                 motor_holder_so101_wrist_v1.stl
moving_jaw_so101_v1.stl           rotation_pitch_so101_v1.stl
sts3215_03a_no_horn_v1.stl        sts3215_03a_v1.stl
under_arm_so101_v1.stl            upper_arm_so101_v1.stl
waveshare_mounting_plate_so101_v2.stl  wrist_roll_follower_so101_v1.stl
wrist_roll_pitch_so101_v2.stl
```

`sts3215_03a_v1.stl` (motor body) is referenced 4× in the XML for different arm segments. `sts3215_03a_no_horn_v1.stl` is used once (wrist servo, no servo horn).

---

## 6. Scene and Camera (OWG Custom)

### 6.1 Scene Assembly

The official repo provides a minimal `scene.xml` (floor + light + skybox). OWG uses `_build_scene_xml()` in `env_soarm.py` to dynamically assemble a full scene combining:
- `so101.xml` robot model (parsed and inlined as XML string fragments)
- Table geometry (`TABLE_TOP_Z = 0.785`)
- Object pool bodies with freejoints (pool size ≥ number of active objects)
- Weld equality constraints (one per pool slot, inactive by default)
- Overhead camera at `pos="0.05 -0.52 1.9"`, `fovy=55`
- Delivery tray at `TARGET_ZONE_POS = [0.20, 0.25, 0.785]`

**Robot base placement:** `ROBOT_BASE_POS = "0 0 0.785"` — robot base sits at table surface level.

### 6.2 Camera

```xml
<camera name="overhead" pos="0.05 -0.52 1.9" euler="0 0 0" fovy="55"/>
```

| Parameter | Value | Notes |
|---|---|---|
| Position | (0.05, −0.52, 1.9) | Tuned for table workspace coverage |
| FOV | 55° | `FOVY = 55.0` |
| Render size | 224×224 | `IMG_SIZE = 224` |
| Pixel conversion | 277.0 px/m | `PIX_CONV = 277.0` |
| Image rotation | −π × 0.54 rad | `IMG_ROT` — aligns with GR-ConvNet coordinate frame |

Official defines no cameras. The `_MockCamera` shim in `env_soarm.py` provides `camera.width`, `camera.near`, `camera.far` for PyBullet API compatibility.

### 6.3 Scene Constants

```python
TABLE_TOP_Z          = 0.785   # table surface Z in world frame
GRASP_Z_TABLE_MARGIN = 0.020   # jaw midpoint offset above obj CoM
                                # = half_jaw_span(10mm) + sphere_r(6mm) + safety(4mm)
ROBOT_BASE_POS       = "0 0 0.785"
TARGET_ZONE_POS      = [0.20, 0.25, 0.785]
```

`GRASP_Z_TABLE_MARGIN` was added to make the jaw offset geometry explicit. It is used in `_ik_jaw_geom_topdown()` when computing the approach height.

---

## 7. Custom OWG Pipeline Modules

### 7.1 `physics_weld_after_bilateral` Grasp Mode

The core research contribution. Constant: `GRASP_MODE_PHYSICS_WELD = "physics_weld_after_bilateral"`.

**Phase 1 (contact detection):**
- IK descent to jaw midpoint above object (target = `get_obj_com_pos() + [0,0,GRASP_Z_TABLE_MARGIN]`)
- Gripper close
- Post-close check: both `wrist_roll_follower` and `moving_jaw_so101_v1` geoms must contact the target object (`bilateral_contacts == 1`)

**Phase 2 (kinematic lift):**
- Gate: bilateral contacts required
- Activate weld equality constraint between `gripper` body and target object body
- Lift EEF to `TABLE_TOP_Z + 0.35` m
- Success = object CoM clears `TABLE_TOP_Z + 0.07` m; weld released

Post-grasp metrics stored in `env.last_grasp_metrics`:
```python
{
    "bilateral_contacts": int,   # 0 or 1
    "left_contacts":      int,
    "right_contacts":     int,
    "symmetry_score":     float, # |left−right| / (left+right+ε)
    "jaw_obj_xy_gap":     float, # XY distance between jaw midpoint and obj
    "ori_err_norm":       float, # IK orientation residual
    "eef_z_axis":         list,  # EEF approach direction in world frame
}
```

`GRASP_MODE_PHYSICS = "physics"` is kept as a legacy alias mapping to the same function.

### 7.2 IK Solver Suite

All IK implemented via MuJoCo `mj_jacSite` (Damped Least Squares):

| Function | Type | DOF | Use case |
|---|---|---|---|
| `_ik_step()` / `_solve_ik()` | Position-only | 5 arm joints | Legacy warm-start |
| `_ik_step_6dof()` | Position + orientation | 5 arm joints | Full 6-DOF step |
| `_solve_ik_6dof()` | Two-phase: pos warm-start → 6-DOF | 5 arm joints | General 6-DOF target |
| `_ik_jaw_geom_topdown()` | Jaw-midpoint top-down with yaw | 5 arm joints | Preferred for grasping |

IK parameters: `IK_ITERS=200`, `IK_DAMPING=0.05`, `IK_TOL=5e-4`.

`_solve_ik_6dof()` uses a two-phase strategy: position-only warm-start for the first half of iterations, then 6-DOF refinement with `w_pos=10.0 >> w_ori=0.3` to anchor XYZ while gently correcting orientation.

### 7.3 Semantic Grounding and LGGSN Reranking

`owg_robot/grasp_ranker_lggsn.py` — pairwise BPR-trained reranking of GR-ConvNet grasp candidates conditioned on language prompts. 14-dimensional feature vector (12 base + 2 episode-relative context: `dist_to_centroid`, `z_rel`). See `paper_final.tex` for full methodology.

### 7.4 6-DoF Grasp Generator

`grasp_6dof/grasp_generator_6dof.py` + `grasp_6dof/grasp_sampler.py`:
- Open3D mesh surface sampling
- PCA-based elongation detection and lateral prior (confirmed effective: Banana 75% success)
- Geometric scoring: workspace AABB filter, surface normal alignment, table clearance, `world_yaw` alignment

### 7.5 Diverse Benchmark Harness

Added 2026-05-19; committed to main 2026-05-25:

| Module | Purpose |
|---|---|
| `benchmark/diverse_runner.py` | Difficulty modes (easy/medium/hard), random object yaw, optional clutter |
| `benchmark/scene_generator.py` | Scene generation with `DifficultyConfig`, `DIFFICULTY_PRESETS` |
| `configs/benchmark/diverse_*.yaml` | 3 configs, 50 seeds × 5 objects each |
| `scripts/run_diverse_benchmark.py` | Entrypoint with `--list-configs` and robust config resolution |

### 7.6 Home Pose and Gripper Limits

```python
HOME_QPOS   = np.array([0.0, -0.4, 0.8, -0.4, 0.0])  # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
GRIP_OPEN   = 1.0   # rad — 57% of ctrlrange max (1.74533); tuned for grasping clearance
GRIP_CLOSED = 0.05  # rad — near-zero, leaves small gap for contact detection
```

Official defines no home pose. `HOME_QPOS` was chosen empirically for IK reachability and stability over the table workspace (`x ∈ [−0.1, 0.5]`, `y ∈ [−0.4, 0.4]`).

---

## 8. What Cannot Be Directly Reused from Official

| Official Asset | Reason Not Usable As-Is |
|---|---|
| `joints_properties.xml` (kp=17.8) | 56× lower gain than new-calib; arm collapses under gravity |
| Official `scene.xml` | No objects, no camera, no table; incompatible with dynamic object pool |
| `so101_old_calib.xml` | Asymmetric shoulder_lift range (−3.316 to +0.175 rad); HOME_QPOS would be outside limits |
| Official URDF (`so101_new_calib.urdf`) | MuJoCo simulation uses XML directly; URDF is for other frameworks only |

---

## 9. Migration Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Accidentally using `joints_properties.xml` gains (kp=17.8) | **High** — arm collapses | Always use `so101.xml` (new_calib); never import `joints_properties.xml` separately |
| Upstream STL mesh update | **Medium** — jaw geom sphere positions shift | Re-run `_simplify_jaw_collision()`; re-validate Banana success rate (75% baseline) |
| Upstream joint limit changes | **Medium** — `HOME_QPOS` may violate new limits | Verify `HOME_QPOS` stays within new limits; re-run workspace reachability check |
| `gripperframe` site position change | **Medium** — IK targets and `GRASP_Z_TABLE_MARGIN` shift | Re-calibrate margin; re-run end-to-end benchmark |
| `wrist_roll` asymmetric limit (−2.744 / +2.841) | Low — handled by ctrlrange | No action; actuator clamps correctly |
| Backlash joints activated in upstream | Low — currently zero active backlash | If upstream adds `class="backlash"` joints, check effect on contact detection |
| `forcerange` class vs. per-actuator confusion | Low — per-actuator always wins | Always check per-actuator `forcerange` in `<actuator>` block, not class default |

---

## 10. Recommended Sync Procedure

When pulling upstream SO-101 changes into `owg_robot/assets/so101/`:

1. `diff owg_robot/assets/so101/so101.xml <upstream_so101_new_calib.xml>`
2. Check: joint ranges, actuator `kp`/`kv`/`forcerange`, `gripperframe` site position/quat, body frame positions
3. If joint limits changed → verify `HOME_QPOS` is still within limits
4. If `gripperframe` position changed → update `GRASP_Z_TABLE_MARGIN`; re-run Banana benchmark
5. If STL meshes replaced → re-run `_simplify_jaw_collision()` and confirm jaw sphere centres still land 1–2 mm outside object faces (verify with a Cylinder or MustardBottle contact trace)
6. Never replace actuator gains with `joints_properties.xml` values
7. Never switch to `so101_old_calib.xml` — the asymmetric shoulder_lift range breaks the current workspace

---

## 11. Files Summary

### Reused verbatim from official `so101_new_calib.xml`
- `owg_robot/assets/so101/so101.xml` — robot model (identical to upstream)
- `owg_robot/assets/so101/assets/*.stl` — all 13 mesh files

### Custom OWG (not in official)
- `owg_robot/env_soarm.py` — full simulation environment, IK suite, grasp modes
- `owg_robot/grasp_ranker_lggsn.py` — pairwise BPR reranker
- `grasp_6dof/grasp_sampler.py` — 6-DoF grasp generation
- `grasp_6dof/grasp_generator_6dof.py` — CLI grasp generator
- `benchmark/replay_soarm_grasp.py` — scene replay tool
- `benchmark/runner.py` — evaluation harness
- `benchmark/diverse_runner.py` — difficulty-aware benchmark runner
- `benchmark/scene_generator.py` — scene generation with clutter support
- `configs/benchmark/diverse_*.yaml` — benchmark configurations
- `docs/` — all documentation files

### Legacy (archived to `legacy/pybullet_panda/`)
- `validate_grasps_panda*.py`, `env_panda.py`, `robot_panda.py`, etc.
- No longer active; preserved for reference only
