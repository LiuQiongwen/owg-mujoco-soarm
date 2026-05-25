# SO-ARM101 Official Alignment Audit

**Date:** 2026-05-19  
**Branch:** feature/vis-aabb  
**Official source:** `TheRobotStudio/SO-ARM100` → `Simulation/SO101/`  
**Local robot model:** `owg_robot/assets/so101/so101.xml`  
**Primary env:** `owg_robot/env_soarm.py`

---

## 1. Summary Verdict

| Component | Status |
|-----------|--------|
| Robot XML (`so101.xml`) | ✅ Identical to upstream `so101_new_calib.xml` |
| All 13 STL mesh files | ✅ Present in `owg_robot/assets/so101/assets/` |
| Joint names | ✅ Exact match |
| Joint limits | ✅ Exact match (new-calib ranges) |
| Actuator names | ✅ Exact match |
| Actuator gains (kp/kv) | ✅ Using high-gain version (kp=998.22) — correct |
| EEF site name & position | ✅ Exact match (`gripperframe`) |
| Gripper body/jaw names | ✅ Exact match |
| Cameras | ⚠️ Custom (official defines none) |
| Scene / table geometry | ⚠️ Custom OWG additions |
| Home pose | ⚠️ Custom (official defines none) |
| IK solver | ✅ Custom (official defines none) |
| Collision simplification | ✅ Custom — required, keep |
| `physics_weld_after_bilateral` | ✅ Custom — core contribution, keep |

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
| `shoulder_pan` | ±1.9199 | ±1.91986 | ✅ |
| `shoulder_lift` | ±1.7453 | ±1.74533 | ✅ |
| `elbow_flex` | ±1.6900 | ±1.69 | ✅ |
| `wrist_flex` | ±1.6581 | ±1.65806 | ✅ |
| `wrist_roll` | −2.7438 / +2.8412 | −2.74385 / +2.84121 | ✅ |
| `gripper` | −0.1745 / +1.7453 | −0.17453 / +1.74533 | ✅ |

### 2.3 Actuator Gains

**OWG uses:** `kp=998.22`, `kv=2.731`, `forcerange="-3.35 3.35"` (per-actuator override)  
**Official (`so101_new_calib.xml` / our `so101.xml`):** same values — sourced from RBE501-RL servo gain calculations at servo proportional gain = 16.

> ⚠️ **Do NOT use** `joints_properties.xml` from the official repo — it sets `kp=17.8` which is 56× lower gain. This is a standalone file for a different calibration context and is incompatible with our dynamics.

### 2.4 Joint Dynamics (class `sts3215`)

| Parameter | Official | OWG | Match |
|---|---|---|---|
| `damping` | 0.60 | inherited from XML | ✅ |
| `frictionloss` | 0.052 | inherited from XML | ✅ |
| `armature` | 0.028 | inherited from XML | ✅ |
| Backlash | ±0.5° = ±0.00873 rad | inherited from XML | ✅ |

---

## 3. Gripper Geometry

### 3.1 Body and Joint Structure

| Element | Official | OWG | Match |
|---|---|---|---|
| Fixed jaw body | `gripper` | `"gripper"` in `_cache_ids()` | ✅ |
| Moving jaw body | `moving_jaw_so101_v1` | `"moving_jaw_so101_v1"` in `_cache_ids()` | ✅ |
| Fixed jaw tip geom | `wrist_roll_follower_so101_v1` | `_find_collision_geom("gripper", "wrist_roll_follower_so101_v1")` | ✅ |
| Moving jaw tip geom | `moving_jaw_so101_v1` | `_find_collision_geom("moving_jaw_so101_v1", "moving_jaw_so101_v1")` | ✅ |
| Gripper joint axis | Z-axis hinge | inherited from XML | ✅ |

### 3.2 Collision Geometry — OWG Custom Modification

**Official:** Jaw collision geometry uses full STL meshes (convex hulls spanning ~10 cm along jaw arm).

**OWG (`_simplify_jaw_collision()`):** Replaces the two jaw tip mesh collision geoms with 6 mm radius spheres at the same local positions, and disables the sts3215 motor collision geom on the gripper body:

```python
_R = 0.006  # 6 mm spheres
self.model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_SPHERE
self.model.geom_size[gid, 0] = _R
```

**Why this is necessary:** The mesh convex hulls create 2.8–3.9 cm penetrations when the arm is teleported to a grasp configuration, generating explosive contact impulses. The 6 mm spheres produce controlled 4–5 mm penetrations that preserve bilateral contact detection without destabilizing the scene.

**Risk:** Grip force is lower with small spheres. This is why `physics_weld_after_bilateral` was developed — pure contact friction is insufficient to lift ~0.15 kg objects. The bilateral contact gate provides the success criterion; the kinematic weld provides the actual lift.

---

## 4. End-Effector Site

| Property | Official | OWG | Match |
|---|---|---|---|
| Site name | `gripperframe` | `EEF_SITE = "gripperframe"` | ✅ |
| Position in gripper body | `(-0.0079, -0.000218121, -0.0981274)` | inherited from `so101.xml` | ✅ |
| Orientation | `quat="0.707107 0 0.707107 0"` (Z→X forward) | used via `site_xmat` and `site_xpos` | ✅ |

**OWG note on site orientation:**  
The `gripperframe` site Z-axis points in the approach direction (the closing/gripping axis). The function `make_topdown_rotation(yaw)` constructs a 3×3 rotation so that site Z → world −Z (straight down) with a yaw rotation:

```python
def make_topdown_rotation(yaw):
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, sy, 0], [sy, -cy, 0], [0, 0, -1]])
```

This convention must be maintained when integrating any external IK or control.

---

## 5. Scene and Camera (OWG Custom)

### 5.1 Scene Assembly

The official repo provides a minimal `scene.xml` (floor + light + skybox). OWG uses `_build_scene_xml()` in `env_soarm.py` to dynamically assemble a full scene combining:
- `so101.xml` robot model (via XML string include)
- Table geometry (`TABLE_TOP_Z = 0.785`)
- Object pool bodies with freejoints
- Weld equality constraints (one per pool slot, inactive by default)
- Overhead camera
- Tray (delivery target zone)

**Robot base placement:** `ROBOT_BASE_POS = "0 0 0.785"` — placed on table surface.

### 5.2 Camera

```xml
<camera name="overhead" pos="0.05 -0.52 1.9" euler="0 0 0" fovy="55"/>
```

**Official:** No cameras defined.  
**OWG:** Custom overhead camera. Its position and FOV (55°) are tuned for the table workspace. The rendering pipeline uses `IMG_SIZE=224`, `PIX_CONV=277.0`.

---

## 6. Custom OWG Pipeline Modules (Not in Official)

### 6.1 `physics_weld_after_bilateral` Grasp Mode

The core research contribution. Two-phase execution:

**Phase 1 (contact detection):**
- IK descent to jaw midpoint above object
- Gripper close
- Post-close bilateral contact check: both `wrist_roll_follower` and `moving_jaw_so101_v1` geoms must contact the target object

**Phase 2 (kinematic lift):**
- If and only if bilateral contacts detected → activate weld equality constraint between `gripper` body and target object
- Lift EEF to `GRIPPER_GRASPED_LIFT_HEIGHT = TABLE_TOP_Z + 0.35`
- Success = object clears `TABLE_TOP_Z + 0.07`

**Why kinematic weld:** 6 mm sphere collision geoms cannot generate sufficient friction force to lift 0.15 kg objects against gravity at the current actuator force scale. The bilateral contact gate ensures the weld is only triggered by genuine jaw contact, preserving the validity of the success signal.

### 6.2 IK Solver Suite

OWG implements DLS (Damped Least Squares) Jacobian IK via `mj_jacSite`:
- `_ik_step()` — position-only warm-start (5 DOF, arm only)
- `_ik_step_6dof()` — position + orientation (6 DOF)
- `_solve_ik_jaw_topdown()` — top-down orientation with jaw yaw
- `_ik_jaw_geom_topdown()` — jaw midpoint target (preferred for grasping)

None of these are provided by the official repository.

### 6.3 Semantic Grounding and LGGSN Reranking

`owg_robot/grasp_ranker_lggsn.py` — pairwise BPR-trained reranking of grasp candidates conditioned on language prompts. Entirely custom; no equivalent in official.

### 6.4 6-DoF Grasp Generator

`grasp_6dof/grasp_generator_6dof.py` + `grasp_6dof/grasp_sampler.py`:
- Open3D mesh surface sampling
- PCA-based elongation detection and lateral prior
- Geometric scoring (workspace AABB, surface normal alignment, table clearance)
- `world_yaw` field for gripper orientation alignment

Not in official.

### 6.5 Home Pose

```python
HOME_QPOS = np.array([0.0, -0.4, 0.8, -0.4, 0.0])  # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
```

Official defines no home pose. This was chosen empirically for stability and IK reachability over the table workspace.

```python
GRIP_OPEN   = 1.0   # rad — gripper actuator target for "open" (~57% of 1.745 max)
GRIP_CLOSED = 0.05  # rad — gripper actuator target for "closed"
```

> Note: `GRIP_OPEN = 1.0` is not fully open (max ctrlrange = 1.74533). This was tuned for grasping clearance. Increase to ~1.5 if wider jaw opening is needed for larger objects.

---

## 7. What Cannot Be Directly Reused from Official

| Official Asset | Reason Not Usable |
|---|---|
| `joints_properties.xml` (kp=17.8) | 56× lower gain than our `so101.xml`; mixing would create inconsistent dynamics |
| Official `scene.xml` | Too minimal — no objects, no camera, no table; we need the dynamic assembly |
| Old-calibration XML (`so101_old_calib.xml`) | Asymmetric joint ranges (shoulder_lift: −3.316 to +0.175); incompatible with our workspace |

---

## 8. Migration Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Accidentally switching to `joints_properties.xml` gains (kp=17.8) | **High** — arm collapses under gravity | Always use `so101.xml` (new_calib); never import `joints_properties.xml` |
| Upstream SO-101 mesh update (new STL versions) | Medium — contact geom positions would shift | Re-run jaw-geom sphere calibration; re-validate banana success rate |
| Upstream joint limit changes | Medium — could make current HOME_QPOS invalid | Verify `HOME_QPOS` stays within new limits before merging upstream |
| `gripperframe` site position change | Medium — IK targets would shift | Re-calibrate `GRASP_Z_TABLE_MARGIN`; re-run end-to-end benchmark |
| `wrist_roll` asymmetric limit (−2.744 / +2.841) | Low — already handled by ctrlrange in XML | No action needed; actuator clamps correctly |
| Backlash joints in upstream (±0.5°) | Low | Present in our `so101.xml`; does not affect current evaluation |

---

## 9. Recommended Sync Procedure

When pulling upstream SO-101 changes:

1. `diff owg_robot/assets/so101/so101.xml <upstream_so101_new_calib.xml>`
2. Check for changes to: joint ranges, actuator kp/kv/forcerange, gripperframe site position, body structure
3. If joint limits changed: verify `HOME_QPOS` is still valid
4. If site position changed: update `GRASP_Z_TABLE_MARGIN` and re-run banana benchmark
5. If mesh files replaced: re-run `_simplify_jaw_collision()` logic and confirm sphere positions still hit object faces
6. Never replace actuator gains with `joints_properties.xml` values

---

## 10. Files Summary

### Reused from official (verbatim)
- `owg_robot/assets/so101/so101.xml` — identical to `so101_new_calib.xml`
- `owg_robot/assets/so101/assets/*.stl` — all 13 mesh files

### Custom OWG (not in official)
- `owg_robot/env_soarm.py` — full simulation environment
- `owg_robot/grasp_ranker_lggsn.py` — semantic reranking
- `owg_robot/assets/scenes/benchmark_scene.xml` — static scene for inspection
- `grasp_6dof/grasp_sampler.py` — 6-DoF grasp generation
- `grasp_6dof/grasp_generator_6dof.py` — CLI grasp generator
- `benchmark/replay_soarm_grasp.py` — scene replay tool
- `benchmark/runner.py`, `benchmark/diverse_runner.py` — evaluation harness

### Legacy (moved to `legacy/pybullet_panda/`)
- All `validate_grasps_panda*.py`, `env_panda.py`, `robot_panda.py`, etc.
