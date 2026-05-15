# PyBullet → MuJoCo Migration Report

**Branch**: `feature/vis-aabb`  
**Date**: 2026-05-15  
**Status**: Observation pipeline working; grasp execution stubs in place

---

## What Works

| Component | Status | Notes |
|-----------|--------|-------|
| `get_obs()` → `{image, depth, seg, obj_ids, points}` | ✅ | All keys present; aliases `rgb`, `segmentation` added |
| Segmentation mask `seg == obj_id` | ✅ | ~275 px for YcbBanana; logical IDs start from 1 |
| Metric depth (metres from camera) | ✅ | MuJoCo 3.x returns metric depth directly |
| `load_obj()` / `load_isolated_obj()` | ✅ | Path or name; handles `YcbBanana` / `Banana` forms |
| `remove_obj()` / `remove_all_obj()` | ✅ | Teleports to z=-100 (no model rebuild needed) |
| `set_obj_grasps()` / `get_obj_grasps()` | ✅ | Stored per logical obj_id |
| `pick_obj_by_id(grasp_indices=...)` | ✅ | Case A (grasp objects), B (int indices), fallback |
| `put_obj_in_tray()` / `put_obj_in_loc()` | ✅ | Pick + deliver to tray / target location |
| `get_obj_pose()` → `{position, quaternion}` | ✅ | Matches grasp_ranker interface |
| `_MockCamera` shim | ✅ | `.width`, `.near`, `.far` for GraspGenerator |
| mujoco_validator | ✅ | All 7 steps pass (syntax, import, instantiate, reset, sim_step, object_load, joint_count) |

---

## Bugs Fixed in This Session

### 1. Segmentation all-background (`seg_raw` unique `[-1]`)
**Root cause**: `CAM_EULER = "3.14159 0 0"` with `<compiler angle="radian"/>` rotates the camera 180° around X → camera looks **up**, sees nothing.  
**Fix**: Changed `CAM_EULER` to `"0 0 0"` (MuJoCo default camera orientation already looks -Z/down).

### 2. Negative metric depth
**Root cause**: Conversion formula `near / (1 - depth_raw * (1 - near/far))` assumed normalized Z-buffer [0,1], but MuJoCo 3.x `enable_depth_rendering()` returns metric depth in metres directly.  
**Fix**: Removed conversion; use `depth_raw` as-is.

### 3. Logical `obj_id=0` collides with background
**Root cause**: First object got `logical_id = len(obj_ids) = 0`, same as background in seg mask.  
**Fix**: Changed to `logical_id = max(obj_ids) + 1` if non-empty, else `1`.

### 4. Segmentation update_scene ordering
**Fix**: `update_scene()` is now called after each `enable_*_rendering()` call (required in MuJoCo 3.x to encode geom IDs into the scene).

---

## Remaining PyBullet Assumptions

### IK / Arm Motion (TODO markers in code)
- `_ik_step()`: position-only (xyz). Orientation (roll/yaw) is **accepted but ignored**.
- `move_ee()`: passes orientation argument but does not use it.
- `_execute_grasp()`: approach roll not applied.
- **Impact**: grasps with non-zero yaw will be attempted at the correct position but with the wrong gripper orientation. Success rate will be lower than PyBullet panda.
- **TODO marker**: 5× `TODO(6dof-ik)` in env_soarm.py

### Grasp Format Compatibility
- Pipeline produces `(x, y, z, yaw, opening_len, obj_height)` tuples from GR-ConvNet / GraspGenerator.
- `pick_obj_by_id` reads indices 0..5 safely. ✅
- `put_obj_in_tray` / `put_obj_in_loc` relay `grasp_indices` unchanged. ✅

### Contact Detection
- Uses MuJoCo contact array (`data.contact`). Gripper body names (`gripper`, `moving_jaw_so101_v1`) must match SO-101 XML.
- Fallback: if body lookup fails, `check_grasped()` returns False (safe).

### Seg Mask Size
- 275 pixels out of 50176 (0.5%) for a banana. GR-ConvNet needs visible object to generate grasps; this is sufficient but tight.
- Camera FOV / position matches PyBullet overhead camera config.

---

## Stage Compatibility

| Stage | Status | Notes |
|-------|--------|-------|
| Stage 1 | ✅ (structural) | PyBullet GrConvNet grasp generation + execution stub |
| Stage 2 | ✅ | 6-DoF grasp sampling runs standalone (no env needed) |
| Stage 3 | ⚠️  needs test | Semantic grounding + 6DoF sampler; IK orientation gap |
| Stage 4 | ⚠️  needs test | Stage 3 + LGGSN ranker; same IK gap |

---

## Environment Dependency Gap

The full OWG pipeline requires `open3d`, `clip`, `pybullet`, and related packages.
These are missing from the `bridge` conda env (which has MuJoCo 3.8.1 but not open3d/clip).
The original `owg2` env referenced in CLAUDE.md no longer exists.

**Blockers for end-to-end Stage 3/4 demo**:
- `owg/visual_prompt.py` imports `open3d` at module level (needed for point cloud grounding)
- `owg_robot/camera.py` imports `pybullet` at module level
- Neither is available in `bridge` env; numpy 2.x in bridge conflicts with system scipy

**Workaround options**:
1. Create new env from `requirements.txt` with both MuJoCo and open3d/clip
2. Install mujoco in an existing env that already has the full OWG stack
3. Make `owg/visual_prompt.py` open3d import optional (deferred until grounding needed)

## Next Steps

1. **Unblock demo**: Set up a conda env with both `mujoco>=3.x` and the OWG policy stack.
   Minimal additions to `bridge`: `open3d`, `clip`, `pybullet`
   Or install mujoco into whichever env has `open3d + clip + pybullet`.
2. **Implement 6-DoF IK** at `TODO(6dof-ik)` markers in `env_soarm.py`.
3. **Validate grasp success rate** on MuJoCo vs PyBullet baseline.
