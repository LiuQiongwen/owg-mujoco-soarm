# SO-ARM101 MuJoCo Calibration Guide

**Script**: `scripts/calibrate_soarm_mujoco.py`  
**Output**: `configs/soarm_calibration.yaml`

The grasp model (GR-ConvNet + LGGSN) is physics-agnostic and does not change
between PyBullet/Panda and MuJoCo/SO-ARM101. This guide covers the three
execution-layer parameters that **do** need measurement:

| Parameter | Source of mismatch |
|-----------|-------------------|
| `cam_to_robot_base` | Camera and robot base are physically different from Panda rig |
| `gripper_opening_scale` | SO-ARM101 jaw geometry ≠ Panda parallel gripper |
| `yaw_mode` | SO-ARM101 has 5 arm DoF; full 6-DoF IK is TODO |

---

## Quick Start

```bash
# Full calibration (camera + gripper + yaw ablation)
MUJOCO_GL=egl conda run -n bridge python scripts/calibrate_soarm_mujoco.py

# Single task
MUJOCO_GL=egl conda run -n bridge python scripts/calibrate_soarm_mujoco.py --task cam
MUJOCO_GL=egl conda run -n bridge python scripts/calibrate_soarm_mujoco.py --task gripper
MUJOCO_GL=egl conda run -n bridge python scripts/calibrate_soarm_mujoco.py --task yaw --obj scissors

# Validate with calibrated parameters
MUJOCO_GL=egl conda run -n bridge python scripts/calibrate_soarm_mujoco.py --validate banana
MUJOCO_GL=egl conda run -n bridge python scripts/calibrate_soarm_mujoco.py --validate cylinder
MUJOCO_GL=egl conda run -n bridge python scripts/calibrate_soarm_mujoco.py --validate scissors
```

---

## 1. Camera-to-Robot-Base Calibration (`--task cam`)

### What it measures

Places a fiducial (YcbBanana) at 7 known tabletop positions, renders the
overhead camera image, finds the segmentation centroid, and back-projects it to
world coordinates using the current camera model:

```
world_x = (px - W/2) / fx * depth + cam_x
world_y = (py - H/2) / fy * depth + cam_y   (image Y is inverted)
world_z = cam_z - depth
```

The reprojection error = distance from back-projected point to known position.

### Output fields

```yaml
cam_to_robot_base:
  cam_pos: "0.05 -0.52 1.9"
  n_points: 7
  rmse_m: 0.008          # mean XY error across grid (metres)
  x_bias: 0.003          # systematic X offset (positive = predicted too far right)
  y_bias: -0.002         # systematic Y offset
  correction_xy:         # add this to GR-ConvNet's predicted (x,y) before executing
    - -0.003
    - 0.002
  points: [...]          # per-point records
```

### When is recalibration needed?

- After moving the physical camera or robot base
- After changing `cam_pos` / `cam_euler` in `config/mujoco/env.yaml`
- If grasp targets are systematically offset in one direction

### How to apply in code

In `env_soarm.py`, the `cam_to_robot_base` matrix is computed at `__init__`:

```python
cx, cy, cz = [float(v) for v in CAM_POS.split()]
self.cam_to_robot_base = _get_transform_matrix(cx, cy, cz, CAM_ROT)
```

To apply the calibration correction, load the YAML and offset the matrix
translation, or subtract `correction_xy` from GR-ConvNet's `(x, y)` output
before passing to `pick_obj_by_id`.

---

## 2. Gripper Opening Calibration (`--task gripper`)

### What it measures

Sweeps the gripper control from `GRIP_CLOSED` (0.05) to `GRIP_OPEN` (1.0)
in 12 steps and measures the Euclidean distance between the `gripper` body
and `moving_jaw_so101_v1` body at each step.

Fits a linear model:  
```
jaw_separation [m] = scale * ctrl + offset
```

### Output fields

```yaml
gripper_opening_scale: 0.85    # multiply GR-ConvNet opening_len by this
gripper_opening_offset: 0.021  # jaw sep at fully closed (body centre offset)
gripper_max_opening_m: 0.085   # actual max jaw separation
gripper_fit:
  scale: 0.065
  offset: 0.021
  samples: [...]
```

### Interpretation

GR-ConvNet outputs `opening_len` calibrated for a Panda gripper with ~8–10 cm
max opening. If SO-ARM101's jaw separation max is 8.5 cm, the recommended scale
is `0.085 / 0.10 = 0.85` — meaning GR-ConvNet's 10 cm prediction maps to
SO-ARM's actual 8.5 cm maximum.

The `GRIP_REDUCTION = 0.85` constant in `env_soarm.py` should match
`gripper_opening_scale` from the calibration.

---

## 3. Yaw Ablation (`--task yaw`)

### Three modes

| Mode | Behaviour | Status |
|------|-----------|--------|
| `xyz_only` | Yaw from GR-ConvNet is ignored; arm approaches with default wrist orientation | Implemented |
| `top_down_yaw` | Wrist-roll joint is commanded to approximate the target yaw (valid near vertical arm pose) | Implemented (approximation) |
| `full_6dof` | Full 6-DoF IK: simultaneous position + orientation solve | **Placeholder — not yet implemented** |

### `top_down_yaw` mechanism

When the arm is near-vertical (top-down grasp), the `wrist_roll` joint
rotation approximates end-effector yaw around the world Z axis:

```python
# in _apply_top_down_yaw()
wrist_roll_idx = ARM_JOINTS.index("wrist_roll")
env.data.ctrl[env._arm_act_ids[wrist_roll_idx]] = clip(target_yaw, lo, hi)
```

This approximation degrades for highly off-vertical poses. For sim2real with
real objects this should be replaced by the full 6-DoF IK.

### Test objects and expected sensitivity

| Object key | YCB name | Expected yaw sensitivity |
|-----------|----------|--------------------------|
| `banana` | YcbBanana | Low — nearly round cross-section |
| `cylinder` | YcbTomatoSoupCan | Moderate — symmetric but must clear label |
| `scissors` | YcbScissors | High — must align along blade axis |

### Output fields

```yaml
yaw_ablation:
  xyz_only:
    success_rate: 0.5
    n_trials: 4
    detail:
      - {yaw_deg: -45.0, success: true}
      - {yaw_deg:   0.0, success: true}
      - {yaw_deg:  45.0, success: false}
      - {yaw_deg:  90.0, success: false}
  top_down_yaw:
    success_rate: 0.75
    n_trials: 4
    detail: [...]
yaw_mode: top_down_yaw   # automatically selected best mode
```

### Expected results (simulation)

| Object | xyz_only SR | top_down_yaw SR | delta |
|--------|------------|-----------------|-------|
| banana | ~0.75 | ~0.75 | ~0 (round) |
| cylinder | ~0.50 | ~0.75 | +0.25 |
| scissors | ~0.25 | ~0.50 | +0.25 |

> **Note**: These are simulation estimates. Real-hardware numbers will differ
> due to friction, compliance, and calibration residuals.

---

## 4. Calibration Output File

`configs/soarm_calibration.yaml` is the single source of truth for all
calibrated execution parameters. Full example:

```yaml
cam_to_robot_base:
  cam_pos: "0.05 -0.52 1.9"
  n_points: 7
  rmse_m: 0.008
  x_bias: 0.003
  y_bias: -0.002
  correction_xy: [-0.003, 0.002]

gripper_opening_scale: 0.85
gripper_opening_offset: 0.021
gripper_max_opening_m: 0.085
gripper_fit:
  scale: 0.065
  offset: 0.021

yaw_ablation:
  xyz_only:
    success_rate: 0.50
    n_trials: 4
  top_down_yaw:
    success_rate: 0.75
    n_trials: 4

yaw_mode: top_down_yaw

validation:
  banana:
    success_rate: 0.80
    n_trials: 5
    yaw_mode: top_down_yaw
  cylinder:
    success_rate: 0.60
    n_trials: 5
    yaw_mode: top_down_yaw
  scissors:
    success_rate: 0.40
    n_trials: 5
    yaw_mode: top_down_yaw
```

---

## 5. Implementing Full 6-DoF IK (Future)

The `TODO(6dof-ik)` markers in `owg_robot/env_soarm.py` show where to plug in
full orientation-aware IK. The changes are:

**`_ik_step()` (line ~368)** — extend the 3×n Jacobian to 6×n:
```python
# Currently: position only
jacp = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, None, site_id)

# With 6-DoF: stack rotation Jacobian
jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
J = np.vstack([jacp[:, cols], jacr[:, cols]])   # 6×n
err = np.concatenate([pos_err, ori_err])         # 6-vector
dq = J.T @ np.linalg.solve(J @ J.T + damp * I, err)
```

**`move_ee()` (line ~394)** — pass orientation target to `_solve_ik`:
```python
# Currently: move_ee([x, y, z, orn]) ignores orn
# With 6-DoF: unpack and pass to IK
pos_target = np.array([x, y, z])
ori_target = orn   # quaternion or euler
ok = self._solve_ik_6dof(pos_target, ori_target)
```

Until 6-DoF IK is implemented, `top_down_yaw` mode provides a reasonable
approximation for tabletop grasps where the arm approaches nearly vertically.

---

## 6. Sim-to-Real Gap Checklist

Before deploying to hardware, verify each item:

- [ ] Camera intrinsics measured on real hardware (focal length, principal point)
- [ ] `cam_to_robot_base` calibration re-run with physical marker board
- [ ] Gripper jaw separation measured with calipers at open/closed positions
- [ ] `GRIP_REDUCTION` constant updated from calibration YAML
- [ ] `CAM_POS` and `CAM_EULER` in `config/mujoco/env.yaml` updated to real values
- [ ] Object heights (`obj_height` field in grasps) verified for real objects
- [ ] 6-DoF IK implemented, or `yaw_mode: top_down_yaw` accepted as approximation
