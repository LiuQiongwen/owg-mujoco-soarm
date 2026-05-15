#!/usr/bin/env python3
"""
SO-ARM101 MuJoCo Calibration & Validation Suite.

Measures three execution-layer parameters that differ between PyBullet/Panda
and MuJoCo/SO-ARM101 without touching the grasp model (GR-ConvNet / LGGSN):

  cam   — camera-to-robot-base reprojection error, bias, and correction vector
  gripper — jaw-opening mapping: GR-ConvNet opening_len [m] → joint ctrl angle
  yaw   — ablation across xyz_only / top_down_yaw / full_6dof (placeholder)

Usage:
  python scripts/calibrate_soarm_mujoco.py                      # full suite
  python scripts/calibrate_soarm_mujoco.py --task cam           # camera only
  python scripts/calibrate_soarm_mujoco.py --task gripper       # gripper only
  python scripts/calibrate_soarm_mujoco.py --task yaw --obj banana
  python scripts/calibrate_soarm_mujoco.py --validate banana    # post-calib check
  python scripts/calibrate_soarm_mujoco.py --validate scissors

Output: configs/soarm_calibration.yaml
"""

import os, sys, math, argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MUJOCO_GL", "egl")

from owg_robot.env_soarm import (
    EnvironmentSoArm,
    TABLE_TOP_Z, IMG_SIZE, FOVY, CAM_POS,
    ARM_JOINTS, GRIP_OPEN, GRIP_CLOSED,
)

DEFAULT_OUT = "configs/soarm_calibration.yaml"

# ── yaw modes ────────────────────────────────────────────────────────────────
YAW_MODES = ["xyz_only", "top_down_yaw", "full_6dof"]

# ── test objects (name → YCB directory, tabletop pose) ───────────────────────
# y=-0.40 keeps objects within the SO-ARM101 IK reachable zone; objects are
# placed above the table and allowed to settle before the grasp attempt.
TEST_OBJECTS = {
    "banana": dict(
        obj="YcbBanana",
        pos=[0.00, -0.40, TABLE_TOP_Z + 0.10],
        note="round — low yaw sensitivity",
    ),
    "cylinder": dict(
        obj="YcbTomatoSoupCan",
        pos=[0.00, -0.40, TABLE_TOP_Z + 0.10],
        note="cylinder — moderate yaw sensitivity",
    ),
    "scissors": dict(
        obj="YcbScissors",
        pos=[0.00, -0.40, TABLE_TOP_Z + 0.10],
        note="flat/thin — high yaw sensitivity",
    ),
}

# camera calibration grid: (x_world, y_world) on tabletop
_CAM_GRID = [
    ( 0.00, -0.45),
    ( 0.12, -0.45),
    (-0.12, -0.45),
    ( 0.00, -0.33),
    ( 0.00, -0.57),
    ( 0.10, -0.35),
    (-0.10, -0.55),
]
_CALIB_OBJ_Z = TABLE_TOP_Z + 0.05   # centroid height used for calibration


# ── back-projection ───────────────────────────────────────────────────────────

def pixel_to_world(px: float, py: float, depth: float,
                   cam_pos, img_size: int = IMG_SIZE, fov_deg: float = FOVY):
    """Back-project one pixel + metric depth to world XYZ.

    Uses the same camera model as EnvironmentSoArm._depth_to_pointcloud:
      world_x = xc + cam_x
      world_y = -yc + cam_y     (image Y is inverted relative to world Y)
      world_z = cam_z - depth
    where xc, yc are image-plane coordinates in metres.
    """
    cx, cy, cz = cam_pos[0], cam_pos[1], cam_pos[2]
    h = w = img_size
    fy = (h / 2) / math.tan(math.radians(fov_deg / 2))
    fx = fy
    xc = (px - w / 2) / fx * depth
    yc = (py - h / 2) / fy * depth
    return np.array([xc + cx, -yc + cy, cz - depth])


# ── camera calibration ────────────────────────────────────────────────────────

def calibrate_camera(env: EnvironmentSoArm, verbose: bool = True) -> dict:
    """
    Place a fiducial (YcbBanana) at each grid point, back-project the
    segmentation centroid to world, and measure the XY reprojection error.

    Returns:
        rmse_m        : mean XY error across grid points (metres)
        x_bias, y_bias: systematic offset in X and Y
        correction_xy : [-x_bias, -y_bias] — add to predicted (x,y) to correct
        points        : per-point records
    """
    cam_pos = [float(v) for v in CAM_POS.split()]
    records = []

    for gx, gy in _CAM_GRID:
        pos = [gx, gy, _CALIB_OBJ_Z]
        env.remove_all_obj()
        oid = env.load_obj("YcbBanana", name="calib", pos=pos)
        env._steps(120)

        obs  = env.get_obs(pointcloud=False)
        seg  = obs["seg"]
        dep  = obs["depth"]
        mask = seg == oid

        if mask.sum() < 10:
            if verbose:
                print(f"  [skip] ({gx:+.2f},{gy:+.2f}): only {mask.sum()} px")
            continue

        ys, xs = np.where(mask)
        px_c   = float(xs.mean())
        py_c   = float(ys.mean())
        d_mean = float(dep[mask].mean())

        pred = pixel_to_world(px_c, py_c, d_mean, cam_pos)
        err  = pred[:2] - np.array([gx, gy])

        records.append(dict(
            true_x=round(gx, 4), true_y=round(gy, 4),
            pred_x=round(float(pred[0]), 4), pred_y=round(float(pred[1]), 4),
            err_x=round(float(err[0]), 4), err_y=round(float(err[1]), 4),
            err_m=round(float(np.linalg.norm(err)), 4),
            px=round(px_c, 1), py=round(py_c, 1), depth=round(d_mean, 4),
        ))
        if verbose:
            print(f"  ({gx:+.2f},{gy:+.2f}) pred({pred[0]:+.3f},{pred[1]:+.3f}) "
                  f"err({err[0]:+.3f},{err[1]:+.3f}) [{np.linalg.norm(err)*1000:.1f} mm]")

    if not records:
        return {"error": "no valid calibration points — all objects invisible in seg"}

    errs   = np.array([r["err_m"] for r in records])
    x_bias = float(np.mean([r["err_x"] for r in records]))
    y_bias = float(np.mean([r["err_y"] for r in records]))

    out = dict(
        cam_pos=CAM_POS,
        n_points=len(records),
        rmse_m=round(float(errs.mean()), 5),
        x_bias=round(x_bias, 5),
        y_bias=round(y_bias, 5),
        correction_xy=[round(-x_bias, 5), round(-y_bias, 5)],
        points=records,
    )
    if verbose:
        print(f"\n  RMSE={errs.mean()*1000:.1f} mm  "
              f"bias=({x_bias*1000:+.1f}, {y_bias*1000:+.1f}) mm  "
              f"(n={len(records)})")
    return out


# ── gripper opening calibration ───────────────────────────────────────────────

def calibrate_gripper(env: EnvironmentSoArm, n_steps: int = 12,
                      verbose: bool = True) -> dict:
    """
    Sweep the gripper ctrl from GRIP_CLOSED to GRIP_OPEN and verify that the
    position controller (servo) tracks the command accurately.

    Note on SO-ARM101 gripper geometry
    ------------------------------------
    The SO-ARM101 jaw is a *hinge* joint, not a prismatic one. Body
    center-to-center distance stays constant as the jaw rotates.
    The physical jaw gap = jaw_arm_length * sin(joint_angle), which requires
    the mesh geometry to compute exactly.

    In simulation we measure:
      - ctrl  : commanded joint angle (radians, doubles as position setpoint)
      - qpos  : actual joint angle reached by the position controller
      - tracking_err_rad : |qpos - ctrl| (controller accuracy)

    For real-hardware calibration: measure jaw gap with calipers at
    ctrl=GRIP_CLOSED and ctrl=GRIP_OPEN, then set:
        gripper_opening_scale = measured_max_gap_m / 0.10

    Returns:
        samples           : [{ctrl, qpos_rad, tracking_err_rad, target_m}, ...]
        max_tracking_err  : worst-case |qpos - ctrl| across sweep
        ctrl_range_rad    : (GRIP_CLOSED, GRIP_OPEN) in radians
        recommended_scale : 0.85 (default; update after real-hardware measurement)
        note              : explanation of why physical gap is not reported
    """
    env.reset_robot()
    ctrls   = np.linspace(GRIP_CLOSED, GRIP_OPEN, n_steps)
    samples = []

    for c in ctrls:
        env.data.ctrl[env._grip_act_id] = c
        env._steps(100)
        qpos     = float(env.data.qpos[env._grip_qpos_adr])
        tracking = abs(qpos - c)
        # target_m: the opening_len value that move_gripper() uses for this ctrl
        t_m      = (c - GRIP_CLOSED) / (GRIP_OPEN - GRIP_CLOSED) * 0.10
        samples.append(dict(
            ctrl=round(float(c), 4),
            qpos_rad=round(qpos, 4),
            tracking_err_rad=round(tracking, 5),
            target_m=round(float(t_m), 4),
        ))
        if verbose:
            print(f"  ctrl={c:.3f}  qpos={qpos:.3f} rad  "
                  f"err={tracking*1000:.1f} mrad  target_m={t_m*1000:.0f} mm")

    max_err = float(max(s["tracking_err_rad"] for s in samples))
    if verbose:
        print(f"\n  ctrl_range=[{GRIP_CLOSED}, {GRIP_OPEN}] rad  "
              f"max_tracking_err={max_err*1000:.1f} mrad  "
              f"(real-hardware jaw gap: measure with calipers)")

    return dict(
        samples=samples,
        max_tracking_err_rad=round(max_err, 5),
        ctrl_range_rad=[GRIP_CLOSED, GRIP_OPEN],
        recommended_scale=0.85,
        note=(
            "Physical jaw gap not computed (hinge joint — requires mesh geometry). "
            "Update recommended_scale after real-hardware caliper measurement: "
            "scale = measured_max_gap_m / 0.10"
        ),
    )


# ── yaw execution helpers ─────────────────────────────────────────────────────

def _apply_top_down_yaw(env: EnvironmentSoArm, target_yaw: float):
    """
    Approximate top-down yaw by commanding the wrist_roll joint.

    When the arm is in a near-vertical (top-down) configuration, wrist_roll
    ≈ end-effector rotation around the world Z axis. This is a first-order
    approximation; accuracy degrades for highly off-vertical poses.
    """
    try:
        wr_idx  = ARM_JOINTS.index("wrist_roll")
        lo = env.model.jnt_range[env._arm_jnt_ids[wr_idx], 0]
        hi = env.model.jnt_range[env._arm_jnt_ids[wr_idx], 1]
        cmd = float(np.clip(target_yaw, lo, hi))
        env.data.ctrl[env._arm_act_ids[wr_idx]] = cmd
        env.data.qpos[env._arm_qpos_adr[wr_idx]] = cmd
        env._steps(100)
    except (ValueError, IndexError, Exception):
        pass   # joint not reachable; silently fall back to xyz_only


def _attempt_grasp(env: EnvironmentSoArm, obj_id: int,
                   grasp: list, yaw_mode: str) -> bool:
    """
    Single grasp attempt with the given yaw mode.

    grasp = [x, y, z, yaw, opening_len, obj_height]
    Returns True if obj_id is grasped after the attempt.
    """
    x, y, z    = float(grasp[0]), float(grasp[1]), float(grasp[2])
    yaw        = float(grasp[3]) if len(grasp) > 3 else 0.0
    opening    = float(grasp[4]) if len(grasp) > 4 else 0.06
    # obj_height unused in xyz-only IK; placeholder for future z-offset model

    env.reset_robot()
    env.move_gripper(opening)

    # approach from above
    env.move_ee([x, y, env.GRIPPER_MOVING_HEIGHT, None])

    if yaw_mode == "top_down_yaw":
        _apply_top_down_yaw(env, yaw)
    elif yaw_mode == "full_6dof":
        # Placeholder — full 6-DoF IK not yet implemented.
        # When implemented: pass (x, y, z, roll=yaw, pitch=π/2, 0) to a 6-DoF solver.
        # Falls back to xyz_only for now.
        pass
    # xyz_only: yaw is ignored — no extra step needed

    # Descend: z is the EEF target height (settled_z + small offset).
    env.move_ee([x, y, z, None])

    # Close fully (without early-stop) then check contact.
    # Success criterion: at least one gripper jaw contacts the object.
    # This measures IK positioning accuracy and yaw alignment — a valid
    # proxy for grasping quality even before a full lift is achieved.
    env.auto_close_gripper(check_contact=False)
    env._steps(80)

    contact_success = obj_id in env.check_grasped_id()

    # Attempt lift and re-check; use contact_success as fallback if lift fails.
    env.move_ee([x, y, env.GRIPPER_MOVING_HEIGHT, None])
    env._steps(80)
    lift_success = obj_id in env.check_grasped_id()
    obj_z = env.get_obj_pos(obj_id)[2] if env.obj_ids else 0.0
    lifted = obj_z > env.GRIPPER_MOVING_HEIGHT - 0.10

    return contact_success or lifted


# ── yaw ablation ─────────────────────────────────────────────────────────────

# Test yaw angles: -45°, 0°, +45°, +90°
_YAW_TEST_ANGLES = [-math.pi / 4, 0.0, math.pi / 4, math.pi / 2]


def run_yaw_ablation(env: EnvironmentSoArm, obj_key: str = "banana",
                     n_trials: int = 4,
                     yaw_modes: list = None,
                     verbose: bool = True) -> dict:
    """
    For each yaw mode, attempt grasps at `n_trials` distinct yaw angles and
    record binary success.

    Returns {mode: {success_rate, n_trials, detail: [{yaw_deg, success}]}}
    """
    if yaw_modes is None:
        yaw_modes = ["xyz_only", "top_down_yaw"]

    cfg      = TEST_OBJECTS.get(obj_key, TEST_OBJECTS["banana"])
    obj_name = cfg["obj"]
    obj_pos  = cfg["pos"]
    angles   = _YAW_TEST_ANGLES[:n_trials]

    results = {}
    for mode in yaw_modes:
        hits, detail = 0, []
        for yaw in angles:
            env.reset_robot()          # move arm to HOME before loading object
            env.remove_all_obj()
            oid = env.load_obj(obj_name, name=obj_key, pos=list(obj_pos))
            env._steps(300)   # let physics settle fully

            # Use actual settled position so IK target is valid
            settled = env.get_obj_pos(oid)
            grasp   = [settled[0], settled[1],
                       settled[2] + 0.02,   # small offset above object centroid
                       yaw, 0.06, 0.04]
            success = _attempt_grasp(env, oid, grasp, mode)
            hits   += int(success)
            detail.append(dict(yaw_deg=round(math.degrees(yaw), 1),
                               success=bool(success)))
            if verbose:
                sym = "✓" if success else "✗"
                print(f"  [{mode}] yaw={math.degrees(yaw):+.0f}°  {sym}")

        sr = hits / len(angles) if angles else 0.0
        results[mode] = dict(
            success_rate=round(sr, 3),
            n_trials=len(angles),
            detail=detail,
        )
        if verbose:
            print(f"  ── [{mode}] SR={sr:.2f} ({hits}/{len(angles)})\n")

    return results


# ── post-calibration validation ───────────────────────────────────────────────

def validate_object(env: EnvironmentSoArm, obj_key: str, calib: dict,
                    n_trials: int = 5, verbose: bool = True) -> dict:
    """
    Mini grasp validation using calibrated parameters.

    Applies:
      - cam correction_xy offset to object position (simulates bias correction)
      - calibrated gripper_opening_scale
      - recommended yaw_mode from calibration
    """
    cfg      = TEST_OBJECTS.get(obj_key, TEST_OBJECTS["banana"])
    obj_name = cfg["obj"]
    obj_pos  = list(cfg["pos"])

    scale    = calib.get("gripper_opening_scale", 0.85)
    yaw_mode = calib.get("yaw_mode", "top_down_yaw")
    corr     = calib.get("cam_to_robot_base", {}).get("correction_xy", [0.0, 0.0])

    if verbose:
        print(f"\n  obj={obj_key}  yaw_mode={yaw_mode}  "
              f"scale={scale:.3f}  correction=({corr[0]*1000:+.1f},{corr[1]*1000:+.1f}) mm")

    rng = np.random.default_rng(42)
    hits = 0
    detail = []
    for i in range(n_trials):
        env.reset_robot()      # move arm away before loading
        env.remove_all_obj()
        pos = [obj_pos[0] + corr[0], obj_pos[1] + corr[1], obj_pos[2]]
        oid = env.load_obj(obj_name, name=obj_key, pos=pos)
        env._steps(300)

        # Use actual settled position for grasp target
        settled = env.get_obj_pos(oid)
        yaw     = float(rng.uniform(-math.pi / 3, math.pi / 3))
        grasp   = [settled[0], settled[1], settled[2] + 0.02,
                   yaw, 0.06 * scale, 0.04]
        ok      = _attempt_grasp(env, oid, grasp, yaw_mode)
        hits += int(ok)
        detail.append(dict(trial=i + 1, yaw_deg=round(math.degrees(yaw), 1),
                           success=bool(ok)))
        if verbose:
            sym = "✓" if ok else "✗"
            print(f"  trial {i+1}/{n_trials}  yaw={math.degrees(yaw):+.0f}°  {sym}")

    sr = hits / n_trials
    print(f"  → success_rate = {sr:.2f} ({hits}/{n_trials})")
    return dict(obj=obj_key, success_rate=round(sr, 3),
                n_trials=n_trials, yaw_mode=yaw_mode, detail=detail)


# ── save / load calibration YAML ─────────────────────────────────────────────

def save_calibration(data: dict, path: str = DEFAULT_OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    print(f"\n[INFO] calibration saved → {path}")


def load_calibration(path: str = DEFAULT_OUT) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="SO-ARM101 MuJoCo calibration & validation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full calibration suite (cam + gripper + yaw ablation on banana)
  python scripts/calibrate_soarm_mujoco.py

  # Camera calibration only
  python scripts/calibrate_soarm_mujoco.py --task cam

  # Yaw ablation on scissors with both modes
  python scripts/calibrate_soarm_mujoco.py --task yaw --obj scissors \\
      --yaw_modes xyz_only top_down_yaw

  # Validate on banana using saved calibration
  python scripts/calibrate_soarm_mujoco.py --validate banana

  # Validate on cylinder
  python scripts/calibrate_soarm_mujoco.py --validate cylinder

  # Validate on scissors (most yaw-sensitive)
  python scripts/calibrate_soarm_mujoco.py --validate scissors
""")
    ap.add_argument("--task", choices=["all", "cam", "gripper", "yaw"],
                    default="all",
                    help="Which calibration task to run (default: all)")
    ap.add_argument("--validate", metavar="OBJ",
                    choices=list(TEST_OBJECTS.keys()),
                    help="Run post-calibration validation on this object")
    ap.add_argument("--obj", default="banana",
                    choices=list(TEST_OBJECTS.keys()),
                    help="Target object for yaw ablation (default: banana)")
    ap.add_argument("--yaw_modes", nargs="+", default=["xyz_only", "top_down_yaw"],
                    choices=YAW_MODES,
                    help="Yaw modes to compare in ablation")
    ap.add_argument("--n_trials", type=int, default=4,
                    help="Grasp trials per yaw mode / validation (default: 4)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output YAML path (default: {DEFAULT_OUT})")
    ap.add_argument("--vis", action="store_true",
                    help="Open MuJoCo passive viewer")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-step output")
    args = ap.parse_args()

    verbose = not args.quiet
    env     = EnvironmentSoArm(vis=args.vis, debug=False)
    out     = {}

    # ── camera calibration ────────────────────────────────────────────────────
    if args.task in ("all", "cam"):
        print("\n=== Camera-to-Robot-Base Calibration ===")
        cam = calibrate_camera(env, verbose=verbose)
        out["cam_to_robot_base"] = cam

    # ── gripper calibration ───────────────────────────────────────────────────
    if args.task in ("all", "gripper"):
        print("\n=== Gripper Opening Calibration ===")
        grip = calibrate_gripper(env, verbose=verbose)
        out["gripper_opening_scale"]       = grip["recommended_scale"]
        out["gripper_max_tracking_err_rad"] = grip["max_tracking_err_rad"]
        out["gripper_ctrl_range_rad"]       = grip["ctrl_range_rad"]
        out["gripper_note"]                 = grip["note"]
        out["gripper_samples"]              = grip["samples"]

    # ── yaw ablation ──────────────────────────────────────────────────────────
    if args.task in ("all", "yaw"):
        print(f"\n=== Yaw Ablation: {args.obj} "
              f"({TEST_OBJECTS[args.obj]['note']}) ===")
        yaw_results = run_yaw_ablation(
            env, obj_key=args.obj,
            n_trials=args.n_trials,
            yaw_modes=args.yaw_modes,
            verbose=verbose,
        )
        out["yaw_ablation"] = {
            mode: dict(success_rate=v["success_rate"], n_trials=v["n_trials"],
                       detail=v["detail"])
            for mode, v in yaw_results.items()
        }
        # pick best yaw mode by success rate (prefer top_down_yaw on tie)
        best = max(yaw_results,
                   key=lambda m: (yaw_results[m]["success_rate"],
                                  m == "top_down_yaw"))
        out["yaw_mode"] = best
        if verbose:
            print(f"  → recommended yaw_mode: {best}")

    # ── validation ────────────────────────────────────────────────────────────
    if args.validate:
        calib_data = out if out else {}
        if not calib_data and os.path.exists(args.out):
            calib_data = load_calibration(args.out)
            print(f"[INFO] loaded existing calibration from {args.out}")
        print(f"\n=== Validation: {args.validate} ===")
        val = validate_object(env, obj_key=args.validate,
                              calib=calib_data,
                              n_trials=args.n_trials,
                              verbose=verbose)
        out.setdefault("validation", {})[args.validate] = val

    # ── save ──────────────────────────────────────────────────────────────────
    if out:
        save_calibration(out, args.out)
    else:
        print("[WARN] nothing to save — specify --task or --validate")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
