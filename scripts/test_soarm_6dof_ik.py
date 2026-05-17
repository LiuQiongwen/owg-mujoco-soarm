#!/usr/bin/env python3
"""
Validate 6-DoF orientation-aware IK for SO-ARM101.

Tests (in order):
  1. make_topdown_rotation() — orthonormality + Z-axis direction
  2. IK convergence: 6-DoF IK reduces jaw-object XY gap vs xyz_only baseline
  3. Debug metrics: get_grasp_debug_metrics() keys and finite values
  4. Backward compat: move_ee() xyz_only mode still works
  5. Bilateral contacts: _execute_grasp_physics_topdown achieves bilateral contacts
     NOTE: actual lift remains 0% — SO-ARM101 oblique approach geometry means both
     jaws contact the cube's +Y face (push contact) rather than clamping from
     opposite sides. This is the confirmed physics-mode baseline documented in
     calib_logs/CALIBRATION_FINDINGS.md.

Usage:
  conda run -n owg2 python scripts/test_soarm_6dof_ik.py
  conda run -n owg2 python scripts/test_soarm_6dof_ik.py --skip-grasp
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from owg_robot.env_soarm import (
    EnvironmentSoArm,
    GRASP_MODE_PHYSICS,
    TABLE_TOP_Z,
    IK_MODE_XYZ_ONLY,
    IK_MODE_YAW_TOPDOWN,
    IK_MODE_FULL_6DOF,
    make_topdown_rotation,
    register_primitive_geom,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def report(label, ok, detail=""):
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    return ok


# ── Test 1: make_topdown_rotation ─────────────────────────────────────────────

def test_rotation_math():
    print("\n=== Test 1: make_topdown_rotation() ===")
    all_ok = True
    for yaw in [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3]:
        R = make_topdown_rotation(yaw)
        ortho  = np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        det    = np.linalg.det(R)
        z_down = np.allclose(R[:, 2], [0, 0, -1], atol=1e-10)
        ok     = ortho and abs(det - 1.0) < 1e-10 and z_down
        all_ok &= ok
        report(f"yaw={yaw:.3f}  orthonormal={ortho}  det={det:.4f}  Z_down={z_down}", ok)
    return all_ok


# ── Test 2: IK jaw-gap improvement ────────────────────────────────────────────

def test_ik_jaw_gap(env):
    """6-DoF IK should reduce jaw midpoint-to-object XY gap vs xyz_only."""
    print("\n=== Test 2: 6-DoF IK jaw-gap improvement ===")
    test_pts = [
        (0.0, -0.25, TABLE_TOP_Z + 0.10),
        (0.0, -0.20, TABLE_TOP_Z + 0.08),
    ]
    all_ok = True
    for x, y, z in test_pts:
        tgt = np.array([x, y, z])

        # xyz_only baseline
        env.reset_robot()
        env._solve_ik(tgt)
        jaw_fix = env.data.xpos[env._jaw_body_id].copy()
        jaw_mv  = env.data.xpos[env._jaw_mv_body_id].copy()
        mid_xy_base = 0.5 * (jaw_fix[:2] + jaw_mv[:2])
        gap_base = float(np.linalg.norm(mid_xy_base - tgt[:2]))

        # 6-DoF IK
        env.reset_robot()
        R_tgt = make_topdown_rotation(0.0)
        ok_ik, pe, oe = env._solve_ik_6dof(tgt, R_tgt, iters=400)
        jaw_fix = env.data.xpos[env._jaw_body_id].copy()
        jaw_mv  = env.data.xpos[env._jaw_mv_body_id].copy()
        mid_xy_6d = 0.5 * (jaw_fix[:2] + jaw_mv[:2])
        gap_6dof = float(np.linalg.norm(mid_xy_6d - tgt[:2]))

        improved = gap_6dof < gap_base
        all_ok &= improved
        report(
            f"({x:+.2f},{y:+.2f},{z:.3f})  gap_base={gap_base*100:.1f}cm → gap_6dof={gap_6dof*100:.1f}cm",
            improved,
            f"pe={pe*1000:.1f}mm oe={oe:.3f}rad",
        )
    return all_ok


# ── Test 3: debug metrics ─────────────────────────────────────────────────────

def test_debug_metrics(env):
    print("\n=== Test 3: get_grasp_debug_metrics() ===")
    env.reset_robot()
    m = env.get_grasp_debug_metrics()
    ok_keys    = all(k in m for k in [
        "ori_err_norm", "jaw_obj_xy_gap", "bilateral_contacts",
        "left_contacts", "right_contacts", "symmetry_score", "eef_z_axis",
    ])
    ori_finite = np.isfinite(m["ori_err_norm"])
    eef_z_unit = abs(np.linalg.norm(m["eef_z_axis"]) - 1.0) < 1e-2
    ok = ok_keys and ori_finite and eef_z_unit
    report("keys present + ori_err finite + eef_z_axis unit", ok,
           f"ori_err={m['ori_err_norm']:.4f} eef_z={m['eef_z_axis']}")
    return ok


# ── Test 4: xyz_only backward compat ─────────────────────────────────────────

def test_xyz_only_compat(env):
    print("\n=== Test 4: move_ee() xyz_only backward compat ===")
    env.reset_robot()
    target = [0.05, -0.25, TABLE_TOP_Z + 0.10, None]
    try:
        ok_ret, (pos, orn) = env.move_ee(target, ik_mode=IK_MODE_XYZ_ONLY)
        ok = True  # call completed without exception
        report("move_ee xyz_only returns without exception", ok,
               f"eef={pos.round(3)}")
    except Exception as e:
        report("move_ee xyz_only returns without exception", False, str(e))
        ok = False
    return ok


# ── Test 5: bilateral contacts via topdown ────────────────────────────────────

def test_bilateral_contacts(env):
    """_execute_grasp_physics_topdown achieves bilateral jaw-cube contacts."""
    print("\n=== Test 5: bilateral contacts via topdown grasp ===")
    cube_pool = register_primitive_geom(
        "box", (0.025, 0.025, 0.025), 0.150, (1.5, 0.05, 0.01), (0.2, 0.5, 0.8, 1.0))

    bilateral_count = 0
    n_trials = 5
    for seed in range(n_trials):
        env.remove_all_obj()
        env.reset_robot()
        obj_id = env.load_obj(cube_pool, name="obj",
                              pos=[0.0, -0.25, TABLE_TOP_Z + 0.025 + 0.06])
        env._steps(300)
        env.wait_until_all_still(max_wait_epochs=100)
        obj_pos = env.get_obj_pos(obj_id)

        env._execute_grasp_physics_topdown(
            pos=(float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]) + 0.01),
            yaw=0.0, gripper_opening_length=0.09, obj_height=0.05,
        )
        m = env.get_grasp_debug_metrics(obj_id)
        if m["bilateral_contacts"] > 0:
            bilateral_count += 1

    rate = bilateral_count / n_trials
    ok = rate >= 0.6   # bilateral contact in at least 60% of trials
    report(f"bilateral contacts in {bilateral_count}/{n_trials} trials", ok,
           "NOTE: lift=0% expected — push contacts only, see CALIBRATION_FINDINGS.md")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-grasp", action="store_true",
                    help="Skip physics grasp tests (runs IK/metric tests only)")
    args = ap.parse_args()

    print("Creating environment …")
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS)

    results = []

    def run(fn, *a, **kw):
        try:
            r = fn(*a, **kw)
            results.append(bool(r))
        except Exception as exc:
            print(f"  [EXCEPTION] {fn.__name__}: {exc}")
            results.append(False)

    run(test_rotation_math)
    run(test_ik_jaw_gap, env)
    run(test_debug_metrics, env)
    run(test_xyz_only_compat, env)

    if not args.skip_grasp:
        run(test_bilateral_contacts, env)

    env.close()

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{total} test groups passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
