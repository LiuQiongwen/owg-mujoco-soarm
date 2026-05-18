#!/usr/bin/env python3
"""
Validate 6-DoF orientation-aware IK for SO-ARM101.

Tests (in order):
  1. make_topdown_rotation() — orthonormality + Z-axis direction
  2. Jaw-midpoint IK: _solve_ik_jaw_topdown reduces jaw-mid→object gap vs xyz_only
  3. Debug metrics: get_grasp_debug_metrics() keys and finite values
  4. Backward compat: move_ee() xyz_only mode still works
  5. Bilateral + lift: _execute_grasp_physics_topdown achieves bilateral clamping
     and lifts a 2 cm cube (within the ~3.5 cm jaw opening gap).

Object size note:
  The SO-ARM101 scissor jaw has a body-centre separation of ~3.5 cm at maximum
  opening. Objects wider than ~3 cm cannot fit between the jaws for clamping;
  use half-size ≤ 0.015 m for reliable bilateral contact.

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
    IK_MODE_JAW_TOPDOWN,
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


# ── Test 2: jaw-midpoint IK ────────────────────────────────────────────────────

def test_jaw_midpoint_ik(env):
    """_solve_ik_jaw_topdown reduces jaw-midpoint→target gap vs xyz_only baseline."""
    print("\n=== Test 2: jaw-midpoint IK vs xyz_only baseline ===")
    test_pts = [
        (0.0, -0.25, TABLE_TOP_Z + 0.02),
        (0.0, -0.20, TABLE_TOP_Z + 0.02),
    ]
    all_ok = True
    for x, y, z in test_pts:
        tgt = np.array([x, y, z])

        # xyz_only baseline: jaw midpoint gap after position-only IK
        env.reset_robot()
        env._solve_ik(tgt)
        gap_base = float(np.linalg.norm(env._get_jaw_midpoint()[:2] - tgt[:2]))

        # jaw-midpoint IK
        env.reset_robot()
        ok_ik, jaw_pe, oe = env._solve_ik_jaw_topdown(tgt, yaw=0.0, iters=600)
        gap_jaw = float(np.linalg.norm(env._get_jaw_midpoint()[:2] - tgt[:2]))

        improved = gap_jaw < gap_base
        all_ok &= improved
        report(
            f"({x:+.2f},{y:+.2f},{z:.3f})  jaw_gap: {gap_base*100:.1f}cm → {gap_jaw*100:.1f}cm",
            improved,
            f"jaw_pe={jaw_pe*100:.1f}cm oe={np.degrees(oe):.0f}°",
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
        ok = True
        report("move_ee xyz_only returns without exception", ok,
               f"eef={pos.round(3)}")
    except Exception as e:
        report("move_ee xyz_only returns without exception", False, str(e))
        ok = False
    return ok


# ── Test 5: bilateral clamping + lift ─────────────────────────────────────────

def test_bilateral_lift(env):
    """jaw-midpoint IK achieves bilateral clamping and lifts a tall thin box.

    The SO-ARM101 base is mounted on the table surface.  Its elbow link collides
    with the table when the arm tries to reach below z ≈ TABLE_TOP_Z + 0.075 m.
    Therefore the test object is a tall thin box (3.4 cm × 3.4 cm × 15 cm) whose
    centre sits at TABLE_TOP_Z + 0.075 = 0.860 m — within the arm's physical
    reach — and whose width (3.4 cm) fits inside the ~3.6 cm jaw geom gap.

    The jaw-geom-midpoint IK places the jaw tips ~1.2 mm outside each object face
    so that closing produces opposing bilateral clamping forces.
    Bilateral contacts ≥ 60% and lift success ≥ 50% over 6 trials is the criterion.
    """
    print("\n=== Test 5: bilateral clamping + lift (tall thin box) ===")

    # Tall thin box: 3.4 cm × 3.4 cm × 15 cm
    # Half-size z=0.075 → centre at TABLE_TOP_Z + 0.075 = 0.860 m (reachable).
    # Half-size x,y=0.017 → 3.4 cm width.  Jaw geom Y-gap ≈ 3.6 cm at IK
    # solution → ~1.2 mm clearance per side, within the ~4 mm closing motion.
    OBJ_HALF_Z = 0.075
    OBJ_HALF_XY = 0.017
    box_pool = register_primitive_geom(
        "box", (OBJ_HALF_XY, OBJ_HALF_XY, OBJ_HALF_Z), 0.100,
        (1.5, 0.05, 0.01), (0.2, 0.5, 0.8, 1.0))

    bilateral_count = 0
    lift_count      = 0
    n_trials        = 6

    for seed in range(n_trials):
        env.remove_all_obj()
        env.reset_robot()
        obj_id = env.load_obj(box_pool, name="obj",
                              pos=[0.0, -0.25, TABLE_TOP_Z + OBJ_HALF_Z + 0.06])
        env._steps(300)
        env.wait_until_all_still(max_wait_epochs=100)
        obj_pos = env.get_obj_pos(obj_id)

        ok, _ = env._execute_grasp_physics_topdown(
            pos=(float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])),
            yaw=0.0,
            gripper_opening_length=0.09,
            obj_height=2 * OBJ_HALF_Z,
        )
        m = env.last_grasp_metrics or {}
        if m.get("bilateral_contacts", 0) > 0:
            bilateral_count += 1
        if ok:
            lift_count += 1

    bil_rate  = bilateral_count / n_trials
    lift_rate = lift_count / n_trials
    ok_bil  = bil_rate  >= 0.60
    ok_lift = lift_rate >= 0.50

    report(f"bilateral contacts ≥ 60%: {bilateral_count}/{n_trials} = {bil_rate:.0%}", ok_bil)
    report(f"lift success       ≥ 50%: {lift_count}/{n_trials} = {lift_rate:.0%}", ok_lift)
    return ok_bil and ok_lift


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
            import traceback
            print(f"  [EXCEPTION] {fn.__name__}: {exc}")
            traceback.print_exc()
            results.append(False)

    run(test_rotation_math)
    run(test_jaw_midpoint_ik, env)
    run(test_debug_metrics, env)
    run(test_xyz_only_compat, env)

    if not args.skip_grasp:
        run(test_bilateral_lift, env)

    env.close()

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{total} test groups passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
