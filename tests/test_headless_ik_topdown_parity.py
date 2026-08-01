# -*- coding: utf-8 -*-
"""Cross-check: HeadlessIKSolver.solve_ik_jaw_topdown vs
EnvironmentSoArm._solve_ik_jaw_topdown -- the Tier-3 per-candidate IK
feature's actual code path (see
EnvironmentSoArm.compute_ik_reachability_per_candidate).

Proves the extracted headless topdown solver reproduces the original sim IK
exactly (within float tolerance), which is the precondition for the
lggsn_tier3_ik_pilot (experiments/lggsn_tier3_ik_pilot.yaml) trusting
HeadlessIKSolver as a stand-in for EnvironmentSoArm under the MVP4
restricted-execution harness, whose fixed RLIMIT_AS is too small for
EnvironmentSoArm's unconditional GL/EGL renderer. Mirrors
test_headless_ik_parity.py's structure exactly, extended to yaw/orientation.
Run with:

    conda run -n tango python tests/test_headless_ik_topdown_parity.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from tango_robot.env_soarm import EnvironmentSoArm
from tango_robot.headless_ik import HeadlessIKSolver


def test_headless_ik_topdown_parity():
    env = EnvironmentSoArm(vis=False)   # ground truth -- allowed to use GL/renderer
    headless = HeadlessIKSolver()       # under test -- must not need GL

    home_mid = env._get_jaw_geom_midpoint().copy()
    print("HOME jaw geom midpoint (world):", home_mid.round(4))

    # (offset, yaw) pairs -- same style of small perturbations around HOME as
    # the position-only parity test, plus nonzero yaw to exercise the
    # orientation-aware branch that solve_ik_jaw_pos_only never touches.
    cases = [
        (np.array([0.00, 0.00, 0.00]), 0.0),
        (np.array([0.02, 0.00, 0.02]), 0.0),
        (np.array([-0.02, 0.03, 0.00]), 0.5),
        (np.array([0.00, -0.02, 0.03]), -0.3),
        (np.array([0.03, 0.02, -0.02]), 1.2),
        (np.array([-0.03, -0.02, 0.02]), -1.0),
    ]

    max_q_dev = 0.0
    max_pe_dev = 0.0
    rows = []
    for i, (offset, yaw) in enumerate(cases):
        target = home_mid + offset

        ok_a, pe_a, oe_a = env._solve_ik_jaw_topdown(np.asarray(target), yaw=yaw)
        q_a = np.array([env.data.qpos[adr] for adr in env._arm_qpos_adr])

        ok_b, pe_b, oe_b = headless.solve_ik_jaw_topdown(np.asarray(target), yaw=yaw)
        q_b = headless.get_arm_qpos()

        q_dev = float(np.max(np.abs(q_a - q_b)))
        pe_dev = abs(pe_a - pe_b)
        max_q_dev = max(max_q_dev, q_dev)
        max_pe_dev = max(max_pe_dev, pe_dev)

        rows.append((i, ok_a, ok_b, pe_a, pe_b, q_dev))
        print(f"case[{i}] yaw={yaw:+.2f} converged(env={ok_a}, headless={ok_b}) "
              f"pe_env={pe_a*1000:.3f}mm pe_headless={pe_b*1000:.3f}mm "
              f"max|dq|={q_dev:.3e}rad ({np.degrees(q_dev):.4f}deg)")

    print()
    print(f"MAX joint-angle deviation across all cases: {max_q_dev:.3e} rad "
          f"({np.degrees(max_q_dev):.5f} deg)")
    print(f"MAX pos-error-metric deviation: {max_pe_dev*1000:.4f} mm")

    all_conv_match = all(a == b for _, a, b, *_ in rows)
    print(f"convergence flags match on every case: {all_conv_match}")

    # Same 1e-6 rad bound as test_headless_ik_parity.py, for the same reason:
    # unreasonable to demand bitwise identity across two independently
    # compiled MjModel/MjData instances, but ~1000x tighter than any
    # SO-ARM101 servo's real angular resolution.
    tol_rad = 1e-6
    passed = max_q_dev < tol_rad and all_conv_match
    print()
    print("RESULT:", "PASS -- exact equivalence" if passed else "FAIL -- divergence detected")
    assert passed, (
        f"topdown IK parity check failed: max_q_dev={max_q_dev:.3e} rad "
        f"(tol={tol_rad:.0e} rad), convergence flags match={all_conv_match}"
    )


if __name__ == "__main__":
    test_headless_ik_topdown_parity()
