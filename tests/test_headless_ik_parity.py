# -*- coding: utf-8 -*-
"""Cross-check: HeadlessIKSolver vs EnvironmentSoArm._solve_ik_jaw_pos_only.

Proves the extracted headless solver reproduces the original sim IK exactly
(within float tolerance), which is the precondition for trusting it on real
hardware — see docs/w1_headless_ik notes. Run with:

    conda run -n tango python tests/test_headless_ik_parity.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from tango_robot.env_soarm import EnvironmentSoArm
from tango_robot.headless_ik import HeadlessIKSolver


def main():
    env = EnvironmentSoArm(vis=False)   # ground truth — allowed to use GL/renderer
    headless = HeadlessIKSolver()       # under test — must not need GL

    # HOME-frame jaw geom midpoint as an anchor: perturbations around it stay
    # inside the reachable workspace so IK convergence isn't the confound.
    home_mid = env._get_jaw_geom_midpoint().copy()
    print("HOME jaw geom midpoint (world):", home_mid.round(4))

    targets = [
        home_mid + np.array([0.00, 0.00, 0.00]),
        home_mid + np.array([0.05, 0.00, 0.00]),
        home_mid + np.array([-0.05, 0.03, 0.00]),
        home_mid + np.array([0.00, -0.05, 0.05]),
        home_mid + np.array([0.03, 0.04, -0.03]),
    ]

    max_q_dev = 0.0
    max_pe_dev = 0.0
    rows = []
    for i, target in enumerate(targets):
        ok_a, pe_a, _ = env._solve_ik_jaw_pos_only(np.asarray(target), silent=True)
        q_a = np.array([env.data.qpos[adr] for adr in env._arm_qpos_adr])

        ok_b, pe_b, _ = headless.solve_ik_jaw_pos_only(np.asarray(target), silent=True)
        q_b = headless.get_arm_qpos()

        q_dev = float(np.max(np.abs(q_a - q_b)))
        pe_dev = abs(pe_a - pe_b)
        max_q_dev = max(max_q_dev, q_dev)
        max_pe_dev = max(max_pe_dev, pe_dev)

        rows.append((i, ok_a, ok_b, pe_a, pe_b, q_dev))
        print(f"target[{i}] converged(env={ok_a}, headless={ok_b}) "
              f"pe_env={pe_a*1000:.3f}mm pe_headless={pe_b*1000:.3f}mm "
              f"max|dq|={q_dev:.3e}rad ({np.degrees(q_dev):.4f}deg)")

    print()
    print(f"MAX joint-angle deviation across all targets: {max_q_dev:.3e} rad "
          f"({np.degrees(max_q_dev):.5f} deg)")
    print(f"MAX pos-error-metric deviation: {max_pe_dev*1000:.4f} mm")

    all_conv_match = all(a == b for _, a, b, *_ in rows)
    print(f"convergence flags match on every target: {all_conv_match}")

    # 1e-9 rad is unreasonable to demand bitwise across two independently
    # compiled MjModel/MjData instances (float summation order, etc. isn't
    # guaranteed identical). 1e-6 rad (~5.7e-5 deg) is still ~1000x tighter
    # than any SO-ARM101 servo's real angular resolution, so it's a safe
    # "these are the same solution" bound rather than a loosened test.
    tol_rad = 1e-6
    passed = max_q_dev < tol_rad and all_conv_match
    print()
    print("RESULT:", "PASS — exact equivalence" if passed else "FAIL — divergence detected")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
