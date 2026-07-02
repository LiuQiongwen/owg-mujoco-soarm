import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from grasp_6dof.grasp_sampler import (
    local_point_density, normal_consistency, contact_width_ratio,
)


def _scissors():
    """Thin blade: 0.15×0.02×0.002m lying in world XY-plane.
    Gripper jaw (local x-axis) = world-z, so it closes across the 2mm thickness."""
    rng = np.random.default_rng(42)
    blade = rng.uniform([-0.075, -0.01, -0.001], [0.075, 0.01, 0.001], (500, 3))
    bg    = rng.uniform([-0.5, -0.5, 0.], [0.5, 0.5, 0.5], (9500, 3))
    ep    = np.vstack([blade, bg])
    # Column 0 of R = [0,0,1] (world-z) → local-x = world-z component
    R = np.array([[0., 0., -1.],
                  [0., 1.,  0.],
                  [1., 0.,  0.]], dtype=np.float64)
    return ep, np.zeros(3), R, 0.08


def _box():
    """CrackerBox surface: 6 faces of a box that fits fully inside the gripper bbox.
    half_xy=0.03 < hw=0.04, half_z=0.015 < hd=0.02 → all 600 face pts are inside bbox."""
    rng = np.random.default_rng(42)
    half_xy, half_z = 0.03, 0.015
    faces = []
    for ax, half in enumerate([half_xy, half_xy, half_z]):
        for sgn in (-1., 1.):
            f = rng.uniform([-half_xy, -half_xy, -half_z],
                            [ half_xy,  half_xy,  half_z], (100, 3))
            f[:, ax] = sgn * half
            faces.append(f)
    bg = rng.uniform([-0.5, -0.5, 0.], [0.5, 0.5, 0.5], (400, 3))
    ep = np.vstack(faces + [bg])
    return ep, np.zeros(3), np.eye(3), 0.08


def test_local_point_density():
    sc_ep, sc_xyz, sc_R, sc_w = _scissors()
    bx_ep, bx_xyz, bx_R, bx_w = _box()
    sc = local_point_density(sc_xyz, sc_R, sc_w, sc_ep)
    bx = local_point_density(bx_xyz, bx_R, bx_w, bx_ep)
    assert sc < 0.10, f"flat density={sc:.4f} should be <0.10"
    assert bx > 0.55, f"box density={bx:.4f} should be >0.55"
    assert sc < bx,   f"flat={sc:.4f} should be < box={bx:.4f}"
    print(f"  local_point_density  flat={sc:.4f}  box={bx:.4f}  PASS")


def test_normal_consistency():
    sc_ep, sc_xyz, sc_R, sc_w = _scissors()
    bx_ep, bx_xyz, bx_R, bx_w = _box()
    sc = normal_consistency(sc_xyz, sc_R, sc_w, sc_ep)
    bx = normal_consistency(bx_xyz, bx_R, bx_w, bx_ep)
    assert sc < bx, f"flat normal_consistency={sc:.4f} should be < box={bx:.4f}"
    print(f"  normal_consistency    flat={sc:.4f}  box={bx:.4f}  PASS")


def test_contact_width_ratio():
    sc_ep, sc_xyz, sc_R, sc_w = _scissors()
    bx_ep, bx_xyz, bx_R, bx_w = _box()
    sc = contact_width_ratio(sc_xyz, sc_R, sc_w, sc_ep)
    bx = contact_width_ratio(bx_xyz, bx_R, bx_w, bx_ep)
    assert sc < 0.30, f"flat ratio={sc:.4f} should be <0.30"
    assert bx > 0.50, f"box ratio={bx:.4f} should be >0.50"
    assert sc < bx,   f"flat={sc:.4f} should be < box={bx:.4f}"
    print(f"  contact_width_ratio  flat={sc:.4f}  box={bx:.4f}  PASS")


if __name__ == "__main__":
    for fn in [test_local_point_density, test_normal_consistency, test_contact_width_ratio]:
        fn()
    print("All tests passed.")
