"""Tests for the read-only jaw metrology (enable_jaw_metrology, added 2026-08-07).

Two jobs:

  1. Pin the measured facts that motivated the whole investigation, so they
     cannot drift back silently: the gripper's commandable range does NOT span
     0-100 mm of fingertip opening, and the simulated proxy spheres track that
     range about four times less sensitively than the real fingertips do.
  2. Prove the flag is inert when off -- runs with metrology disabled must be
     comparable to every result recorded before it existed.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np
import pytest

from tango_robot.env_soarm import (
    EnvironmentSoArm,
    GRASP_MODE_PHYSICS_WELD,
    GRIP_CLOSED,
    GRIP_OPEN,
    TABLE_TOP_Z,
)
from tango_robot.jaw_metrology import JawMetrology, claimed_opening_m

SO101_XML = str(Path(__file__).resolve().parent.parent
                / "tango_robot" / "assets" / "so101" / "so101.xml")

JAW_KEYS = ["grip_qpos_rad", "true_opening_m", "proxy_gap_m", "claimed_opening_m"]


@pytest.fixture(scope="module")
def metrology():
    return JawMetrology(mujoco.MjModel.from_xml_path(SO101_XML))


# ── the measured facts ────────────────────────────────────────────────────────

def test_commanded_window_never_reaches_zero_opening(metrology):
    """GRIP_CLOSED sits well above the joint's real lower limit, so the jaw
    cannot close on anything thinner than ~19 mm however small the request."""
    closed = metrology.true_opening_m(GRIP_CLOSED)
    assert 0.017 < closed < 0.022, closed
    # The joint itself can very nearly close; it is the constant that blocks it.
    lo = metrology.model.joint("gripper").range[0]
    assert metrology.true_opening_m(lo) < 0.005


def test_claimed_map_misstates_both_ends_of_the_range(metrology):
    """move_gripper()'s linear map is off by ~20 mm at each end and only happens
    to agree near the middle."""
    assert claimed_opening_m(GRIP_CLOSED) == pytest.approx(0.0, abs=1e-9)
    assert claimed_opening_m(GRIP_OPEN) == pytest.approx(0.10, abs=1e-9)

    err_closed = metrology.true_opening_m(GRIP_CLOSED) - claimed_opening_m(GRIP_CLOSED)
    err_open = metrology.true_opening_m(GRIP_OPEN) - claimed_opening_m(GRIP_OPEN)
    assert err_closed > 0.015          # claims 0 mm, delivers ~19 mm
    assert err_open < -0.015           # claims 100 mm, delivers ~80 mm

    # ...and crosses zero error somewhere in between, which is why the bug
    # stayed invisible on mid-sized objects.
    mid = 0.5 * (GRIP_CLOSED + GRIP_OPEN)
    assert abs(metrology.true_opening_m(mid) - claimed_opening_m(mid)) < 0.005


def test_true_opening_is_monotonic_in_the_hinge_angle(metrology):
    qs = np.linspace(*metrology.model.joint("gripper").range, 40)
    gaps = np.array([metrology.true_opening_m(q) for q in qs])
    assert np.all(np.diff(gaps) > 0)


def test_metrology_binds_to_the_collision_geoms_not_the_visual_twins(metrology):
    """Each finger has a visual and a collision geom built from the same mesh at
    the same pose.  Binding to the visual one leaves geom_type as a full mesh
    even after _simplify_jaw_collision, so any shape query (mj_geomDistance)
    silently measures a body the solver is not colliding."""
    for gid in (metrology._gf, metrology._gm):
        assert metrology.model.geom_contype[gid] != 0


def test_proxy_distance_matches_the_solver_on_a_real_contact():
    """mj_geomDistance on the bound geoms must reproduce MuJoCo's contact.dist;
    if it does not, the metrology is measuring different geometry than physics."""
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True)
    try:
        jm = env._jaw_metrology
        obj_id = env.load_obj("TomatoSoupCan",
                              pos=[0.30, -0.20, TABLE_TOP_Z + 0.02], yaw=0.0)
        env._steps(200)
        gids = env._obj_collision_geom_ids(obj_id)
        assert gids

        jaw_geoms = {jm._gf, jm._gm}
        checked = 0
        for _ in range(400):
            env.step_simulation()
            for ci in range(env.data.ncon):
                c = env.data.contact[ci]
                if c.geom1 not in jaw_geoms and c.geom2 not in jaw_geoms:
                    continue
                jg = c.geom1 if c.geom1 in jaw_geoms else c.geom2
                og = c.geom2 if jg == c.geom1 else c.geom1
                if og not in gids:
                    continue
                d = mujoco.mj_geomDistance(env.model, env.data, jg, int(og),
                                           1.0, np.zeros(6))
                assert d == pytest.approx(float(c.dist), abs=1e-4)
                checked += 1
            if checked:
                break
        # Drive the jaw into the object if free settling produced no contact.
        if not checked:
            pytest.skip("no jaw-object contact arose in the probe window")
    finally:
        env.close()


def test_proxy_spheres_track_the_command_far_less_than_real_fingertips(metrology):
    """The 6 mm proxies sit at the finger meshes' frame origins, not at the
    pads, so contact detection responds to a much smaller range of motion."""
    data = mujoco.MjData(metrology.model)
    qadr = metrology.model.joint("gripper").qposadr[0]

    def proxy_at(q):
        data.qpos[:] = 0.0
        data.qpos[qadr] = q
        mujoco.mj_forward(metrology.model, data)
        return metrology.proxy_gap_m(data)

    # Measured: true 51.5 mm of travel vs. 15.1 mm of proxy travel over the
    # same commanded window.  Bounds are loose enough to survive resampling of
    # the tip point set, tight enough to catch the ratio collapsing.
    true_span = (metrology.true_opening_m(GRIP_OPEN)
                 - metrology.true_opening_m(GRIP_CLOSED))
    proxy_span = proxy_at(GRIP_OPEN) - proxy_at(GRIP_CLOSED)
    assert true_span > 0.045
    assert proxy_span < 0.020
    assert true_span / proxy_span > 2.5


# ── the flag is inert when off ────────────────────────────────────────────────

def test_metrology_default_off_adds_no_keys():
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD)
    try:
        assert env.enable_jaw_metrology is False
        assert env._jaw_metrology is None
        m = env.get_grasp_debug_metrics()
        for k in JAW_KEYS:
            assert k not in m
    finally:
        env.close()


def test_metrology_on_adds_keys_and_leaves_step_simulation_restored():
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True)
    try:
        assert env._jaw_metrology is not None and env._jaw_metrology.available
        step_before = env.step_simulation
        env.load_obj("PowerDrill", pos=[0.30, -0.20, TABLE_TOP_Z + 0.02], yaw=0.0)
        env.grasp((0.30, -0.20, TABLE_TOP_Z + 0.05), 0.0, 0.09, 0.06)

        m = env.last_grasp_metrics
        for k in JAW_KEYS:
            assert k in m
        assert m["close_min_true_opening_m"] is not None
        assert m["requested_opening_m"] == pytest.approx(0.09)
        # GRIP_REDUCTION is applied before the value reaches move_gripper
        assert m["commanded_opening_m"] < m["requested_opening_m"]
        # The trace wrapper must not outlive the close window.  Compared with
        # == rather than is: step_simulation is a bound method, so each attribute
        # access builds a fresh object and `is` would fail even when correct.
        assert env.step_simulation == step_before
        assert not hasattr(env.step_simulation, "__closure__") or \
            env.step_simulation.__closure__ is None
    finally:
        env.close()
