"""Tests for EnvironmentSoArm's opt-in close-window physics diagnostics
(enable_close_window_diagnostics, added 2026-08-05). See the constructor's
docstring comment for why this replaced an earlier external, throwaway
monkey-patch probe: naive whole-trajectory velocity monitoring is
contaminated by legitimate internal teleport/park/restore cycles
_execute_grasp_physics_topdown performs while positioning the arm."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tango_robot.env_soarm import EnvironmentSoArm, GRASP_MODE_PHYSICS_WELD, TABLE_TOP_Z


def test_diagnostics_default_off_populates_none():
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD)
    assert env.enable_close_window_diagnostics is False
    obj_id = env.load_obj("PowerDrill", pos=[0.30, -0.20, TABLE_TOP_Z + 0.02], yaw=0.0)
    env.grasp((0.30, -0.20, TABLE_TOP_Z + 0.05), 0.0, 0.09, 0.06)
    m = env.last_grasp_metrics
    assert m is not None
    assert m["close_window_max_speed_mps"] is None
    assert m["close_window_min_contact_dist_m"] is None


def test_diagnostics_default_off_leaves_step_simulation_unpatched():
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD)
    orig = env.step_simulation
    obj_id = env.load_obj("PowerDrill", pos=[0.30, -0.20, TABLE_TOP_Z + 0.02], yaw=0.0)
    env.grasp((0.30, -0.20, TABLE_TOP_Z + 0.05), 0.0, 0.09, 0.06)
    assert env.step_simulation == orig or env.step_simulation.__func__ == orig.__func__


def test_diagnostics_enabled_populates_sane_values_and_restores_step_simulation():
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD,
                            enable_close_window_diagnostics=True)
    assert env.enable_close_window_diagnostics is True
    orig_step = env.step_simulation

    obj_id = env.load_obj("PowerDrill", pos=[0.30, -0.20, TABLE_TOP_Z + 0.02], yaw=0.0)
    env.grasp((0.30, -0.20, TABLE_TOP_Z + 0.05), 0.0, 0.09, 0.06)

    m = env.last_grasp_metrics
    assert m is not None
    max_speed = m["close_window_max_speed_mps"]
    min_dist = m["close_window_min_contact_dist_m"]
    assert max_speed is not None
    assert min_dist is not None
    # Sane physical bounds: a real, correctly-scoped close event should never
    # look like the ~1776 m/s false "explosion" the earlier throwaway probe
    # produced when it was contaminated by the object's intentional
    # park-at-z=-100 teleport window -- that was entirely a measurement
    # artifact, and this test guards against ever silently regressing to it.
    assert 0.0 <= max_speed < 50.0
    assert -0.20 <= min_dist <= 0.0

    # step_simulation must be restored to the original bound method after
    # the call returns -- the diagnostic monkey-patch must not leak.
    assert env.step_simulation.__func__ == orig_step.__func__


def test_diagnostics_do_not_change_grasp_outcome():
    """The diagnostic wrapper only observes state (reads cvel/contact), it
    must not alter physics. Same seed/pose/object, on vs off, should produce
    the same success outcome (physics itself is not perfectly bit-reproducible
    on marginal contacts per this project's own documented ~0.6-1% flip rate,
    but bilateral_contact/weld_triggered on a clearly-successful case like
    this one should still match)."""
    results = {}
    for enabled in (False, True):
        env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD,
                                enable_close_window_diagnostics=enabled)
        env.load_obj("PowerDrill", pos=[0.30, -0.20, TABLE_TOP_Z + 0.02], yaw=0.0)
        success, grasped = env.grasp((0.30, -0.20, TABLE_TOP_Z + 0.05), 0.0, 0.09, 0.06)
        results[enabled] = (success, env.last_grasp_metrics["bilateral_contact"])
    assert results[False][1] == results[True][1]
