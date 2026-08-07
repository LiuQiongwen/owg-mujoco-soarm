"""Tests for the read-only pad-contact fidelity diagnostic (tango_robot/pad_fidelity.py).

Split in two:
  - classifier/aggregator tests: pure Python, no MuJoCo, cover the four
    correctness properties the task specified.
  - env integration tests: verify the recording hook changes nothing about
    legacy physics/success, only adds an opt-in metrics key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tango_robot.pad_fidelity import (
    GeometricVerdict,
    PadFidelityConfig,
    PadFidelitySample,
    PadFidelityTrial,
    PadState,
    classify_step,
    find_runs,
)

CFG = PadFidelityConfig()   # defaults: contact_tol=1mm, plausible_max=6mm, persistence=8


# ── property 1: positive separation cannot be plausible bilateral ────────────

def test_positive_separation_cannot_be_plausible_bilateral():
    for d in (0.002, 0.01, 0.05, 0.10):
        assert classify_step(d, d, CFG) != PadState.PLAUSIBLE_BILATERAL
        assert classify_step(d, d, CFG) == PadState.NO_BILATERAL


def test_just_inside_contact_tolerance_is_not_clear():
    # exactly at the boundary is inclusive on the "touching" side
    d = CFG.contact_tol_m
    assert classify_step(d, d, CFG) == PadState.PLAUSIBLE_BILATERAL
    # a hair over the boundary is clear, not plausible
    over = CFG.contact_tol_m + 1e-6
    assert classify_step(over, over, CFG) == PadState.NO_BILATERAL


def test_mixed_positive_and_excessive_is_never_plausible():
    """A clear positive side paired with a deeply excessive side must not
    average out into anything resembling bilateral contact."""
    assert classify_step(0.02, -0.02, CFG) != PadState.PLAUSIBLE_BILATERAL


# ── property 2: unilateral near-contact cannot be bilateral ──────────────────

def test_unilateral_near_contact_cannot_be_bilateral():
    # one pad touching, the other clearly clear
    verdict = classify_step(0.0, 0.02, CFG)
    assert verdict != PadState.PLAUSIBLE_BILATERAL
    assert verdict == PadState.AMBIGUOUS


def test_unilateral_touch_both_orders():
    assert classify_step(-0.0005, 0.05, CFG) == PadState.AMBIGUOUS
    assert classify_step(0.05, -0.0005, CFG) == PadState.AMBIGUOUS


def test_both_touching_is_bilateral():
    assert classify_step(-0.001, 0.0005, CFG) == PadState.PLAUSIBLE_BILATERAL


# ── property 3: excessive penetration is never classified as (geometrically)
# successful, i.e. never reported as PLAUSIBLE_ENGAGEMENT, regardless of what
# the legacy success/bilateral labels say and regardless of other samples in
# the same trial also being plausible ─────────────────────────────────────────

def test_single_excessive_side_dominates_the_step():
    assert classify_step(-0.02, 0.0, CFG) == PadState.EXCESSIVE_PENETRATION
    assert classify_step(0.0, -0.02, CFG) == PadState.EXCESSIVE_PENETRATION
    assert classify_step(-0.02, -0.02, CFG) == PadState.EXCESSIVE_PENETRATION


def _make_trial(dist_pairs, **legacy):
    samples = [PadFidelitySample(step=i, pad_obj_dist_fixed_m=f,
                                 pad_obj_dist_moving_m=m)
               for i, (f, m) in enumerate(dist_pairs)]
    return PadFidelityTrial(samples=samples, cfg=CFG, **legacy)


def test_excessive_penetration_dominant_even_with_a_plausible_run_present():
    # 10 plausible-bilateral steps, THEN 10 excessive steps -- both runs are
    # individually persistent (>= persistence_steps=8)
    pairs = [(0.0, 0.0)] * 10 + [(-0.05, -0.05)] * 10
    trial = _make_trial(pairs, final_success=True, final_bilateral_contact=True)
    assert trial.geometric_verdict() == GeometricVerdict.EXCESSIVE_PENETRATION_DOMINANT
    assert trial.geometric_verdict() != GeometricVerdict.PLAUSIBLE_ENGAGEMENT


def test_excessive_penetration_dominant_regardless_of_legacy_labels():
    """The verdict must be identical whether or not the legacy fields claim
    success -- it is computed purely from geometry."""
    pairs = [(-0.05, -0.05)] * 10
    trial_a = _make_trial(pairs, final_success=True, final_bilateral_contact=True,
                          final_weld_triggered=True, final_lifted=True)
    trial_b = _make_trial(pairs, final_success=False, final_bilateral_contact=False)
    assert trial_a.geometric_verdict() == GeometricVerdict.EXCESSIVE_PENETRATION_DOMINANT
    assert trial_b.geometric_verdict() == GeometricVerdict.EXCESSIVE_PENETRATION_DOMINANT


def test_brief_excessive_penetration_below_persistence_does_not_dominate():
    # only 3 excessive steps (< persistence_steps=8): noise, not sustained
    pairs = [(0.0, 0.0)] * 10 + [(-0.05, -0.05)] * 3 + [(0.0, 0.0)] * 10
    trial = _make_trial(pairs)
    assert trial.geometric_verdict() == GeometricVerdict.PLAUSIBLE_ENGAGEMENT


def test_geometric_verdict_never_reads_legacy_fields():
    """Construct two trials with IDENTICAL geometry but opposite legacy
    labels; verdicts must match, proving the computation path never
    references final_success/final_bilateral_contact/etc."""
    pairs = [(-0.05, -0.05)] * 10
    optimistic = _make_trial(pairs, final_success=True, final_bilateral_contact=True,
                             final_weld_triggered=True, final_lifted=True,
                             final_retained=True)
    pessimistic = _make_trial(pairs, final_success=False, final_bilateral_contact=False,
                              final_weld_triggered=False, final_lifted=False,
                              final_retained=False)
    assert optimistic.geometric_verdict() == pessimistic.geometric_verdict()


# ── persistence / aggregation correctness ─────────────────────────────────────

def test_find_runs_requires_minimum_length():
    states = [PadState.NO_BILATERAL] * 3 + [PadState.PLAUSIBLE_BILATERAL] * 8 + [PadState.NO_BILATERAL] * 2
    runs = find_runs(states, min_len=8)
    assert runs[PadState.PLAUSIBLE_BILATERAL] == [(3, 11)]
    assert runs[PadState.NO_BILATERAL] == []   # both runs of len 3,2 < 8


def test_thresholds_are_configurable_without_editing_the_module():
    strict_cfg = PadFidelityConfig(plausible_penetration_max_m=0.001)
    # 5mm penetration: plausible under the default 6mm threshold...
    assert classify_step(-0.005, -0.005, CFG) == PadState.PLAUSIBLE_BILATERAL
    # ...but excessive under a caller-supplied 1mm threshold
    assert classify_step(-0.005, -0.005, strict_cfg) == PadState.EXCESSIVE_PENETRATION


def test_summary_reports_distances_and_durations_without_touching_legacy_labels():
    pairs = [(0.02, 0.02)] * 5 + [(0.0, 0.0)] * 10 + [(-0.05, -0.05)] * 8
    trial = _make_trial(pairs, object_name="TestObj", seed=7,
                        final_success=True, final_bilateral_contact=True)
    s = trial.summary()
    assert s["object"] == "TestObj" and s["seed"] == 7
    assert s["min_pad_dist_fixed_m"] == pytest.approx(-0.05)
    assert s["final_pad_dist_fixed_m"] == pytest.approx(-0.05)
    assert s["bilateral_engagement_samples"] == 10
    assert s["excessive_penetration_samples"] == 8
    assert s["geometric_verdict"] == GeometricVerdict.EXCESSIVE_PENETRATION_DOMINANT.value
    # the legacy fields are REPORTED, not overwritten or consulted
    assert s["legacy_success"] is True
    assert s["legacy_bilateral_contact"] is True


def test_confusion_row_pairs_legacy_and_geometric_per_step():
    samples = [
        PadFidelitySample(step=0, pad_obj_dist_fixed_m=0.02, pad_obj_dist_moving_m=0.02,
                          bilateral_contact=False),   # NO_BILATERAL, legacy agrees
        PadFidelitySample(step=1, pad_obj_dist_fixed_m=-0.05, pad_obj_dist_moving_m=-0.05,
                          bilateral_contact=True),    # EXCESSIVE_PENETRATION, legacy says bilateral
    ]
    trial = PadFidelityTrial(samples=samples, cfg=CFG)
    row = trial.confusion_row()
    assert row[(False, "NO_BILATERAL")] == 1
    assert row[(True, "EXCESSIVE_PENETRATION")] == 1


def test_missing_distances_are_ambiguous_not_silently_dropped():
    assert classify_step(None, 0.0, CFG) == PadState.AMBIGUOUS
    assert classify_step(None, None, CFG) == PadState.AMBIGUOUS


def test_empty_trial_has_a_defined_verdict():
    trial = PadFidelityTrial(samples=[], cfg=CFG)
    assert trial.geometric_verdict() == GeometricVerdict.AMBIGUOUS
    s = trial.summary()
    assert s["min_pad_dist_fixed_m"] is None
    assert s["n_samples"] == 0


# ── property 4: the diagnostic is read-only; legacy rollouts stay unaffected ─
#
# These need MuJoCo, unlike everything above. Import lazily so the pure tests
# above still run (and stay fast) even in an environment without it.

import os  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from tango_robot.env_soarm import (  # noqa: E402
    EnvironmentSoArm,
    GRASP_MODE_PHYSICS_WELD,
    JAW_CONTACT_MEASURED_PADS_AIMED,
    JAW_CONTACT_PROXY_SPHERES,
    TABLE_TOP_Z,
)


def _spawn(seed):
    rng = np.random.default_rng(seed)
    return [float(rng.uniform(-0.06, 0.06)),
            -0.40 + float(rng.uniform(-0.04, 0.04)),
            TABLE_TOP_Z + 0.12]


def test_flag_defaults_off_and_is_the_default_constructor_value():
    env = EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD)
    try:
        assert env.enable_pad_fidelity_trace is False
        assert env._pad_fidelity_trial is None
    finally:
        env.close()


def test_flag_rejects_proxy_spheres_jaw_contact_model():
    with pytest.raises(ValueError, match="measured_pads"):
        EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD,
                         jaw_contact_model=JAW_CONTACT_PROXY_SPHERES,
                         enable_jaw_metrology=True,
                         enable_pad_fidelity_trace=True)


def test_flag_requires_jaw_metrology():
    with pytest.raises(ValueError, match="enable_jaw_metrology"):
        EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD,
                         jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED,
                         enable_jaw_metrology=False,
                         enable_pad_fidelity_trace=True)


def test_default_off_leaves_metrics_key_set_untouched():
    """No caller that doesn't opt in should ever see a pad_fidelity_summary
    key -- adding it unconditionally would change last_grasp_metrics's shape
    for every existing caller of this codebase."""
    env = EnvironmentSoArm(obj_names=["HammerC"], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED,
                           enable_jaw_metrology=True,
                           enable_pad_fidelity_trace=False)
    try:
        env.reset_robot(); env.remove_all_obj()
        oid = env.load_obj("HammerC", name="HammerC", pos=_spawn(3))
        env._steps(240)
        p = env.get_obj_pos(oid).copy()
        env._execute_grasp(pos=(float(p[0]), float(p[1]), float(p[2])), roll=0.0,
                           gripper_opening_length=0.065,
                           obj_height=float(p[2] - TABLE_TOP_Z))
        assert "pad_fidelity_summary" not in env.last_grasp_metrics
        assert env._pad_fidelity_trial is None
    finally:
        env.close()


def _run_grasp(env, obj_key, seed):
    env.reset_robot(); env.remove_all_obj()
    oid = env.load_obj(obj_key, name=obj_key, pos=_spawn(seed))
    env._steps(240)
    p = env.get_obj_pos(oid).copy()
    ok, _ = env._execute_grasp(pos=(float(p[0]), float(p[1]), float(p[2])), roll=0.0,
                               gripper_opening_length=0.065,
                               obj_height=float(p[2] - TABLE_TOP_Z))
    return ok, env.data.qpos.copy(), dict(env.last_grasp_metrics or {})


def test_recording_does_not_change_physics_or_outcome():
    """Same scene, same seed, deterministic MuJoCo: with the trace on, the
    resulting qpos trajectory and grasp outcome must be identical to it being
    off. If they diverge, the recording hook is doing more than reading."""
    kwargs = dict(obj_names=["HammerC"], vis=False,
                 grasp_mode=GRASP_MODE_PHYSICS_WELD,
                 jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED,
                 enable_jaw_metrology=True)
    env_off = EnvironmentSoArm(enable_pad_fidelity_trace=False, **kwargs)
    env_on = EnvironmentSoArm(enable_pad_fidelity_trace=True, **kwargs)
    try:
        ok_off, q_off, m_off = _run_grasp(env_off, "HammerC", 5)
        ok_on, q_on, m_on = _run_grasp(env_on, "HammerC", 5)
        assert ok_off == ok_on
        assert np.array_equal(q_off, q_on)
        for k in ("bilateral_contact", "weld_triggered", "lifted", "success",
                 "final_z"):
            assert m_off.get(k) == m_on.get(k)
        assert "pad_fidelity_summary" not in m_off
        assert "pad_fidelity_summary" in m_on
    finally:
        env_off.close()
        env_on.close()


def test_step_wrapper_is_restored_after_the_close_window():
    env = EnvironmentSoArm(obj_names=["HammerC"], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED,
                           enable_jaw_metrology=True,
                           enable_pad_fidelity_trace=True)
    try:
        step_before = env.step_simulation
        _run_grasp(env, "HammerC", 2)
        assert env.step_simulation == step_before
    finally:
        env.close()


def test_summary_populates_with_real_geometry_and_reports_legacy_alongside():
    env = EnvironmentSoArm(obj_names=["HammerC"], vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           jaw_contact_model=JAW_CONTACT_MEASURED_PADS_AIMED,
                           enable_jaw_metrology=True,
                           enable_pad_fidelity_trace=True)
    try:
        ok, _, m = _run_grasp(env, "HammerC", 1)
        s = m["pad_fidelity_summary"]
        assert s["n_samples"] > 0
        assert s["geometric_verdict"] in {v.value for v in GeometricVerdict}
        # legacy fields are REPORTED alongside, and must match what the
        # legacy pipeline itself decided -- this diagnostic never overwrites them
        assert s["legacy_success"] == bool(ok)
        assert s["legacy_bilateral_contact"] == bool(m["bilateral_contact"])
    finally:
        env.close()
