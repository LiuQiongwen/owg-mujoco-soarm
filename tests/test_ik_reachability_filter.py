"""Tests for the opt-in hard pre-execution IK-reachability filter added to
benchmark/runner.py 2026-08-05 (Phase C of the jaw-collision investigation:
confirmed the default candidate pipeline had zero collision/reachability
filtering anywhere before physical execution -- see SamplingConfig's
ik_reachability_filter docstring)."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from benchmark.runner import SamplingConfig, _filter_ik_reachable
from tango_robot.env_soarm import EnvironmentSoArm, TABLE_TOP_Z


def test_sampling_config_filter_defaults_to_off():
    sc = SamplingConfig()
    assert sc.ik_reachability_filter is False
    assert sc.ik_reachability_residual_max > 0


def test_filter_empty_candidates_is_noop():
    env = EnvironmentSoArm(vis=False)
    candidates = np.zeros((0, 6), dtype=np.float32)
    out, stats = _filter_ik_reachable(candidates, env, SamplingConfig())
    assert len(out) == 0
    assert stats == {"n_input": 0, "n_ik_rejected": 0, "n_output": 0}


def test_filter_rejects_unreachable_and_keeps_reachable():
    env = EnvironmentSoArm(vis=False)
    sc = SamplingConfig()

    # A candidate well within the arm's normal working area (matches the
    # kind of pose _sample_candidates would actually produce near a
    # tabletop object) should IK-converge and be kept.
    reachable = [0.30, -0.20, TABLE_TOP_Z + 0.10, 0.0, 0.07, 0.05]
    # A candidate far outside any physically reachable region (several
    # metres away) cannot possibly IK-converge and must be rejected.
    unreachable = [5.0, 5.0, TABLE_TOP_Z + 0.10, 0.0, 0.07, 0.05]

    candidates = np.array([reachable, unreachable], dtype=np.float32)
    out, stats = _filter_ik_reachable(candidates, env, sc)

    assert stats["n_input"] == 2
    assert stats["n_ik_rejected"] == 1
    assert stats["n_output"] == 1
    assert len(out) == 1
    # the surviving row must be the reachable one, not the unreachable one
    assert np.allclose(out[0][:2], reachable[:2])


def test_filter_falls_back_to_unfiltered_if_all_rejected():
    env = EnvironmentSoArm(vis=False)
    sc = SamplingConfig()
    # Two candidates, both absurdly far away -- everything gets rejected,
    # so the function must return the ORIGINAL candidates rather than an
    # empty array (an empty pool downstream just produces a confusing
    # generic "all_attempts_failed" instead of the real reason).
    candidates = np.array([
        [5.0, 5.0, TABLE_TOP_Z + 0.10, 0.0, 0.07, 0.05],
        [-5.0, -5.0, TABLE_TOP_Z + 0.10, 0.0, 0.07, 0.05],
    ], dtype=np.float32)
    out, stats = _filter_ik_reachable(candidates, env, sc)
    assert stats["n_input"] == 2
    assert stats["n_ik_rejected"] == 2
    assert stats["n_output"] == 0
    assert len(out) == 2  # fallback: unfiltered candidates returned, not empty


def test_filter_does_not_mutate_real_simulation_state():
    """compute_ik_reachability_per_candidate is documented to run on a
    scratch MjData -- confirm the real env's qpos is unchanged after
    filtering, i.e. this really cannot leak into the actual trial."""
    env = EnvironmentSoArm(vis=False)
    qpos_before = env.data.qpos.copy()
    candidates = np.array([
        [0.30, -0.20, TABLE_TOP_Z + 0.10, 0.0, 0.07, 0.05],
    ], dtype=np.float32)
    _filter_ik_reachable(candidates, env, SamplingConfig())
    assert np.array_equal(env.data.qpos, qpos_before)
