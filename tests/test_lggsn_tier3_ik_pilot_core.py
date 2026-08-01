# -*- coding: utf-8 -*-
"""Unit tests for research_agent_pilots/lggsn_tier3_ik/pilot_core.py -- the
pure-stdlib validation/aggregation logic behind the LGGSN Tier-3 pilot
(experiments/lggsn_tier3_ik_pilot.yaml). No mujoco/numpy/tango_robot
dependency, so this runs under the research-agent venv exactly like the
other 292 research-agent tests.

Run: PREFECT_LOGGING_LEVEL=WARNING python -m pytest -q tests/test_lggsn_tier3_ik_pilot_core.py
"""
import hashlib
import json
import math
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PILOT_DIR = os.path.join(_REPO_ROOT, "research_agent_pilots", "lggsn_tier3_ik")
sys.path.insert(0, _PILOT_DIR)

import pilot_core  # noqa: E402

FIXTURE_PATH = os.path.join(_PILOT_DIR, "fixtures", "candidate_poses.json")

# Pinned to the same values as experiments/lggsn_tier3_ik_pilot.yaml's
# required_metrics -- both are checked against the one real fixture file, so
# a drift between them (e.g. someone edits the fixture without updating the
# spec) is caught here rather than only surfacing as a live MVP4 FAIL.
EXPECTED_FIXTURE_SHA256 = "1f6777ca28bd3756b5e18a8e1bfb9b8f96d02c4f6945c8602a6062b94de2e2c8"


def _good_result(converged=True, residual=0.001, max_joint_delta=1.5):
    return {"ik_converged": converged, "ik_residual": residual, "max_joint_delta": max_joint_delta}


# ── load_fixture ─────────────────────────────────────────────────────────────

def test_load_fixture_reads_the_committed_fixture():
    candidates, digest = pilot_core.load_fixture(FIXTURE_PATH)
    assert 3 <= len(candidates) <= 10
    for cand in candidates:
        assert set(cand) == {"x", "y", "z", "yaw"}
        assert all(isinstance(v, float) for v in cand.values())


def test_load_fixture_digest_matches_raw_file_bytes():
    with open(FIXTURE_PATH, "rb") as f:
        raw = f.read()
    expected = hashlib.sha256(raw).hexdigest()
    _, digest = pilot_core.load_fixture(FIXTURE_PATH)
    assert digest == expected
    # Cross-check against the value pinned in experiments/lggsn_tier3_ik_pilot.yaml
    assert digest == EXPECTED_FIXTURE_SHA256


def test_load_fixture_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad_fixture.json"
    bad.write_text(json.dumps({"candidates": [{"x": 0.0, "y": 0.0, "z": 0.8}]}))  # missing yaw
    with pytest.raises(pilot_core.FixtureError):
        pilot_core.load_fixture(str(bad))


def test_load_fixture_rejects_too_few_candidates(tmp_path):
    bad = tmp_path / "bad_fixture.json"
    bad.write_text(json.dumps({"candidates": [
        {"x": 0.0, "y": -0.4, "z": 0.8, "yaw": 0.0},
        {"x": 0.01, "y": -0.4, "z": 0.8, "yaw": 0.0},
    ]}))  # only 2, below the required 3-10
    with pytest.raises(pilot_core.FixtureError):
        pilot_core.load_fixture(str(bad))


def test_load_fixture_rejects_too_many_candidates(tmp_path):
    bad = tmp_path / "bad_fixture.json"
    cands = [{"x": 0.0, "y": -0.4, "z": 0.8, "yaw": 0.0} for _ in range(11)]
    bad.write_text(json.dumps({"candidates": cands}))
    with pytest.raises(pilot_core.FixtureError):
        pilot_core.load_fixture(str(bad))


def test_load_fixture_rejects_empty_candidates_list(tmp_path):
    bad = tmp_path / "bad_fixture.json"
    bad.write_text(json.dumps({"candidates": []}))
    with pytest.raises(pilot_core.FixtureError):
        pilot_core.load_fixture(str(bad))


# ── build_candidate_features ─────────────────────────────────────────────────

def test_build_candidate_features_merges_pose_and_result():
    candidates = [{"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.5}]
    results = [_good_result()]
    features = pilot_core.build_candidate_features(candidates, results)
    assert len(features) == 1
    f = features[0]
    assert f["x"] == 1.0 and f["y"] == 2.0 and f["z"] == 3.0 and f["yaw"] == 0.5
    assert f["ik_converged"] is True
    assert f["ik_residual"] == 0.001
    assert f["max_joint_delta"] == 1.5
    assert f["candidate_index"] == 0


def test_build_candidate_features_rejects_missing_candidate():
    """A dropped candidate (fewer raw results than fixture poses) must be a
    hard error, never silently truncated or padded."""
    candidates = [{"x": 0.0, "y": 0.0, "z": 0.8, "yaw": 0.0}] * 3
    results = [_good_result(), _good_result()]  # one short
    with pytest.raises(pilot_core.CandidateCountMismatchError):
        pilot_core.build_candidate_features(candidates, results)


def test_build_candidate_features_rejects_extra_result():
    candidates = [{"x": 0.0, "y": 0.0, "z": 0.8, "yaw": 0.0}] * 2
    results = [_good_result(), _good_result(), _good_result()]  # one extra
    with pytest.raises(pilot_core.CandidateCountMismatchError):
        pilot_core.build_candidate_features(candidates, results)


def test_build_candidate_features_rejects_malformed_result():
    candidates = [{"x": 0.0, "y": 0.0, "z": 0.8, "yaw": 0.0}]
    results = [{"ik_converged": True}]  # missing ik_residual/max_joint_delta
    with pytest.raises(pilot_core.FixtureError):
        pilot_core.build_candidate_features(candidates, results)


# ── compute_metrics ───────────────────────────────────────────────────────────

def _features_from(results, n_pos=3):
    candidates = [{"x": float(i), "y": 0.0, "z": 0.8, "yaw": 0.0} for i in range(n_pos)]
    return pilot_core.build_candidate_features(candidates, results)


def test_compute_metrics_schema_and_types():
    results = [_good_result(True, 0.001, 1.0), _good_result(False, 0.02, 2.0), _good_result(True, 0.003, 1.2)]
    features = _features_from(results)
    metrics = pilot_core.compute_metrics(features, "deadbeef")

    assert isinstance(metrics["candidate_count"], int) and metrics["candidate_count"] == 3
    assert isinstance(metrics["converged_count"], int) and metrics["converged_count"] == 2
    assert isinstance(metrics["convergence_rate"], float)
    assert 0.0 <= metrics["convergence_rate"] <= 1.0
    assert isinstance(metrics["all_residuals_finite"], bool) and metrics["all_residuals_finite"] is True
    assert isinstance(metrics["all_joint_deltas_finite"], bool) and metrics["all_joint_deltas_finite"] is True
    assert isinstance(metrics["max_residual"], float)
    assert isinstance(metrics["max_joint_delta"], float)
    assert isinstance(metrics["deterministic_digest"], str) and len(metrics["deterministic_digest"]) == 64
    assert isinstance(metrics["pilot_ok"], bool)
    assert metrics["fixture_sha256"] == "deadbeef"
    assert metrics["pilot_ok"] is True
    assert metrics["max_residual"] == pytest.approx(0.02)


def test_compute_metrics_convergence_rate_in_unit_interval_always():
    for converged_flags in ([True], [False], [True, False], [False, False, False]):
        results = [_good_result(c) for c in converged_flags]
        features = _features_from(results, n_pos=len(converged_flags))
        metrics = pilot_core.compute_metrics(features, "deadbeef")
        assert 0.0 <= metrics["convergence_rate"] <= 1.0


def test_compute_metrics_rejects_non_finite_without_crashing():
    """A non-finite residual/joint-delta must be faithfully reported, not
    silently dropped and not an uncaught exception."""
    results = [
        _good_result(True, 0.001, 1.0),
        _good_result(True, float("nan"), 2.0),
        _good_result(True, 0.002, float("inf")),
    ]
    features = _features_from(results)
    metrics = pilot_core.compute_metrics(features, "deadbeef")  # must not raise

    assert metrics["candidate_count"] == 3  # nothing was dropped
    assert metrics["all_residuals_finite"] is False
    assert metrics["all_joint_deltas_finite"] is False
    assert metrics["pilot_ok"] is False
    # max_* aggregates must themselves stay finite (computed only over the
    # finite subset), so a verifier reading them never trips over NaN/Inf.
    assert math.isfinite(metrics["max_residual"])
    assert math.isfinite(metrics["max_joint_delta"])


def test_compute_metrics_all_finite_when_all_inputs_finite():
    results = [_good_result(True, 0.001, 1.0), _good_result(False, 0.05, 2.5)]
    features = _features_from(results, n_pos=2)
    metrics = pilot_core.compute_metrics(features, "deadbeef")
    assert metrics["all_residuals_finite"] is True
    assert metrics["all_joint_deltas_finite"] is True


# ── determinism ───────────────────────────────────────────────────────────────

def test_deterministic_digest_repeatable_for_same_input():
    results = [_good_result(True, 0.001, 1.0), _good_result(False, 0.05, 2.5)]
    features = _features_from(results, n_pos=2)
    d1 = pilot_core.deterministic_digest(features, "deadbeef")
    d2 = pilot_core.deterministic_digest(features, "deadbeef")
    assert d1 == d2
    m1 = pilot_core.compute_metrics(features, "deadbeef")
    m2 = pilot_core.compute_metrics(features, "deadbeef")
    assert m1["deterministic_digest"] == m2["deterministic_digest"] == d1


def test_deterministic_digest_changes_with_different_fixture_digest():
    results = [_good_result()]
    features = _features_from(results, n_pos=1)
    d1 = pilot_core.deterministic_digest(features, "aaaa")
    d2 = pilot_core.deterministic_digest(features, "bbbb")
    assert d1 != d2


def test_deterministic_digest_changes_with_different_candidate_features():
    features_a = _features_from([_good_result(True, 0.001, 1.0)], n_pos=1)
    features_b = _features_from([_good_result(False, 0.02, 1.0)], n_pos=1)
    assert pilot_core.deterministic_digest(features_a, "deadbeef") != \
        pilot_core.deterministic_digest(features_b, "deadbeef")
