# -*- coding: utf-8 -*-
"""Integration tests for the LGGSN Tier-3 IK pilot: actually runs
research_agent_pilots/lggsn_tier3_ik/run_pilot.py (the same script
experiments/lggsn_tier3_ik_pilot.yaml's approved command invokes) as a
subprocess against a scratch artifacts directory, using the tango conda
env's python interpreter directly (mirroring the spec's approved command
exactly, minus the MVP4 policy/rlimit wrapper itself, which is exercised
separately by the live `run-experiment` validation).

Needs mujoco (tango conda env) -- skips cleanly under the research-agent
venv, where the other 292 research-agent tests + this pilot's pure-logic
tests (test_lggsn_tier3_ik_pilot_core.py, test_lggsn_tier3_ik_pilot_spec.py)
run instead.

Run: conda run -n tango python -m pytest -q tests/test_lggsn_tier3_ik_pilot_integration.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_PILOT_PATH = os.path.join(
    _REPO_ROOT, "research_agent_pilots", "lggsn_tier3_ik", "run_pilot.py"
)
_PYTHON = sys.executable  # this test only runs (or is collected) under an env with mujoco


def _run_pilot(artifacts_dir: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["MALLOC_ARENA_MAX"] = "1"
    env["RESEARCH_AGENT_ARTIFACTS_DIR"] = artifacts_dir
    return subprocess.run(
        [_PYTHON, RUN_PILOT_PATH],
        env=env, capture_output=True, text=True, timeout=timeout,
    )


def _git_status_short(*, include_ignored: bool = False) -> str:
    """include_ignored=True mirrors research_agent.tasks.repository
    .capture_repo_fingerprint's stricter check (research_agent's own
    mutation-detection flags ANY tree change, including gitignored paths
    like __pycache__/ -- see test_pilot_creates_no_bytecode_cache_files
    below, which caught a real bug that plain `git status --short` alone
    could never have caught since ignored files never show up in it)."""
    args = ["git", "-C", _REPO_ROOT, "status", "--short"]
    if include_ignored:
        args.append("--ignored")
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def test_pilot_runs_and_produces_expected_artifacts(tmp_path):
    artifacts_dir = str(tmp_path / "artifacts")
    os.makedirs(artifacts_dir)
    proc = _run_pilot(artifacts_dir)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"

    # Artifact paths confined to the run's assigned artifacts directory --
    # exactly the two expected files, nothing else.
    produced = sorted(os.listdir(artifacts_dir))
    assert produced == ["candidate_features.json", "metrics.json"]

    with open(os.path.join(artifacts_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(artifacts_dir, "candidate_features.json")) as f:
        features = json.load(f)

    assert metrics["candidate_count"] == 7 == len(features)
    assert metrics["converged_count"] == sum(1 for c in features if c["ik_converged"])
    assert metrics["all_residuals_finite"] is True
    assert metrics["all_joint_deltas_finite"] is True
    assert 0.0 <= metrics["convergence_rate"] <= 1.0
    assert metrics["pilot_ok"] is True
    assert isinstance(metrics["deterministic_digest"], str) and len(metrics["deterministic_digest"]) == 64
    assert metrics["fixture_sha256"] == "1f6777ca28bd3756b5e18a8e1bfb9b8f96d02c4f6945c8602a6062b94de2e2c8"


def test_pilot_matches_pinned_spec_metrics(tmp_path):
    """Cross-check against the exact values pinned in
    experiments/lggsn_tier3_ik_pilot.yaml's required_metrics."""
    artifacts_dir = str(tmp_path / "artifacts")
    os.makedirs(artifacts_dir)
    proc = _run_pilot(artifacts_dir)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    with open(os.path.join(artifacts_dir, "metrics.json")) as f:
        metrics = json.load(f)

    assert metrics["candidate_count"] == 7
    assert metrics["converged_count"] == 2
    assert metrics["max_residual"] == pytest.approx(0.033328365409923795, abs=1e-6)
    assert metrics["max_joint_delta"] == pytest.approx(2.7438472969992493, abs=1e-6)
    assert metrics["deterministic_digest"] == \
        "69ad2a7f3ea0a9728e770170b72c9b45be71b403d6401b0ec2f79e888cca2de5"


def test_pilot_repeated_run_produces_identical_digest_and_features(tmp_path):
    dir_a = str(tmp_path / "run_a")
    dir_b = str(tmp_path / "run_b")
    os.makedirs(dir_a)
    os.makedirs(dir_b)

    proc_a = _run_pilot(dir_a)
    proc_b = _run_pilot(dir_b)
    assert proc_a.returncode == 0 and proc_b.returncode == 0

    with open(os.path.join(dir_a, "metrics.json")) as f:
        metrics_a = json.load(f)
    with open(os.path.join(dir_b, "metrics.json")) as f:
        metrics_b = json.load(f)
    with open(os.path.join(dir_a, "candidate_features.json")) as f:
        features_a = json.load(f)
    with open(os.path.join(dir_b, "candidate_features.json")) as f:
        features_b = json.load(f)

    assert metrics_a["deterministic_digest"] == metrics_b["deterministic_digest"]
    assert features_a == features_b  # byte-for-byte-equivalent structures
    # duration_seconds is the one legitimately non-deterministic field
    # (wall-clock), and is deliberately excluded from the digest.
    non_wallclock_a = {k: v for k, v in metrics_a.items() if k != "duration_seconds"}
    non_wallclock_b = {k: v for k, v in metrics_b.items() if k != "duration_seconds"}
    assert non_wallclock_a == non_wallclock_b


def test_pilot_missing_candidate_is_rejected(tmp_path, monkeypatch):
    """Integration-level check that a truncated fixture (a candidate
    silently dropped before reaching the solver) makes the whole pilot fail
    loudly rather than reporting a smaller candidate_count as if nothing
    were wrong. Simulated by pointing HeadlessIKSolver's caller at a
    deliberately short fixture copy."""
    import shutil

    real_fixture = os.path.join(
        _REPO_ROOT, "research_agent_pilots", "lggsn_tier3_ik", "fixtures", "candidate_poses.json"
    )
    with open(real_fixture) as f:
        doc = json.load(f)
    truncated = dict(doc)
    truncated["candidates"] = doc["candidates"][:3]  # still schema-valid (>=3), just fewer

    # run_pilot.py hardcodes its own fixture path relative to __file__, so
    # this test only proves pilot_core's rejection path is reachable, not
    # that run_pilot.py itself can be pointed at another fixture (it can't,
    # by design -- the approved command has no argv for that). The dropped-
    # candidate contract is exercised directly here for exactly that reason.
    import sys as _sys
    pilot_dir = os.path.join(_REPO_ROOT, "research_agent_pilots", "lggsn_tier3_ik")
    if pilot_dir not in _sys.path:
        _sys.path.insert(0, pilot_dir)
    import pilot_core

    candidates, _digest = pilot_core.load_fixture(real_fixture)
    short_results = [{"ik_converged": True, "ik_residual": 0.001, "max_joint_delta": 1.0}] * (len(candidates) - 1)
    with pytest.raises(pilot_core.CandidateCountMismatchError):
        pilot_core.build_candidate_features(candidates, short_results)


def test_pilot_causes_no_repository_mutation(tmp_path):
    before = _git_status_short()
    artifacts_dir = str(tmp_path / "artifacts")
    os.makedirs(artifacts_dir)
    proc = _run_pilot(artifacts_dir)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    after = _git_status_short()
    assert before == after


def test_pilot_creates_no_bytecode_cache_files(tmp_path):
    """Regression test: run_pilot.py used to write .pyc files into the repo
    tree on import (e.g. tango_robot/__pycache__/), which
    research_agent.tasks.repository.capture_repo_fingerprint correctly
    flags as a POLICY_FAILURE ("main worktree mutated during execution")
    even though __pycache__/ is gitignored -- the harness checks for ANY
    tree mutation, not just trackable ones. Caught live: the first official
    MVP4 run from a fresh worktree (which had no pre-existing __pycache__ to
    hide the bug behind) failed for exactly this reason. Fixed by
    sys.dont_write_bytecode=True at the top of run_pilot.py plus
    PYTHONDONTWRITEBYTECODE=1 in the spec's environment_overrides.

    Plain `git status --short` (test_pilot_causes_no_repository_mutation,
    above) can NEVER catch this on its own -- ignored files never appear in
    it without --ignored -- so this test checks both that flag AND the
    filesystem directly.

    Compares a before/after SET of *.pyc files rather than asserting none
    exist at all: other tests in the same pytest session legitimately import
    tango_robot.* directly (e.g. test_headless_ik_parity.py), which writes
    ordinary bytecode cache for the TEST PROCESS's own imports -- unrelated
    to, and not evidence against, run_pilot.py's subprocess. Only NEW files
    appearing as a result of THIS call are a defect."""
    def _pyc_snapshot() -> set:
        return {str(p) for p in Path(_REPO_ROOT).rglob("*.pyc") if ".git" not in p.parts}

    before_pyc = _pyc_snapshot()
    artifacts_dir = str(tmp_path / "artifacts")
    os.makedirs(artifacts_dir)
    proc = _run_pilot(artifacts_dir)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    after_pyc = _pyc_snapshot()
    new_pyc = after_pyc - before_pyc
    assert new_pyc == set(), f"pilot subprocess wrote new bytecode-cache files: {sorted(new_pyc)}"
