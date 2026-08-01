"""MVP4 artifact-policy tests: research_agent.policies.artifact_policy.

Covers (see the MVP4 task contract's "Fake execution tests" list):
  17. artifact created in allowed directory
  18. artifact path escape
  19. artifact symlink escape
  20. too many artifact files
  21. artifact byte limit exceeded
  44. no research dataset/output modified (structural: artifacts dir is
      always a fresh run-scoped directory, never data/datasets/results/...)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.models import ExperimentSpec
from research_agent.policies import artifact_policy as ap


def _spec(**limits_overrides) -> ExperimentSpec:
    execution = {
        "approved_commands": [[sys.executable, "-c", "pass"]],
        "limits": {"max_artifact_files": 3, "max_artifact_bytes": 1000, "max_artifact_file_bytes": 500, **limits_overrides},
    }
    return ExperimentSpec.model_validate({
        "task_id": "artifactpolicy_test", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": ["python", "-c", "pass"], "execution": execution,
    })


@pytest.fixture()
def artifacts_dir(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


def test_clean_artifact_created_in_allowed_directory(artifacts_dir):
    (artifacts_dir / "metrics.json").write_text(json.dumps({"ok": True}))
    spec = _spec()
    records, violations = ap.scan_artifacts(artifacts_dir, spec)
    assert violations == []
    assert any(r.relative_path == "metrics.json" and r.artifact_type == "file" for r in records)
    assert records[0].sha256 is not None


def test_symlink_escape_detected(artifacts_dir):
    os.symlink("/etc/passwd", artifacts_dir / "escape")
    spec = _spec()
    records, violations = ap.scan_artifacts(artifacts_dir, spec)
    assert any("escape" in v.lower() for v in violations)


def test_nested_symlink_directory_escape_detected(artifacts_dir):
    outside = artifacts_dir.parent / "outside_dir"
    outside.mkdir()
    (outside / "leaked.txt").write_text("secret")
    os.symlink(outside, artifacts_dir / "linked_dir")
    spec = _spec()
    records, violations = ap.scan_artifacts(artifacts_dir, spec)
    assert any("escape" in v.lower() for v in violations)


def test_too_many_artifact_files_rejected(artifacts_dir):
    for i in range(5):
        (artifacts_dir / f"f{i}.json").write_text("{}")
    spec = _spec(max_artifact_files=3)
    records, violations = ap.scan_artifacts(artifacts_dir, spec)
    assert any("max_artifact_files" in v for v in violations)


def test_total_artifact_byte_limit_exceeded(artifacts_dir):
    (artifacts_dir / "big.json").write_text("0" * 2000)
    spec = _spec(max_artifact_bytes=1000, max_artifact_file_bytes=5000)
    records, violations = ap.scan_artifacts(artifacts_dir, spec)
    assert any("max_artifact_bytes" in v for v in violations)


def test_per_file_byte_limit_exceeded(artifacts_dir):
    (artifacts_dir / "big.json").write_text("0" * 2000)
    spec = _spec(max_artifact_bytes=100000, max_artifact_file_bytes=500)
    records, violations = ap.scan_artifacts(artifacts_dir, spec)
    assert any("max_artifact_file_bytes" in v for v in violations)


def test_allowed_output_paths_enforced(artifacts_dir):
    (artifacts_dir / "metrics.json").write_text("{}")
    (artifacts_dir / "unexpected.txt").write_text("x")
    execution = {
        "approved_commands": [[sys.executable, "-c", "pass"]],
        "allowed_output_paths": ["metrics.json"],
    }
    spec = ExperimentSpec.model_validate({
        "task_id": "artifactpolicy_allowedpaths", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": ["python", "-c", "pass"], "execution": execution,
    })
    records, violations = ap.scan_artifacts(artifacts_dir, spec)
    assert any("unexpected.txt" in v and "allowed_output_paths" in v for v in violations)


def test_nested_git_repo_rejected(artifacts_dir):
    (artifacts_dir / "sub" / ".git").mkdir(parents=True)
    spec = _spec()
    records, violations = ap.scan_artifacts(artifacts_dir, spec)
    assert any("nested Git" in v for v in violations)


def test_fifo_rejected(artifacts_dir):
    fifo_path = artifacts_dir / "a_fifo"
    os.mkfifo(fifo_path)
    spec = _spec()
    records, violations = ap.scan_artifacts(artifacts_dir, spec)
    assert any("FIFO" in v for v in violations)


def test_missing_artifacts_dir_is_empty_not_an_error(tmp_path):
    spec = _spec()
    records, violations = ap.scan_artifacts(tmp_path / "does_not_exist", spec)
    assert records == []
    assert violations == []
