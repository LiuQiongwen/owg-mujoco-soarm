"""MVP4 deterministic metric-verifier tests: research_agent.tasks.metric_verifier.

Covers (see the MVP4 task contract's "Fake execution tests" list):
  22. malformed metrics JSON
  23. missing metric
  24. wrong metric type
  25. float tolerance pass/fail
  26. required artifact missing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.models import ExperimentSpec
from research_agent.tasks.metric_verifier import run_execution_verifier


def _spec(required_metrics, required_artifacts=None) -> ExperimentSpec:
    """required_artifacts=None means 'use the default (["metrics.json"])';
    pass an explicit [] to test the 'no required artifacts at all' case."""
    if required_artifacts is None:
        required_artifacts = ["metrics.json"]
    execution = {
        "approved_commands": [[sys.executable, "-c", "pass"]],
        "required_metrics": required_metrics,
        "required_artifacts": required_artifacts,
    }
    return ExperimentSpec.model_validate({
        "task_id": "metricverifier_test", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": ["python", "-c", "pass"], "execution": execution,
    })


def _verify(tmp_path, metrics_content, required_metrics, required_artifacts=None, *, write=True):
    if write:
        (tmp_path / "metrics.json").write_text(metrics_content)
    spec = _spec(required_metrics, required_artifacts)
    return run_execution_verifier(spec=spec, artifacts_dir=tmp_path, task_id="t", run_id="r", attempt_index=0)


def test_bool_equals_pass(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"ok": True}), [{"key": "ok", "check": "bool_equals", "value": True}])
    assert result.verdict == "PASS"


def test_bool_equals_fail(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"ok": False}), [{"key": "ok", "check": "bool_equals", "value": True}])
    assert result.verdict == "FAIL"
    assert result.issues


def test_str_equals(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"status": "done"}), [{"key": "status", "check": "str_equals", "value": "done"}])
    assert result.verdict == "PASS"
    result2, _ = _verify(tmp_path, json.dumps({"status": "pending"}), [{"key": "status", "check": "str_equals", "value": "done"}])
    assert result2.verdict == "FAIL"


def test_int_equals_and_range(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"count": 5}), [{"key": "count", "check": "int_equals", "value": 5}])
    assert result.verdict == "PASS"
    result2, _ = _verify(tmp_path, json.dumps({"count": 5}), [{"key": "count", "check": "int_range", "min_value": 1, "max_value": 10}])
    assert result2.verdict == "PASS"
    result3, _ = _verify(tmp_path, json.dumps({"count": 50}), [{"key": "count", "check": "int_range", "min_value": 1, "max_value": 10}])
    assert result3.verdict == "FAIL"


def test_float_tolerance_pass_and_fail(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"value": 1.0001}), [{"key": "value", "check": "float_equals", "value": 1.0, "tolerance": 0.001}])
    assert result.verdict == "PASS"
    result2, _ = _verify(tmp_path, json.dumps({"value": 1.1}), [{"key": "value", "check": "float_equals", "value": 1.0, "tolerance": 0.001}])
    assert result2.verdict == "FAIL"


def test_float_range(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"value": 0.5}), [{"key": "value", "check": "float_range", "min_value": 0.0, "max_value": 1.0}])
    assert result.verdict == "PASS"


def test_exists_check(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"anything": None}), [{"key": "anything", "check": "exists"}])
    assert result.verdict == "PASS"
    result2, _ = _verify(tmp_path, json.dumps({}), [{"key": "missing_key", "check": "exists"}])
    assert result2.verdict == "FAIL"


def test_type_is_check(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"payload": [1, 2, 3]}), [{"key": "payload", "check": "type_is", "json_type": "array"}])
    assert result.verdict == "PASS"
    result2, _ = _verify(tmp_path, json.dumps({"payload": "not an array"}), [{"key": "payload", "check": "type_is", "json_type": "array"}])
    assert result2.verdict == "FAIL"


def test_wrong_metric_type_rejected(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"ok": "true"}), [{"key": "ok", "check": "bool_equals", "value": True}])
    assert result.verdict == "FAIL"
    assert any("expected bool" in i for i in result.issues)


def test_missing_metric_key_rejected(tmp_path):
    result, _ = _verify(tmp_path, json.dumps({"other": 1}), [{"key": "ok", "check": "bool_equals", "value": True}])
    assert result.verdict == "FAIL"
    assert any("not present" in i for i in result.issues)


def test_malformed_metrics_json_rejected(tmp_path):
    (tmp_path / "metrics.json").write_text("{this is not valid json")
    spec = _spec([{"key": "ok", "check": "bool_equals", "value": True}])
    result, _ = run_execution_verifier(spec=spec, artifacts_dir=tmp_path, task_id="t", run_id="r", attempt_index=0)
    assert result.verdict == "FAIL"
    assert any("not valid JSON" in i for i in result.issues)


def test_required_artifact_missing_rejected(tmp_path):
    spec = _spec([], required_artifacts=["metrics.json", "extra.json"])
    (tmp_path / "metrics.json").write_text("{}")
    result, _ = run_execution_verifier(spec=spec, artifacts_dir=tmp_path, task_id="t", run_id="r", attempt_index=0)
    assert result.verdict == "FAIL"
    assert any("extra.json" in i for i in result.issues)


def test_command_and_policy_issues_fail_verifier_even_with_perfect_metrics(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"ok": True}))
    spec = _spec([{"key": "ok", "check": "bool_equals", "value": True}])
    result, _ = run_execution_verifier(
        spec=spec, artifacts_dir=tmp_path, task_id="t", run_id="r", attempt_index=0,
        command_issues=["command 0 exited nonzero"],
    )
    assert result.verdict == "FAIL"


def test_no_required_metrics_or_artifacts_passes_trivially(tmp_path):
    spec = _spec([], required_artifacts=[])
    result, _ = run_execution_verifier(spec=spec, artifacts_dir=tmp_path, task_id="t", run_id="r", attempt_index=0)
    assert result.verdict == "PASS"
