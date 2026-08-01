"""MVP4 deterministic metric/artifact verifier.

Checks research_agent.models.ExecutionSpec.required_artifacts /
.required_metrics against actual files under the run's artifacts directory.
Neither Codex nor Claude ever decides pass/fail here -- every check is a
plain filesystem read, a JSON parse, and a Python-level comparison. Command-
execution outcomes (nonzero exit, timeout) and artifact-policy violations
(symlink escape, count/byte limits, ...) are NOT decided here -- the caller
(research_agent.execution_flow) passes them in as already-computed
`command_issues`/`policy_issues`, so this module's `issues` list (and hence
`verdict`) is the single place a run's overall pass/fail is decided, without
this module needing to know how a command was run or how artifacts were
scanned.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from research_agent.models import ExecutionVerifierResult

if TYPE_CHECKING:
    from research_agent.models import ExperimentSpec, MetricCheck


def _load_json(path: Path) -> tuple[Optional[object], Optional[str]]:
    if not path.exists():
        return None, f"required file missing: {path.name}"
    try:
        return json.loads(path.read_text()), None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        return None, f"{path.name} is not valid JSON: {e}"


_TYPE_MAP = {
    "object": dict, "array": list, "string": str, "number": (int, float),
    "integer": int, "boolean": bool, "null": type(None),
}


def _check_metric(check: "MetricCheck", value: object) -> Optional[str]:
    """Returns None on pass, else a human-readable failure reason."""
    kind = check.check
    if kind == "exists":
        return None  # presence itself is already required before this is called
    if kind == "bool_equals":
        if not isinstance(value, bool):
            return f"expected bool, got {type(value).__name__}: {value!r}"
        return None if value == check.value else f"expected {check.value!r}, got {value!r}"
    if kind == "str_equals":
        if not isinstance(value, str):
            return f"expected str, got {type(value).__name__}: {value!r}"
        return None if value == check.value else f"expected {check.value!r}, got {value!r}"
    if kind == "int_equals":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"expected int, got {type(value).__name__}: {value!r}"
        return None if value == check.value else f"expected {check.value!r}, got {value!r}"
    if kind == "int_range":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"expected int, got {type(value).__name__}: {value!r}"
        if check.min_value is not None and value < check.min_value:
            return f"{value} is below min_value={check.min_value}"
        if check.max_value is not None and value > check.max_value:
            return f"{value} is above max_value={check.max_value}"
        return None
    if kind in ("float_equals", "float_range"):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"expected a number, got {type(value).__name__}: {value!r}"
        if math.isnan(value) or math.isinf(value):
            return f"value is not finite: {value}"
        if kind == "float_equals":
            if abs(float(value) - float(check.value)) > float(check.tolerance):
                return f"expected {check.value!r} +/- {check.tolerance}, got {value!r}"
            return None
        if check.min_value is not None and value < check.min_value:
            return f"{value} is below min_value={check.min_value}"
        if check.max_value is not None and value > check.max_value:
            return f"{value} is above max_value={check.max_value}"
        return None
    if kind == "type_is":
        expected = _TYPE_MAP[check.json_type]
        ok = isinstance(value, expected)
        if check.json_type in ("number", "integer") and isinstance(value, bool):
            ok = False
        return None if ok else f"expected JSON type {check.json_type!r}, got {type(value).__name__}: {value!r}"
    return f"unknown check kind: {kind!r}"


def run_execution_verifier(
    *,
    spec: "ExperimentSpec",
    artifacts_dir: Path,
    task_id: str,
    run_id: str,
    attempt_index: int,
    command_checks: Optional[list[str]] = None,
    command_issues: Optional[list[str]] = None,
    policy_checks: Optional[list[str]] = None,
    policy_issues: Optional[list[str]] = None,
) -> tuple[ExecutionVerifierResult, dict[str, object]]:
    """Returns (result, metrics_by_file): metrics_by_file maps every
    distinct artifact-relative filename actually read to its parsed JSON
    content, for the caller to persist as the run's top-level metrics.json
    aggregate."""
    execution = spec.execution
    artifacts_dir = Path(artifacts_dir)

    artifact_checks: list[str] = []
    metric_checks: list[str] = []
    issues: list[str] = []
    evidence: list[str] = []
    loaded: dict[str, object] = {}
    load_errors: dict[str, str] = {}

    required_artifacts = list(execution.required_artifacts) if execution else []
    required_metrics = list(execution.required_metrics) if execution else []

    for rel_path in required_artifacts:
        check_name = f"required_artifact:{rel_path}"
        if not (artifacts_dir / rel_path).exists():
            issues.append(f"required artifact missing: {rel_path}")
            evidence.append(check_name)
        else:
            artifact_checks.append(check_name)

    def _get_loaded(rel_path: str):
        if rel_path in loaded:
            return loaded[rel_path], None
        if rel_path in load_errors:
            return None, load_errors[rel_path]
        data, err = _load_json(artifacts_dir / rel_path)
        if err is not None:
            load_errors[rel_path] = err
            return None, err
        loaded[rel_path] = data
        return data, None

    for check in required_metrics:
        artifact_rel = check.artifact or "metrics.json"
        check_name = f"metric:{artifact_rel}:{check.key}:{check.check}"
        data, err = _get_loaded(artifact_rel)
        if err is not None:
            issues.append(f"{check_name}: {err}")
            evidence.append(err)
            continue
        if not isinstance(data, dict):
            issues.append(f"{check_name}: {artifact_rel} does not contain a JSON object")
            continue
        if check.key not in data:
            issues.append(f"{check_name}: required key {check.key!r} not present in {artifact_rel}")
            continue
        value = data[check.key]
        failure = _check_metric(check, value)
        if failure is not None:
            issues.append(f"{check_name}: {failure}")
            evidence.append(f"{check.key}={value!r}")
        else:
            metric_checks.append(check_name)

    issues.extend(command_issues or [])
    issues.extend(policy_issues or [])
    verdict = "FAIL" if issues else "PASS"

    result = ExecutionVerifierResult(
        task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict=verdict,
        artifact_checks=artifact_checks, metric_checks=metric_checks,
        command_checks=list(command_checks or []), policy_checks=list(policy_checks or []),
        issues=issues, evidence=evidence,
    )
    return result, loaded
