"""Static checks and artifact verification -- both deterministic, neither an
LLM. The verifier relies only on saved exit codes, file existence, and JSON
content; it never trusts planner/executor/reviewer natural-language claims.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from research_agent import subprocess_runner
from research_agent.models import CommandResult, ExperimentSpec, RunPaths, VerificationResult


def run_static_checks(
    *, worktree_dir: Path, spec: ExperimentSpec, run_paths: RunPaths, python_executable: str = sys.executable
) -> CommandResult:
    """`python -m compileall` over whatever allowed paths actually exist in
    the isolated worktree -- a fixed tool invocation the flow constructs
    itself, not an LLM-authored command, so it is not subject to the
    command-approval gate."""
    targets = [str(worktree_dir / p) for p in spec.allowed_paths if (worktree_dir / p).exists()]
    if not targets:
        targets = [str(worktree_dir)]
    command = [python_executable, "-m", "compileall", "-q", *targets]
    return subprocess_runner.run_command(
        command,
        cwd=worktree_dir,
        run_dir=run_paths.commands_dir,
        name="static_checks",
        timeout=spec.timeouts.verifier_seconds,
    )


def verify_artifacts(
    spec: ExperimentSpec, run_paths: RunPaths, smoke_result: CommandResult
) -> VerificationResult:
    details: list[str] = []
    artifacts_ok = True

    if smoke_result.timed_out:
        details.append("smoke command timed out")
    if smoke_result.returncode != 0:
        details.append(f"smoke command exited with nonzero code {smoke_result.returncode}")

    metrics_path = run_paths.artifacts_dir / "metrics.json"
    metrics: dict = {}
    if not metrics_path.exists():
        artifacts_ok = False
        details.append(f"missing required artifact: {metrics_path}")
    else:
        try:
            loaded = json.loads(metrics_path.read_text())
        except json.JSONDecodeError as e:
            artifacts_ok = False
            details.append(f"metrics.json is not valid JSON: {e}")
            loaded = None
        if isinstance(loaded, dict):
            metrics = loaded
        elif loaded is not None:
            artifacts_ok = False
            details.append(f"metrics.json must contain a JSON object, got {type(loaded).__name__}")

    required_metrics_ok = True
    for rm in spec.required_metrics:
        if rm.name not in metrics:
            required_metrics_ok = False
            details.append(f"missing required metric: {rm.name!r}")
            continue
        value = metrics[rm.name]

        if rm.type == "bool":
            type_ok = isinstance(value, bool)
        elif rm.type == "int":
            type_ok = isinstance(value, int) and not isinstance(value, bool)
        elif rm.type == "float":
            type_ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:  # "str"
            type_ok = isinstance(value, str)
        if not type_ok:
            required_metrics_ok = False
            details.append(f"metric {rm.name!r} has wrong type: expected {rm.type}, got {type(value).__name__}")
            continue

        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            required_metrics_ok = False
            details.append(f"metric {rm.name!r} is not finite: {value}")
            continue
        if rm.min_value is not None and isinstance(value, (int, float)) and value < rm.min_value:
            required_metrics_ok = False
            details.append(f"metric {rm.name!r}={value} is below min_value={rm.min_value}")
        if rm.max_value is not None and isinstance(value, (int, float)) and value > rm.max_value:
            required_metrics_ok = False
            details.append(f"metric {rm.name!r}={value} is above max_value={rm.max_value}")

    command_ok = smoke_result.returncode == 0 and not smoke_result.timed_out
    verdict = "PASS" if (command_ok and artifacts_ok and required_metrics_ok) else "FAIL"

    return VerificationResult(
        verdict=verdict,
        required_metrics_ok=required_metrics_ok,
        artifacts_ok=artifacts_ok,
        details=details,
    )
