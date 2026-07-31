"""Experiment specification loading and run bookkeeping.

`init_run` is the sole place a run directory is created: it always mkdir's
without `exist_ok`, so a colliding run_id raises RunAlreadyExistsError
instead of silently reusing (and thereby overwriting) a prior run.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from research_agent import subprocess_runner
from research_agent.models import ExperimentSpec, RunPaths
from research_agent.policies import commands as command_policy


class RunAlreadyExistsError(RuntimeError):
    pass


class RunLimitExceededError(RuntimeError):
    pass


def load_spec(path: Path) -> ExperimentSpec:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"experiment specification not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"experiment specification must be a YAML mapping, got {type(raw).__name__}")
    return ExperimentSpec.model_validate(raw)


def generate_run_id(task_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{task_id}_{ts}_{uuid.uuid4().hex[:8]}"


def count_existing_runs(runs_root: Path, task_id: str) -> int:
    """Count prior runs for task_id by reading each run's saved spec.yaml,
    never by parsing run_id -- run_id is caller-choosable (see --run-id on
    the CLI) and must not be relied on to encode the task_id."""
    if not runs_root.exists():
        return 0
    count = 0
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        spec_file = run_dir / "spec.yaml"
        if not spec_file.exists():
            continue
        try:
            data = yaml.safe_load(spec_file.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and data.get("task_id") == task_id:
            count += 1
    return count


def assert_run_count_within_limit(runs_root: Path, task_id: str, max_run_count: int) -> None:
    existing = count_existing_runs(runs_root, task_id)
    if existing >= max_run_count:
        raise RunLimitExceededError(
            f"task '{task_id}' already has {existing} run(s) under {runs_root}, "
            f"at or above max_run_count={max_run_count}"
        )


def init_run(*, runs_root: Path, run_id: str, spec_source_path: Path) -> RunPaths:
    """Create a brand-new, immutable run directory. Never overwrites an
    existing run: raises RunAlreadyExistsError if run_id collides."""
    run_paths = RunPaths(root=runs_root, run_id=run_id)
    if run_paths.run_dir.exists():
        raise RunAlreadyExistsError(f"run already exists and will not be overwritten: {run_paths.run_dir}")
    run_paths.run_dir.mkdir(parents=True)
    run_paths.commands_dir.mkdir(parents=True)
    run_paths.artifacts_dir.mkdir(parents=True)
    run_paths.spec_path.write_text(Path(spec_source_path).read_text())
    return run_paths


def run_smoke_command(spec: ExperimentSpec, run_paths: RunPaths, *, cwd: Path):
    """The deterministic runner: executes exactly `spec.smoke_command`, the
    array taken verbatim from the specification. Neither LLM ever reaches
    this function or any subprocess it spawns."""
    command_policy.assert_command_approved(spec.smoke_command, spec)
    run_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "RESEARCH_AGENT_RUN_ID": run_paths.run_id,
        "RESEARCH_AGENT_ARTIFACTS_DIR": str(run_paths.artifacts_dir),
    }
    return subprocess_runner.run_command(
        spec.smoke_command,
        cwd=cwd,
        run_dir=run_paths.commands_dir,
        name="smoke_command",
        timeout=spec.timeouts.smoke_seconds,
        env=env,
    )
