"""MVP4 run bookkeeping: run directory initialization and atomic
state.json persistence for research_agent.execution_flow -- mirrors
research_agent.tasks.repair's init_repair_run/persist_state, extended with
the additional execution/ and execution_cwd/ directories the restricted-
execution phase needs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from research_agent.models import ExecutionRunPaths, ExecutionRunStateRecord
from research_agent.tasks import reporting as reporting_tasks
from research_agent.tasks.experiment import RunAlreadyExistsError

__all__ = ["init_execution_run", "persist_execution_state"]


def init_execution_run(*, runs_root: Path, run_id: str, spec_source_path: Path) -> ExecutionRunPaths:
    """Create a brand-new, immutable MVP4 run directory. Never overwrites an
    existing run: raises RunAlreadyExistsError if run_id collides (same
    invariant as tasks.experiment.init_run / tasks.repair.init_repair_run)."""
    run_paths = ExecutionRunPaths(root=runs_root, run_id=run_id)
    if run_paths.run_dir.exists():
        raise RunAlreadyExistsError(f"run already exists and will not be overwritten: {run_paths.run_dir}")
    run_paths.run_dir.mkdir(parents=True)
    run_paths.commands_dir.mkdir(parents=True)
    run_paths.prompts_dir.mkdir(parents=True)
    run_paths.artifacts_dir.mkdir(parents=True)
    run_paths.attempts_dir.mkdir(parents=True)
    run_paths.diagnoses_dir.mkdir(parents=True)
    run_paths.repairs_dir.mkdir(parents=True)
    run_paths.execution_dir.mkdir(parents=True)
    run_paths.execution_cwd_dir.mkdir(parents=True)
    run_paths.spec_path.write_text(Path(spec_source_path).read_text())
    return run_paths


def persist_execution_state(
    run_paths: ExecutionRunPaths,
    *,
    run_id: str,
    task_id: str,
    state: str,
    attempt_index: int,
    history: list[str],
    detail: Optional[str] = None,
) -> ExecutionRunStateRecord:
    """Atomically overwrite state.json after every state-machine transition
    -- see the MVP4 task contract's 'No run may remain in an active state
    after process exit' requirement, enforced by execution_flow's outer
    try/except always calling this one more time with a terminal state
    before returning."""
    now = reporting_tasks.utcnow_iso()
    record = ExecutionRunStateRecord(
        run_id=run_id, task_id=task_id, state=state, attempt_index=attempt_index,
        updated_at=now, history=[*history, f"{state}@{now}"], detail=detail,
    )
    reporting_tasks.save_json_artifact(run_paths.state_path, record)
    return record
