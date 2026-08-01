"""MVP3 repair-run bookkeeping: run directory initialization, atomic
state.json persistence, and worktree-local git inspection helpers that write
their artifacts into an attempt-scoped directory (research_agent.models
.RepairRunPaths.attempt_dir) rather than a single shared run-wide commands
directory -- required so "each attempt must be immutable after completion"
holds for the raw git/subprocess evidence too, not just the JSON summaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from research_agent import subprocess_runner
from research_agent.models import RepairRunPaths, RunStateRecord
from research_agent.tasks import reporting as reporting_tasks
from research_agent.tasks.experiment import RunAlreadyExistsError

__all__ = [
    "init_repair_run",
    "persist_state",
    "collect_worktree_status",
    "write_attempt_git_artifacts",
]


def init_repair_run(*, runs_root: Path, run_id: str, spec_source_path: Path) -> RepairRunPaths:
    """Create a brand-new, immutable repair-run directory. Never overwrites
    an existing run: raises RunAlreadyExistsError if run_id collides (same
    invariant as research_agent.tasks.experiment.init_run)."""
    run_paths = RepairRunPaths(root=runs_root, run_id=run_id)
    if run_paths.run_dir.exists():
        raise RunAlreadyExistsError(f"run already exists and will not be overwritten: {run_paths.run_dir}")
    run_paths.run_dir.mkdir(parents=True)
    run_paths.commands_dir.mkdir(parents=True)
    run_paths.prompts_dir.mkdir(parents=True)
    run_paths.attempts_dir.mkdir(parents=True)
    run_paths.diagnoses_dir.mkdir(parents=True)
    run_paths.repairs_dir.mkdir(parents=True)
    run_paths.spec_path.write_text(Path(spec_source_path).read_text())
    return run_paths


def persist_state(
    run_paths: RepairRunPaths,
    *,
    run_id: str,
    task_id: str,
    state: str,
    attempt_index: int,
    history: list[str],
    detail: Optional[str] = None,
) -> RunStateRecord:
    """Atomically overwrite state.json (via reporting_tasks.save_json_artifact's
    temp-file-plus-os.replace pattern) after every state-machine transition.
    `history` must be the PRIOR record's history list; this function appends
    exactly one new entry and returns the record it wrote, which the caller
    should pass back in as `history` on the next transition."""
    now = reporting_tasks.utcnow_iso()
    record = RunStateRecord(
        run_id=run_id,
        task_id=task_id,
        state=state,
        attempt_index=attempt_index,
        updated_at=now,
        history=[*history, f"{state}@{now}"],
        detail=detail,
    )
    reporting_tasks.save_json_artifact(run_paths.state_path, record)
    return record


def collect_worktree_status(
    worktree_dir: Path, *, command_results_dir: Path, name_prefix: str, timeout: float = 30.0
) -> list[tuple[str, str]]:
    """Same `git status --porcelain=v1 --untracked-files=all` parsing as
    research_agent.execution_worktree.collect_worktree_status, but writes
    subprocess sidecar artifacts into an arbitrary `command_results_dir`
    (an attempt-scoped directory) instead of requiring a full RunPaths."""
    command_results_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess_runner.run_command(
        ["git", "-C", str(worktree_dir), "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=worktree_dir, run_dir=command_results_dir, name=f"{name_prefix}_git_status", timeout=timeout,
    )
    raw = Path(result.stdout_path).read_text()
    entries: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        code = line[:2]
        rest = line[3:] if len(line) > 3 else line.strip()
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        entries.append((code.strip() or code, rest.strip().strip('"')))
    return entries


def write_attempt_git_artifacts(worktree_dir: Path, attempt_dir: Path, *, name_prefix: str, timeout: float = 30.0) -> None:
    """Writes git_status.txt, git_diff.patch, and git_diff_name_status.txt
    directly into attempt_dir -- the human-readable, per-attempt-immutable
    artifacts required by the MVP3 task contract's "Deterministic verifier"
    section, distinct from the raw stdout/stderr/meta.json sidecar files
    subprocess_runner writes under attempt_dir/command_results/."""
    command_results_dir = attempt_dir / "command_results"
    command_results_dir.mkdir(parents=True, exist_ok=True)

    status = subprocess_runner.run_command(
        ["git", "-C", str(worktree_dir), "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=worktree_dir, run_dir=command_results_dir, name=f"{name_prefix}_git_status_snapshot", timeout=timeout,
    )
    (attempt_dir / "git_status.txt").write_text(Path(status.stdout_path).read_text())

    diff = subprocess_runner.run_command(
        ["git", "-C", str(worktree_dir), "diff", "--binary"],
        cwd=worktree_dir, run_dir=command_results_dir, name=f"{name_prefix}_git_diff_snapshot", timeout=timeout,
    )
    (attempt_dir / "git_diff.patch").write_text(Path(diff.stdout_path).read_text())

    name_status = subprocess_runner.run_command(
        ["git", "-C", str(worktree_dir), "diff", "--name-status"],
        cwd=worktree_dir, run_dir=command_results_dir, name=f"{name_prefix}_git_diff_name_status", timeout=timeout,
    )
    (attempt_dir / "git_diff_name_status.txt").write_text(Path(name_status.stdout_path).read_text())


def worktree_head(worktree_dir: Path, *, command_results_dir: Path, name_prefix: str, timeout: float = 30.0) -> str:
    """Defense in depth: Claude never has Bash access in this build (see
    agents/claude_executor.py's --tools Read,Write,Edit), so a `git commit`
    should be structurally impossible -- but repair_flow.py still checks
    HEAD is unchanged after every Claude call rather than assuming the tool
    restriction holds forever. Compares against the worktree's recorded
    base_commit, NOT "exactly one commit total" -- the main repository this
    worktree was branched from may have arbitrarily deep history, so `git
    log --oneline` inside the worktree is not a valid proxy for "no new
    commit was made here"."""
    command_results_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess_runner.run_command(
        ["git", "-C", str(worktree_dir), "rev-parse", "HEAD"],
        cwd=worktree_dir, run_dir=command_results_dir, name=f"{name_prefix}_git_rev_parse_head", timeout=timeout,
    )
    return Path(result.stdout_path).read_text().strip()
