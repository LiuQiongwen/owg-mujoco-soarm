"""research_agent.cli's worktree-status/worktree-cleanup dual report-format
support: an MVP2 `execute` run persists report.json (ExecuteReport); an MVP3
`repair` run persists final_report.json (RepairFinalReport) instead. Both
commands must detect which one a given run produced -- from which
structured artifact is actually present and schema-valid, never guessed
from run_id text -- and compare the live execution worktree against the
correct recorded evidence for that run type.

Covers (see the MVP3 cleanup-compatibility task):
  1.  successful MVP3 cleanup with recorded changes
  2.  dry-run
  3.  status for a repair run
  4.  modified-after-report refusal (same recorded path, different content)
  5.  untracked-after-report refusal (a new, unrecorded path appears)
  6.  malformed/missing final_report.json refusal
  7.  MVP2 report.json cleanup regression
  8.  terminal FAILED repair run cleanup when its exact state is recorded
  9.  active-state refusal
  10. wrong recorded worktree path refusal (identity cross-check)
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.agents.claude_executor import MockClaudeExecutorAgent
from research_agent.agents.codex import MockCodexAgent
from research_agent.cli import cmd_worktree_cleanup, cmd_worktree_status
from research_agent.execute_flow import execute_flow
from research_agent.repair_flow import repair_flow
from tests.test_repair_flow import (
    ScriptedClaude,
    ScriptedCodex,
    _fix_content,
    _init_repo,
    _leave_wrong,
    _write_spec,
    _write_wrong,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _args(*, run_id: str, repo_root: Path, runs_root: Path, dry_run: bool = False, delete_branch: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=run_id, repo_root=str(repo_root), runs_root=str(runs_root), dry_run=dry_run, delete_branch=delete_branch,
    )


def _call(func, args) -> tuple[int, dict]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = func(args)
    text = out.getvalue() or err.getvalue()
    return code, json.loads(text)


def _make_repair_pass_run(tmp_path: Path, *, run_id: str = "repair_run1"):
    """A repair run that PASSES on attempt 1 (one repair round), worktree
    left exactly as the run finished it -- the common case cleanup must
    accept."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id=f"{run_id}_task")
    runs_root = tmp_path / "runs"
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_fix_content)
    codex = ScriptedCodex()
    report = repair_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id=run_id,
        execution_worktrees_root=tmp_path / "execution_worktrees",
        codex_agent=codex, claude_agent=claude,
    )
    assert report.overall_status == "PASS"
    return report, repo_root, runs_root


def _make_repair_retry_exhausted_run(tmp_path: Path, *, run_id: str = "repair_run_exhausted"):
    """A repair run that never gets fixed and terminates RETRY_EXHAUSTED --
    the "terminal failed repair run" case."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(
        tmp_path, task_id=f"{run_id}_task",
        repair_limits={
            "max_repair_rounds": 1, "max_total_claude_invocations": 2, "max_total_codex_invocations": 2,
            "max_total_changed_files": 10, "max_total_changed_bytes": 200000, "max_wall_clock_seconds": 300,
        },
    )
    runs_root = tmp_path / "runs"
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_leave_wrong)
    codex = ScriptedCodex()
    report = repair_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id=run_id,
        execution_worktrees_root=tmp_path / "execution_worktrees",
        codex_agent=codex, claude_agent=claude,
    )
    assert report.overall_status == "FAIL"
    assert report.final_state == "RETRY_EXHAUSTED"
    return report, repo_root, runs_root


def _make_execute_run(tmp_path: Path, *, run_id: str = "execute_run1"):
    """An MVP2 `execute` run (report.json, not final_report.json)."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id=f"{run_id}_task", allowed_modify_paths=["research_agent_sandbox"])
    runs_root = tmp_path / "runs"
    claude = MockClaudeExecutorAgent(execute_write_relpath="research_agent_sandbox/out.txt", execute_write_content="hello")
    report = execute_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id=run_id,
        execution_worktrees_root=tmp_path / "execution_worktrees",
        codex_agent=MockCodexAgent(), claude_agent=claude,
    )
    assert report.overall_status == "PASS"
    return report, repo_root, runs_root


# ── 1. successful MVP3 cleanup with recorded changes ────────────────────────

def test_repair_run_cleanup_succeeds_with_matching_recorded_changes(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)
    assert worktree_path.exists()

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["status"] == "REMOVED"
    assert out["run_type"] == "repair"
    assert not worktree_path.exists()


# ── 2. dry-run ───────────────────────────────────────────────────────────────

def test_repair_run_cleanup_dry_run_removes_nothing(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root, dry_run=True))
    assert out["status"] == "DRY_RUN"
    assert out["run_type"] == "repair"
    assert out["would_remove_worktree"] == str(worktree_path)
    assert worktree_path.exists(), "dry-run must never actually remove the worktree"


# ── 3. status for a repair run ───────────────────────────────────────────────

def test_worktree_status_for_repair_run(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)

    code, out = _call(cmd_worktree_status, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["run_type"] == "repair"
    assert out["report_summary"]["run_type"] == "repair"
    assert out["report_summary"]["overall_status"] == "PASS"
    assert out["report_summary"]["passing_attempt_index"] == 1
    assert out["report_error"] is None
    assert out["active_state"] is None


# ── 4. modified-after-report refusal ─────────────────────────────────────────

def test_cleanup_refuses_when_recorded_path_modified_in_place_after_report(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)
    target = worktree_path / "research_agent_sandbox" / "mvp3_validation.txt"
    assert target.exists()
    # Same path (git status line for it is unchanged: still "??"), but the
    # actual bytes now differ from what the report recorded.
    target.write_text("TANGO_MVP3_REPAIR_PASS_TAMPERED")

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["error"] == "REFUSED_MODIFIED_AFTER_REPORT"
    assert "research_agent_sandbox/mvp3_validation.txt" in out["modified_paths"]
    assert worktree_path.exists()


# ── 5. untracked-after-report refusal ────────────────────────────────────────

def test_cleanup_refuses_when_new_untracked_file_appears_after_report(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)
    (worktree_path / "research_agent_sandbox" / "surprise.txt").write_text("unexpected")

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["error"] == "REFUSED_UNRECORDED_CHANGES"
    assert "research_agent_sandbox/surprise.txt" in out["current_changed_paths"]
    assert worktree_path.exists()


# ── 6. malformed/missing final_report.json refusal ──────────────────────────

def test_cleanup_refuses_malformed_final_report(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)
    (runs_root / "repair_run1" / "final_report.json").write_text("{not valid json")

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["error"] == "REFUSED_MALFORMED_REPORT"
    assert worktree_path.exists()


def test_status_reports_malformed_final_report_error(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    (runs_root / "repair_run1" / "final_report.json").write_text(json.dumps({"not": "a valid RepairFinalReport"}))

    code, out = _call(cmd_worktree_status, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["run_type"] == "repair"
    assert out["report_summary"] is None
    assert out["report_error"] is not None


def test_cleanup_refuses_missing_attempt_manifest_even_with_valid_report(tmp_path):
    """final_report.json itself is schema-valid, but the passing attempt's
    changed_file_manifest.json (the actual recorded evidence) is gone --
    must refuse rather than silently trust the worktree's current state."""
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)
    manifest_path = runs_root / "repair_run1" / "attempts" / "attempt_01" / "changed_file_manifest.json"
    assert manifest_path.exists()
    manifest_path.unlink()

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["error"] == "REFUSED_UNTRUSTWORTHY_RECORD"
    assert worktree_path.exists()


# ── 7. MVP2 report.json cleanup regression ───────────────────────────────────

def test_execute_run_report_json_cleanup_regression(tmp_path):
    report, repo_root, runs_root = _make_execute_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)
    assert worktree_path.exists()
    assert (runs_root / "execute_run1" / "report.json").exists()
    assert not (runs_root / "execute_run1" / "final_report.json").exists()

    status_code, status_out = _call(cmd_worktree_status, _args(run_id="execute_run1", repo_root=repo_root, runs_root=runs_root))
    assert status_out["run_type"] == "execute"
    assert status_out["report_summary"]["run_type"] == "execute"
    assert status_out["report_summary"]["terminal_state"] == "EXECUTION_PASS"

    code, out = _call(cmd_worktree_cleanup, _args(run_id="execute_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["status"] == "REMOVED"
    assert out["run_type"] == "execute"
    assert not worktree_path.exists()


def test_execute_run_cleanup_still_refuses_unrecorded_changes(tmp_path):
    report, repo_root, runs_root = _make_execute_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)
    (worktree_path / "research_agent_sandbox" / "another.txt").write_text("surprise")

    code, out = _call(cmd_worktree_cleanup, _args(run_id="execute_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["error"] == "REFUSED_UNRECORDED_CHANGES"
    assert worktree_path.exists()


# ── 8. terminal FAILED repair run cleanup when its exact state is recorded ──

def test_retry_exhausted_repair_run_cleanup_succeeds_with_matching_state(tmp_path):
    report, repo_root, runs_root = _make_repair_retry_exhausted_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)
    assert worktree_path.exists()

    status_code, status_out = _call(cmd_worktree_status, _args(run_id="repair_run_exhausted", repo_root=repo_root, runs_root=runs_root))
    assert status_out["report_summary"]["overall_status"] == "FAIL"
    assert status_out["report_summary"]["final_state"] == "RETRY_EXHAUSTED"
    assert status_out["report_summary"]["passing_attempt_index"] is None

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run_exhausted", repo_root=repo_root, runs_root=runs_root))
    assert out["status"] == "REMOVED"
    assert not worktree_path.exists()


# ── 9. active-state refusal ──────────────────────────────────────────────────

def test_cleanup_refuses_active_repair_run(tmp_path):
    """Simulates a process killed mid-repair: execution_worktree.json and a
    real (untouched, zero-diff) worktree exist, state.json is non-terminal,
    and final_report.json was never written. Must refuse even though the
    worktree itself currently has no uncommitted changes -- the old "no
    report + no changes -> allowed" fallback must NOT fire for an active run."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="active_run_task")
    runs_root = tmp_path / "runs"

    # Reach PLAN_VALIDATED + a created worktree deterministically, then bail
    # out before attempt 0 ever runs by starving the Claude-invocation budget
    # to zero repair rounds worth of work -- simplest: just drive repair_flow
    # normally to completion, then hand-edit its artifacts to simulate an
    # interruption (delete final_report.json, rewrite state.json non-terminal).
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_fix_content)
    codex = ScriptedCodex()
    report = repair_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="active_run",
        execution_worktrees_root=tmp_path / "execution_worktrees",
        codex_agent=codex, claude_agent=claude,
    )
    assert report.overall_status == "PASS"
    worktree_path = Path(report.execution_worktree.worktree_path)

    run_dir = runs_root / "active_run"
    (run_dir / "final_report.json").unlink()
    state_data = json.loads((run_dir / "state.json").read_text())
    state_data["state"] = "REPAIRING"
    (run_dir / "state.json").write_text(json.dumps(state_data))

    # Revert the worktree's file back to a clean, zero-diff state so the OLD
    # "no report + no changes" fallback would have wrongly allowed this.
    subprocess.run(["git", "-C", str(worktree_path), "clean", "-fd"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(worktree_path), "checkout", "--", "."], check=True, capture_output=True)
    status = subprocess.run(
        ["git", "-C", str(worktree_path), "status", "--porcelain"], check=True, capture_output=True, text=True,
    ).stdout
    assert status.strip() == ""

    code, out = _call(cmd_worktree_cleanup, _args(run_id="active_run", repo_root=repo_root, runs_root=runs_root))
    assert out["error"] == "REFUSED_RUN_ACTIVE"
    assert worktree_path.exists()

    status_code, status_out = _call(cmd_worktree_status, _args(run_id="active_run", repo_root=repo_root, runs_root=runs_root))
    assert status_out["active_state"] == "REPAIRING"


# ── 10. wrong recorded worktree path refusal (identity cross-check) ─────────

def test_cleanup_refuses_inconsistent_worktree_identity(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)

    final_report_path = runs_root / "repair_run1" / "final_report.json"
    data = json.loads(final_report_path.read_text())
    data["execution_worktree"]["worktree_path"] = str(worktree_path) + "_TAMPERED_PATH"
    final_report_path.write_text(json.dumps(data))

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["error"] == "REFUSED_INCONSISTENT_IDENTITY"
    assert worktree_path.exists()


def test_cleanup_refuses_inconsistent_branch_identity(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)

    ewt_path = runs_root / "repair_run1" / "execution_worktree.json"
    data = json.loads(ewt_path.read_text())
    data["branch_name"] = "research-agent-execute/some-other-run"
    ewt_path.write_text(json.dumps(data))

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["error"] == "REFUSED_INCONSISTENT_IDENTITY"
    assert worktree_path.exists()


# ── main-worktree protection (defense in depth on top of creation-time checks) ─

def test_cleanup_refuses_if_recorded_worktree_path_is_main_repo(tmp_path):
    report, repo_root, runs_root = _make_repair_pass_run(tmp_path)
    worktree_path = Path(report.execution_worktree.worktree_path)

    ewt_path = runs_root / "repair_run1" / "execution_worktree.json"
    data = json.loads(ewt_path.read_text())
    data["worktree_path"] = str(repo_root)
    ewt_path.write_text(json.dumps(data))

    code, out = _call(cmd_worktree_cleanup, _args(run_id="repair_run1", repo_root=repo_root, runs_root=runs_root))
    assert out["error"] == "REFUSED_MAIN_WORKTREE_TARGET"
    assert worktree_path.exists()  # the real (correct) worktree was never touched
    assert repo_root.exists() and (repo_root / ".git").exists()  # and obviously main itself is untouched
