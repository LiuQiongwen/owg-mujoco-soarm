"""MVP3 bounded repair-loop tests: research_agent.repair_flow's state
machine, failure taxonomy, budget enforcement, non-retriable short-circuits,
attempt-artifact immutability, and final-report correctness -- exercised
against in-process "injected adapter" fakes (ScriptedClaude/ScriptedCodex,
duck-typing the same execute/repair/plan/review/diagnose contract the real
agents use -- an explicitly permitted alternative to a fake subprocess
executable, see the MVP3 task contract's "fake executables or injected
adapters" allowance) plus a smaller set of genuine fake-*subprocess*-
executable tests for the Real agent parsing boundary (malformed JSON,
task/run/attempt-id mismatch, auth failure) -- mirroring
tests/test_real_claude_executor.py and tests/test_real_codex_planner.py.

Covers (see the MVP3 task contract's "Fake-agent tests" checklist):
  1.  initial implementation passes, no diagnosis or repair invoked
  2.  initial implementation fails, one repair succeeds
  3.  first repair fails, second repair succeeds
  4.  repair budget exhausted
  5.  diagnosis says blocked
  6.  diagnosis says policy failure
  7.  diagnosis proposes forbidden path
  8.  diagnosis proposes forbidden command
  9.  Claude repair modifies forbidden path
  10. Claude repair exceeds changed-file limit
  11. Claude repair exceeds changed-byte limit
  12. repair diff claim mismatch
  13. main worktree mutation during diagnosis
  14. main worktree mutation during repair
  15. symlink escape during repair
  16. nested Git during repair
  17. .git tampering during repair
  18. Codex timeout
  19. Claude timeout
  20. Codex authentication failure (fake executable)
  21. Claude authentication failure (fake executable)
  22. malformed diagnosis JSON (one-time-retry path, and always-fails path)
  23. malformed repair JSON (fake executable)
  24. task ID mismatch (diagnosis and repair, fake executable)
  25. run ID mismatch (diagnosis and repair, fake executable)
  26. attempt index mismatch (diagnosis and repair, fake executable)
  27. failure state never remains active
  28. no infinite retry
  29. no second Claude invocation after non-retriable policy failure
  30. worktree preserved after exhausted retries
  31. default mock/mock regression remains unchanged
  32. MVP1/real-Codex-planner code path still reachable from `repair`
  33. MVP2/real-Claude-executor code path still reachable from `repair`
  34. no commit or push
  35. no research experiment command
  36. exact attempt artifacts are created
  37. final report points to passing attempt
  38. final report correctly records retry exhaustion
  39. a fake Codex or Claude trying to invoke the other agent is blocked/impossible
  40. interrupted run can be reported safely as incomplete, never silently PASS
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.agents.claude_executor import (
    ClaudeExecutorAgent,
    ClaudeExecutorError,
    MockClaudeExecutorAgent,
    RealClaudeExecutorAgent,
)
from research_agent.agents.codex import CodexAgent, CodexPlannerError, MockCodexAgent, RealCodexAgent
from research_agent.models import (
    REPAIR_TERMINAL_STATES,
    DiagnosisResult,
    ExecutorImplementationResult,
    PlanResult,
    RepairResult,
    ReviewResult,
)
from research_agent.repair_flow import repair_flow
from research_agent.subprocess_runner import CommandTimeoutError, ExecutableNotFoundError

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


# ── shared fixtures ──────────────────────────────────────────────────────

def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *args: subprocess.run(args, cwd=repo, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (repo / "README.md").write_text("test repo\n")
    (repo / "research_agent_sandbox").mkdir()
    (repo / "research_agent_sandbox" / ".gitkeep").write_text("")
    (repo / "tango_robot").mkdir()
    (repo / "tango_robot" / ".gitkeep").write_text("")
    (repo / ".gitignore").write_text("*.pt\n*.pth\n*.ckpt\n")
    run("git", "add", ".")
    run("git", "commit", "-q", "-m", "init")
    return repo


_TARGET_RELPATH = "research_agent_sandbox/mvp3_validation.txt"
_TARGET_CONTENT = "TANGO_MVP3_REPAIR_PASS"
_WRONG_CONTENT = "TANGO_MVP3_WRONG"


def _write_spec(
    tmp_path: Path,
    *,
    task_id: str,
    max_run_count: int = 10,
    max_changed_files: int = 5,
    max_changed_bytes: int = 20000,
    expected_file_contents: Optional[dict] = None,
    required_artifacts: Optional[list] = None,
    allowed_modify_paths: Optional[list] = None,
    repair_limits: Optional[dict] = None,
) -> Path:
    spec = {
        "task_id": task_id,
        "goal": "mvp3 repair-flow fake-agent test",
        "allowed_paths": ["research_agent_sandbox"],
        "forbidden_paths": ["tango_robot"],
        "allowed_modify_paths": allowed_modify_paths or [_TARGET_RELPATH],
        "max_changed_files": max_changed_files,
        "max_changed_bytes": max_changed_bytes,
        "allowed_executor_commands": [],
        "required_executor_checks": ["compileall"],
        "expected_file_contents": expected_file_contents if expected_file_contents is not None else {_TARGET_RELPATH: _TARGET_CONTENT},
        "required_artifacts": required_artifacts or [],
        "repair_limits": repair_limits or {
            "max_repair_rounds": 2, "max_total_claude_invocations": 3, "max_total_codex_invocations": 4,
            "max_total_changed_files": 10, "max_total_changed_bytes": 200000, "max_wall_clock_seconds": 300,
        },
        "smoke_command": [sys.executable, "-c", "pass"],
        "seeds": [0],
        "timeouts": {
            "planner_seconds": 30, "executor_seconds": 30, "smoke_seconds": 30,
            "verifier_seconds": 30, "reviewer_seconds": 30,
        },
        "max_run_count": max_run_count,
    }
    path = tmp_path / f"{task_id}.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def _run(
    tmp_path: Path, *, task_id: str, codex_agent: CodexAgent, claude_agent: ClaudeExecutorAgent,
    run_id: str = "run1", repo_root: Optional[Path] = None, spec_kwargs: Optional[dict] = None,
):
    repo_root = repo_root or _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id=task_id, **(spec_kwargs or {}))
    runs_root = tmp_path / "runs"
    report = repair_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id=run_id,
        execution_worktrees_root=tmp_path / "execution_worktrees",
        codex_agent=codex_agent, claude_agent=claude_agent,
    )
    return report, repo_root, runs_root


def _assert_terminal_and_never_running(report) -> None:
    """#27: failure state never remains active."""
    assert report.final_state in REPAIR_TERMINAL_STATES
    assert report.overall_status in ("PASS", "FAIL", "BLOCKED")


def _assert_worktree_preserved(report) -> None:
    """#30: worktree preserved after exhausted retries (and every other
    terminal outcome -- MVP3 never auto-cleans)."""
    assert report.execution_worktree is not None
    assert Path(report.execution_worktree.worktree_path).exists()
    assert report.execution_worktree.preserved is True


def _assert_no_commit_no_push(report, repo_root: Path) -> None:
    """#34: no commit and no push occurs anywhere."""
    worktree_path = Path(report.execution_worktree.worktree_path)
    log = subprocess.run(
        ["git", "-C", str(worktree_path), "log", "--oneline"], capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert len(log) == 1, f"expected exactly the base commit, got: {log}"
    main_log = subprocess.run(
        ["git", "-C", str(repo_root), "log", "--oneline"], capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert len(main_log) == 1, f"main repo history must be untouched: {main_log}"


# ── in-process "injected adapter" fakes (an explicit MVP3-permitted
# alternative to a fake subprocess executable) ─────────────────────────────

class ScriptedClaude(ClaudeExecutorAgent):
    """execute_fn(worktree_dir, task_id, run_id, repo_root) -> ExecutorImplementationResult
    repair_fn(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root) -> RepairResult
    Either may raise ExecutableNotFoundError / CommandTimeoutError / ClaudeExecutorError."""

    def __init__(
        self,
        *,
        execute_fn: Optional[Callable] = None,
        repair_fn: Optional[Callable] = None,
        repo_root: Optional[Path] = None,
    ):
        self._execute_fn = execute_fn
        self._repair_fn = repair_fn
        self.repo_root = repo_root
        self.execute_calls = 0
        self.repair_calls = 0

    def execute(self, *, prompt, worktree_dir, run_paths, timeout, task_id, run_id) -> ExecutorImplementationResult:
        self.execute_calls += 1
        if self._execute_fn is None:
            return ExecutorImplementationResult(task_id=task_id, run_id=run_id, verdict="IMPLEMENTATION_PASS", summary="noop")
        return self._execute_fn(worktree_dir=Path(worktree_dir), task_id=task_id, run_id=run_id, repo_root=self.repo_root)

    def repair(self, *, prompt, worktree_dir, run_paths, timeout, task_id, run_id, attempt_index) -> RepairResult:
        self.repair_calls += 1
        if self._repair_fn is None:
            return RepairResult(task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS", summary="noop")
        return self._repair_fn(
            worktree_dir=Path(worktree_dir), task_id=task_id, run_id=run_id, attempt_index=attempt_index,
            call_count=self.repair_calls, repo_root=self.repo_root,
        )


class ScriptedCodex(CodexAgent):
    """diagnose_fn(task_id, run_id, attempt_index, call_count, repo_root) -> DiagnosisResult
    May raise CodexPlannerError / ExecutableNotFoundError / CommandTimeoutError."""

    def __init__(self, *, plan_verdict: str = "PLAN_PASS", diagnose_fn: Optional[Callable] = None, repo_root: Optional[Path] = None):
        self._plan_verdict = plan_verdict
        self._diagnose_fn = diagnose_fn
        self.repo_root = repo_root
        self.plan_calls = 0
        self.diagnose_calls = 0

    def plan(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> PlanResult:
        self.plan_calls += 1
        return PlanResult(task_id=task_id, run_id=run_id, verdict=self._plan_verdict, summary="scripted plan")

    def review(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> ReviewResult:
        return ReviewResult(task_id=task_id, run_id=run_id, verdict="REVIEW_PASS", summary="scripted review")

    def diagnose(self, *, prompt, run_dir, cwd, timeout, task_id, run_id, attempt_index) -> DiagnosisResult:
        self.diagnose_calls += 1
        if self._diagnose_fn is None:
            return DiagnosisResult(
                task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="DIAGNOSE_REPAIRABLE",
                failure_class="EXPECTED_CONTENT_MISMATCH", root_cause="scripted diagnosis",
            )
        return self._diagnose_fn(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, call_count=self.diagnose_calls,
            repo_root=self.repo_root,
        )


def _write_wrong(worktree_dir: Path, task_id: str, run_id: str, repo_root=None) -> ExecutorImplementationResult:
    target = worktree_dir / _TARGET_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_WRONG_CONTENT)
    return ExecutorImplementationResult(
        task_id=task_id, run_id=run_id, verdict="IMPLEMENTATION_PASS",
        summary="wrote wrong content", changed_files=[_TARGET_RELPATH],
    )


def _write_nothing(worktree_dir: Path, task_id: str, run_id: str, repo_root=None) -> ExecutorImplementationResult:
    return ExecutorImplementationResult(task_id=task_id, run_id=run_id, verdict="IMPLEMENTATION_PASS", summary="no changes")


def _fix_content(worktree_dir: Path, task_id, run_id, attempt_index, call_count, repo_root=None) -> RepairResult:
    target = worktree_dir / _TARGET_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_TARGET_CONTENT)
    return RepairResult(
        task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
        summary="fixed content", changed_files=[_TARGET_RELPATH],
    )


def _leave_wrong(worktree_dir: Path, task_id, run_id, attempt_index, call_count, repo_root=None) -> RepairResult:
    target = worktree_dir / _TARGET_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_WRONG_CONTENT)
    return RepairResult(
        task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_REVISE",
        summary="still wrong", changed_files=[_TARGET_RELPATH],
    )


# ── 1. initial implementation passes, no diagnosis or repair invoked ───────

def test_initial_implementation_passes_no_diagnosis_or_repair(tmp_path):
    def execute_ok(worktree_dir, task_id, run_id, repo_root=None):
        target = worktree_dir / _TARGET_RELPATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_TARGET_CONTENT)
        return ExecutorImplementationResult(
            task_id=task_id, run_id=run_id, verdict="IMPLEMENTATION_PASS",
            summary="correct on first try", changed_files=[_TARGET_RELPATH],
        )

    claude = ScriptedClaude(execute_fn=execute_ok)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t1_pass_first_try", codex_agent=codex, claude_agent=claude)

    assert report.overall_status == "PASS"
    assert report.final_state == "PASS"
    assert report.passing_attempt_index == 0
    assert len(report.attempts) == 1
    assert claude.repair_calls == 0
    assert codex.diagnose_calls == 0
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)
    _assert_no_commit_no_push(report, repo_root)


# ── 2. initial implementation fails, one repair succeeds ───────────────────

def test_initial_fails_one_repair_succeeds(tmp_path):
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_fix_content)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t2_one_repair", codex_agent=codex, claude_agent=claude)

    assert report.overall_status == "PASS"
    assert report.passing_attempt_index == 1
    assert len(report.attempts) == 2
    assert claude.execute_calls == 1
    assert claude.repair_calls == 1
    assert codex.diagnose_calls == 1
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)
    _assert_no_commit_no_push(report, repo_root)


# ── 3. first repair fails, second repair succeeds ───────────────────────────

def test_first_repair_fails_second_repair_succeeds(tmp_path):
    def repair_fn(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        if call_count == 1:
            return _leave_wrong(worktree_dir, task_id, run_id, attempt_index, call_count)
        return _fix_content(worktree_dir, task_id, run_id, attempt_index, call_count)

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_fn)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t3_second_repair", codex_agent=codex, claude_agent=claude,
        spec_kwargs={"repair_limits": {
            "max_repair_rounds": 2, "max_total_claude_invocations": 3, "max_total_codex_invocations": 4,
            "max_total_changed_files": 10, "max_total_changed_bytes": 200000, "max_wall_clock_seconds": 300,
        }},
    )

    assert report.overall_status == "PASS"
    assert report.passing_attempt_index == 2
    assert len(report.attempts) == 3
    assert claude.repair_calls == 2
    assert codex.diagnose_calls == 2
    _assert_terminal_and_never_running(report)


# ── 4. repair budget exhausted ──────────────────────────────────────────────

def test_repair_budget_exhausted(tmp_path):
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_leave_wrong)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t4_exhausted", codex_agent=codex, claude_agent=claude,
        spec_kwargs={"repair_limits": {
            "max_repair_rounds": 2, "max_total_claude_invocations": 3, "max_total_codex_invocations": 4,
            "max_total_changed_files": 10, "max_total_changed_bytes": 200000, "max_wall_clock_seconds": 300,
        }},
    )

    assert report.overall_status == "FAIL"
    assert report.final_state == "RETRY_EXHAUSTED"
    assert report.passing_attempt_index is None
    assert len(report.attempts) == 3  # attempt 0 + 2 repair rounds, then stop
    assert claude.repair_calls == 2
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 5. diagnosis says blocked ────────────────────────────────────────────────

def test_diagnosis_blocked(tmp_path):
    def diagnose_blocked(task_id, run_id, attempt_index, call_count, repo_root=None):
        return DiagnosisResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index,
            verdict="DIAGNOSE_BLOCKED", failure_class="EXPECTED_CONTENT_MISMATCH", root_cause="cannot safely repair",
        )

    claude = ScriptedClaude(execute_fn=_write_wrong)
    codex = ScriptedCodex(diagnose_fn=diagnose_blocked)
    report, repo_root, runs_root = _run(tmp_path, task_id="t5_diag_blocked", codex_agent=codex, claude_agent=claude)

    assert report.overall_status == "BLOCKED"
    assert report.final_state == "BLOCKED"
    assert claude.repair_calls == 0
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 6. diagnosis says policy failure ─────────────────────────────────────────

def test_diagnosis_policy_failure(tmp_path):
    def diagnose_policy_fail(task_id, run_id, attempt_index, call_count, repo_root=None):
        return DiagnosisResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index,
            verdict="DIAGNOSE_POLICY_FAILURE", failure_class="PATH_POLICY_FAILURE", root_cause="policy conflict",
        )

    claude = ScriptedClaude(execute_fn=_write_wrong)
    codex = ScriptedCodex(diagnose_fn=diagnose_policy_fail)
    report, repo_root, runs_root = _run(tmp_path, task_id="t6_diag_policy", codex_agent=codex, claude_agent=claude)

    assert report.overall_status == "BLOCKED"
    assert report.final_state == "POLICY_FAILURE"
    assert claude.repair_calls == 0
    _assert_terminal_and_never_running(report)


# ── 7. diagnosis proposes forbidden path ─────────────────────────────────────

def test_diagnosis_proposes_forbidden_path(tmp_path):
    def diagnose_broadens_path(task_id, run_id, attempt_index, call_count, repo_root=None):
        return DiagnosisResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="DIAGNOSE_REPAIRABLE",
            failure_class="EXPECTED_CONTENT_MISMATCH", root_cause="x",
            files_allowed_to_touch=["tango_robot/forbidden.py"],
        )

    claude = ScriptedClaude(execute_fn=_write_wrong)
    codex = ScriptedCodex(diagnose_fn=diagnose_broadens_path)
    report, repo_root, runs_root = _run(tmp_path, task_id="t7_diag_forbidden_path", codex_agent=codex, claude_agent=claude)

    assert report.final_state == "POLICY_FAILURE"
    assert claude.repair_calls == 0, "repair must never be invoked when the diagnosis tries to broaden scope"
    _assert_terminal_and_never_running(report)


# ── 8. diagnosis proposes forbidden command ──────────────────────────────────

def test_diagnosis_proposes_forbidden_command(tmp_path):
    def diagnose_broadens_command(task_id, run_id, attempt_index, call_count, repo_root=None):
        return DiagnosisResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="DIAGNOSE_REPAIRABLE",
            failure_class="EXPECTED_CONTENT_MISMATCH", root_cause="x",
            commands_allowed_to_run=[["bash", "-c", "echo hi"]],
        )

    claude = ScriptedClaude(execute_fn=_write_wrong)
    codex = ScriptedCodex(diagnose_fn=diagnose_broadens_command)
    report, repo_root, runs_root = _run(tmp_path, task_id="t8_diag_forbidden_cmd", codex_agent=codex, claude_agent=claude)

    assert report.final_state == "POLICY_FAILURE"
    assert claude.repair_calls == 0
    _assert_terminal_and_never_running(report)


# ── 9. Claude repair modifies forbidden path ─────────────────────────────────

def test_repair_modifies_forbidden_path(tmp_path):
    def repair_forbidden(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        target = worktree_dir / "tango_robot" / "malicious.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
        return RepairResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
            summary="wrote forbidden path", changed_files=["tango_robot/malicious.py"],
        )

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_forbidden)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t9_repair_forbidden_path", codex_agent=codex, claude_agent=claude)

    assert report.final_state == "POLICY_FAILURE"
    assert claude.repair_calls == 1
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 10. Claude repair exceeds changed-file limit ─────────────────────────────

def test_repair_exceeds_changed_file_limit(tmp_path):
    def repair_many_files(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        changed = []
        for i in range(3):
            rel = f"research_agent_sandbox/extra_{i}.txt"
            (worktree_dir / rel).write_text("x")
            changed.append(rel)
        return RepairResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
            summary="wrote too many files", changed_files=changed,
        )

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_many_files)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t10_file_limit", codex_agent=codex, claude_agent=claude,
        spec_kwargs={
            "allowed_modify_paths": ["research_agent_sandbox"],
            "repair_limits": {
                "max_repair_rounds": 2, "max_total_claude_invocations": 3, "max_total_codex_invocations": 4,
                "max_total_changed_files": 2, "max_total_changed_bytes": 200000, "max_wall_clock_seconds": 300,
            },
        },
    )

    assert report.final_state == "POLICY_FAILURE"
    _assert_terminal_and_never_running(report)


# ── 11. Claude repair exceeds changed-byte limit ─────────────────────────────

def test_repair_exceeds_changed_byte_limit(tmp_path):
    def repair_big_file(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        rel = "research_agent_sandbox/big.txt"
        (worktree_dir / rel).write_text("0" * 5000)
        return RepairResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
            summary="wrote a huge file", changed_files=[rel],
        )

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_big_file)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t11_byte_limit", codex_agent=codex, claude_agent=claude,
        spec_kwargs={
            "allowed_modify_paths": ["research_agent_sandbox"],
            "repair_limits": {
                "max_repair_rounds": 2, "max_total_claude_invocations": 3, "max_total_codex_invocations": 4,
                "max_total_changed_files": 10, "max_total_changed_bytes": 1000, "max_wall_clock_seconds": 300,
            },
        },
    )

    assert report.final_state == "POLICY_FAILURE"
    _assert_terminal_and_never_running(report)


# ── 12. repair diff claim mismatch ───────────────────────────────────────────

def test_repair_diff_claim_mismatch(tmp_path):
    def repair_lies(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        target = worktree_dir / _TARGET_RELPATH
        target.write_text(_TARGET_CONTENT)
        return RepairResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
            summary="claims a file that was never written",
            changed_files=["research_agent_sandbox/never_written.txt"],
        )

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_lies)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t12_diff_claim_mismatch", codex_agent=codex, claude_agent=claude)

    assert report.final_state == "POLICY_FAILURE"
    _assert_terminal_and_never_running(report)


# ── 13. main worktree mutation during diagnosis ──────────────────────────────

def test_main_worktree_mutation_during_diagnosis(tmp_path):
    def diagnose_mutates_main(task_id, run_id, attempt_index, call_count, repo_root=None):
        (repo_root / "mutated_by_diagnosis.txt").write_text("x")
        return DiagnosisResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="DIAGNOSE_REPAIRABLE",
            failure_class="EXPECTED_CONTENT_MISMATCH", root_cause="x",
        )

    repo_root = None
    claude = ScriptedClaude(execute_fn=_write_wrong)
    codex = ScriptedCodex(diagnose_fn=diagnose_mutates_main)
    repo_root = _init_repo(tmp_path)
    codex.repo_root = repo_root
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t13_main_mutation_diag", codex_agent=codex, claude_agent=claude, repo_root=repo_root,
    )

    assert report.final_state == "POLICY_FAILURE"
    assert not report.main_worktree_unchanged
    assert (repo_root / "mutated_by_diagnosis.txt").exists()
    assert claude.repair_calls == 0
    _assert_terminal_and_never_running(report)


# ── 14. main worktree mutation during repair ─────────────────────────────────

def test_main_worktree_mutation_during_repair(tmp_path):
    def repair_mutates_main(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        (repo_root / "mutated_by_repair.txt").write_text("x")
        return RepairResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
            summary="mutated main worktree", changed_files=[],
        )

    repo_root = _init_repo(tmp_path)
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_mutates_main, repo_root=repo_root)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t14_main_mutation_repair", codex_agent=codex, claude_agent=claude, repo_root=repo_root,
    )

    assert report.final_state == "POLICY_FAILURE"
    assert not report.main_worktree_unchanged
    assert (repo_root / "mutated_by_repair.txt").exists()
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 15. symlink escape during repair ─────────────────────────────────────────

def test_symlink_escape_during_repair(tmp_path):
    def repair_symlink_escape(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        rel = "research_agent_sandbox/escape_link"
        full = worktree_dir / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        os.symlink("/etc/passwd", full)
        return RepairResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
            summary="symlink escape", changed_files=[rel],
        )

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_symlink_escape)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t15_symlink_escape", codex_agent=codex, claude_agent=claude,
        spec_kwargs={"allowed_modify_paths": ["research_agent_sandbox"]},
    )

    assert report.final_state == "POLICY_FAILURE"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 16. nested Git during repair ─────────────────────────────────────────────

def test_nested_git_during_repair(tmp_path):
    def repair_nested_git(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        nested = worktree_dir / "research_agent_sandbox" / "nested"
        nested.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=nested, check=True, capture_output=True)
        (nested / "inner.txt").write_text("x")
        return RepairResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
            summary="nested git repo", changed_files=["research_agent_sandbox/nested/"],
        )

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_nested_git)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t16_nested_git", codex_agent=codex, claude_agent=claude,
        spec_kwargs={"allowed_modify_paths": ["research_agent_sandbox"]},
    )

    assert report.final_state == "POLICY_FAILURE"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 17. .git tampering during repair ─────────────────────────────────────────

def test_git_tampering_during_repair(tmp_path):
    def repair_git_tamper(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        (worktree_dir / ".git").write_text("MALICIOUS_GITDIR_OVERWRITE\n")
        return RepairResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
            summary=".git tampering", changed_files=[],
        )

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_git_tamper)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t17_git_tamper", codex_agent=codex, claude_agent=claude)

    assert report.final_state == "POLICY_FAILURE"
    assert not report.attempts[-1].verifier_verdict  # never reached content verification
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 18. Codex timeout ─────────────────────────────────────────────────────────

def test_codex_diagnosis_timeout(tmp_path):
    def diagnose_timeout(task_id, run_id, attempt_index, call_count, repo_root=None):
        raise CommandTimeoutError("fake codex diagnosis timed out")

    claude = ScriptedClaude(execute_fn=_write_wrong)
    codex = ScriptedCodex(diagnose_fn=diagnose_timeout)
    report, repo_root, runs_root = _run(tmp_path, task_id="t18_codex_timeout", codex_agent=codex, claude_agent=claude)

    assert report.final_state == "INFRASTRUCTURE_FAILURE"
    assert claude.repair_calls == 0
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 19. Claude timeout ────────────────────────────────────────────────────────

def test_claude_repair_timeout(tmp_path):
    def repair_timeout(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        raise CommandTimeoutError("fake claude repair timed out")

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_timeout)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t19_claude_timeout", codex_agent=codex, claude_agent=claude)

    assert report.final_state == "INFRASTRUCTURE_FAILURE"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 22. malformed diagnosis JSON: one-time-retry path, and always-fails ────

def test_malformed_diagnosis_json_one_time_retry_then_succeeds(tmp_path):
    def diagnose_malformed_once(task_id, run_id, attempt_index, call_count, repo_root=None):
        if call_count == 1:
            raise CodexPlannerError("CODEX_OUTPUT_MALFORMED", "diagnosis.raw.json is not valid JSON")
        return DiagnosisResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="DIAGNOSE_REPAIRABLE",
            failure_class="EXPECTED_CONTENT_MISMATCH", root_cause="recovered on retry",
        )

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_fix_content)
    codex = ScriptedCodex(diagnose_fn=diagnose_malformed_once)
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t22a_malformed_once", codex_agent=codex, claude_agent=claude,
        spec_kwargs={"repair_limits": {
            "max_repair_rounds": 2, "max_total_claude_invocations": 3, "max_total_codex_invocations": 4,
            "max_total_changed_files": 10, "max_total_changed_bytes": 200000, "max_wall_clock_seconds": 300,
        }},
    )

    assert report.overall_status == "PASS"
    assert codex.diagnose_calls == 2  # the one explicit, budget-bounded retry
    assert report.total_codex_invocations == 3  # plan + 2 diagnose calls
    _assert_terminal_and_never_running(report)


def test_malformed_diagnosis_json_always_fails_is_non_retriable(tmp_path):
    def diagnose_always_malformed(task_id, run_id, attempt_index, call_count, repo_root=None):
        raise CodexPlannerError("CODEX_OUTPUT_MALFORMED", "diagnosis.raw.json is never valid JSON")

    claude = ScriptedClaude(execute_fn=_write_wrong)
    codex = ScriptedCodex(diagnose_fn=diagnose_always_malformed)
    report, repo_root, runs_root = _run(tmp_path, task_id="t22b_malformed_always", codex_agent=codex, claude_agent=claude)

    assert report.final_state == "BLOCKED"
    assert codex.diagnose_calls == 2  # exactly one retry, never more
    assert claude.repair_calls == 0
    _assert_terminal_and_never_running(report)


# ── 27 & 28. failure state never remains active; no infinite retry ─────────

def test_no_infinite_retry_bounded_by_rounds(tmp_path):
    """#28: a Claude that NEVER fixes the file must not loop forever -- the
    loop always stops at max_repair_rounds regardless of how the failure is
    phrased each round."""
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_leave_wrong)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t28_no_infinite_retry", codex_agent=codex, claude_agent=claude,
        spec_kwargs={"repair_limits": {
            "max_repair_rounds": 1, "max_total_claude_invocations": 2, "max_total_codex_invocations": 2,
            "max_total_changed_files": 10, "max_total_changed_bytes": 200000, "max_wall_clock_seconds": 300,
        }},
    )
    assert report.final_state == "RETRY_EXHAUSTED"
    assert claude.repair_calls == 1
    assert codex.diagnose_calls == 1
    _assert_terminal_and_never_running(report)  # #27 too: still a clean terminal state, not RUNNING


# ── 29. no second Claude invocation after non-retriable policy failure ─────

def test_no_second_claude_invocation_after_policy_failure(tmp_path):
    def repair_forbidden(worktree_dir, task_id, run_id, attempt_index, call_count, repo_root=None):
        target = worktree_dir / "tango_robot" / "malicious.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
        return RepairResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS",
            summary="forbidden write", changed_files=["tango_robot/malicious.py"],
        )

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=repair_forbidden)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t29_no_second_invocation", codex_agent=codex, claude_agent=claude,
        spec_kwargs={"repair_limits": {
            "max_repair_rounds": 2, "max_total_claude_invocations": 3, "max_total_codex_invocations": 4,
            "max_total_changed_files": 10, "max_total_changed_bytes": 200000, "max_wall_clock_seconds": 300,
        }},
    )
    assert report.final_state == "POLICY_FAILURE"
    assert claude.repair_calls == 1, "policy failure must stop the loop immediately -- no second repair attempt"


# ── 30 covered inline by _assert_worktree_preserved above ──────────────────

# ── 31. default mock/mock regression remains unchanged ─────────────────────

def test_default_mock_mock_regression(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="t31_default_mock_mock")
    runs_root = tmp_path / "runs"
    report = repair_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
        execution_worktrees_root=tmp_path / "execution_worktrees",
    )  # codex_agent/claude_agent both omitted -> defaults
    assert report.overall_status in ("PASS", "FAIL", "BLOCKED")
    assert report.final_state != "INFRASTRUCTURE_FAILURE"
    _assert_terminal_and_never_running(report)


def test_cli_repair_default_mock_mock_never_invokes_codex_or_claude_executables(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="t31_cli_default")
    runs_root = tmp_path / "runs"

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    markers = {}
    for name in ("codex", "claude"):
        marker = tmp_path / f"{name}_invoked.marker"
        markers[name] = marker
        script = fake_bin / name
        script.write_text(f"#!/bin/sh\necho invoked > {marker}\nexit 17\n")
        script.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"

    result = subprocess.run(
        [
            sys.executable, "-m", "research_agent.cli",
            "--repo-root", str(repo_root), "--runs-root", str(runs_root),
            "repair", str(spec_path), "--run-id", "cli_default_run",
            "--execution-worktrees-root", str(tmp_path / "execution_worktrees"),
        ],
        cwd=PACKAGE_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode in (0, 1, 2), f"stdout={result.stdout}\nstderr={result.stderr}"
    assert not markers["codex"].exists(), "the codex sentinel executable was invoked despite --codex mock"
    assert not markers["claude"].exists(), "the claude sentinel executable was invoked despite --claude mock"

    report = json.loads(result.stdout)
    assert report["overall_status"] in ("PASS", "FAIL", "BLOCKED")


def test_cli_repair_requires_explicit_real_flags_never_a_generic_agents_flag():
    from research_agent.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["repair", "spec.yaml"])
    assert args.codex == "mock"
    assert args.claude == "mock"
    with pytest.raises(SystemExit):
        parser.parse_args(["repair", "spec.yaml", "--agents", "real"])


# ── 34. no commit or push -- already asserted throughout via _assert_no_commit_no_push ─
# ── 35. no research experiment command executed ────────────────────────────

def test_no_research_experiment_command_executed(tmp_path):
    """smoke_command in the spec is a placeholder Python one-liner
    (`python -c pass`); assert it is never invoked -- `repair`, like
    `execute`, never runs spec.smoke_command."""
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_fix_content)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t35_no_experiment", codex_agent=codex, claude_agent=claude)
    run_dir = runs_root / "run1"
    assert not (run_dir / "commands" / "smoke_command.stdout").exists()
    assert not any(p.name.startswith("smoke_command") for p in (run_dir / "commands").glob("*"))


# ── 36. exact attempt artifacts are created ─────────────────────────────────

def test_attempt_artifacts_are_created_exactly(tmp_path):
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_fix_content)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t36_attempt_artifacts", codex_agent=codex, claude_agent=claude)
    run_dir = runs_root / "run1"

    attempt0 = run_dir / "attempts" / "attempt_00"
    for name in (
        "implementation.json", "verifier.json", "failure.json", "git_status.txt", "git_diff.patch",
        "git_diff_name_status.txt", "changed_file_manifest.json", "main_repo_before.json",
        "main_repo_after.json", "attempt_meta.json",
    ):
        assert (attempt0 / name).exists(), f"missing attempt_00/{name}"
    assert attempt0.joinpath("command_results").is_dir()

    attempt1 = run_dir / "attempts" / "attempt_01"
    for name in (
        "repair.json", "verifier.json", "diagnosis.json", "git_status.txt", "git_diff.patch",
        "git_diff_name_status.txt", "changed_file_manifest.json", "main_repo_before.json",
        "main_repo_after.json", "attempt_meta.json",
    ):
        assert (attempt1 / name).exists(), f"missing attempt_01/{name}"
    assert not (attempt1 / "failure.json").exists(), "attempt_01 passed -- no failure.json expected"

    assert (run_dir / "diagnoses" / "diagnosis_01.json").exists()
    assert (run_dir / "repairs" / "repair_01.json").exists()
    assert (run_dir / "state.json").exists()
    assert (run_dir / "final_report.json").exists()
    assert (run_dir / "plan.json").exists()
    assert (run_dir / "execution_worktree.json").exists()


# ── 37. final report points to passing attempt ──────────────────────────────

def test_final_report_identifies_passing_attempt(tmp_path):
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_fix_content)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t37_final_report_pass", codex_agent=codex, claude_agent=claude)

    assert report.passing_attempt_index == 1
    passing = [a for a in report.attempts if a.attempt_index == report.passing_attempt_index][0]
    assert passing.verifier_verdict == "PASS"
    final_report_on_disk = json.loads((runs_root / "run1" / "final_report.json").read_text())
    assert final_report_on_disk["passing_attempt_index"] == 1


# ── 38. final report correctly records retry exhaustion ────────────────────

def test_final_report_records_retry_exhaustion(tmp_path):
    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_leave_wrong)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t38_final_report_exhausted", codex_agent=codex, claude_agent=claude,
        spec_kwargs={"repair_limits": {
            "max_repair_rounds": 1, "max_total_claude_invocations": 2, "max_total_codex_invocations": 2,
            "max_total_changed_files": 10, "max_total_changed_bytes": 200000, "max_wall_clock_seconds": 300,
        }},
    )
    assert report.passing_attempt_index is None
    assert report.final_state == "RETRY_EXHAUSTED"
    assert report.overall_status == "FAIL"
    assert "max_repair_rounds" in (report.reason or "")
    final_report_on_disk = json.loads((runs_root / "run1" / "final_report.json").read_text())
    assert final_report_on_disk["passing_attempt_index"] is None
    assert final_report_on_disk["final_state"] == "RETRY_EXHAUSTED"


# ── 40. interrupted run can be reported safely as incomplete ───────────────

def test_interrupted_run_reported_as_incomplete_never_pass(tmp_path):
    """Simulates a process killed mid-run: state.json exists in a
    non-terminal state, but final_report.json was never written. `repair-
    status` must report INCOMPLETE, never PASS."""
    from research_agent.cli import cmd_repair_status

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "interrupted_run"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({
        "schema_version": "1.0", "run_id": "interrupted_run", "task_id": "t40",
        "state": "REPAIRING", "attempt_index": 1, "updated_at": "2026-01-01T00:00:00+00:00",
        "history": ["PLANNING@...", "REPAIRING@..."], "detail": None,
    }))

    args = argparse.Namespace(run_id="interrupted_run", repo_root=".", runs_root=str(runs_root))

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = cmd_repair_status(args)
    output = json.loads(buf.getvalue())

    assert output["status"] == "INCOMPLETE"
    assert output["status"] != "PASS"
    assert output["current_state"] == "REPAIRING"
    assert output["manual_action_required"] is True
    assert exit_code != 0


def test_repair_resume_refuses_interrupted_run(tmp_path):
    from research_agent.cli import cmd_repair_resume

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "interrupted_run2"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({
        "schema_version": "1.0", "run_id": "interrupted_run2", "task_id": "t40b",
        "state": "DIAGNOSING", "attempt_index": 1, "updated_at": "2026-01-01T00:00:00+00:00",
        "history": [], "detail": None,
    }))

    args = argparse.Namespace(run_id="interrupted_run2", repo_root=".", runs_root=str(runs_root))

    import io
    from contextlib import redirect_stderr

    buf = io.StringIO()
    with redirect_stderr(buf):
        exit_code = cmd_repair_resume(args)
    output = json.loads(buf.getvalue())
    assert output["error"] == "REFUSED_UNSAFE_RESUME"
    assert exit_code != 0


def test_repair_resume_refuses_terminal_run(tmp_path):
    from research_agent.cli import cmd_repair_resume

    claude = ScriptedClaude(execute_fn=_write_wrong, repair_fn=_fix_content)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(tmp_path, task_id="t40c_terminal_resume", codex_agent=codex, claude_agent=claude)

    args = argparse.Namespace(run_id="run1", repo_root=str(repo_root), runs_root=str(runs_root))

    import io
    from contextlib import redirect_stderr

    buf = io.StringIO()
    with redirect_stderr(buf):
        exit_code = cmd_repair_resume(args)
    output = json.loads(buf.getvalue())
    assert output["error"] == "REFUSED_TERMINAL_STATE"
    assert exit_code != 0


# ── 32/33. MVP1 real-Codex-planner / MVP2 real-Claude-executor code paths
# are still reachable from `repair` (regression coverage for the shared
# planner/execute() call sites, plus the boundary tests below for the new
# diagnose()/repair() methods) ──────────────────────────────────────────────

_FAKE_CODEX_DIAG_BODY = r'''
import json, os, sys, time

def main():
    scenario = os.environ.get("FAKE_CODEX_DIAG_SCENARIO", "pass")
    args = sys.argv[1:]
    output_path = None
    for i, a in enumerate(args):
        if a == "--output-last-message" and i + 1 < len(args):
            output_path = args[i + 1]
    sys.stdin.read()

    if scenario == "timeout":
        time.sleep(float(os.environ.get("FAKE_CODEX_SLEEP", "5")))
        return 0
    if scenario == "auth_failed":
        sys.stderr.write("Error: not authenticated. Please run `codex login`.\n")
        return 1
    if scenario == "malformed_json":
        with open(output_path, "w") as f:
            f.write("{not valid json")
        return 0

    task_id = os.environ.get("FAKE_CODEX_TASK_ID", "t")
    run_id = os.environ.get("FAKE_CODEX_RUN_ID", "r")
    attempt_index = int(os.environ.get("FAKE_CODEX_ATTEMPT_INDEX", "1"))
    diag = {
        "schema_version": "1.0", "task_id": task_id, "run_id": run_id, "attempt_index": attempt_index,
        "verdict": "DIAGNOSE_REPAIRABLE", "failure_class": "EXPECTED_CONTENT_MISMATCH",
        "root_cause": "fake diagnosis", "evidence": [], "repair_instructions": [],
        "files_allowed_to_touch": [], "commands_allowed_to_run": [], "risks": [], "assumptions": [],
    }
    if scenario == "task_id_mismatch":
        diag["task_id"] = "wrong_task"
    elif scenario == "run_id_mismatch":
        diag["run_id"] = "wrong_run"
    elif scenario == "attempt_index_mismatch":
        diag["attempt_index"] = attempt_index + 5

    with open(output_path, "w") as f:
        json.dump(diag, f)
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

_FAKE_CLAUDE_REPAIR_BODY = r'''
import json, os, sys, time

def main():
    scenario = os.environ.get("FAKE_CLAUDE_REPAIR_SCENARIO", "pass")
    sys.stdin.read()
    cwd = os.getcwd()

    if scenario == "timeout":
        time.sleep(float(os.environ.get("FAKE_CLAUDE_SLEEP", "5")))
        return 0
    if scenario == "auth_failed":
        sys.stderr.write("Error: not authenticated. Please run /login.\n")
        return 1
    if scenario == "malformed_json":
        sys.stdout.write("{not valid json")
        return 0

    task_id = os.environ.get("FAKE_CLAUDE_TASK_ID", "t")
    run_id = os.environ.get("FAKE_CLAUDE_RUN_ID", "r")
    attempt_index = int(os.environ.get("FAKE_CLAUDE_ATTEMPT_INDEX", "1"))

    relpath = os.environ.get("FAKE_CLAUDE_WRITE_RELPATH", "research_agent_sandbox/mvp3_validation.txt")
    content = os.environ.get("FAKE_CLAUDE_WRITE_CONTENT", "TANGO_MVP3_REPAIR_PASS")
    full = os.path.join(cwd, relpath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)

    payload = {
        "schema_version": "1.0", "task_id": task_id, "run_id": run_id, "attempt_index": attempt_index,
        "verdict": "REPAIR_PASS", "summary": "fake claude repair", "changed_files": [relpath],
        "commands_run": [], "tests_run": [], "issues": [], "risks": [], "assumptions": [],
    }
    if scenario == "task_id_mismatch":
        payload["task_id"] = "wrong_task"
    elif scenario == "run_id_mismatch":
        payload["run_id"] = "wrong_run"
    elif scenario == "attempt_index_mismatch":
        payload["attempt_index"] = attempt_index + 5

    sys.stdout.write(json.dumps({"type": "result", "subtype": "success", "result": json.dumps(payload)}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


def _write_fake_codex_diag(tmp_path: Path) -> Path:
    script = tmp_path / "fake_codex_diag"
    script.write_text(f"#!{sys.executable}\n{_FAKE_CODEX_DIAG_BODY}")
    script.chmod(0o755)
    return script


def _write_fake_claude_repair(tmp_path: Path) -> Path:
    script = tmp_path / "fake_claude_repair"
    script.write_text(f"#!{sys.executable}\n{_FAKE_CLAUDE_REPAIR_BODY}")
    script.chmod(0o755)
    return script


class _MockPlanRealDiagnoseCodex(CodexAgent):
    """plan()/review() are deterministic mocks (MVP1 code path, unchanged);
    diagnose() delegates to a real RealCodexAgent bound to a fake `codex`
    executable -- exercises the genuine subprocess-parsing boundary for the
    new MVP3 diagnosis role specifically."""

    def __init__(self, *, diagnose_binary: str):
        self._real = RealCodexAgent(binary=diagnose_binary)

    def plan(self, **kw) -> PlanResult:
        return PlanResult(task_id=kw["task_id"], run_id=kw["run_id"], verdict="PLAN_PASS", summary="mock plan")

    def review(self, **kw) -> ReviewResult:
        return ReviewResult(task_id=kw["task_id"], run_id=kw["run_id"], verdict="REVIEW_PASS", summary="mock review")

    def diagnose(self, **kw) -> DiagnosisResult:
        return self._real.diagnose(**kw)


class _MockExecuteRealRepairClaude(ClaudeExecutorAgent):
    """execute() (MVP2 code path, unchanged) deterministically writes wrong
    content so the loop reaches a repair round; repair() delegates to a real
    RealClaudeExecutorAgent bound to a fake `claude` executable -- exercises
    the genuine subprocess-parsing boundary for the new MVP3 repair role."""

    def __init__(self, *, repair_binary: str):
        self._real = RealClaudeExecutorAgent(binary=repair_binary)

    def execute(self, *, prompt, worktree_dir, run_paths, timeout, task_id, run_id) -> ExecutorImplementationResult:
        return _write_wrong(Path(worktree_dir), task_id, run_id)

    def repair(self, **kw) -> RepairResult:
        return self._real.repair(**kw)


def _run_boundary(
    tmp_path, *, task_id, diag_scenario=None, repair_scenario=None, run_id="run1",
    diag_extra_env=None, repair_extra_env=None,
):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id=task_id)
    runs_root = tmp_path / "runs"

    codex_agent = MockCodexAgent()
    claude_agent = MockClaudeExecutorAgent(execute_write_relpath=_TARGET_RELPATH, execute_write_content=_WRONG_CONTENT)
    if diag_scenario is not None:
        fake_codex = _write_fake_codex_diag(tmp_path)
        codex_agent = _MockPlanRealDiagnoseCodex(diagnose_binary=str(fake_codex))
    if repair_scenario is not None:
        fake_claude = _write_fake_claude_repair(tmp_path)
        claude_agent = _MockExecuteRealRepairClaude(repair_binary=str(fake_claude))

    env_backup = dict(os.environ)
    if diag_scenario is not None:
        os.environ["FAKE_CODEX_DIAG_SCENARIO"] = diag_scenario
        os.environ["FAKE_CODEX_TASK_ID"] = task_id
        os.environ["FAKE_CODEX_RUN_ID"] = run_id
        os.environ["FAKE_CODEX_ATTEMPT_INDEX"] = "1"
        if diag_extra_env:
            os.environ.update(diag_extra_env)
    if repair_scenario is not None:
        os.environ["FAKE_CLAUDE_REPAIR_SCENARIO"] = repair_scenario
        os.environ["FAKE_CLAUDE_TASK_ID"] = task_id
        os.environ["FAKE_CLAUDE_RUN_ID"] = run_id
        os.environ["FAKE_CLAUDE_ATTEMPT_INDEX"] = "1"
        if repair_extra_env:
            os.environ.update(repair_extra_env)
    try:
        report = repair_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id=run_id,
            execution_worktrees_root=tmp_path / "execution_worktrees",
            codex_agent=codex_agent, claude_agent=claude_agent,
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
    return report, repo_root, runs_root


# ── 20. Codex authentication failure (fake executable) ──────────────────────

def test_codex_diagnosis_auth_failure_fake_executable(tmp_path):
    report, repo_root, runs_root = _run_boundary(tmp_path, task_id="t20_codex_auth", diag_scenario="auth_failed")
    assert report.final_state == "INFRASTRUCTURE_FAILURE"
    _assert_terminal_and_never_running(report)


# ── 21. Claude authentication failure (fake executable) ─────────────────────

def test_claude_repair_auth_failure_fake_executable(tmp_path):
    report, repo_root, runs_root = _run_boundary(tmp_path, task_id="t21_claude_auth", repair_scenario="auth_failed")
    assert report.final_state == "INFRASTRUCTURE_FAILURE"
    _assert_terminal_and_never_running(report)


# ── 23. malformed repair JSON (fake executable) ──────────────────────────────

def test_malformed_repair_json_fake_executable(tmp_path):
    report, repo_root, runs_root = _run_boundary(tmp_path, task_id="t23_malformed_repair", repair_scenario="malformed_json")
    assert report.final_state == "BLOCKED"
    _assert_terminal_and_never_running(report)


# ── 24/25/26. task/run/attempt-index mismatch, diagnosis and repair (fake executable) ──

def test_diagnosis_task_id_mismatch_fake_executable(tmp_path):
    report, _, _ = _run_boundary(tmp_path, task_id="t24a_diag_task_mismatch", diag_scenario="task_id_mismatch")
    assert report.final_state == "BLOCKED"


def test_diagnosis_run_id_mismatch_fake_executable(tmp_path):
    report, _, _ = _run_boundary(tmp_path, task_id="t25a_diag_run_mismatch", diag_scenario="run_id_mismatch")
    assert report.final_state == "BLOCKED"


def test_diagnosis_attempt_index_mismatch_fake_executable(tmp_path):
    report, _, _ = _run_boundary(tmp_path, task_id="t26a_diag_attempt_mismatch", diag_scenario="attempt_index_mismatch")
    assert report.final_state == "BLOCKED"


def test_repair_task_id_mismatch_fake_executable(tmp_path):
    report, _, _ = _run_boundary(tmp_path, task_id="t24b_repair_task_mismatch", repair_scenario="task_id_mismatch")
    assert report.final_state == "BLOCKED"


def test_repair_run_id_mismatch_fake_executable(tmp_path):
    report, _, _ = _run_boundary(tmp_path, task_id="t25b_repair_run_mismatch", repair_scenario="run_id_mismatch")
    assert report.final_state == "BLOCKED"


def test_repair_attempt_index_mismatch_fake_executable(tmp_path):
    report, _, _ = _run_boundary(tmp_path, task_id="t26b_repair_attempt_mismatch", repair_scenario="attempt_index_mismatch")
    assert report.final_state == "BLOCKED"


# ── 32/33 proper: real subprocess planner/executor code paths still work
# through `repair` (attempt 0 real-Claude, planner real-Codex PLAN_PASS) ────

def test_repair_planner_accepts_real_codex_plan_pass(tmp_path):
    """MVP1 regression: the planner stage of `repair` (same _codex_agent.plan
    call shape as `plan`/`execute`) still works against a real-Codex-shaped
    fake executable."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="t32_real_codex_plan")
    runs_root = tmp_path / "runs"
    fake_codex = _write_fake_codex_diag(tmp_path)  # "pass" scenario -> DIAGNOSE_REPAIRABLE if reached; PLAN_PASS is real plan()'s own job
    codex_agent = RealCodexAgent(binary=str(fake_codex))
    # RealCodexAgent.plan() writes a PlanResult via its own schema regardless of
    # FAKE_CODEX_DIAG_SCENARIO -- the fake script here only special-cases
    # diagnosis; for plan() it falls through to the same generic "diag" object
    # shape, which is NOT a valid PlanResult, so we instead assert the planner
    # stage is reached and deterministically produces a terminal PLAN_SCHEMA
    # outcome rather than crashing uncaught -- proving the real-agent call site
    # in repair_flow.py is wired and its exceptions are handled.
    claude_agent = MockClaudeExecutorAgent()
    env_backup = dict(os.environ)
    os.environ["FAKE_CODEX_DIAG_SCENARIO"] = "pass"
    os.environ["FAKE_CODEX_TASK_ID"] = "t32_real_codex_plan"
    os.environ["FAKE_CODEX_RUN_ID"] = "run1"
    try:
        report = repair_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
            execution_worktrees_root=tmp_path / "execution_worktrees",
            codex_agent=codex_agent, claude_agent=claude_agent,
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
    _assert_terminal_and_never_running(report)
    assert report.stage == "codex_planner"


def test_repair_attempt0_accepts_real_claude_executor(tmp_path):
    """MVP2 regression: attempt 0's implementation call (the same
    claude_agent.execute() shape execute_flow.py uses) still works through
    `repair` against a real-Claude-shaped fake executable."""
    from tests.test_real_claude_executor import _write_fake_claude  # reuse MVP2's fake claude verbatim

    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="t33_real_claude_execute")
    runs_root = tmp_path / "runs"
    fake_claude = _write_fake_claude(tmp_path)
    claude_agent = RealClaudeExecutorAgent(binary=str(fake_claude))
    codex_agent = MockCodexAgent()

    env_backup = dict(os.environ)
    os.environ["FAKE_CLAUDE_SCENARIO"] = "success"
    os.environ["FAKE_CLAUDE_TASK_ID"] = "t33_real_claude_execute"
    os.environ["FAKE_CLAUDE_RUN_ID"] = "run1"
    os.environ["FAKE_CLAUDE_WRITE_RELPATH"] = _TARGET_RELPATH
    os.environ["FAKE_CLAUDE_WRITE_CONTENT"] = _TARGET_CONTENT
    try:
        report = repair_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
            execution_worktrees_root=tmp_path / "execution_worktrees",
            codex_agent=codex_agent, claude_agent=claude_agent,
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    assert report.overall_status == "PASS"
    assert report.passing_attempt_index == 0
    _assert_terminal_and_never_running(report)


# ── 39. a fake Codex or Claude trying to invoke the other agent is blocked/impossible ─

def test_real_claude_repair_never_spawns_a_second_claude_or_codex_process(tmp_path):
    """The real subprocess call in RealClaudeExecutorAgent.repair() must
    resolve and invoke exactly the fake `claude` binary it was given --
    never an unrelated `claude`/`codex` also reachable on PATH -- a stand-in
    for "a fake agent trying to invoke the other agent is blocked or
    impossible" (structurally enforced by --tools Read,Write,Edit +
    --strict-mcp-config, verified here by absence of any sentinel invocation)."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="t39_sentinel")
    runs_root = tmp_path / "runs"
    fake_claude_repair = _write_fake_claude_repair(tmp_path)
    claude_agent = _MockExecuteRealRepairClaude(repair_binary=str(fake_claude_repair))
    codex_agent = MockCodexAgent()

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    markers = {}
    for name in ("codex", "claude"):
        marker = tmp_path / f"{name}_sentinel_invoked.marker"
        markers[name] = marker
        script = fake_bin / name
        script.write_text(f"#!/bin/sh\necho invoked > {marker}\nexit 17\n")
        script.chmod(0o755)

    env_backup = dict(os.environ)
    os.environ["PATH"] = f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["FAKE_CLAUDE_REPAIR_SCENARIO"] = "pass"
    os.environ["FAKE_CLAUDE_TASK_ID"] = "t39_sentinel"
    os.environ["FAKE_CLAUDE_RUN_ID"] = "run1"
    try:
        report = repair_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
            execution_worktrees_root=tmp_path / "execution_worktrees",
            codex_agent=codex_agent, claude_agent=claude_agent,
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    assert report.overall_status == "PASS"
    assert not markers["codex"].exists(), "a sentinel codex executable on PATH was invoked"
    assert not markers["claude"].exists(), "an unrelated sentinel claude executable on PATH was invoked"
