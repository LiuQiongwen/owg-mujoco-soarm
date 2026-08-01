"""MVP4 restricted-execution flow integration tests: research_agent
.execution_flow.run_experiment_flow and the `run-experiment` /
`experiment-status` / `experiment-cleanup` CLI commands.

No test here runs real research code, requires a GPU, or touches a robot or
the network -- every "experiment command" is a fixed, deterministic
`python -c` one-liner authored inline below. Agents are either the real
Mock* classes (offline, deterministic) or small in-process "injected
adapter" fakes duck-typing the same execute/repair/plan/diagnose contract
the real agents use (the same pattern tests/test_repair_flow.py uses).

Covers the MVP4 task contract's "Fake execution tests" checklist (items not
already covered by tests/test_experiment_commands_policy.py,
tests/test_environment_policy.py, tests/test_artifact_policy.py,
tests/test_restricted_subprocess.py, tests/test_metric_verifier.py):
  1.  no --execute means no subprocess runs
  2.  approved deterministic command passes
  11. confirmatory mode blocked
  12. timeout (flow level)
  13. nonzero exit (flow level)
  14/15. stdout/stderr captured under the run directory
  17. artifact created in allowed directory (flow level)
  18/19. artifact path/symlink escape (flow level)
  20. too many artifact files (flow level)
  21. artifact byte limit exceeded (flow level)
  22/23/24/26. malformed metrics / missing metric / wrong type / missing
      artifact (flow level)
  27. main worktree mutation detected
  28. execution worktree mutation outside allowed paths
  33. retry after repair succeeds
  34. execution retry budget exhausted
  35. non-retriable policy failure does not repair
  36. final report records passing attempt
  37. final report records failed attempts
  38. state never remains active
  39/40. no commit, no push
  44. no research dataset/output modified (main worktree unchanged)
  45. default mock/mock regression
  46/47/48. MVP1/MVP2/MVP3 regression
  49. cleanup/status understands MVP4 final_report
  50. interrupted execution reported safely
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.agents.claude_executor import ClaudeExecutorAgent, MockClaudeExecutorAgent
from research_agent.agents.codex import CodexAgent, MockCodexAgent
from research_agent.execution_flow import ConfirmatoryRejected, run_experiment_flow
from research_agent.models import (
    EXECUTION_TERMINAL_STATES,
    DiagnosisResult,
    ExecutorImplementationResult,
    PlanResult,
    RepairResult,
    ReviewResult,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

WRITE_OK_SCRIPT = (
    'import json, os\n'
    'open(os.path.join(os.environ["RESEARCH_AGENT_ARTIFACTS_DIR"], "metrics.json"), "w")'
    '.write(json.dumps({"ok": True, "value": 1.0}))\n'
)
EXIT_NONZERO_SCRIPT = 'import sys; sys.exit(7)'
MALFORMED_JSON_SCRIPT = (
    'import os\n'
    'open(os.path.join(os.environ["RESEARCH_AGENT_ARTIFACTS_DIR"], "metrics.json"), "w")'
    '.write("{not valid json")\n'
)
MISSING_METRIC_SCRIPT = (
    'import json, os\n'
    'open(os.path.join(os.environ["RESEARCH_AGENT_ARTIFACTS_DIR"], "metrics.json"), "w")'
    '.write(json.dumps({"other": 1}))\n'
)
WRONG_TYPE_SCRIPT = (
    'import json, os\n'
    'open(os.path.join(os.environ["RESEARCH_AGENT_ARTIFACTS_DIR"], "metrics.json"), "w")'
    '.write(json.dumps({"ok": "not-a-bool"}))\n'
)
NO_ARTIFACT_SCRIPT = 'pass\n'
SLEEP_SCRIPT = 'import time; time.sleep(30)\n'
SYMLINK_ESCAPE_SCRIPT = (
    'import os\n'
    'os.symlink("/etc/passwd", os.path.join(os.environ["RESEARCH_AGENT_ARTIFACTS_DIR"], "escape"))\n'
)
TOO_MANY_FILES_SCRIPT = (
    'import os\n'
    'd = os.environ["RESEARCH_AGENT_ARTIFACTS_DIR"]\n'
    'for i in range(5):\n'
    '    open(os.path.join(d, f"f{i}.txt"), "w").write("x")\n'
)
BIG_FILE_SCRIPT = (
    'import os\n'
    'open(os.path.join(os.environ["RESEARCH_AGENT_ARTIFACTS_DIR"], "big.bin"), "w").write("0" * 5000)\n'
)
FLAG_FILE_SCRIPT = (
    'import json, os, sys\n'
    'from pathlib import Path\n'
    'flag = Path("research_agent_sandbox/flag.txt")\n'
    'ok = flag.exists() and flag.read_text().strip() == "READY"\n'
    'open(os.path.join(os.environ["RESEARCH_AGENT_ARTIFACTS_DIR"], "metrics.json"), "w")'
    '.write(json.dumps({"ok": ok}))\n'
    'sys.exit(0 if ok else 7)\n'
)


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
    run("git", "add", ".")
    run("git", "commit", "-q", "-m", "init")
    return repo


def _write_spec(
    tmp_path: Path, *, task_id: str, script: str, required_metrics=None, required_artifacts=None,
    allowed_output_paths=None, limits=None, needs_implementation: bool = False,
    working_directory_policy: str = "isolated_run_directory", confirmatory_indicator: bool = False,
) -> Path:
    execution: dict = {
        "execution_mode": "restricted",
        "working_directory_policy": working_directory_policy,
        "approved_commands": [[sys.executable, "-c", script]],
        "required_artifacts": required_artifacts if required_artifacts is not None else ["metrics.json"],
        "required_metrics": required_metrics if required_metrics is not None else [{"key": "ok", "check": "bool_equals", "value": True}],
        "limits": limits or {
            "max_commands": 1, "max_execution_attempts": 1, "max_wall_clock_seconds": 30,
            "per_command_timeout_seconds": 5, "max_repair_rounds": 1,
            "max_total_codex_invocations": 3, "max_total_claude_invocations": 3,
        },
    }
    if allowed_output_paths is not None:
        execution["allowed_output_paths"] = allowed_output_paths

    spec: dict = {
        "task_id": task_id,
        "goal": "confirmatory marker" if confirmatory_indicator else "mvp4 execution-flow fake-agent test",
        "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": [sys.executable, "-c", "pass"],
        "seeds": [0],
        "timeouts": {"planner_seconds": 20, "executor_seconds": 20, "smoke_seconds": 20, "verifier_seconds": 20, "reviewer_seconds": 20},
        "max_run_count": 10,
        "execution": execution,
    }
    if needs_implementation:
        spec["allowed_modify_paths"] = ["research_agent_sandbox/flag.txt"]
        spec["allowed_executor_commands"] = [[sys.executable, "-m", "compileall"]]

    path = tmp_path / f"{task_id}.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


class ScriptedClaude(ClaudeExecutorAgent):
    def __init__(self, *, execute_fn=None, repair_fn=None):
        self._execute_fn = execute_fn
        self._repair_fn = repair_fn
        self.execute_calls = 0
        self.repair_calls = 0

    def execute(self, *, prompt, worktree_dir, run_paths, timeout, task_id, run_id) -> ExecutorImplementationResult:
        self.execute_calls += 1
        if self._execute_fn is None:
            return ExecutorImplementationResult(task_id=task_id, run_id=run_id, verdict="IMPLEMENTATION_PASS", summary="noop")
        return self._execute_fn(worktree_dir=Path(worktree_dir), task_id=task_id, run_id=run_id)

    def repair(self, *, prompt, worktree_dir, run_paths, timeout, task_id, run_id, attempt_index) -> RepairResult:
        self.repair_calls += 1
        if self._repair_fn is None:
            return RepairResult(task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS", summary="noop")
        return self._repair_fn(worktree_dir=Path(worktree_dir), task_id=task_id, run_id=run_id, attempt_index=attempt_index)


class ScriptedCodex(CodexAgent):
    def __init__(self, *, diagnose_fn=None, repo_root: Optional[Path] = None):
        self._diagnose_fn = diagnose_fn
        self.repo_root = repo_root
        self.diagnose_calls = 0

    def plan(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> PlanResult:
        return PlanResult(task_id=task_id, run_id=run_id, verdict="PLAN_PASS", summary="scripted plan")

    def review(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> ReviewResult:
        return ReviewResult(task_id=task_id, run_id=run_id, verdict="REVIEW_PASS", summary="scripted review")

    def diagnose(self, *, prompt, run_dir, cwd, timeout, task_id, run_id, attempt_index) -> DiagnosisResult:
        self.diagnose_calls += 1
        if self._diagnose_fn is None:
            return DiagnosisResult(
                task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="DIAGNOSE_REPAIRABLE",
                failure_class="EXECUTION_NONZERO_EXIT", root_cause="scripted diagnosis",
                files_allowed_to_touch=["research_agent_sandbox/flag.txt"],
            )
        return self._diagnose_fn(task_id=task_id, run_id=run_id, attempt_index=attempt_index, repo_root=self.repo_root)


def _write_ready_flag(worktree_dir: Path, task_id, run_id, attempt_index=None) -> RepairResult:
    target = worktree_dir / "research_agent_sandbox" / "flag.txt"
    target.write_text("READY")
    kwargs = dict(task_id=task_id, run_id=run_id, verdict="REPAIR_PASS", summary="wrote flag", changed_files=["research_agent_sandbox/flag.txt"])
    if attempt_index is not None:
        kwargs["attempt_index"] = attempt_index
    return RepairResult(**kwargs)


def _assert_terminal_and_never_running(report) -> None:
    assert report.final_state in EXECUTION_TERMINAL_STATES
    assert report.overall_status in ("PASS", "FAIL", "BLOCKED")


def _assert_no_commit_no_push(report, repo_root: Path) -> None:
    if report.execution_worktree is not None:
        worktree_path = Path(report.execution_worktree.worktree_path)
        log = subprocess.run(
            ["git", "-C", str(worktree_path), "log", "--oneline"], capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        assert len(log) == 1, f"expected exactly the base commit, got: {log}"
    main_log = subprocess.run(
        ["git", "-C", str(repo_root), "log", "--oneline"], capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert len(main_log) == 1, f"main repo history must be untouched: {main_log}"


def _run(
    tmp_path, *, task_id, script, execute=True, codex_agent=None, claude_agent=None, run_id="run1",
    repo_root: Optional[Path] = None, spec_kwargs: Optional[dict] = None,
):
    repo_root = repo_root or _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id=task_id, script=script, **(spec_kwargs or {}))
    runs_root = tmp_path / "runs"
    report = run_experiment_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id=run_id,
        execution_worktrees_root=tmp_path / "ewt", codex_agent=codex_agent, claude_agent=claude_agent, execute=execute,
    )
    return report, repo_root, runs_root


# ── 1. no --execute means no subprocess runs ────────────────────────────────

def test_no_execute_means_no_subprocess_runs(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t1_no_execute", script=WRITE_OK_SCRIPT, execute=False)
    assert report.final_state == "EXECUTION_NOT_REQUESTED"
    assert report.execution_attempts == []
    run_dir = runs_root / "run1"
    assert not (run_dir / "execution").exists() or not any((run_dir / "execution").iterdir())
    _assert_terminal_and_never_running(report)


# ── 2. approved deterministic command passes ────────────────────────────────

def test_approved_deterministic_command_passes(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t2_pass", script=WRITE_OK_SCRIPT)
    assert report.final_state == "PASS"
    assert report.passing_attempt_index == 0
    assert report.metrics["metrics.json"]["ok"] is True
    _assert_terminal_and_never_running(report)
    _assert_no_commit_no_push(report, repo_root)
    assert report.main_worktree_unchanged is True


# ── 11. confirmatory mode blocked ────────────────────────────────────────────

def test_confirmatory_execution_mode_rejected_at_spec_load(tmp_path):
    from pydantic import ValidationError

    from research_agent.tasks import experiment as experiment_tasks

    spec_path = tmp_path / "confirmatory.yaml"
    spec_path.write_text(yaml.safe_dump({
        "task_id": "t11_confirmatory", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": [sys.executable, "-c", "pass"],
        "execution": {"execution_mode": "confirmatory", "approved_commands": [[sys.executable, "-c", "pass"]]},
    }))
    with pytest.raises(ValidationError) as exc:
        experiment_tasks.load_spec(spec_path)
    assert "CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL" in str(exc.value)


def test_confirmatory_true_flag_rejected_at_spec_load(tmp_path):
    from pydantic import ValidationError

    from research_agent.tasks import experiment as experiment_tasks

    spec_path = tmp_path / "confirmatory2.yaml"
    spec_path.write_text(yaml.safe_dump({
        "task_id": "t11b_confirmatory", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": [sys.executable, "-c", "pass"],
        "execution": {"confirmatory": True, "approved_commands": [[sys.executable, "-c", "pass"]]},
    }))
    with pytest.raises(ValidationError) as exc:
        experiment_tasks.load_spec(spec_path)
    assert "CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL" in str(exc.value)


def test_confirmatory_task_id_marker_rejected_by_flow(tmp_path):
    """Defense in depth: even a structurally 'restricted' spec is rejected
    if its task_id/goal names 'paper_final'/'final_result'/'confirmatory'."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="paper_final_results_run", script=WRITE_OK_SCRIPT)
    with pytest.raises(ConfirmatoryRejected):
        run_experiment_flow(
            spec_path, repo_root=repo_root, runs_root=tmp_path / "runs", run_id="run1",
            execution_worktrees_root=tmp_path / "ewt", execute=True,
        )


# ── 12/13. timeout / nonzero exit (flow level) ──────────────────────────────

def test_timeout_without_repairable_scope_is_execution_failed(tmp_path):
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t12_timeout", script=SLEEP_SCRIPT,
        spec_kwargs={"limits": {"max_commands": 1, "max_execution_attempts": 1, "max_wall_clock_seconds": 10, "per_command_timeout_seconds": 1}},
    )
    assert report.final_state in ("EXECUTION_FAILED", "RETRY_EXHAUSTED")
    assert report.execution_attempts[0].commands[0].timed_out is True
    _assert_terminal_and_never_running(report)


def test_nonzero_exit_without_implementation_scope_is_execution_failed(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t13_nonzero", script=EXIT_NONZERO_SCRIPT)
    assert report.final_state == "EXECUTION_FAILED"
    assert report.execution_attempts[0].failure_class == "EXECUTION_NONZERO_EXIT"
    _assert_terminal_and_never_running(report)


# ── 14/15. stdout/stderr captured under the run directory ──────────────────

def test_stdout_and_stderr_captured_on_disk(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t14_capture", script=WRITE_OK_SCRIPT)
    command_dir = runs_root / "run1" / "execution" / "attempt_00" / "command_00"
    assert (command_dir / "stdout").exists()
    assert (command_dir / "stderr").exists()
    assert (command_dir / "exit_code").read_text().strip() == "0"


# ── 17/18/19. artifact created / path escape / symlink escape ──────────────

def test_artifact_created_in_allowed_directory_recorded_in_manifest(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t17_artifact_ok", script=WRITE_OK_SCRIPT)
    assert any(a.relative_path == "metrics.json" for a in report.artifact_manifest)


def test_artifact_symlink_escape_blocks_run(tmp_path):
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t19_symlink_escape", script=SYMLINK_ESCAPE_SCRIPT, spec_kwargs={"required_metrics": [], "required_artifacts": []},
    )
    assert report.final_state == "POLICY_FAILURE"
    assert report.execution_attempts[-1].failure_class == "ARTIFACT_POLICY_FAILURE"
    _assert_terminal_and_never_running(report)


# ── 20. too many artifact files ─────────────────────────────────────────────

def test_too_many_artifact_files_blocks_run(tmp_path):
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t20_too_many_files", script=TOO_MANY_FILES_SCRIPT,
        spec_kwargs={"required_metrics": [], "required_artifacts": [], "limits": {
            "max_commands": 1, "max_execution_attempts": 1, "max_wall_clock_seconds": 30,
            "per_command_timeout_seconds": 5, "max_artifact_files": 2,
        }},
    )
    assert report.final_state == "POLICY_FAILURE"
    assert report.execution_attempts[-1].failure_class == "ARTIFACT_POLICY_FAILURE"


# ── 21. artifact byte limit exceeded ────────────────────────────────────────

def test_artifact_byte_limit_exceeded_blocks_run(tmp_path):
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t21_byte_limit", script=BIG_FILE_SCRIPT,
        spec_kwargs={"required_metrics": [], "required_artifacts": [], "limits": {
            "max_commands": 1, "max_execution_attempts": 1, "max_wall_clock_seconds": 30,
            "per_command_timeout_seconds": 5, "max_artifact_file_bytes": 100,
        }},
    )
    assert report.final_state == "POLICY_FAILURE"
    assert report.execution_attempts[-1].failure_class == "ARTIFACT_POLICY_FAILURE"


# ── 22/23/24/26. malformed / missing metric / wrong type / missing artifact ─

def test_malformed_metrics_json_without_repair_scope_is_verification_failed(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t22_malformed", script=MALFORMED_JSON_SCRIPT)
    assert report.final_state == "VERIFICATION_FAILED"
    _assert_terminal_and_never_running(report)


def test_missing_metric_key_is_verification_failed(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t23_missing_metric", script=MISSING_METRIC_SCRIPT)
    assert report.final_state == "VERIFICATION_FAILED"


def test_wrong_metric_type_is_verification_failed(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t24_wrong_type", script=WRONG_TYPE_SCRIPT)
    assert report.final_state == "VERIFICATION_FAILED"


def test_missing_required_artifact_is_verification_failed(tmp_path):
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t26_missing_artifact", script=NO_ARTIFACT_SCRIPT,
        spec_kwargs={"required_artifacts": ["metrics.json"], "required_metrics": []},
    )
    assert report.final_state == "VERIFICATION_FAILED"
    assert any("metrics.json" in i for i in report.verifier.issues)


# ── 27. main worktree mutation detected ─────────────────────────────────────

def test_main_worktree_mutation_during_diagnosis_is_policy_failure(tmp_path):
    def diagnose_mutates_main(*, task_id, run_id, attempt_index, repo_root):
        (repo_root / "mutated_by_diagnosis.txt").write_text("x")
        return DiagnosisResult(
            task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="DIAGNOSE_REPAIRABLE",
            failure_class="EXECUTION_NONZERO_EXIT", root_cause="x", files_allowed_to_touch=["research_agent_sandbox/flag.txt"],
        )

    repo_root = _init_repo(tmp_path)
    claude = ScriptedClaude()
    codex = ScriptedCodex(diagnose_fn=diagnose_mutates_main, repo_root=repo_root)
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t27_main_mutation", script=FLAG_FILE_SCRIPT, codex_agent=codex, claude_agent=claude,
        repo_root=repo_root, spec_kwargs={"needs_implementation": True, "working_directory_policy": "execution_worktree"},
    )
    assert report.final_state == "POLICY_FAILURE"
    assert report.main_worktree_unchanged is False
    assert (repo_root / "mutated_by_diagnosis.txt").exists()
    assert claude.repair_calls == 0


# ── 28. execution worktree mutation outside allowed paths ──────────────────

def test_repair_writes_outside_allowed_paths_is_policy_failure(tmp_path):
    def repair_forbidden(*, worktree_dir, task_id, run_id, attempt_index):
        target = worktree_dir / "forbidden.py"
        target.write_text("x")
        return RepairResult(task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS", summary="forbidden write", changed_files=["forbidden.py"])

    claude = ScriptedClaude(repair_fn=repair_forbidden)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t28_forbidden_path", script=FLAG_FILE_SCRIPT, codex_agent=codex, claude_agent=claude,
        spec_kwargs={"needs_implementation": True, "working_directory_policy": "execution_worktree"},
    )
    assert report.final_state == "POLICY_FAILURE"
    assert claude.repair_calls == 1


# ── 33. retry after repair succeeds ─────────────────────────────────────────

def test_post_execution_repair_then_retry_succeeds(tmp_path):
    claude = ScriptedClaude(repair_fn=_write_ready_flag)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t33_repair_retry", script=FLAG_FILE_SCRIPT, codex_agent=codex, claude_agent=claude,
        spec_kwargs={"needs_implementation": True, "working_directory_policy": "execution_worktree", "limits": {
            "max_commands": 1, "max_execution_attempts": 3, "max_wall_clock_seconds": 30, "per_command_timeout_seconds": 5,
            "max_repair_rounds": 1, "max_total_codex_invocations": 3, "max_total_claude_invocations": 3,
        }},
    )
    assert report.final_state == "PASS"
    assert report.passing_attempt_index == 1
    assert len(report.execution_attempts) == 2
    assert report.execution_attempts[0].failure_class == "EXECUTION_NONZERO_EXIT"
    assert codex.diagnose_calls == 1
    assert claude.repair_calls == 1
    _assert_terminal_and_never_running(report)


# ── 34. execution retry budget exhausted ────────────────────────────────────

def test_execution_retry_budget_exhausted(tmp_path):
    def repair_never_fixes(*, worktree_dir, task_id, run_id, attempt_index):
        return RepairResult(task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict="REPAIR_PASS", summary="did nothing useful", changed_files=[])

    claude = ScriptedClaude(repair_fn=repair_never_fixes)
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t34_exhausted", script=FLAG_FILE_SCRIPT, codex_agent=codex, claude_agent=claude,
        spec_kwargs={"needs_implementation": True, "working_directory_policy": "execution_worktree", "limits": {
            "max_commands": 1, "max_execution_attempts": 2, "max_wall_clock_seconds": 30, "per_command_timeout_seconds": 5,
            "max_repair_rounds": 1, "max_total_codex_invocations": 3, "max_total_claude_invocations": 3,
        }},
    )
    assert report.final_state == "RETRY_EXHAUSTED"
    assert report.overall_status == "FAIL"
    assert report.passing_attempt_index is None
    _assert_terminal_and_never_running(report)


# ── 35. non-retriable policy failure does not repair ────────────────────────

def test_non_retriable_artifact_policy_failure_never_triggers_repair(tmp_path):
    claude = ScriptedClaude()
    codex = ScriptedCodex()
    report, repo_root, runs_root = _run(
        tmp_path, task_id="t35_no_repair_on_policy_failure", script=SYMLINK_ESCAPE_SCRIPT, codex_agent=codex, claude_agent=claude,
        spec_kwargs={"needs_implementation": True, "required_metrics": [], "required_artifacts": []},
    )
    assert report.final_state == "POLICY_FAILURE"
    assert claude.repair_calls == 0
    assert codex.diagnose_calls == 0


# ── 36/37. final report records passing and failed attempts ────────────────

def test_final_report_records_passing_attempt(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t36_passing_attempt", script=WRITE_OK_SCRIPT)
    assert report.passing_attempt_index == 0
    on_disk = json.loads((runs_root / "run1" / "final_report.json").read_text())
    assert on_disk["passing_attempt_index"] == 0


def test_final_report_records_failed_attempts(tmp_path):
    report, repo_root, runs_root = _run(tmp_path, task_id="t37_failed_attempts", script=EXIT_NONZERO_SCRIPT)
    assert len(report.execution_attempts) == 1
    assert report.execution_attempts[0].failure_class == "EXECUTION_NONZERO_EXIT"
    on_disk = json.loads((runs_root / "run1" / "final_report.json").read_text())
    assert len(on_disk["execution_attempts"]) == 1


# ── 38. state never remains active (broad sweep) ────────────────────────────

@pytest.mark.parametrize("script,expected_not_in", [
    (WRITE_OK_SCRIPT, ()),
    (EXIT_NONZERO_SCRIPT, ()),
    (MALFORMED_JSON_SCRIPT, ()),
])
def test_state_never_remains_active_across_outcomes(tmp_path, script, expected_not_in):
    report, repo_root, runs_root = _run(tmp_path, task_id=f"t38_{id(script)}", script=script)
    state_on_disk = json.loads((runs_root / "run1" / "state.json").read_text())
    assert state_on_disk["state"] == report.final_state
    _assert_terminal_and_never_running(report)


# ── 44. no research dataset/output modified / main worktree unchanged ──────

def test_main_worktree_unchanged_across_pass_and_fail(tmp_path):
    for i, script in enumerate((WRITE_OK_SCRIPT, EXIT_NONZERO_SCRIPT)):
        sub = tmp_path / f"case_{i}"
        sub.mkdir()
        repo_root = _init_repo(sub)
        before = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        report, _, _ = _run(sub, task_id=f"t44_case_{i}", script=script, repo_root=repo_root, run_id="run1")
        after = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        assert before == after
        assert report.main_worktree_unchanged is True


# ── 45. default mock/mock regression ────────────────────────────────────────

def test_default_mock_mock_regression(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="t45_default_mock_mock", script=WRITE_OK_SCRIPT)
    report = run_experiment_flow(
        spec_path, repo_root=repo_root, runs_root=tmp_path / "runs", run_id="run1",
        execution_worktrees_root=tmp_path / "ewt", execute=True,
    )  # codex_agent/claude_agent both omitted -> MockCodexAgent()/MockClaudeExecutorAgent()
    assert report.overall_status == "PASS"
    _assert_terminal_and_never_running(report)


def test_cli_run_experiment_default_mock_mock_never_invokes_codex_or_claude_executables(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="t45_cli_default", script=WRITE_OK_SCRIPT)
    runs_root = tmp_path / "runs"

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    markers = {}
    for name in ("codex", "claude"):
        marker = tmp_path / f"{name}_invoked.marker"
        markers[name] = marker
        script_path = fake_bin / name
        script_path.write_text(f"#!/bin/sh\necho invoked > {marker}\nexit 17\n")
        script_path.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"

    result = subprocess.run(
        [
            sys.executable, "-m", "research_agent.cli",
            "--repo-root", str(repo_root), "--runs-root", str(runs_root),
            "run-experiment", str(spec_path), "--run-id", "cli_default_run",
            "--execution-worktrees-root", str(tmp_path / "ewt"), "--execute",
        ],
        cwd=PACKAGE_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode in (0, 1, 2), f"stdout={result.stdout}\nstderr={result.stderr}"
    assert not markers["codex"].exists(), "the codex sentinel executable was invoked despite --codex mock"
    assert not markers["claude"].exists(), "the claude sentinel executable was invoked despite --claude mock"
    report = json.loads(result.stdout)
    assert report["overall_status"] == "PASS"


def test_cli_run_experiment_requires_explicit_real_flags_never_a_generic_agents_flag():
    from research_agent.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run-experiment", "spec.yaml"])
    assert args.codex == "mock"
    assert args.claude == "mock"
    assert args.execute is False
    with pytest.raises(SystemExit):
        parser.parse_args(["run-experiment", "spec.yaml", "--agents", "real"])


# ── 46/47/48. MVP1/MVP2/MVP3 regression ─────────────────────────────────────

def test_mvp1_plan_flow_still_works(tmp_path):
    from research_agent.flow import plan_flow

    repo_root = _init_repo(tmp_path)
    spec_path = tmp_path / "mvp1.yaml"
    spec_path.write_text(yaml.safe_dump({
        "task_id": "t46_mvp1", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": [sys.executable, "-c", "pass"],
    }))
    result = plan_flow(spec_path, repo_root=repo_root, runs_root=tmp_path / "runs", run_id="run1")
    assert result.verdict == "PLAN_PASS"


def test_mvp2_execute_flow_still_works(tmp_path):
    from research_agent.execute_flow import execute_flow

    repo_root = _init_repo(tmp_path)
    spec_path = tmp_path / "mvp2.yaml"
    spec_path.write_text(yaml.safe_dump({
        "task_id": "t47_mvp2", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": [sys.executable, "-c", "pass"],
        "allowed_modify_paths": ["research_agent_sandbox"],
    }))
    report = execute_flow(spec_path, repo_root=repo_root, runs_root=tmp_path / "runs", run_id="run1", execution_worktrees_root=tmp_path / "ewt")
    assert report.overall_status in ("PASS", "FAIL", "BLOCKED")


def test_mvp3_repair_flow_still_works(tmp_path):
    from research_agent.repair_flow import repair_flow

    repo_root = _init_repo(tmp_path)
    spec_path = tmp_path / "mvp3.yaml"
    spec_path.write_text(yaml.safe_dump({
        "task_id": "t48_mvp3", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": [sys.executable, "-c", "pass"],
        "allowed_modify_paths": ["research_agent_sandbox"],
    }))
    report = repair_flow(spec_path, repo_root=repo_root, runs_root=tmp_path / "runs", run_id="run1", execution_worktrees_root=tmp_path / "ewt")
    assert report.overall_status in ("PASS", "FAIL", "BLOCKED")


# ── 49. cleanup/status understands MVP4 final_report ────────────────────────

def _cli_args(*, run_id, repo_root, runs_root, dry_run=False, delete_branch=False):
    return argparse.Namespace(run_id=run_id, repo_root=str(repo_root), runs_root=str(runs_root), dry_run=dry_run, delete_branch=delete_branch)


def _call(func, args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = func(args)
    text = out.getvalue() or err.getvalue()
    return code, json.loads(text)


def test_experiment_status_and_cleanup_understand_mvp4_reports(tmp_path):
    from research_agent.cli import cmd_experiment_cleanup, cmd_experiment_status

    report, repo_root, runs_root = _run(tmp_path, task_id="t49_status_cleanup", script=WRITE_OK_SCRIPT)
    assert report.overall_status == "PASS"

    code, status_out = _call(cmd_experiment_status, _cli_args(run_id="run1", repo_root=repo_root, runs_root=runs_root))
    assert status_out["run_type"] == "run-experiment"
    assert status_out["status"] == "PASS"

    code, cleanup_out = _call(
        cmd_experiment_cleanup, _cli_args(run_id="run1", repo_root=repo_root, runs_root=runs_root, dry_run=True),
    )
    assert cleanup_out["status"] == "DRY_RUN"
    assert cleanup_out["run_type"] == "run-experiment"


def test_worktree_status_generic_command_also_understands_mvp4(tmp_path):
    from research_agent.cli import cmd_worktree_status

    report, repo_root, runs_root = _run(tmp_path, task_id="t49b_worktree_status", script=WRITE_OK_SCRIPT)
    code, out = _call(cmd_worktree_status, _cli_args(run_id="run1", repo_root=repo_root, runs_root=runs_root))
    assert out["run_type"] == "run-experiment"
    assert out["report_summary"]["overall_status"] == "PASS"


# ── 50. interrupted execution reported safely ───────────────────────────────

def test_interrupted_execution_reported_as_incomplete_never_pass(tmp_path):
    from research_agent.cli import cmd_experiment_status

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "interrupted_run"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({
        "schema_version": "1.0", "run_id": "interrupted_run", "task_id": "t50",
        "state": "EXECUTING", "attempt_index": 0, "updated_at": "2026-01-01T00:00:00+00:00",
        "history": ["PLANNING@...", "EXECUTING@..."], "detail": None,
    }))
    code, out = _call(cmd_experiment_status, _cli_args(run_id="interrupted_run", repo_root=".", runs_root=runs_root))
    assert out["status"] == "INCOMPLETE"
    assert out["status"] != "PASS"
    assert out["current_state"] == "EXECUTING"
    assert out["manual_action_required"] is True
    assert code != 0
