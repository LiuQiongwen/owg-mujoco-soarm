"""Real-Claude-executor tests: RealClaudeExecutorAgent, execute_flow's
deterministic path/command/diff/main-worktree checks, the before/after
ignored-file manifest, execution-worktree preservation, and CLI-level
Codex/Claude selection for the MVP2 `execute` command -- all exercised
offline against a single fake `claude` executable, never a real Claude
installation.

The fake executable (see _write_fake_claude below) emulates the actual CLI
contract closely enough for these tests to matter: it consumes the prompt on
stdin, inspects `FAKE_CLAUDE_SCENARIO` (and a handful of scenario-specific
env vars) to decide what to do inside its current working directory (which
`RealClaudeExecutorAgent.execute` always sets to the execution worktree),
and prints a Claude `--output-format json`-shaped envelope
(`{"result": "<json string>", ...}`) to stdout.

Covers (see the MVP2 task's "Fake Claude tests" checklist):
  1.  successful harmless file creation
  2.  executable unavailable
  3.  timeout
  4.  nonzero exit
  5.  authentication-style failure
  6.  missing structured output
  7.  malformed JSON
  8.  unknown schema field
  9.  invalid verdict
  10. task ID mismatch
  11. run ID mismatch
  12. forbidden path modification
  13. unapproved command claim
  14. changed-file count exceeded
  15. changed-byte limit exceeded
  16. symlink escape
  17. nested Git repository creation
  18. .git modification attempt
  19. main worktree mutation
  20. implementation report differs from actual Git diff
  21. real Codex is not invoked when --codex mock
  22. second Claude process is not invoked
  23. default mock/mock behavior remains unchanged
  24. failure state is never left as RUNNING
  25. execution worktree is preserved after failure
  26. no commit and no push occurs
  27. ignored-file in-place mutation is detected where covered by manifest
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.agents.claude_executor import RealClaudeExecutorAgent
from research_agent.agents.codex import MockCodexAgent
from research_agent.execute_flow import execute_flow
from research_agent.models import EXECUTE_TERMINAL_STATES

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

_FAKE_CLAUDE_BODY = r'''
import json, os, subprocess, sys, time
from pathlib import Path

def _payload(task_id, run_id, verdict, summary, **extra):
    p = {
        "schema_version": "1.0", "task_id": task_id, "run_id": run_id,
        "verdict": verdict, "summary": summary,
        "changed_files": [], "commands_run": [], "tests_run": [],
        "issues": [], "risks": [], "assumptions": [],
    }
    p.update(extra)
    return p

def main():
    scenario = os.environ.get("FAKE_CLAUDE_SCENARIO", "success")
    sys.stdin.read()  # consume the prompt, mirroring the real CLI contract
    task_id = os.environ.get("FAKE_CLAUDE_TASK_ID", "example_mvp2_sandbox")
    run_id = os.environ.get("FAKE_CLAUDE_RUN_ID", "run")
    cwd = os.getcwd()

    if scenario == "timeout":
        time.sleep(float(os.environ.get("FAKE_CLAUDE_SLEEP", "5")))
        return 0

    if scenario == "nonzero":
        sys.stderr.write("claude: internal executor error\n")
        return 7

    if scenario == "auth_failed":
        sys.stderr.write("Error: not authenticated. Please run /login.\n")
        return 1

    if scenario == "missing_output":
        return 0  # exit 0 but print nothing

    if scenario == "malformed_json":
        sys.stdout.write("{not valid json")
        return 0

    payload = _payload(task_id, run_id, "IMPLEMENTATION_PASS", "fake claude executor")

    if scenario == "success":
        relpath = os.environ.get("FAKE_CLAUDE_WRITE_RELPATH", "research_agent_sandbox/out.txt")
        content = os.environ.get("FAKE_CLAUDE_WRITE_CONTENT", "hello")
        full = os.path.join(cwd, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        payload["changed_files"] = [relpath]

    elif scenario == "unknown_field":
        payload["unexpected_field"] = "oops"

    elif scenario == "invalid_verdict":
        payload["verdict"] = "IMPLEMENTATION_MAYBE"

    elif scenario == "task_id_mismatch":
        payload["task_id"] = "wrong_task"

    elif scenario == "run_id_mismatch":
        payload["run_id"] = "wrong_run"

    elif scenario == "forbidden_path":
        full = os.path.join(cwd, "data", "malicious.txt")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write("malicious")
        payload["changed_files"] = ["data/malicious.txt"]

    elif scenario == "unapproved_command_claim":
        full = os.path.join(cwd, "research_agent_sandbox", "ok.txt")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write("ok")
        payload["changed_files"] = ["research_agent_sandbox/ok.txt"]
        payload["commands_run"] = [["bash", "-c", "echo hi"]]

    elif scenario == "changed_file_limit":
        n = int(os.environ.get("FAKE_CLAUDE_FILE_COUNT", "10"))
        changed = []
        for i in range(n):
            rel = f"research_agent_sandbox/f{i}.txt"
            full = os.path.join(cwd, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write("x")
            changed.append(rel)
        payload["changed_files"] = changed

    elif scenario == "changed_bytes_limit":
        rel = "research_agent_sandbox/big.bin"
        full = os.path.join(cwd, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write("0" * int(os.environ.get("FAKE_CLAUDE_BIG_SIZE", "50000")))
        payload["changed_files"] = [rel]

    elif scenario == "symlink_escape":
        rel = "research_agent_sandbox/escape_link"
        full = os.path.join(cwd, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        os.symlink("/etc/passwd", full)
        payload["changed_files"] = [rel]

    elif scenario == "nested_git":
        nested = os.path.join(cwd, "research_agent_sandbox", "nested")
        os.makedirs(nested, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=nested, check=True, capture_output=True)
        (Path(nested) / "inner.txt").write_text("x")
        payload["changed_files"] = ["research_agent_sandbox/nested/"]

    elif scenario == "git_modification":
        with open(os.path.join(cwd, ".git"), "w") as f:
            f.write("MALICIOUS_GITDIR_OVERWRITE\n")
        payload["changed_files"] = []

    elif scenario == "main_worktree_mutation":
        target = os.environ.get("FAKE_CLAUDE_MAIN_REPO_MUTATION_PATH")
        if target:
            with open(target, "w") as f:
                f.write("mutated by fake claude\n")
        payload["changed_files"] = []

    elif scenario == "diff_mismatch":
        rel = "research_agent_sandbox/real.txt"
        full = os.path.join(cwd, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write("real content")
        payload["changed_files"] = ["research_agent_sandbox/other_file_that_was_never_written.txt"]

    elif scenario == "ignored_mutation":
        rel = "research_agent_sandbox/checkpoint.pt"
        full = os.path.join(cwd, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(b"\x00" * 64)
        payload["changed_files"] = []

    elif scenario == "implementation_blocked":
        payload["verdict"] = "IMPLEMENTATION_BLOCKED"
        payload["summary"] = "blocked: policy conflict, stopping rather than bypassing"
        payload["issues"] = ["blocked by design"]

    sys.stdout.write(json.dumps({"type": "result", "subtype": "success", "result": json.dumps(payload)}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


def _write_fake_claude(tmp_path: Path) -> Path:
    script = tmp_path / "fake_claude"
    script.write_text(f"#!{sys.executable}\n{_FAKE_CLAUDE_BODY}")
    script.chmod(0o755)
    return script


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
    (repo / ".gitignore").write_text("*.pt\n*.pth\n*.ckpt\n")
    run("git", "add", ".")
    run("git", "commit", "-q", "-m", "init")
    return repo


def _write_spec(
    tmp_path: Path, *, task_id: str, max_run_count: int = 10, max_changed_files: int = 5,
    max_changed_bytes: int = 20000, allowed_executor_commands=None,
) -> Path:
    spec = {
        "task_id": task_id,
        "goal": "mvp2 fake-claude executor test",
        "allowed_paths": ["research_agent_sandbox"],
        "forbidden_paths": [],
        "allowed_modify_paths": ["research_agent_sandbox"],
        "max_changed_files": max_changed_files,
        "max_changed_bytes": max_changed_bytes,
        "allowed_executor_commands": allowed_executor_commands or [],
        "required_executor_checks": ["compileall"],
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


def _run_execute(
    tmp_path, *, task_id, scenario, run_id="run1", extra_env=None, executor_seconds=None,
    max_changed_files=5, max_changed_bytes=20000, allowed_executor_commands=None,
):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(
        tmp_path, task_id=task_id, max_changed_files=max_changed_files,
        max_changed_bytes=max_changed_bytes, allowed_executor_commands=allowed_executor_commands,
    )
    if executor_seconds is not None:
        data = yaml.safe_load(spec_path.read_text())
        data["timeouts"]["executor_seconds"] = executor_seconds
        spec_path.write_text(yaml.safe_dump(data))
    runs_root = tmp_path / "runs"
    fake_claude = _write_fake_claude(tmp_path)

    env_backup = dict(os.environ)
    os.environ["FAKE_CLAUDE_SCENARIO"] = scenario
    os.environ["FAKE_CLAUDE_TASK_ID"] = task_id
    os.environ["FAKE_CLAUDE_RUN_ID"] = run_id
    if extra_env:
        os.environ.update(extra_env)
    try:
        report = execute_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id=run_id,
            execution_worktrees_root=tmp_path / "execution_worktrees",
            codex_agent=MockCodexAgent(), claude_agent=RealClaudeExecutorAgent(binary=str(fake_claude)),
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
    return report, repo_root, runs_root


def _assert_terminal_and_never_running(report):
    """#24: failure state is never left as RUNNING."""
    assert report.overall_status in ("PASS", "FAIL", "BLOCKED")
    assert isinstance(report.terminal_state, str) and report.terminal_state
    assert report.terminal_state != "RUNNING"
    if report.terminal_state != "EXECUTION_PASS":
        assert report.terminal_state in EXECUTE_TERMINAL_STATES or report.terminal_state in (
            "STATIC_CHECKS_FAILED", "CODEX_COMMAND_POLICY_FAILED",
        )


def _assert_worktree_preserved(report):
    """#25: execution worktree is preserved after failure."""
    assert report.execution_worktree is not None
    assert Path(report.execution_worktree.worktree_path).exists()
    assert report.execution_worktree.preserved is True


def _assert_no_commit_no_push(report, repo_root: Path):
    """#26: no commit and no push occurs, in either the execution worktree
    or the main repository."""
    worktree_path = Path(report.execution_worktree.worktree_path)
    log = subprocess.run(
        ["git", "-C", str(worktree_path), "log", "--oneline"], capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert len(log) == 1, f"expected exactly the base commit, got: {log}"
    main_log = subprocess.run(
        ["git", "-C", str(repo_root), "log", "--oneline"], capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert len(main_log) == 1, f"main repo history must be untouched: {main_log}"
    remotes = subprocess.run(
        ["git", "-C", str(repo_root), "remote"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert remotes == ""  # no remote configured at all -- a push is not even possible


# ── 1. successful harmless file creation ────────────────────────────────────

def test_success_writes_allowed_file_and_passes(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="success_case", scenario="success")
    assert report.overall_status == "PASS"
    assert report.terminal_state == "EXECUTION_PASS"
    assert report.changed_file_count == 1
    assert report.changed_files[0].path == "research_agent_sandbox/out.txt"
    run_dir = runs_root / "run1"
    assert (run_dir / "prompts" / "claude_executor_prompt.md").exists()
    assert (run_dir / "commands" / "claude_executor.command.json").exists()
    assert (run_dir / "commands" / "claude_executor.stdout").exists()
    assert (run_dir / "commands" / "claude_executor.stderr").exists()
    assert (run_dir / "commands" / "claude_executor.exit_code").exists()
    assert (run_dir / "commands" / "claude_executor.result.json").exists()
    assert (run_dir / "implementation.raw.json").exists()
    assert (run_dir / "implementation.json").exists()
    assert (run_dir / "execution_worktree.json").exists()
    assert (run_dir / "ignored_file_manifest.before.json").exists()
    assert (run_dir / "ignored_file_manifest.after.json").exists()
    command_array = json.loads((run_dir / "commands" / "claude_executor.command.json").read_text())
    assert command_array[1:] == [
        "-p", "--output-format", "json", "--permission-mode", "acceptEdits",
        "--tools", "Read,Write,Edit", "--strict-mcp-config", "--no-session-persistence",
        "--json-schema", command_array[-1],
    ]
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)
    _assert_no_commit_no_push(report, repo_root)


# ── 2. executable unavailable ───────────────────────────────────────────────

def test_executable_unavailable_is_blocked_deterministically(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="missing_exe_case")
    runs_root = tmp_path / "runs"
    report = execute_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
        execution_worktrees_root=tmp_path / "execution_worktrees",
        codex_agent=MockCodexAgent(), claude_agent=RealClaudeExecutorAgent(binary="definitely_not_a_real_claude_xyz"),
    )
    assert report.terminal_state == "CLAUDE_EXECUTABLE_UNAVAILABLE"
    assert report.overall_status == "BLOCKED"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 3. timeout ───────────────────────────────────────────────────────────────

def test_timeout_is_blocked_deterministically(tmp_path):
    report, repo_root, runs_root = _run_execute(
        tmp_path, task_id="timeout_case", scenario="timeout",
        extra_env={"FAKE_CLAUDE_SLEEP": "2"}, executor_seconds=1,
    )
    assert report.terminal_state == "CLAUDE_TIMEOUT"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 4. nonzero exit ──────────────────────────────────────────────────────────

def test_nonzero_exit_is_blocked_deterministically(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="nonzero_case", scenario="nonzero")
    assert report.terminal_state == "CLAUDE_NONZERO_EXIT"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 5. authentication-style failure ─────────────────────────────────────────

def test_authentication_failure_is_blocked_deterministically(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="auth_case", scenario="auth_failed")
    assert report.terminal_state == "CLAUDE_AUTHENTICATION_FAILED"
    stderr_text = (runs_root / "run1" / "commands" / "claude_executor.stderr").read_text()
    assert "not authenticated" in stderr_text
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 6. missing structured output ────────────────────────────────────────────

def test_missing_output_is_blocked_deterministically(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="missing_output_case", scenario="missing_output")
    assert report.terminal_state == "CLAUDE_OUTPUT_MISSING"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 7. malformed JSON ────────────────────────────────────────────────────────

def test_malformed_json_is_blocked_deterministically(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="malformed_case", scenario="malformed_json")
    assert report.terminal_state == "CLAUDE_OUTPUT_MALFORMED"
    assert (runs_root / "run1" / "implementation.raw.json").exists()
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 8. unknown schema field ──────────────────────────────────────────────────

def test_unknown_field_is_blocked_deterministically(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="unknown_field_case", scenario="unknown_field")
    assert report.terminal_state == "CLAUDE_SCHEMA_INVALID"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 9. invalid verdict ───────────────────────────────────────────────────────

def test_invalid_verdict_is_blocked_deterministically(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="invalid_verdict_case", scenario="invalid_verdict")
    assert report.terminal_state == "CLAUDE_SCHEMA_INVALID"
    _assert_terminal_and_never_running(report)


# ── 10. task ID mismatch ─────────────────────────────────────────────────────

def test_task_id_mismatch_is_blocked_deterministically(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="task_mismatch_case", scenario="task_id_mismatch")
    assert report.terminal_state == "CLAUDE_TASK_ID_MISMATCH"
    _assert_terminal_and_never_running(report)


# ── 11. run ID mismatch ──────────────────────────────────────────────────────

def test_run_id_mismatch_is_blocked_deterministically(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="run_mismatch_case", scenario="run_id_mismatch")
    assert report.terminal_state == "CLAUDE_RUN_ID_MISMATCH"
    _assert_terminal_and_never_running(report)


# ── 12. forbidden path modification ─────────────────────────────────────────

def test_forbidden_path_modification_is_blocked(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="forbidden_path_case", scenario="forbidden_path")
    assert report.terminal_state == "CLAUDE_PATH_POLICY_FAILED"
    assert report.overall_status == "FAIL"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)
    _assert_no_commit_no_push(report, repo_root)


# ── 13. unapproved command claim ─────────────────────────────────────────────

def test_unapproved_command_claim_is_blocked(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="bad_cmd_claim_case", scenario="unapproved_command_claim")
    assert report.terminal_state == "CLAUDE_COMMAND_POLICY_FAILED"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 14. changed-file count exceeded ─────────────────────────────────────────

def test_changed_file_limit_exceeded_is_blocked(tmp_path):
    report, repo_root, runs_root = _run_execute(
        tmp_path, task_id="file_limit_case", scenario="changed_file_limit",
        extra_env={"FAKE_CLAUDE_FILE_COUNT": "10"}, max_changed_files=3,
    )
    assert report.terminal_state == "CLAUDE_CHANGED_FILE_LIMIT_EXCEEDED"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 15. changed-byte limit exceeded ─────────────────────────────────────────

def test_changed_bytes_limit_exceeded_is_blocked(tmp_path):
    report, repo_root, runs_root = _run_execute(
        tmp_path, task_id="byte_limit_case", scenario="changed_bytes_limit",
        extra_env={"FAKE_CLAUDE_BIG_SIZE": "50000"}, max_changed_bytes=1000,
    )
    assert report.terminal_state == "CLAUDE_CHANGED_BYTES_LIMIT_EXCEEDED"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 16. symlink escape ───────────────────────────────────────────────────────

def test_symlink_escape_is_detected_and_blocked(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="symlink_case", scenario="symlink_escape")
    assert report.terminal_state == "CLAUDE_SYMLINK_ESCAPE_DETECTED"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 17. nested Git repository creation ──────────────────────────────────────

def test_nested_git_repo_is_detected_and_blocked(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="nested_git_case", scenario="nested_git")
    assert report.terminal_state == "CLAUDE_NESTED_GIT_DETECTED"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 18. .git modification attempt ───────────────────────────────────────────

def test_git_pointer_file_modification_is_detected_and_blocked(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="git_mod_case", scenario="git_modification")
    assert report.terminal_state == "CLAUDE_DIFF_MISMATCH"
    assert not report.ignored_file_manifest_ok
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 19. main worktree mutation ──────────────────────────────────────────────

def test_main_worktree_mutation_is_detected_and_blocked(tmp_path):
    repo_root = _init_repo(tmp_path)
    mutation_path = repo_root / "mutated_by_fake_claude.txt"
    report, _, runs_root = None, None, None
    spec_path = _write_spec(tmp_path, task_id="main_mutation_case")
    runs_root = tmp_path / "runs"
    fake_claude = _write_fake_claude(tmp_path)

    env_backup = dict(os.environ)
    os.environ["FAKE_CLAUDE_SCENARIO"] = "main_worktree_mutation"
    os.environ["FAKE_CLAUDE_TASK_ID"] = "main_mutation_case"
    os.environ["FAKE_CLAUDE_RUN_ID"] = "run1"
    os.environ["FAKE_CLAUDE_MAIN_REPO_MUTATION_PATH"] = str(mutation_path)
    try:
        report = execute_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
            execution_worktrees_root=tmp_path / "execution_worktrees",
            codex_agent=MockCodexAgent(), claude_agent=RealClaudeExecutorAgent(binary=str(fake_claude)),
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    assert report.terminal_state == "CLAUDE_MAIN_WORKTREE_CHANGED"
    assert not report.main_worktree_unchanged
    assert mutation_path.exists()  # the mutation genuinely happened; it was still caught
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 20. implementation report differs from actual Git diff ─────────────────

def test_diff_mismatch_between_claim_and_actual_git_diff_is_blocked(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="diff_mismatch_case", scenario="diff_mismatch")
    assert report.terminal_state == "CLAUDE_DIFF_MISMATCH"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── 21 & 22. sentinel Codex/Claude are never invoked beyond the intended ───

def test_execute_with_mock_codex_never_invokes_sentinel_codex(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="sentinel_codex_case")
    runs_root = tmp_path / "runs"
    fake_claude = _write_fake_claude(tmp_path)

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    codex_marker = tmp_path / "codex_invoked.marker"
    sentinel = fake_bin / "codex"
    sentinel.write_text(f"#!/bin/sh\necho invoked > {codex_marker}\nexit 17\n")
    sentinel.chmod(0o755)

    env_backup = dict(os.environ)
    os.environ["FAKE_CLAUDE_SCENARIO"] = "success"
    os.environ["FAKE_CLAUDE_TASK_ID"] = "sentinel_codex_case"
    os.environ["FAKE_CLAUDE_RUN_ID"] = "run1"
    os.environ["PATH"] = f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        report = execute_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
            execution_worktrees_root=tmp_path / "execution_worktrees",
            codex_agent=MockCodexAgent(), claude_agent=RealClaudeExecutorAgent(binary=str(fake_claude)),
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    assert report.overall_status == "PASS"
    assert not codex_marker.exists(), "the codex sentinel executable was invoked despite --codex mock"


def test_real_claude_executor_never_spawns_a_second_claude_process(tmp_path):
    """The RealClaudeExecutorAgent must resolve and invoke exactly the fake
    `claude` binary it was given -- never an unrelated `claude` also
    reachable on PATH (a stand-in for "Claude invoking another Claude
    process")."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="sentinel_claude_case")
    runs_root = tmp_path / "runs"
    fake_claude = _write_fake_claude(tmp_path)

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    sentinel_marker = tmp_path / "sentinel_claude_invoked.marker"
    sentinel = fake_bin / "claude"
    sentinel.write_text(f"#!/bin/sh\necho invoked > {sentinel_marker}\nexit 17\n")
    sentinel.chmod(0o755)

    env_backup = dict(os.environ)
    os.environ["FAKE_CLAUDE_SCENARIO"] = "success"
    os.environ["FAKE_CLAUDE_TASK_ID"] = "sentinel_claude_case"
    os.environ["FAKE_CLAUDE_RUN_ID"] = "run1"
    os.environ["PATH"] = f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        report = execute_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
            execution_worktrees_root=tmp_path / "execution_worktrees",
            codex_agent=MockCodexAgent(), claude_agent=RealClaudeExecutorAgent(binary=str(fake_claude)),
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    assert report.overall_status == "PASS"
    assert not sentinel_marker.exists(), "an unrelated sentinel `claude` on PATH was invoked"


# ── 23. default mock/mock behavior remains unchanged (through the real CLI) ─

def test_cli_execute_default_mock_mock_never_invokes_codex_or_claude_executables(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="cli_default_execute_case")
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
            "execute", str(spec_path), "--run-id", "cli_default_run",
            "--execution-worktrees-root", str(tmp_path / "execution_worktrees"),
        ],
        cwd=PACKAGE_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert not markers["codex"].exists(), "the codex sentinel executable was invoked"
    assert not markers["claude"].exists(), "the claude sentinel executable was invoked"

    report = json.loads(result.stdout)
    assert report["overall_status"] == "PASS"
    assert report["changed_file_count"] == 0  # MockClaudeExecutorAgent never writes anything


def test_cli_execute_requires_explicit_claude_real_never_a_generic_agents_flag():
    """There must be no generic --agents flag on `execute` -- real Claude
    is only reachable via the explicit --claude real."""
    from research_agent.cli import build_parser

    parser = build_parser()
    execute_args = parser.parse_args(["execute", "spec.yaml"])
    assert execute_args.codex == "mock"
    assert execute_args.claude == "mock"
    with pytest.raises(SystemExit):
        parser.parse_args(["execute", "spec.yaml", "--agents", "real"])


def test_cli_execute_prints_warning_before_real_claude(tmp_path):
    """The CLI must print a clear warning before starting a real Claude
    subprocess -- verified here by using --claude real with an
    unresolvable binary (so no real API call is made) and checking stderr."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="warning_case")
    runs_root = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable, "-m", "research_agent.cli",
            "--repo-root", str(repo_root), "--runs-root", str(runs_root),
            "execute", str(spec_path), "--run-id", "warning_run",
            "--execution-worktrees-root", str(tmp_path / "execution_worktrees"),
            "--claude", "real",
        ],
        cwd=PACKAGE_ROOT, capture_output=True, text=True, timeout=60,
        env={**os.environ, "PATH": "/definitely/not/a/real/path"},
    )
    assert "WARNING" in result.stderr and "--claude real" in result.stderr


# ── 27. ignored-file in-place mutation is detected where covered by manifest ─

def test_ignored_file_mutation_is_detected_via_manifest(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="ignored_mutation_case", scenario="ignored_mutation")
    assert report.terminal_state == "CLAUDE_DIFF_MISMATCH"
    assert not report.ignored_file_manifest_ok
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── implementation-level BLOCKED verdict is honored as a distinct outcome ───

def test_implementation_blocked_verdict_is_surfaced(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="impl_blocked_case", scenario="implementation_blocked")
    assert report.terminal_state == "CLAUDE_IMPLEMENTATION_BLOCKED"
    assert report.overall_status == "BLOCKED"
    _assert_terminal_and_never_running(report)
    _assert_worktree_preserved(report)


# ── worktree path is validated: never collides, never inside the main tree ──

def test_execution_worktree_is_outside_main_repository_worktree(tmp_path):
    report, repo_root, runs_root = _run_execute(tmp_path, task_id="worktree_location_case", scenario="success")
    worktree_path = Path(report.execution_worktree.worktree_path).resolve()
    assert repo_root.resolve() not in worktree_path.parents
    assert worktree_path != repo_root.resolve()
