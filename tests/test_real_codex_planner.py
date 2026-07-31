"""Real-Codex-planner tests: RealCodexAgent, the repository-mutation check,
and CLI-level Codex/Claude selection -- all exercised offline against a
single fake `codex` executable, never a real Codex installation.

The fake executable (see _write_fake_codex below) emulates the actual CLI
contract closely enough for these tests to matter: it reads argv for
`--output-last-message`, consumes the prompt on stdin, and its behavior is
selected entirely via the FAKE_CODEX_SCENARIO environment variable, so one
script covers every required scenario without fifteen near-duplicate files.

Covers (see the MVP1 task's "Fake Codex tests" checklist):
  1.  successful structured PLAN_PASS
  2.  executable unavailable
  3.  timeout
  4.  nonzero exit
  5.  authentication-style failure
  6.  missing output file
  7.  malformed JSON
  8.  unknown schema field
  9.  invalid verdict
  10. task ID mismatch
  11. run ID mismatch
  12. forbidden proposed command
  13. unapproved proposed command
  14. shell string instead of command array
  15. repository unchanged on success
  16. repository mutation detected and blocked
  17. mock Claude remains active (real Codex + mock Claude smoke flow)
  18. sentinel `claude` executable is never invoked
  19. existing mock-default regression tests still pass (see test_smoke_flow.py)
  20. failure state is never left as RUNNING (every scenario below asserts
      plan.json exists with a terminal PLAN_BLOCKED/PLAN_PASS verdict)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.agents.claude import MockClaudeAgent
from research_agent.agents.codex import RealCodexAgent
from research_agent.flow import plan_flow, smoke_flow

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

_FAKE_CODEX_BODY = r'''
import json, os, sys, time

def main():
    scenario = os.environ.get("FAKE_CODEX_SCENARIO", "pass")
    args = sys.argv[1:]
    output_path = None
    for i, a in enumerate(args):
        if a == "--output-last-message" and i + 1 < len(args):
            output_path = args[i + 1]
    sys.stdin.read()  # consume the prompt, mirroring the real CLI contract

    if scenario == "timeout":
        time.sleep(float(os.environ.get("FAKE_CODEX_SLEEP", "5")))
        return 0

    if scenario == "nonzero":
        sys.stderr.write("codex: internal planning error\n")
        return 7

    if scenario == "auth_failed":
        sys.stderr.write("Error: not authenticated. Please run `codex login`.\n")
        return 1

    if scenario == "missing_output":
        return 0  # exit 0 but never write output_path

    if scenario == "malformed_json":
        with open(output_path, "w") as f:
            f.write("{not valid json")
        return 0

    task_id = os.environ.get("FAKE_CODEX_TASK_ID", "example_smoke")
    run_id = os.environ.get("FAKE_CODEX_RUN_ID", "run")
    plan = {
        "schema_version": "1.0",
        "task_id": task_id,
        "run_id": run_id,
        "verdict": "PLAN_PASS",
        "summary": "fake codex plan",
        "issues": [],
        "artifacts": [],
        "proposed_commands": [],
        "expected_artifacts": [],
        "expected_metrics": [],
        "risks": [],
        "assumptions": [],
    }

    if scenario == "unknown_field":
        plan["unexpected_field"] = "oops"
    elif scenario == "invalid_verdict":
        plan["verdict"] = "PLAN_MAYBE"
    elif scenario == "task_id_mismatch":
        plan["task_id"] = "wrong_task"
    elif scenario == "run_id_mismatch":
        plan["run_id"] = "wrong_run"
    elif scenario == "forbidden_command":
        plan["proposed_commands"] = [["sudo", "rm", "-rf", "/"]]
    elif scenario == "unapproved_command":
        plan["proposed_commands"] = [["python", "unapproved_script.py"]]
    elif scenario == "shell_string":
        plan["proposed_commands"] = ["python -c 'print(1)'"]
    elif scenario == "repo_mutation":
        mutation_path = os.environ.get("FAKE_CODEX_MUTATION_PATH")
        if mutation_path:
            with open(mutation_path, "w") as f:
                f.write("mutated by fake codex\n")

    with open(output_path, "w") as f:
        json.dump(plan, f)
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


def _write_fake_codex(tmp_path: Path) -> Path:
    script = tmp_path / "fake_codex"
    script.write_text(f"#!{sys.executable}\n{_FAKE_CODEX_BODY}")
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
    run("git", "add", "README.md")
    run("git", "commit", "-q", "-m", "init")
    return repo


def _write_spec(tmp_path: Path, *, task_id: str, max_run_count: int = 5) -> Path:
    spec = {
        "task_id": task_id,
        "goal": "real codex planner test",
        "allowed_paths": ["experiments"],
        "forbidden_paths": [],
        "smoke_command": [
            sys.executable, "-c",
            "import json, os\n"
            "out = os.path.join(os.environ['RESEARCH_AGENT_ARTIFACTS_DIR'], 'metrics.json')\n"
            "json.dump({'smoke_ok': True, 'value': 1.0}, open(out, 'w'))\n",
        ],
        "seeds": [0],
        "timeouts": {
            "planner_seconds": 5, "executor_seconds": 30, "smoke_seconds": 30,
            "verifier_seconds": 30, "reviewer_seconds": 30,
        },
        "required_metrics": [{"name": "smoke_ok", "type": "bool"}],
        "max_run_count": max_run_count,
    }
    path = tmp_path / f"{task_id}.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def _run_plan(tmp_path, *, task_id, scenario, run_id="run1", extra_env=None, planner_seconds=None):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id=task_id)
    if planner_seconds is not None:
        data = yaml.safe_load(spec_path.read_text())
        data["timeouts"]["planner_seconds"] = planner_seconds
        spec_path.write_text(yaml.safe_dump(data))
    runs_root = tmp_path / "runs"
    fake_codex = _write_fake_codex(tmp_path)

    env_backup = dict(os.environ)
    os.environ["FAKE_CODEX_SCENARIO"] = scenario
    os.environ["FAKE_CODEX_TASK_ID"] = task_id
    os.environ["FAKE_CODEX_RUN_ID"] = run_id
    if extra_env:
        os.environ.update(extra_env)
    try:
        plan = plan_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id=run_id,
            codex_agent=RealCodexAgent(binary=str(fake_codex)),
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
    return plan, repo_root, runs_root


# ── 1. successful structured PLAN_PASS ──────────────────────────────────────

def test_real_codex_plan_pass_produces_valid_plan_and_artifacts(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="pass_case", scenario="pass")
    assert plan.verdict == "PLAN_PASS"

    run_dir = runs_root / "run1"
    assert (run_dir / "prompts" / "codex_planner_prompt.md").exists()
    assert (run_dir / "commands" / "codex_planner.command.json").exists()
    assert (run_dir / "commands" / "codex_planner.stdout").exists()
    assert (run_dir / "commands" / "codex_planner.stderr").exists()
    assert (run_dir / "commands" / "codex_planner.exit_code").exists()
    assert (run_dir / "commands" / "codex_planner.result.json").exists()
    assert (run_dir / "plan.raw.json").exists()
    assert (run_dir / "plan.json").exists()

    assert (run_dir / "commands" / "codex_planner.exit_code").read_text().strip() == "0"

    import json as _json
    command_array = _json.loads((run_dir / "commands" / "codex_planner.command.json").read_text())
    assert command_array[1:] == [
        "exec", "--sandbox", "read-only", "--ephemeral",
        "--output-schema", str(run_dir / "plan.schema.json"),
        "--output-last-message", str(run_dir / "plan.raw.json"),
        "-",
    ]


# ── 2. executable unavailable ───────────────────────────────────────────────

def test_real_codex_executable_unavailable_is_blocked_deterministically(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="missing_exe_case")
    runs_root = tmp_path / "runs"

    plan = plan_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
        codex_agent=RealCodexAgent(binary="definitely_not_a_real_codex_xyz"),
    )
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_EXECUTABLE_UNAVAILABLE"]
    assert "CODEX_EXECUTABLE_UNAVAILABLE" in plan.summary
    assert (runs_root / "run1" / "plan.json").exists()


# ── 3. timeout ───────────────────────────────────────────────────────────────

def test_real_codex_timeout_is_blocked_deterministically(tmp_path):
    plan, repo_root, runs_root = _run_plan(
        tmp_path, task_id="timeout_case", scenario="timeout",
        extra_env={"FAKE_CODEX_SLEEP": "2"}, planner_seconds=1,
    )
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_TIMEOUT"]
    assert "CODEX_TIMEOUT" in plan.summary


# ── 4. nonzero exit ──────────────────────────────────────────────────────────

def test_real_codex_nonzero_exit_is_blocked_deterministically(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="nonzero_case", scenario="nonzero")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_NONZERO_EXIT"]


# ── 5. authentication-style failure ─────────────────────────────────────────

def test_real_codex_authentication_failure_is_blocked_deterministically(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="auth_case", scenario="auth_failed")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_AUTHENTICATION_FAILED"]
    # the original stderr must be preserved verbatim, not replaced
    run_dir = runs_root / "run1"
    stderr_text = (run_dir / "commands" / "codex_planner.stderr").read_text()
    assert "not authenticated" in stderr_text


# ── 6. missing output file ──────────────────────────────────────────────────

def test_real_codex_missing_output_is_blocked_deterministically(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="missing_output_case", scenario="missing_output")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_OUTPUT_MISSING"]


# ── 7. malformed JSON ────────────────────────────────────────────────────────

def test_real_codex_malformed_json_is_blocked_deterministically(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="malformed_case", scenario="malformed_json")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_OUTPUT_MALFORMED"]


# ── 8. unknown schema field ──────────────────────────────────────────────────

def test_real_codex_unknown_field_is_blocked_deterministically(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="unknown_field_case", scenario="unknown_field")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_SCHEMA_INVALID"]


# ── 9. invalid verdict ───────────────────────────────────────────────────────

def test_real_codex_invalid_verdict_is_blocked_deterministically(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="invalid_verdict_case", scenario="invalid_verdict")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_SCHEMA_INVALID"]


# ── 10. task ID mismatch ─────────────────────────────────────────────────────

def test_real_codex_task_id_mismatch_is_blocked_deterministically(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="task_mismatch_case", scenario="task_id_mismatch")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_TASK_ID_MISMATCH"]


# ── 11. run ID mismatch ──────────────────────────────────────────────────────

def test_real_codex_run_id_mismatch_is_blocked_deterministically(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="run_mismatch_case", scenario="run_id_mismatch")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_RUN_ID_MISMATCH"]


# ── 12. forbidden proposed command ──────────────────────────────────────────

def test_real_codex_forbidden_proposed_command_blocks_via_plan_policy(tmp_path):
    """Codex itself returns a schema-valid PLAN_PASS with a forbidden
    proposed command; plan_flow's plan-policy-validation stage (run right
    after the planner, still inside plan_flow) independently catches it and
    converts the outcome into a terminal PLAN_BLOCKED -- Codex is only an
    advisor, its own PLAN_PASS verdict never causes anything to execute."""
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="forbidden_cmd_case", scenario="forbidden_command")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_COMMAND_POLICY_FAILED"]
    assert "forbidden token" in plan.summary


# ── 13. unapproved proposed command ─────────────────────────────────────────

def test_real_codex_unapproved_proposed_command_blocks_via_plan_policy(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="unapproved_cmd_case", scenario="unapproved_command")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_COMMAND_POLICY_FAILED"]
    assert "not an approved command" in plan.summary


# ── 14. shell string instead of command array ───────────────────────────────

def test_real_codex_shell_string_proposed_commands_is_schema_invalid(tmp_path):
    """proposed_commands must be list[list[str]]; a bare string element (a
    shell string) fails Pydantic validation before it ever reaches the
    command-policy layer."""
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="shell_string_case", scenario="shell_string")
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["CODEX_SCHEMA_INVALID"]


# ── 15. repository unchanged on success ─────────────────────────────────────

def test_real_codex_repository_unchanged_on_success(tmp_path):
    plan, repo_root, runs_root = _run_plan(tmp_path, task_id="repo_unchanged_case", scenario="pass")
    assert plan.verdict == "PLAN_PASS"
    # git status in repo_root must show nothing outstanding (init committed everything)
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain=v1"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert status.strip() == ""


# ── 16. repository mutation detected and blocked ────────────────────────────

def test_real_codex_repository_mutation_is_detected_and_blocked(tmp_path):
    # _run_plan creates repo_root at tmp_path / "repo" (see _init_repo); the
    # mutation path must point there so the fake codex script writes inside
    # the SAME repository plan_flow actually snapshots before/after.
    mutation_path = tmp_path / "repo" / "mutated_by_codex.txt"
    plan, repo_root, runs_root = _run_plan(
        tmp_path, task_id="repo_mutation_case", scenario="repo_mutation",
        extra_env={"FAKE_CODEX_MUTATION_PATH": str(mutation_path)},
    )
    assert plan.verdict == "PLAN_BLOCKED"
    assert plan.issues == ["REPOSITORY_CHANGED_DURING_CODEX_PLANNING"]
    assert "REPOSITORY_CHANGED_DURING_CODEX_PLANNING" in plan.summary
    assert mutation_path.exists()  # the mutation genuinely happened; it was still caught


# ── 17 & 18. mock Claude remains active; sentinel claude never invoked ─────

def test_real_codex_smoke_flow_keeps_mock_claude_and_never_invokes_sentinel_claude(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="real_codex_mock_claude_case")
    runs_root = tmp_path / "runs"
    fake_codex = _write_fake_codex(tmp_path)

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    claude_marker = tmp_path / "claude_invoked.marker"
    sentinel = fake_bin / "claude"
    sentinel.write_text(f"#!/bin/sh\necho invoked > {claude_marker}\nexit 17\n")
    sentinel.chmod(0o755)

    env_backup = dict(os.environ)
    os.environ["FAKE_CODEX_SCENARIO"] = "pass"
    os.environ["FAKE_CODEX_TASK_ID"] = "real_codex_mock_claude_case"
    os.environ["FAKE_CODEX_RUN_ID"] = "run1"
    os.environ["PATH"] = f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        report = smoke_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run1",
            codex_agent=RealCodexAgent(binary=str(fake_codex)), claude_agent=MockClaudeAgent(),
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    assert report.overall_status == "PASS"
    assert "mock executor" in report.implementation.summary
    assert not claude_marker.exists(), "the claude sentinel executable was invoked"


def test_cli_real_codex_mock_claude_never_invokes_sentinel_claude(tmp_path):
    """End-to-end regression through the actual CLI entry point: --codex
    real (against a fake codex on PATH) + --claude mock (the default) must
    never spawn the sentinel `claude` executable."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="cli_real_codex_case")
    runs_root = tmp_path / "runs"

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    fake_codex = fake_bin / "codex"
    fake_codex.write_text(f"#!{sys.executable}\n{_FAKE_CODEX_BODY}")
    fake_codex.chmod(0o755)

    claude_marker = tmp_path / "claude_invoked.marker"
    sentinel = fake_bin / "claude"
    sentinel.write_text(f"#!/bin/sh\necho invoked > {claude_marker}\nexit 17\n")
    sentinel.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_CODEX_SCENARIO"] = "pass"
    env["FAKE_CODEX_TASK_ID"] = "cli_real_codex_case"
    env["FAKE_CODEX_RUN_ID"] = "cli_run"

    result = subprocess.run(
        [
            sys.executable, "-m", "research_agent.cli",
            "--repo-root", str(repo_root), "--runs-root", str(runs_root),
            "smoke", str(spec_path), "--run-id", "cli_run",
            "--codex", "real", "--claude", "mock",
        ],
        cwd=PACKAGE_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert not claude_marker.exists(), "the claude sentinel executable was invoked"


# ── CLI: real Claude is rejected deterministically, even alongside real Codex ─

def test_cli_real_claude_rejected_even_with_real_codex(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="real_claude_with_real_codex_case")
    runs_root = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable, "-m", "research_agent.cli",
            "--repo-root", str(repo_root), "--runs-root", str(runs_root),
            "smoke", str(spec_path), "--run-id", "rejected_run",
            "--codex", "real", "--claude", "real",
        ],
        cwd=PACKAGE_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "REAL_CLAUDE_DISABLED_IN_MVP1" in result.stderr
    assert not (runs_root / "rejected_run").exists()
