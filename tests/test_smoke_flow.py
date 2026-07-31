"""End-to-end tests for the Prefect flow, using mock agent adapters only.

Never invokes the real `codex` or `claude` CLIs. Each test builds a
throwaway Git repository under tmp_path (never the real research repository)
so isolated-worktree creation/removal is exercised for real, without ever
touching this repository's own worktrees or branches.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.agents.codex import MockCodexAgent
from research_agent.agents.claude import MockClaudeAgent
from research_agent.cli import build_parser
from research_agent.flow import plan_flow, smoke_flow
from research_agent.tasks.experiment import RunAlreadyExistsError, RunLimitExceededError

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


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


def _write_spec(
    tmp_path: Path,
    *,
    task_id: str = "pytest_smoke",
    max_run_count: int = 5,
    smoke_command=None,
    required_metrics=None,
) -> Path:
    if smoke_command is None:
        smoke_command = [
            sys.executable,
            "-c",
            "import json, os\n"
            "out = os.path.join(os.environ['RESEARCH_AGENT_ARTIFACTS_DIR'], 'metrics.json')\n"
            "json.dump({'smoke_ok': True, 'value': 1.0}, open(out, 'w'))\n",
        ]
    spec = {
        "task_id": task_id,
        "goal": "pytest smoke experiment",
        "allowed_paths": ["experiments"],
        "forbidden_paths": [],
        "smoke_command": smoke_command,
        "seeds": [0],
        "timeouts": {
            "planner_seconds": 30, "executor_seconds": 30, "smoke_seconds": 30,
            "verifier_seconds": 30, "reviewer_seconds": 30,
        },
        "required_metrics": (
            required_metrics if required_metrics is not None else [{"name": "smoke_ok", "type": "bool"}]
        ),
        "max_run_count": max_run_count,
    }
    path = tmp_path / f"{task_id}.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def test_valid_smoke_flow_passes_and_writes_metrics(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="valid_case")
    runs_root = tmp_path / "runs"

    report = smoke_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run_valid",
        codex_agent=MockCodexAgent(), claude_agent=MockClaudeAgent(),
    )

    assert report.overall_status == "PASS"
    assert report.plan.verdict == "PLAN_PASS"
    assert report.implementation.verdict == "IMPLEMENTATION_READY_FOR_REVIEW"
    assert report.verification.verdict == "PASS"
    assert report.review.verdict == "REVIEW_PASS"

    run_dir = runs_root / "run_valid"
    assert (run_dir / "report.json").exists()
    assert (run_dir / "plan.json").exists()
    assert (run_dir / "implementation.json").exists()
    assert (run_dir / "verification.json").exists()
    assert (run_dir / "review.json").exists()
    assert (run_dir / "git_sha.txt").exists()
    assert (run_dir / "git_diff.patch").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "artifacts" / "metrics.json").exists()

    # the isolated worktree must be cleaned up after the run
    assert not (run_dir / "worktree").exists()
    branches = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--list", "research-agent-run/*"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert branches.strip() == ""


def test_existing_run_id_is_not_overwritten(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="collision_case", max_run_count=10)
    runs_root = tmp_path / "runs"

    smoke_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="dup_run",
        codex_agent=MockCodexAgent(), claude_agent=MockClaudeAgent(),
    )
    with pytest.raises(RunAlreadyExistsError):
        smoke_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="dup_run",
            codex_agent=MockCodexAgent(), claude_agent=MockClaudeAgent(),
        )


def test_plan_blocked_short_circuits_before_any_command_runs(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="blocked_case")
    runs_root = tmp_path / "runs"

    report = smoke_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run_blocked",
        codex_agent=MockCodexAgent(plan_verdict="PLAN_BLOCKED"), claude_agent=MockClaudeAgent(),
    )

    assert report.overall_status == "BLOCKED"
    assert report.blocked_stage == "codex_planner"
    assert report.implementation is None
    assert report.verification is None

    run_dir = runs_root / "run_blocked"
    # the deterministic smoke command must never have run
    assert not (run_dir / "artifacts" / "metrics.json").exists()
    # and no worktree should have been created, let alone leaked
    assert not (run_dir / "worktree").exists()


def test_verifier_fail_is_not_overridden_by_reviewer_pass(tmp_path):
    repo_root = _init_repo(tmp_path)
    # smoke_command deliberately never writes metrics.json
    spec_path = _write_spec(
        tmp_path, task_id="verifier_fail_case",
        smoke_command=[sys.executable, "-c", "print('no metrics written')"],
    )
    runs_root = tmp_path / "runs"

    report = smoke_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run_verifier_fail",
        codex_agent=MockCodexAgent(review_verdict="REVIEW_PASS"), claude_agent=MockClaudeAgent(),
    )

    assert report.verification.verdict == "FAIL"
    assert report.review.verdict == "REVIEW_PASS"
    assert report.overall_status == "FAIL"  # verifier FAIL wins regardless of reviewer PASS


def test_max_run_count_is_enforced(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="limited_case", max_run_count=1)
    runs_root = tmp_path / "runs"

    smoke_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run_one",
        codex_agent=MockCodexAgent(), claude_agent=MockClaudeAgent(),
    )
    with pytest.raises(RunLimitExceededError):
        smoke_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run_two",
            codex_agent=MockCodexAgent(), claude_agent=MockClaudeAgent(),
        )


def test_plan_flow_runs_only_the_planner_stage(tmp_path):
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="plan_only_case")
    runs_root = tmp_path / "runs"

    plan = plan_flow(
        spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run_plan_only",
        codex_agent=MockCodexAgent(),
    )

    assert plan.verdict == "PLAN_PASS"
    run_dir = runs_root / "run_plan_only"
    assert (run_dir / "plan.json").exists()
    # plan_flow never creates a worktree or runs the smoke command
    assert not (run_dir / "worktree").exists()
    assert not (run_dir / "artifacts" / "metrics.json").exists()


def test_smoke_flow_never_invokes_a_real_cli_binary(tmp_path):
    """Sanity check that the mock agents genuinely never spawn codex/claude:
    patch subprocess.run to fail loudly if either binary name ever appears."""
    import research_agent.subprocess_runner as sr

    real_run = sr.subprocess.run

    def guarded_run(command, *args, **kwargs):
        assert command[0] not in ("codex", "claude"), f"real CLI invoked: {command}"
        return real_run(command, *args, **kwargs)

    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="no_real_cli_case")
    runs_root = tmp_path / "runs"

    from unittest.mock import patch

    with patch("research_agent.subprocess_runner.subprocess.run", side_effect=guarded_run):
        report = smoke_flow(
            spec_path, repo_root=repo_root, runs_root=runs_root, run_id="run_no_real_cli",
            codex_agent=MockCodexAgent(), claude_agent=MockClaudeAgent(),
        )
    assert report.overall_status == "PASS"


# ── regression: the CLI's default (no --agents flag) must stay mock ────────

def test_cli_default_codex_and_claude_arguments_are_mock():
    """argparse-level guarantee: omitting --codex/--claude must resolve to
    "mock" on both `plan` and `smoke`, so real agents are always an explicit,
    independently-selected opt-in (never bundled behind a single flag)."""
    parser = build_parser()
    plan_args = parser.parse_args(["plan", "spec.yaml"])
    assert plan_args.codex == "mock"
    smoke_args = parser.parse_args(["smoke", "spec.yaml"])
    assert smoke_args.codex == "mock"
    assert smoke_args.claude == "mock"


def test_cli_smoke_rejects_real_claude_deterministically(tmp_path):
    """--claude real must never invoke anything: it is rejected up front
    with the exact deterministic code REAL_CLAUDE_DISABLED_IN_MVP1, and the
    pipeline must never even start (no run directory should be created)."""
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="real_claude_rejected_case")
    runs_root = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable, "-m", "research_agent.cli",
            "--repo-root", str(repo_root), "--runs-root", str(runs_root),
            "smoke", str(spec_path), "--run-id", "rejected_run",
            "--claude", "real",
        ],
        cwd=PACKAGE_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "REAL_CLAUDE_DISABLED_IN_MVP1" in result.stderr
    assert not (runs_root / "rejected_run").exists()


def test_cli_smoke_default_never_invokes_codex_or_claude_executables(tmp_path):
    """End-to-end regression: run the actual CLI entry point (as a
    subprocess, exactly as a user would) with NO --agents flag, on a PATH
    that contains sentinel `codex` and `claude` executables which record
    whether they were ever launched. If the default ever silently invoked a
    real agent, the corresponding sentinel would leave a marker file.
    """
    repo_root = _init_repo(tmp_path)
    spec_path = _write_spec(tmp_path, task_id="cli_default_case")
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
            "smoke", str(spec_path), "--run-id", "cli_default_run",
            # deliberately NOT passing --agents: proves the default is mock
        ],
        cwd=PACKAGE_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert not markers["codex"].exists(), "the codex sentinel executable was invoked"
    assert not markers["claude"].exists(), "the claude sentinel executable was invoked"
