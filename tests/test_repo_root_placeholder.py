# -*- coding: utf-8 -*-
"""Tests for research_agent.policies.repo_root_placeholder -- the ${REPO_ROOT}
mechanism that lets a committed ExperimentSpec's approved_commands stay
portable across the main repository and any future Git worktree, instead of
baking in a development-worktree-specific absolute path.

Covers, at the unit level (this file) and the end-to-end level
(tests/test_execution_flow.py-style run_experiment_flow integration, below):
  - only ${REPO_ROOT} is recognized; any other ${NAME} is rejected
  - '..' path traversal is rejected, both lexically and via a symlink escape
  - no shell/environment expansion is ever used (os.environ is never
    consulted; a bare $VAR with no braces is never touched)
  - the resolved path always lies inside the given repo_root
  - the spec works end-to-end from a differently-named Git worktree
  - both the declared (placeholder) and resolved (absolute) forms of the
    command are persisted to the run directory

Run: PREFECT_LOGGING_LEVEL=WARNING python -m pytest -q tests/test_repo_root_placeholder.py
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
from research_agent.execution_flow import run_experiment_flow
from research_agent.policies import repo_root_placeholder as rrp

# ── unit tests: resolve_token / resolve_command ─────────────────────────────


def test_token_without_placeholder_passes_through_unchanged(tmp_path):
    assert rrp.resolve_token("plain/relative/path.py", tmp_path) == "plain/relative/path.py"
    assert rrp.resolve_token("--flag=value", tmp_path) == "--flag=value"


def test_bare_dollar_without_braces_is_never_treated_as_a_placeholder(tmp_path):
    """No shell-style bare $VAR expansion -- only the exact ${NAME} form is
    ever recognized. A bare $REPO_ROOT (no braces) must stay 100% literal."""
    token = "$REPO_ROOT/script.py"
    assert rrp.resolve_token(token, tmp_path) == token


def test_known_placeholder_is_expanded_to_the_given_repo_root(tmp_path):
    (tmp_path / "sub").mkdir()
    result = rrp.resolve_token("${REPO_ROOT}/sub/script.py", tmp_path)
    assert result == str(tmp_path / "sub" / "script.py")


def test_resolve_command_expands_every_placeholder_token(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "out").mkdir()
    command = ["/usr/bin/python3", "${REPO_ROOT}/a/b.py", "${REPO_ROOT}/out"]
    resolved = rrp.resolve_command(command, tmp_path)
    assert resolved[0] == "/usr/bin/python3"
    assert resolved[1] == str(tmp_path / "a" / "b.py")
    assert resolved[2] == str(tmp_path / "out")


def test_placeholder_embedded_partway_through_a_token_is_rejected(tmp_path):
    """Deliberately narrow: ${REPO_ROOT} is only accepted as the token's OWN
    prefix, never mixed into a larger token like `--out=${REPO_ROOT}/out` --
    that would make the containment check ambiguous about which part of the
    token is actually a path."""
    with pytest.raises(rrp.PlaceholderPolicyViolation) as exc_info:
        rrp.resolve_token("--out=${REPO_ROOT}/out", tmp_path)
    assert exc_info.value.code == "PLACEHOLDER_MUST_BE_TOKEN_PREFIX"


def test_unknown_placeholder_is_rejected(tmp_path):
    with pytest.raises(rrp.PlaceholderPolicyViolation) as exc_info:
        rrp.resolve_token("${NOT_A_REAL_VAR}/script.py", tmp_path)
    assert exc_info.value.code == "UNKNOWN_PLACEHOLDER"


def test_unknown_placeholder_alongside_known_one_still_rejected(tmp_path):
    with pytest.raises(rrp.PlaceholderPolicyViolation) as exc_info:
        rrp.resolve_token("${REPO_ROOT}/${EVIL}/x", tmp_path)
    assert exc_info.value.code == "UNKNOWN_PLACEHOLDER"


def test_traversal_after_placeholder_is_rejected(tmp_path):
    with pytest.raises(rrp.PlaceholderPolicyViolation) as exc_info:
        rrp.resolve_token("${REPO_ROOT}/../../etc/passwd", tmp_path)
    assert exc_info.value.code == "PLACEHOLDER_PATH_TRAVERSAL_REJECTED"


def test_traversal_before_placeholder_is_rejected(tmp_path):
    """Rejected outright: the token doesn't even start with ${REPO_ROOT}
    (the prefix-only rule), which is a strictly stronger guarantee than a
    traversal-specific check -- this shape can never reach the substitution
    step at all."""
    with pytest.raises(rrp.PlaceholderPolicyViolation) as exc_info:
        rrp.resolve_token("../${REPO_ROOT}/x", tmp_path)
    assert exc_info.value.code == "PLACEHOLDER_MUST_BE_TOKEN_PREFIX"


def test_resolve_command_traversal_in_any_token_is_rejected(tmp_path):
    command = ["/usr/bin/python3", "${REPO_ROOT}/ok.py", "${REPO_ROOT}/../escape.py"]
    with pytest.raises(rrp.PlaceholderPolicyViolation) as exc_info:
        rrp.resolve_command(command, tmp_path)
    assert exc_info.value.code == "PLACEHOLDER_PATH_TRAVERSAL_REJECTED"


def test_symlink_escape_via_placeholder_is_rejected(tmp_path):
    """A committed symlink inside repo_root that points OUTSIDE it must not
    let ${REPO_ROOT}/<link>/... escape containment -- mirrors
    execution_policy.assert_within_worktree_no_symlink_escape's threat model."""
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("nope")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "escape_link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(rrp.PlaceholderPolicyViolation) as exc_info:
        rrp.resolve_token("${REPO_ROOT}/escape_link/secret.txt", repo_root)
    assert exc_info.value.code == "PLACEHOLDER_ESCAPES_REPO_ROOT"


def test_valid_subpath_resolves_inside_repo_root(tmp_path):
    (tmp_path / "pkg").mkdir()
    resolved = rrp.resolve_token("${REPO_ROOT}/pkg/mod.py", tmp_path)
    assert Path(resolved).resolve().is_relative_to(tmp_path.resolve())


def test_no_environment_variable_is_ever_consulted(tmp_path, monkeypatch):
    """The placeholder value comes ONLY from the repo_root argument -- never
    from os.environ, even if an env var of the exact same name is set to
    something else entirely (e.g. by a malicious or stale shell)."""
    monkeypatch.setenv("REPO_ROOT", "/totally/different/attacker/path")
    result = rrp.resolve_token("${REPO_ROOT}/x.py", tmp_path)
    assert result == str(tmp_path / "x.py")
    assert "attacker" not in result


def test_no_shell_is_ever_invoked():
    """resolve_token/resolve_command must be implementable, and are
    implemented, entirely with string operations -- no subprocess, no
    os.system, no shell=True, no environment-variable lookup anywhere in
    the module. Checked as call-shaped patterns (not bare substrings) so
    this doesn't also flag the module's own docstrings, which legitimately
    name these as what is deliberately NOT used."""
    import inspect

    source = inspect.getsource(rrp)
    for banned_call in (
        "subprocess.", "os.system(", "shell=True",
        "os.path.expandvars(", "os.environ[", "os.environ.get(", "os.getenv(",
    ):
        assert banned_call not in source, f"{banned_call!r} must never appear in repo_root_placeholder.py"


# ── end-to-end: run_experiment_flow from a differently-named worktree ──────

PILOT_SCRIPT = (
    'import json, os\n'
    'open(os.path.join(os.environ["RESEARCH_AGENT_ARTIFACTS_DIR"], "metrics.json"), "w")'
    '.write(json.dumps({"ok": True, "value": 1.0}))\n'
)


def _init_named_repo(base: Path, name: str) -> Path:
    """Mirrors tests/test_execution_flow.py's _init_repo, but at a caller-
    chosen directory NAME -- specifically NOT 'OWG-agent-pilot1' or anything
    resembling this development worktree, to prove portability."""
    repo = base / name
    repo.mkdir()
    run = lambda *args: subprocess.run(args, cwd=repo, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (repo / "README.md").write_text("test repo\n")
    (repo / "research_agent_sandbox").mkdir()
    (repo / "research_agent_sandbox" / ".gitkeep").write_text("")
    (repo / "pilot_scripts").mkdir()
    (repo / "pilot_scripts" / "hello.py").write_text(PILOT_SCRIPT)
    run("git", "add", ".")
    run("git", "commit", "-q", "-m", "init")
    return repo


def _write_placeholder_spec(tmp_path: Path, *, script_token: str, task_id: str = "t_repo_root") -> Path:
    execution = {
        "execution_mode": "restricted",
        "working_directory_policy": "isolated_run_directory",
        "approved_commands": [[sys.executable, script_token]],
        "required_artifacts": ["metrics.json"],
        "required_metrics": [{"key": "ok", "check": "bool_equals", "value": True}],
        "limits": {
            "max_commands": 1, "max_execution_attempts": 1, "max_wall_clock_seconds": 30,
            "per_command_timeout_seconds": 5, "max_repair_rounds": 0,
            "max_total_codex_invocations": 3, "max_total_claude_invocations": 3,
        },
    }
    spec: dict = {
        "task_id": task_id,
        "goal": "repo_root placeholder portability test",
        "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": [sys.executable, "-c", "pass"],
        "seeds": [0],
        "timeouts": {"planner_seconds": 20, "executor_seconds": 20, "smoke_seconds": 20, "verifier_seconds": 20, "reviewer_seconds": 20},
        "max_run_count": 10,
        "execution": execution,
    }
    path = tmp_path / f"{task_id}.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def test_spec_with_repo_root_placeholder_runs_from_a_differently_named_worktree(tmp_path):
    repo = _init_named_repo(tmp_path, "a_totally_different_worktree_name")
    spec_path = _write_placeholder_spec(tmp_path, script_token="${REPO_ROOT}/pilot_scripts/hello.py")
    runs_root = tmp_path / "runs"

    report = run_experiment_flow(
        spec_path, repo_root=repo, runs_root=runs_root, run_id="run1",
        execution_worktrees_root=tmp_path / "ewt", execute=True,
    )

    assert report.overall_status == "PASS", report.reason
    assert "a_totally_different_worktree_name" in str(repo)


def test_declared_and_resolved_commands_both_persisted(tmp_path):
    repo = _init_named_repo(tmp_path, "another_worktree_layout")
    spec_path = _write_placeholder_spec(tmp_path, script_token="${REPO_ROOT}/pilot_scripts/hello.py")
    runs_root = tmp_path / "runs"

    report = run_experiment_flow(
        spec_path, repo_root=repo, runs_root=runs_root, run_id="run1",
        execution_worktrees_root=tmp_path / "ewt", execute=True,
    )
    assert report.overall_status == "PASS"

    command_dir = runs_root / "run1" / "execution" / "attempt_00" / "command_00"
    on_disk = json.loads((command_dir / "command.json").read_text())

    assert on_disk["approved_command"] == [sys.executable, "${REPO_ROOT}/pilot_scripts/hello.py"]
    expected_resolved = str((repo / "pilot_scripts" / "hello.py").resolve())
    assert on_disk["executed_command"] == [sys.executable, expected_resolved]
    # the resolved script path must lie inside the recorded repository worktree
    assert Path(expected_resolved).is_relative_to(repo.resolve())
    assert "${REPO_ROOT}" not in on_disk["executed_command"][1]


def test_unknown_placeholder_in_spec_blocks_run_with_policy_failure(tmp_path):
    repo = _init_named_repo(tmp_path, "yet_another_worktree")
    spec_path = _write_placeholder_spec(
        tmp_path, script_token="${NOT_A_REAL_VAR}/pilot_scripts/hello.py", task_id="t_unknown_placeholder",
    )
    runs_root = tmp_path / "runs"

    report = run_experiment_flow(
        spec_path, repo_root=repo, runs_root=runs_root, run_id="run1",
        execution_worktrees_root=tmp_path / "ewt", execute=True,
    )
    assert report.final_state == "POLICY_FAILURE"
    assert "UNKNOWN_PLACEHOLDER" in (report.reason or "")


def test_traversal_placeholder_in_spec_blocks_run_with_policy_failure(tmp_path):
    repo = _init_named_repo(tmp_path, "traversal_worktree")
    spec_path = _write_placeholder_spec(
        tmp_path, script_token="${REPO_ROOT}/../../etc/passwd", task_id="t_traversal",
    )
    runs_root = tmp_path / "runs"

    report = run_experiment_flow(
        spec_path, repo_root=repo, runs_root=runs_root, run_id="run1",
        execution_worktrees_root=tmp_path / "ewt", execute=True,
    )
    assert report.final_state == "POLICY_FAILURE"
    assert "PLACEHOLDER_PATH_TRAVERSAL_REJECTED" in (report.reason or "")


def test_no_execute_never_attempts_placeholder_resolution_side_effects(tmp_path):
    """Sanity: even a spec with a valid placeholder never touches the
    filesystem beyond validation when --execute is not passed."""
    repo = _init_named_repo(tmp_path, "no_execute_worktree")
    spec_path = _write_placeholder_spec(tmp_path, script_token="${REPO_ROOT}/pilot_scripts/hello.py", task_id="t_no_exec")
    runs_root = tmp_path / "runs"

    report = run_experiment_flow(
        spec_path, repo_root=repo, runs_root=runs_root, run_id="run1",
        execution_worktrees_root=tmp_path / "ewt", execute=False,
    )
    assert report.final_state == "EXECUTION_NOT_REQUESTED"
