# -*- coding: utf-8 -*-
"""Spec/policy tests for experiments/lggsn_tier3_ik_pilot.yaml -- the LGGSN
Tier-3 IK pilot's ExperimentSpec, checked against the same research_agent
policy modules the MVP4 `run-experiment` CLI actually uses. No mujoco
dependency (research_agent policy code is pure Python), so this runs under
the research-agent venv exactly like the other 292 research-agent tests.

Run: PREFECT_LOGGING_LEVEL=WARNING python -m pytest -q tests/test_lggsn_tier3_ik_pilot_spec.py
"""
import hashlib
import os
from pathlib import Path

import pytest
import yaml

from research_agent.models import ExperimentSpec
from research_agent.policies import experiment_commands, repo_root_placeholder

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPEC_PATH = os.path.join(_REPO_ROOT, "experiments", "lggsn_tier3_ik_pilot.yaml")
FIXTURE_PATH = os.path.join(
    _REPO_ROOT, "research_agent_pilots", "lggsn_tier3_ik", "fixtures", "candidate_poses.json"
)


@pytest.fixture(scope="module")
def spec() -> ExperimentSpec:
    with open(SPEC_PATH) as f:
        data = yaml.safe_load(f)
    return ExperimentSpec.model_validate(data)


@pytest.fixture(scope="module")
def resolved_command(spec) -> list[str]:
    """The one approved command, ${REPO_ROOT}-resolved against THIS
    checkout -- exactly what execution_flow.py computes at runtime before
    authorize_execution ever sees it. Every authorization-gate test below
    exercises this resolved form, matching the real pipeline: authorize_execution
    never does placeholder expansion itself (see repo_root_placeholder.py's
    module docstring)."""
    return repo_root_placeholder.resolve_command(spec.execution.approved_commands[0], Path(_REPO_ROOT))


@pytest.fixture(scope="module")
def resolved_approved_commands(spec) -> list[list[str]]:
    return [repo_root_placeholder.resolve_command(c, Path(_REPO_ROOT)) for c in spec.execution.approved_commands]


def test_spec_loads_and_validates(spec):
    assert spec.task_id == "lggsn_tier3_ik_pilot"
    assert spec.execution is not None
    assert spec.execution.execution_mode == "restricted"


def test_execution_flags_block_gpu_network_robot_training(spec):
    execution = spec.execution
    assert execution.cpu_only is True
    assert execution.network_allowed is False
    assert execution.gpu_allowed is False
    assert execution.robot_allowed is False
    assert execution.training_allowed is False
    assert execution.confirmatory is False


def test_exactly_one_approved_command(spec):
    assert len(spec.execution.approved_commands) == 1


def test_declared_approved_command_uses_repo_root_placeholder_not_a_hardcoded_worktree_path(spec):
    """The whole point of this mechanism: the COMMITTED spec must never
    contain a development-worktree-specific absolute path for the script
    argument -- only ${REPO_ROOT}, expanded at runtime."""
    command = spec.execution.approved_commands[0]
    assert any("${REPO_ROOT}" in tok for tok in command)
    for tok in command:
        assert "OWG-agent-pilot1" not in tok
        assert not tok.startswith("/lena/")


def test_approved_command_passes_full_policy_gate(spec, resolved_command, resolved_approved_commands):
    violations = experiment_commands.validate_approved_commands(spec, commands=resolved_approved_commands)
    assert violations == []
    # authorize_execution additionally requires an exact match against the
    # resolved approved-commands list -- re-run the full gate, not just the
    # spec-load-time subset, to prove the one approved command is actually
    # authorizable at execution time, exactly as execution_flow.py does it.
    experiment_commands.authorize_execution(
        resolved_command, spec, approved_commands_override=resolved_approved_commands
    )  # must not raise


def test_approved_command_uses_only_allowlisted_python_executable(resolved_command):
    basename = os.path.basename(resolved_command[0]).lower()
    assert basename in experiment_commands.ALLOWED_EXECUTABLE_BASENAMES


def test_restricted_policy_rejects_non_exact_command(spec, resolved_command, resolved_approved_commands):
    """A command that merely resembles the approved one (extra flag) must
    be rejected -- proves the policy does exact-match, not prefix/fuzzy
    matching."""
    command = list(resolved_command) + ["--extra-flag"]
    with pytest.raises(experiment_commands.ExperimentCommandPolicyViolation) as exc_info:
        experiment_commands.authorize_execution(command, spec, approved_commands_override=resolved_approved_commands)
    assert exc_info.value.code == "UNAPPROVED_COMMAND"


def test_restricted_policy_rejects_conda_run_variant(spec, resolved_command, resolved_approved_commands):
    """`conda run -n tango python ...` was the first invocation shape tried
    during development and is exactly why the approved command instead
    invokes the tango env's python interpreter by absolute path: `conda` is
    on the MVP4 command policy's forbidden-executable list."""
    command = ["conda", "run", "-n", "tango"] + list(resolved_command)
    with pytest.raises(experiment_commands.ExperimentCommandPolicyViolation) as exc_info:
        experiment_commands.authorize_execution(command, spec, approved_commands_override=resolved_approved_commands)
    assert exc_info.value.code == "FORBIDDEN_EXECUTABLE"


def test_restricted_policy_rejects_shell_wrapped_command(spec, resolved_command, resolved_approved_commands):
    command = ["bash", "-c", " ".join(resolved_command)]
    with pytest.raises(experiment_commands.ExperimentCommandPolicyViolation) as exc_info:
        experiment_commands.authorize_execution(command, spec, approved_commands_override=resolved_approved_commands)
    assert exc_info.value.code == "FORBIDDEN_EXECUTABLE"


@pytest.mark.parametrize("bad_token", [
    "--gpu", "cuda", "nvidia-smi", "--confirmatory", "sudo", "--network",
])
def test_restricted_policy_rejects_gpu_network_confirmatory_tokens(spec, resolved_command, resolved_approved_commands, bad_token):
    command = list(resolved_command) + [bad_token]
    with pytest.raises(experiment_commands.ExperimentCommandPolicyViolation) as exc_info:
        experiment_commands.authorize_execution(command, spec, approved_commands_override=resolved_approved_commands)
    assert exc_info.value.code == "FORBIDDEN_TOKEN"


def test_artifact_paths_are_relative_and_confined(spec):
    execution = spec.execution
    for p in execution.required_artifacts + execution.allowed_output_paths:
        assert not p.startswith("/")
        assert ".." not in p.split("/")
    assert set(execution.required_artifacts) == {"metrics.json", "candidate_features.json"}


def test_no_repair_rounds_and_tight_wall_clock_budget(spec):
    limits = spec.execution.limits
    assert limits.max_repair_rounds == 0
    assert limits.max_commands == 1
    assert limits.max_execution_attempts == 1
    # Task requirement: pilot must complete in well under 30 seconds.
    assert limits.max_wall_clock_seconds <= 30


def test_required_metrics_do_not_invent_a_success_threshold(spec):
    """convergence_rate must only be range-checked into [0, 1] (a structural
    tautology), never pinned to an invented pass/fail bar like '>90%'."""
    checks = {m.key: m for m in spec.execution.required_metrics}
    assert "convergence_rate" in checks
    rate_check = checks["convergence_rate"]
    assert rate_check.check == "float_range"
    assert rate_check.min_value == 0.0
    assert rate_check.max_value == 1.0


def test_pinned_fixture_digest_matches_the_real_committed_fixture(spec):
    """Guards against spec/fixture drift: if the fixture file is ever edited
    without updating the spec (or vice versa), this must fail loudly rather
    than the live MVP4 run silently pinning a stale digest."""
    checks = {m.key: m for m in spec.execution.required_metrics}
    with open(FIXTURE_PATH, "rb") as f:
        actual_digest = hashlib.sha256(f.read()).hexdigest()
    assert checks["fixture_sha256"].value == actual_digest


def test_required_metrics_cover_the_documented_schema(spec):
    keys = {m.key for m in spec.execution.required_metrics}
    expected = {
        "candidate_count", "converged_count", "convergence_rate",
        "all_residuals_finite", "all_joint_deltas_finite",
        "max_residual", "max_joint_delta", "fixture_sha256",
        "deterministic_digest", "pilot_ok",
    }
    assert expected.issubset(keys)


def test_task_id_and_goal_contain_no_confirmatory_indicators(spec):
    """execution_flow._assert_no_confirmatory_indicators does a substring
    scan over task_id+goal for markers like 'confirmatory'/'paper_final' --
    including inside a NEGATED sentence ("not a confirmatory claim") still
    trips it. Caught live during development: the original goal text said
    "Not a confirmatory ... claim" and was rejected with
    CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL before planning even
    started. Guards against that regression."""
    markers = ("confirmatory", "paper_final", "paper-final", "final_result", "final-result")
    haystack = f"{spec.task_id} {spec.goal}".lower()
    for marker in markers:
        assert marker not in haystack, f"goal/task_id contains confirmatory-indicator substring {marker!r}"


def test_fixture_file_is_committed_and_within_size_bounds(spec):
    """3-10 candidates per the task's fixture-size requirement."""
    import json
    with open(FIXTURE_PATH) as f:
        doc = json.load(f)
    assert 3 <= len(doc["candidates"]) <= 10
