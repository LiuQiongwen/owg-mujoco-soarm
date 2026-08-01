# -*- coding: utf-8 -*-
"""Spec/policy tests for experiments/lggsn_suite/*.yaml -- the four
core-matrix LGGSN evaluation specs, checked against the same research_agent
policy modules the MVP4 `run-experiment` CLI actually uses. Pure Python
(research_agent policy code has no torch dependency), runs under the
research-agent venv.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from research_agent.models import ExperimentSpec
from research_agent.policies import experiment_commands, repo_root_placeholder

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SUITE_DIR = os.path.join(_REPO_ROOT, "experiments", "lggsn_suite")
_MATRIX_NAMES = ("base", "nodist", "nozrel", "full_v2")


def _spec_path(name: str) -> str:
    return os.path.join(_SUITE_DIR, f"lggsn_suite_{name}.yaml")


@pytest.fixture(params=_MATRIX_NAMES)
def spec_name(request):
    return request.param


@pytest.fixture
def spec(spec_name) -> ExperimentSpec:
    with open(_spec_path(spec_name)) as f:
        data = yaml.safe_load(f)
    return ExperimentSpec.model_validate(data)


@pytest.fixture
def resolved_approved_commands(spec):
    return [repo_root_placeholder.resolve_command(c, Path(_REPO_ROOT)) for c in spec.execution.approved_commands]


def test_all_four_spec_files_exist():
    for name in _MATRIX_NAMES:
        assert os.path.isfile(_spec_path(name)), name


def test_spec_loads_and_is_restricted_cpu_only(spec):
    execution = spec.execution
    assert execution.execution_mode == "restricted"
    assert execution.cpu_only is True
    assert execution.network_allowed is False
    assert execution.gpu_allowed is False
    assert execution.robot_allowed is False
    assert execution.training_allowed is False
    assert execution.confirmatory is False


def test_declared_command_uses_repo_root_placeholder_not_a_hardcoded_worktree_path(spec):
    command = spec.execution.approved_commands[0]
    assert any("${REPO_ROOT}" in tok for tok in command)
    for tok in command:
        assert "OWG-agent-pilot1" not in tok
        assert not tok.startswith("/lena/")


def test_approved_command_uses_system_python3_not_the_torch_environment(spec):
    """publish_eval.py is torch-free by design (see docs/LGGSN_EVAL_SUITE.md)
    -- it must never be launched with the tango env's interpreter, which
    would defeat the whole point of keeping the MVP4-side command tiny and
    portable."""
    command = spec.execution.approved_commands[0]
    assert command[0] == "/usr/bin/python3"
    assert "tango" not in command[0]


def test_approved_command_passes_full_policy_gate(spec, resolved_approved_commands):
    violations = experiment_commands.validate_approved_commands(spec, commands=resolved_approved_commands)
    assert violations == []
    experiment_commands.authorize_execution(
        resolved_approved_commands[0], spec, approved_commands_override=resolved_approved_commands
    )  # must not raise


def test_task_id_and_goal_contain_no_confirmatory_indicators(spec):
    markers = ("confirmatory", "paper_final", "paper-final", "final_result", "final-result")
    haystack = f"{spec.task_id} {spec.goal}".lower()
    for marker in markers:
        assert marker not in haystack


def test_required_artifacts_are_exactly_the_three_manifest_files(spec):
    assert set(spec.execution.required_artifacts) == {
        "metrics.json", "checkpoint_manifest.json", "evaluation_manifest.json",
    }


def test_required_metrics_pin_the_real_computed_values(spec, spec_name):
    checks = {m.key: m for m in spec.execution.required_metrics}
    metrics_path = os.path.join(
        _REPO_ROOT, "research_agent_pilots", "lggsn_suite", "eval_outputs", spec_name, "metrics.json",
    )
    with open(metrics_path) as f:
        real_metrics = json.load(f)
    assert checks["checkpoint_name"].value == real_metrics["checkpoint_name"] == spec_name
    assert checks["deterministic_digest"].value == real_metrics["deterministic_digest"]
    assert checks["pair_accuracy"].value == pytest.approx(real_metrics["pair_accuracy"])
    assert checks["eligible_group_count"].value == real_metrics["eligible_group_count"]
    assert checks["eligible_pair_count"].value == real_metrics["eligible_pair_count"]


def test_pinned_digest_env_override_matches_required_metric(spec):
    checks = {m.key: m for m in spec.execution.required_metrics}
    assert spec.execution.environment_overrides["LGGSN_PUBLISH_EXPECTED_DIGEST"] == checks["deterministic_digest"].value


def test_convergence_style_metrics_are_not_pinned_to_an_invented_threshold(spec):
    """pair_accuracy is pinned to the exact, real, already-computed value
    (float_equals with a tight tolerance) -- not compared against an
    invented pass/fail bar like '>70%'. Confirms the check kind, not a bound."""
    checks = {m.key: m for m in spec.execution.required_metrics}
    assert checks["pair_accuracy"].check == "float_equals"
    assert checks["pair_accuracy"].tolerance <= 1e-5


def test_all_four_specs_reference_the_same_dataset_sha256(spec):
    checks = {m.key: m for m in spec.execution.required_metrics}
    assert checks["dataset_sha256"].value == "30ca2c398de4bc451691ae6e802e619e7838e0b8bce12a4eba8115b6ff4c42b5"


def test_no_development_worktree_path_anywhere_in_the_committed_yaml(spec_name):
    with open(_spec_path(spec_name)) as f:
        content = f.read()
    assert "OWG-agent-pilot1" not in content
    assert "/lena/projects" not in content
