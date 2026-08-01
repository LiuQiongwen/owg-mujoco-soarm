"""MVP4 command-policy unit tests: research_agent.policies.experiment_commands.

Covers (see the MVP4 task contract's "Fake execution tests" list):
  3.  command argv exact-match enforcement
  4.  unapproved command blocked
  5.  shell string blocked
  6.  bash -c blocked
  7.  sudo/docker/curl/wget blocked
  8.  GPU token/executable blocked
  9.  robot/serial/ROS command blocked
  10. training command blocked
  41. no network command
  42. no GPU
  43. no robot
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.models import ExperimentSpec
from research_agent.policies import experiment_commands as ec


def _spec(approved_commands, **overrides) -> ExperimentSpec:
    execution = {"approved_commands": approved_commands, **overrides}
    return ExperimentSpec.model_validate({
        "task_id": "cmdpolicy_test", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": ["python", "-c", "pass"], "execution": execution,
    })


APPROVED = [sys.executable, "-c", "print(1)"]


def test_exact_match_enforcement_passes_for_approved_command():
    spec = _spec([APPROVED])
    ec.authorize_execution(APPROVED, spec)  # must not raise


def test_exact_match_enforcement_rejects_near_miss():
    spec = _spec([APPROVED])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution([sys.executable, "-c", "print(2)"], spec)
    assert exc.value.code == "UNAPPROVED_COMMAND"


def test_unapproved_command_blocked_even_if_structurally_valid():
    spec = _spec([APPROVED])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution([sys.executable, "-c", "print('totally different')"], spec)
    assert exc.value.code == "UNAPPROVED_COMMAND"


def test_shell_string_blocked():
    spec = _spec([APPROVED])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution("python -c 'print(1)'", spec)
    assert exc.value.code == "SHELL_STRING_REJECTED"


def test_bash_dash_c_blocked():
    spec = _spec([["bash", "-c", "echo hi"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution(["bash", "-c", "echo hi"], spec)
    assert exc.value.code == "FORBIDDEN_EXECUTABLE"


@pytest.mark.parametrize("executable", ["sudo", "docker", "podman", "curl", "wget", "pip", "conda", "apt", "git", "scp", "ssh", "nc", "sh", "zsh"])
def test_forbidden_executables_blocked(executable):
    spec = _spec([[executable, "--version"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution([executable, "--version"], spec)
    assert exc.value.code == "FORBIDDEN_EXECUTABLE"


def test_gpu_token_blocked():
    spec = _spec([[sys.executable, "--gpu"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution([sys.executable, "--gpu"], spec)
    assert exc.value.code == "FORBIDDEN_TOKEN"


def test_gpu_executable_blocked():
    spec = _spec([["nvidia-smi"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution(["nvidia-smi"], spec)
    assert exc.value.code == "FORBIDDEN_EXECUTABLE"


def test_robot_serial_command_blocked():
    spec = _spec([[sys.executable, "-c", "open('/dev/ttyUSB0')"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution([sys.executable, "-c", "open('/dev/ttyUSB0')"], spec)
    assert exc.value.code == "FORBIDDEN_TOKEN"


def test_ros_launch_command_blocked():
    spec = _spec([["roslaunch", "pkg", "file.launch"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution(["roslaunch", "pkg", "file.launch"], spec)
    assert exc.value.code == "FORBIDDEN_EXECUTABLE"


def test_training_command_blocked():
    spec = _spec([[sys.executable, "train_lggsn_pairwise.py"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution([sys.executable, "train_lggsn_pairwise.py"], spec)
    assert exc.value.code == "FORBIDDEN_TOKEN"


def test_url_in_command_blocked():
    spec = _spec([[sys.executable, "-c", "print(1)", "https://example.com"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution([sys.executable, "-c", "print(1)", "https://example.com"], spec)
    assert exc.value.code == "NETWORK_INDICATOR_REJECTED"


def test_executable_not_on_allowlist_blocked():
    spec = _spec([["node", "script.js"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution(["node", "script.js"], spec)
    assert exc.value.code == "FORBIDDEN_EXECUTABLE"


def test_python_c_script_body_exempt_from_metachar_scan():
    """The one argv element immediately following -c is a whole inline
    Python script and legitimately contains ';', '>', braces, etc."""
    script = 'import json; d={"a": 1}; print(d if 2>1 else None)'
    spec = _spec([[sys.executable, "-c", script]])
    ec.authorize_execution([sys.executable, "-c", script], spec)  # must not raise


def test_metacharacter_in_a_non_script_token_is_rejected():
    spec = _spec([[sys.executable, "-c;rm -rf /"]])
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.authorize_execution([sys.executable, "-c;rm -rf /"], spec)
    assert exc.value.code == "SHELL_METACHARACTER_REJECTED"


def test_command_count_limit_enforced_at_spec_construction():
    """ExecutionSpec's own Pydantic validator rejects approved_commands
    longer than limits.max_commands -- an unsafe spec never even loads."""
    two = [[sys.executable, "-c", "print(1)"], [sys.executable, "-c", "print(2)"]]
    with pytest.raises(Exception):
        _spec(two, limits={"max_commands": 1})


def test_malformed_command_shape_rejected():
    with pytest.raises(ec.ExperimentCommandPolicyViolation) as exc:
        ec.assert_command_shape([""])
    assert exc.value.code == "MALFORMED_COMMAND"


def test_validate_approved_commands_flags_unsafe_spec_entries():
    """Defense in depth beyond exact-match: research_agent.execution_flow
    calls validate_approved_commands(spec) once after planning, independent
    of whatever specific command is later authorized -- so a spec whose
    'approved' list itself contains an unsafe entry (curl, a forbidden
    executable, etc.) is caught even before any authorize_execution call."""
    spec = _spec([["curl", "http://example.com"]])
    violations = ec.validate_approved_commands(spec)
    assert violations, "an unsafe approved_commands entry must be flagged at validation time"
    assert any("FORBIDDEN_EXECUTABLE" in v or "curl" in v for v in violations)


def test_validate_approved_commands_clean_for_safe_spec():
    spec = _spec([APPROVED])
    assert ec.validate_approved_commands(spec) == []
