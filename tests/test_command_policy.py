import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.models import ExperimentSpec, PlanResult
from research_agent.policies.commands import (
    CommandPolicyViolation,
    assert_command_approved,
    find_forbidden_token,
    is_command_approved,
    validate_plan_commands,
)


def _spec(**overrides):
    base = dict(
        task_id="demo",
        goal="demo",
        allowed_paths=["experiments"],
        smoke_command=["python", "-c", "print('ok')"],
    )
    base.update(overrides)
    return ExperimentSpec(**base)


# ── exact-match approval (requirement: no LLM command runs unless it
#    exactly matches an approved command in the specification) ─────────────

def test_exact_match_of_smoke_command_is_approved():
    spec = _spec()
    assert is_command_approved(["python", "-c", "print('ok')"], spec)
    assert_command_approved(["python", "-c", "print('ok')"], spec)  # does not raise


def test_near_match_is_not_approved():
    spec = _spec()
    with pytest.raises(CommandPolicyViolation):
        assert_command_approved(["python", "-c", "print('ok')", "--extra"], spec)


def test_unrelated_command_is_not_approved():
    spec = _spec()
    with pytest.raises(CommandPolicyViolation):
        assert_command_approved(["ls", "-la"], spec)


def test_confirmatory_command_not_approved_by_default():
    spec = _spec(confirmatory_command=["python", "train_full.py"])
    assert not is_command_approved(["python", "train_full.py"], spec)
    with pytest.raises(CommandPolicyViolation):
        assert_command_approved(["python", "train_full.py"], spec)


def test_confirmatory_command_approved_only_with_explicit_flag():
    spec = _spec(confirmatory_command=["python", "train_full.py"])
    assert is_command_approved(["python", "train_full.py"], spec, allow_confirmatory=True)
    assert_command_approved(["python", "train_full.py"], spec, allow_confirmatory=True)


# ── forbidden-token scan (GPU / robot / confirmatory / Docker / sudo /
#    git push / full training) ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "command",
    [
        ["sudo", "python", "run.py"],
        ["docker", "run", "image"],
        ["git", "push", "origin", "main"],
        ["python", "run.py", "--gpu"],
        ["python", "run.py", "--device=cuda"],
        ["nvidia-smi"],
        ["python", "scripts/soarm_real_backend.py"],
        ["python", "lerobot_train.py", "--epochs", "100"],
    ],
)
def test_forbidden_tokens_are_detected(command):
    assert find_forbidden_token(command) is not None


def test_forbidden_token_blocks_even_an_exact_match_command():
    spec = _spec(smoke_command=["sudo", "python", "-c", "print('ok')"])
    with pytest.raises(CommandPolicyViolation):
        assert_command_approved(["sudo", "python", "-c", "print('ok')"], spec)


def test_harmless_command_has_no_forbidden_token():
    assert find_forbidden_token(["python", "-c", "print('ok')"]) is None


def test_forbidden_token_detection_is_case_insensitive():
    assert find_forbidden_token(["SUDO", "python", "run.py"]) is not None
    assert find_forbidden_token(["Docker", "run", "x"]) is not None


# ── plan-policy-validation: any LLM-proposed command must also pass the
#    same gate before the executor stage is reached ────────────────────────

def _plan(proposed_commands):
    return PlanResult(
        task_id="demo", run_id="run1", verdict="PLAN_PASS",
        summary="test plan", proposed_commands=proposed_commands,
    )


def test_plan_with_no_proposed_commands_is_clean():
    spec = _spec()
    assert validate_plan_commands(_plan([]), spec) == []


def test_plan_proposing_the_approved_command_is_clean():
    spec = _spec()
    plan = _plan([["python", "-c", "print('ok')"]])
    assert validate_plan_commands(plan, spec) == []


def test_plan_proposing_an_unapproved_command_is_flagged():
    spec = _spec()
    plan = _plan([["python", "-c", "print('ok')"], ["rm", "-rf", "/"]])
    violations = validate_plan_commands(plan, spec)
    assert len(violations) == 1
    assert "rm" in str(violations[0]) or "not an approved command" in violations[0]


def test_plan_proposing_a_forbidden_command_is_flagged():
    spec = _spec()
    plan = _plan([["sudo", "python", "-c", "print('ok')"]])
    violations = validate_plan_commands(plan, spec)
    assert len(violations) == 1
    assert "forbidden token" in violations[0]
