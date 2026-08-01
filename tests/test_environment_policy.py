"""MVP4 environment-policy tests: research_agent.policies.environment_policy.

Covers (see the MVP4 task contract's "Fake execution tests" list):
  29. environment allowlist
  30. secret environment variable not inherited
  31. proxy variables stripped
  32. CUDA disabled variables applied
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.models import ExperimentSpec
from research_agent.policies import environment_policy as ep


def _spec(**execution_overrides) -> ExperimentSpec:
    execution = {"approved_commands": [[sys.executable, "-c", "pass"]], **execution_overrides}
    return ExperimentSpec.model_validate({
        "task_id": "envpolicy_test", "goal": "g", "allowed_paths": ["research_agent_sandbox"],
        "smoke_command": ["python", "-c", "pass"], "execution": execution,
    })


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("HTTPS_PROXY", "https_proxy", "MY_ALLOWED_VAR", "SUPER_SECRET_TOKEN_LOOKALIKE"):
        monkeypatch.delenv(name, raising=False)
    yield


def test_environment_allowlist_only_forwards_named_variables(monkeypatch):
    monkeypatch.setenv("MY_ALLOWED_VAR", "hello")
    monkeypatch.setenv("NOT_ALLOWED_VAR", "should_not_appear")
    spec = _spec(environment_allowlist=["MY_ALLOWED_VAR"])
    env = ep.build_child_environment(spec, run_dir=Path("/tmp/run"), artifacts_dir=Path("/tmp/run/artifacts"))
    assert env.get("MY_ALLOWED_VAR") == "hello"
    assert "NOT_ALLOWED_VAR" not in env


def test_secret_looking_environment_variable_never_inherited(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_TOKEN_LOOKALIKE", "xxx")
    spec = _spec()
    env = ep.build_child_environment(spec, run_dir=Path("/tmp/run"), artifacts_dir=Path("/tmp/run/artifacts"))
    assert "SUPER_SECRET_TOKEN_LOOKALIKE" not in env
    # and it cannot even be added via environment_allowlist -- rejected at spec-validation time.
    with pytest.raises(Exception):
        _spec(environment_allowlist=["SUPER_SECRET_TOKEN_LOOKALIKE"])


def test_proxy_variables_rejected_from_allowlist_at_spec_validation():
    """A proxy variable name is already caught by ExperimentSpec's own
    validator (it matches the 'PROXY' sensitive-name marker) -- a spec can
    never even request one be forwarded."""
    with pytest.raises(Exception):
        _spec(environment_allowlist=["HTTPS_PROXY"])


def test_proxy_variables_always_stripped_from_child_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.internal:8080")
    spec = _spec()
    env = ep.build_child_environment(spec, run_dir=Path("/tmp/run"), artifacts_dir=Path("/tmp/run/artifacts"))
    assert "HTTPS_PROXY" not in env
    assert "https_proxy" not in env


def test_cuda_disabled_variables_always_applied():
    spec = _spec()
    env = ep.build_child_environment(spec, run_dir=Path("/tmp/run"), artifacts_dir=Path("/tmp/run/artifacts"))
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["NVIDIA_VISIBLE_DEVICES"] == ""
    assert env["WANDB_MODE"] == "disabled"


def test_environment_overrides_applied_and_scoped_to_run(monkeypatch):
    spec = _spec(environment_overrides={"MY_EXPERIMENT_FLAG": "on"})
    run_dir = Path("/tmp/run_x")
    artifacts_dir = Path("/tmp/run_x/artifacts")
    env = ep.build_child_environment(spec, run_dir=run_dir, artifacts_dir=artifacts_dir)
    assert env["MY_EXPERIMENT_FLAG"] == "on"
    assert env["RESEARCH_AGENT_RUN_DIR"] == str(run_dir)
    assert env["RESEARCH_AGENT_ARTIFACTS_DIR"] == str(artifacts_dir)


def test_environment_overrides_rejects_sensitive_names_at_spec_validation():
    with pytest.raises(Exception):
        _spec(environment_overrides={"AWS_SECRET_KEY": "xxx"})


def test_no_wandb_hf_anthropic_codex_credentials_ever_inherited(monkeypatch):
    for name in ("WANDB_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SSH_AUTH_SOCK"):
        monkeypatch.setenv(name, "xxx")
    spec = _spec()
    env = ep.build_child_environment(spec, run_dir=Path("/tmp/run"), artifacts_dir=Path("/tmp/run/artifacts"))
    for name in ("WANDB_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SSH_AUTH_SOCK"):
        assert name not in env


def test_default_environment_is_minimal():
    spec = _spec()
    env = ep.build_child_environment(spec, run_dir=Path("/tmp/run"), artifacts_dir=Path("/tmp/run/artifacts"))
    # Only the base allowlist (as present in the test process) plus forced
    # values plus the two run-scoped variables -- never a full os.environ copy.
    assert set(env) <= set(ep.BASE_ALLOWLIST) | set(ep._FORCED_VALUES) | {"RESEARCH_AGENT_RUN_DIR", "RESEARCH_AGENT_ARTIFACTS_DIR"}
