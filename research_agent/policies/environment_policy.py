"""MVP4 restricted-execution environment policy.

Builds the restricted subprocess's environment from a small allowlist --
never from a copy of the parent process environment (contrast with
research_agent.subprocess_runner.run_command, which inherits `os.environ`
and is used only for the harness's OWN deterministic commands, never for an
approved experiment command). This is the ONLY function that constructs the
environment an MVP4 restricted subprocess actually runs with.

Network isolation is policy-based, not kernel-enforced: this module removes
proxy variables and never passes credentials, but nothing here creates a
network namespace or firewall rule. `research_agent.policies
.experiment_commands` provides the other half of defense in depth (rejecting
network-shaped executables/tokens/URLs in the command itself) -- neither
layer alone, nor both together, is a substitute for real OS-level network
isolation; see the MVP4 task contract's "Network restriction" section.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from research_agent.models import ExperimentSpec

# Always present if set in the parent environment (never a secret by
# construction), plus the two run-scoped variables this module always
# injects itself (RESEARCH_AGENT_RUN_DIR/RESEARCH_AGENT_ARTIFACTS_DIR).
BASE_ALLOWLIST = (
    "PATH",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
)

# Values this module always sets itself, regardless of the parent
# environment or environment_allowlist -- CPU-only defense in depth (see
# the task contract: "Do not rely only on environment variables for GPU
# blocking; reject GPU-related executables and tokens too" -- this is only
# ONE of two independent layers, the other being experiment_commands.py).
_FORCED_VALUES = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "",
    "WANDB_MODE": "disabled",
    "PYTHONUNBUFFERED": "1",
}

# Variables that must never be forwarded into a restricted subprocess even
# if they happen to be in os.environ, no matter what environment_allowlist
# says -- see ExecutionSpec's own validators, which already reject a
# sensitive-looking name in environment_allowlist/environment_overrides;
# this is the second, independent enforcement point (belt and suspenders).
_SENSITIVE_NAME_MARKERS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH",
    "WANDB_API", "HF_TOKEN", "HUGGINGFACE", "OPENAI", "ANTHROPIC", "CLAUDE_CODE",
    "CODEX", "SSH", "GITHUB_TOKEN", "GCP", "AWS", "AZURE",
)

# Always stripped, never forwarded, and never overridable via
# environment_allowlist -- proxies are a network-egress vector.
_PROXY_VAR_NAMES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "no_proxy",
)

# Never forwarded regardless of allowlist -- robot/hardware device paths.
_DEVICE_VAR_NAMES = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")


def looks_sensitive_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SENSITIVE_NAME_MARKERS)


def build_child_environment(
    spec: "ExperimentSpec", *, run_dir: "Path", artifacts_dir: "Path"
) -> dict[str, str]:
    """The ONLY environment an MVP4 restricted subprocess ever receives.
    Built entirely from BASE_ALLOWLIST + spec.execution.environment_allowlist
    (both filtered through looks_sensitive_name as a second check), plus
    spec.execution.environment_overrides (already validated not to name a
    sensitive-looking variable at spec-load time), plus the forced
    CPU-only/offline defaults. Proxy variables are always absent regardless
    of any allowlist entry."""
    execution = spec.execution
    allowlist_names = set(BASE_ALLOWLIST)
    overrides: dict[str, str] = {}
    if execution is not None:
        allowlist_names.update(execution.environment_allowlist)
        overrides = dict(execution.environment_overrides)

    env: dict[str, str] = {}
    for name in sorted(allowlist_names):
        if name in _PROXY_VAR_NAMES or looks_sensitive_name(name):
            continue
        if name in os.environ:
            env[name] = os.environ[name]

    env.update(_FORCED_VALUES)
    env["RESEARCH_AGENT_RUN_DIR"] = str(run_dir)
    env["RESEARCH_AGENT_ARTIFACTS_DIR"] = str(artifacts_dir)

    for name, value in overrides.items():
        if name in _PROXY_VAR_NAMES or looks_sensitive_name(name):
            continue  # already rejected at spec-validation time; skip defensively anyway
        env[name] = value

    for name in _PROXY_VAR_NAMES:
        env.pop(name, None)

    return env


def redact_environment_names_only(env: dict[str, str]) -> list[str]:
    """MVP4 never persists environment VALUES to disk (unlike
    subprocess_runner.redact_environment, which persists a value-masked
    copy for the harness's own commands) -- only the variable NAMES, so a
    saved environment.json can never leak a value even for a name that
    slipped past looks_sensitive_name."""
    return sorted(env)
