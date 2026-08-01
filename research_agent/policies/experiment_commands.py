"""MVP4 restricted-execution command policy.

Layered, defense-in-depth gates a command must pass before the restricted
subprocess runner (research_agent.restricted_subprocess) is ever allowed to
launch it. Every gate is independent and re-checked from scratch against the
literal argv array -- never against a shell string, never against anything
Codex or Claude merely claims is safe:

  1. shape       -- argv array of non-empty strings, never a shell string.
  2. metachars    -- no shell metacharacters/command-substitution syntax in
                     any single token (defense in depth; shell=False already
                     makes these inert, but a metacharacter appearing in an
                     "approved" command is itself a smell worth rejecting).
  3. executable   -- resolved executable basename must be on a small fixed
                     allowlist (python/python3 only, in this MVP4 build).
  4. forbidden    -- no forbidden executable, token, or substring anywhere
                     in the command (bash/sh/sudo/docker/curl/wget/pip/
                     conda/apt/git push|commit|reset|clean|worktree/rm -rf/
                     scp/ssh/nc, GPU/CUDA/nvidia-smi, robot/serial/ROS,
                     training entry points).
  5. network      -- no URL scheme or common network flag in any token.
  6. exact match  -- the full argv array must equal, byte-for-byte, one
                     entry in spec.execution.approved_commands. Nothing
                     dynamically generated, and nothing merely "similar to"
                     an approved command, is ever permitted to run.

`authorize_execution` runs every gate in this fixed order and is the ONLY
function research_agent.execution_flow calls before spawning a command --
see that module's "Execution authorization" section for the full 15-point
checklist this module covers gates 1-6 and part of 13/14 of.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from research_agent.models import ExperimentSpec

ALLOWED_EXECUTABLE_BASENAMES = {"python", "python3"}

FORBIDDEN_EXECUTABLE_BASENAMES = {
    "bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "env",
    "sudo", "su", "doas",
    "docker", "docker-compose", "podman", "containerd", "runc", "nsenter",
    "curl", "wget", "nc", "ncat", "netcat", "telnet", "ftp", "tftp",
    "pip", "pip3", "conda", "mamba", "apt", "apt-get", "dpkg", "yum", "dnf", "brew",
    "git", "scp", "sftp", "ssh", "ssh-keygen", "rsync",
    "rm", "dd", "mkfs", "mount", "umount", "chmod", "chown", "chgrp",
    "kill", "killall", "pkill", "reboot", "shutdown", "systemctl", "service",
    "xargs", "eval", "exec", "at", "crontab", "nohup", "setsid",
    "nvidia-smi", "nvcc",
    "roscore", "rosrun", "roslaunch", "ros2",
    "perl", "ruby", "node", "npm", "npx", "make", "cc", "gcc", "g++", "java", "go",
}

FORBIDDEN_SUBSTRINGS = (
    "cuda", "nvidia", "gpu",
    "docker", "podman",
    "robot", "serial", "/dev/tty", "ros1", "ros2",
    "calibrat",
    "lerobot_train", "train_lggsn", "torch.cuda",
    "wandb.init",
)

FORBIDDEN_SOLO_TOKENS = {
    "--gpu", "gpu", "cuda", "nvidia-smi", "--confirmatory", "sudo", "--network",
    "--privileged", "--cap-add",
}

_SHELL_METACHARACTERS = set(";|&`$<>*?~\n\r")
_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_NETWORK_FLAG_PREFIXES = ("--url=", "--host=", "--proxy=")


class ExperimentCommandPolicyViolation(RuntimeError):
    """Carries a deterministic `.code` so callers never have to string-sniff
    a message to decide which terminal failure this maps to."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _executable_basename(token: str) -> str:
    return token.replace("\\", "/").rsplit("/", 1)[-1].lower()


def assert_command_shape(command) -> None:
    if isinstance(command, (str, bytes)):
        raise ExperimentCommandPolicyViolation(
            "SHELL_STRING_REJECTED", f"command must be an argv array, never a shell string: {command!r}"
        )
    command = list(command)
    if not command or any(not isinstance(tok, str) or not tok for tok in command):
        raise ExperimentCommandPolicyViolation(
            "MALFORMED_COMMAND", f"command must be a non-empty list of non-empty strings: {command!r}"
        )


def assert_no_shell_metacharacters(command: Sequence[str]) -> None:
    """Since every command always runs with shell=False (research_agent
    .restricted_subprocess never spawns a shell), a metacharacter inside a
    single argv token is inert -- it cannot chain commands, redirect, or
    substitute anything. This check exists purely as a smell test on the
    STRUCTURE of the command (its executable and flag tokens), not on
    free-form program text: the one argv element immediately following a
    `-c` flag is, by construction, an entire inline Python script and is
    exempt (the preferred MVP4 live-validation command,
    `python -c <fixed script>`, legitimately contains semicolons, `>`,
    braces, etc. as ordinary Python syntax)."""
    command = list(command)
    script_body_index = None
    for i, tok in enumerate(command):
        if tok == "-c" and i + 1 < len(command):
            script_body_index = i + 1
            break
    for i, tok in enumerate(command):
        if i == script_body_index:
            continue
        if any(ch in _SHELL_METACHARACTERS for ch in tok) or "$(" in tok:
            raise ExperimentCommandPolicyViolation(
                "SHELL_METACHARACTER_REJECTED", f"token contains a shell metacharacter or command substitution: {tok!r}"
            )


def assert_executable_allowed(command: Sequence[str]) -> None:
    basename = _executable_basename(command[0])
    if basename in FORBIDDEN_EXECUTABLE_BASENAMES:
        raise ExperimentCommandPolicyViolation(
            "FORBIDDEN_EXECUTABLE", f"executable is on the forbidden list: {command[0]!r}"
        )
    if basename not in ALLOWED_EXECUTABLE_BASENAMES:
        raise ExperimentCommandPolicyViolation(
            "EXECUTABLE_NOT_ALLOWLISTED",
            f"executable {command[0]!r} is not on the fixed MVP4 allowlist {sorted(ALLOWED_EXECUTABLE_BASENAMES)}",
        )


def find_forbidden_token(command: Sequence[str]) -> str | None:
    lowered = [tok.lower() for tok in command]
    for tok in lowered:
        if tok in FORBIDDEN_SOLO_TOKENS:
            return tok
        if _executable_basename(tok) in FORBIDDEN_EXECUTABLE_BASENAMES:
            return tok
        for sub in FORBIDDEN_SUBSTRINGS:
            if sub in tok:
                return sub
    if "git" in lowered:
        for bad in ("push", "commit", "reset", "clean", "worktree"):
            if bad in lowered:
                return f"git {bad}"
    return None


def assert_no_forbidden_tokens(command: Sequence[str]) -> None:
    forbidden = find_forbidden_token(command)
    if forbidden is not None:
        raise ExperimentCommandPolicyViolation(
            "FORBIDDEN_TOKEN", f"command contains forbidden token/substring {forbidden!r}: {list(command)}"
        )


def assert_no_network_indicators(command: Sequence[str]) -> None:
    for tok in command:
        if _URL_SCHEME_RE.match(tok):
            raise ExperimentCommandPolicyViolation("NETWORK_INDICATOR_REJECTED", f"token looks like a URL: {tok!r}")
        lowered = tok.lower()
        if any(lowered.startswith(p) for p in _NETWORK_FLAG_PREFIXES):
            raise ExperimentCommandPolicyViolation("NETWORK_INDICATOR_REJECTED", f"token looks like a network flag: {tok!r}")


def approved_commands(spec: "ExperimentSpec") -> list[list[str]]:
    if spec.execution is None:
        return []
    return [list(c) for c in spec.execution.approved_commands]


def assert_exact_match(
    command: Sequence[str], spec: "ExperimentSpec", *, approved_commands_override: "list[list[str]] | None" = None
) -> None:
    """approved_commands_override lets a caller compare against a RESOLVED
    approved-commands list (research_agent.policies.repo_root_placeholder)
    instead of spec.execution.approved_commands verbatim -- e.g. when a spec
    uses the ${REPO_ROOT} placeholder, both `command` and every entry in the
    override must already be fully resolved (no `${...}` left in either) by
    the caller; this function does no placeholder expansion itself. None
    (the default) preserves the exact prior behavior for every spec that
    never uses a placeholder."""
    command = list(command)
    candidates = approved_commands_override if approved_commands_override is not None else approved_commands(spec)
    if not any(command == approved for approved in candidates):
        raise ExperimentCommandPolicyViolation(
            "UNAPPROVED_COMMAND", f"command does not exactly match an approved command in the specification: {command}"
        )


def assert_command_count_within_limit(commands: Sequence[Sequence[str]], spec: "ExperimentSpec") -> None:
    if spec.execution is None:
        return
    if len(commands) > spec.execution.limits.max_commands:
        raise ExperimentCommandPolicyViolation(
            "COMMAND_COUNT_LIMIT_EXCEEDED",
            f"{len(commands)} commands exceeds limits.max_commands={spec.execution.limits.max_commands}",
        )


def authorize_execution(
    command: Sequence[str], spec: "ExperimentSpec", *, approved_commands_override: "list[list[str]] | None" = None
) -> None:
    """The single entry point research_agent.execution_flow calls before
    spawning any restricted subprocess. Runs every gate in a fixed order;
    raises ExperimentCommandPolicyViolation on the first one that fails.
    Never returns a "maybe" -- either the command is fully authorized, or an
    exception is raised.

    `command` must already be fully resolved (see
    research_agent.policies.repo_root_placeholder) -- this function never
    expands a placeholder itself, it only authorizes what it's given."""
    assert_command_shape(command)
    command = list(command)
    assert_no_shell_metacharacters(command)
    assert_executable_allowed(command)
    assert_no_forbidden_tokens(command)
    assert_no_network_indicators(command)
    assert_exact_match(command, spec, approved_commands_override=approved_commands_override)


def validate_approved_commands(
    spec: "ExperimentSpec", *, commands: "list[list[str]] | None" = None
) -> list[str]:
    """Spec-load-time validation: every entry in `commands` (default:
    spec.execution.approved_commands, unresolved) must independently pass
    every gate EXCEPT the exact-match gate (trivially true against itself)
    -- used by execution_flow's upfront POLICY_FAILURE check so a spec
    containing an unsafe "approved" command is rejected before planning
    ever starts, not discovered only when execution is attempted. Returns
    human-readable violation messages; empty means every approved command
    is clean.

    A caller resolving a ${REPO_ROOT} placeholder (research_agent.policies
    .repo_root_placeholder) should pass the RESOLVED commands here --
    otherwise the literal `${REPO_ROOT}` token's `$` would trip
    assert_no_shell_metacharacters, which has no awareness of placeholder
    syntax at all (deliberately -- see that module's docstring: it is the
    only place ${...} is ever understood)."""
    violations: list[str] = []
    if spec.execution is None:
        return violations
    effective_commands = commands if commands is not None else spec.execution.approved_commands
    if len(effective_commands) > spec.execution.limits.max_commands:
        violations.append(
            f"approved_commands has {len(effective_commands)} entries, exceeding "
            f"limits.max_commands={spec.execution.limits.max_commands}"
        )
    for command in effective_commands:
        try:
            assert_command_shape(command)
            assert_no_shell_metacharacters(command)
            assert_executable_allowed(command)
            assert_no_forbidden_tokens(command)
            assert_no_network_indicators(command)
        except ExperimentCommandPolicyViolation as e:
            violations.append(str(e))
    return violations
