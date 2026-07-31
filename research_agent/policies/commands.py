"""Command allowlist and forbidden-token policy.

Two independent gates:

1. Forbidden-token scan: rejects any command that mentions GPU, real
   hardware/robot, confirmatory execution, Docker, sudo, git push, or full
   training entry points -- the categories this MVP must never run.
2. Exact-match approval: no LLM-generated command may run unless it is
   byte-for-byte equal to `spec.smoke_command` (or, when explicitly
   permitted, `spec.confirmatory_command` -- which stays unreachable in the
   MVP; see policies.confirmatory).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from research_agent.models import ExperimentSpec, PlanResult

# Each entry maps directly to one of the six categories this MVP must never
# run: GPU, robot/real-hardware, confirmatory, Docker, sudo, git push, or
# full training jobs.
FORBIDDEN_EXACT_TOKENS = {
    "sudo",
    "docker",
    "docker-compose",
    "push",
    "--gpu",
    "gpu",
    "cuda",
    "nvidia-smi",
    "--backend=real",
    "--confirmatory",
    "bash",
    "sh",
}

# Plan-policy-validation gate (informational commands an LLM plan proposes,
# never executed directly -- see validate_plan_commands): at most one
# proposed command per plan, so a plan can never smuggle in an additional
# or duplicate smoke run alongside (or instead of) the one the deterministic
# runner will execute.
MAX_PROPOSED_COMMANDS = 1
FORBIDDEN_SUBSTRINGS = (
    "cuda",
    "docker",
    "real_hw",
    "soarm_real_backend",
    "/dev/ttyusb",
    "lerobot_train.py",
    "train_lggsn_pairwise.py",
)


class CommandPolicyViolation(RuntimeError):
    pass


def find_forbidden_token(command: Sequence[str]) -> str | None:
    lowered = [str(tok).lower() for tok in command]
    for tok in lowered:
        if tok in FORBIDDEN_EXACT_TOKENS:
            return tok
        for sub in FORBIDDEN_SUBSTRINGS:
            if sub in tok:
                return sub
    if "git" in lowered and "push" in lowered:
        return "git push"
    return None


def approved_commands(spec: "ExperimentSpec", *, allow_confirmatory: bool = False) -> list[list[str]]:
    commands = [spec.smoke_command]
    if allow_confirmatory and spec.confirmatory_command is not None:
        commands.append(spec.confirmatory_command)
    return commands


def is_command_approved(
    command: Sequence[str], spec: "ExperimentSpec", *, allow_confirmatory: bool = False
) -> bool:
    command = list(command)
    return any(command == approved for approved in approved_commands(spec, allow_confirmatory=allow_confirmatory))


def assert_command_approved(
    command: Sequence[str], spec: "ExperimentSpec", *, allow_confirmatory: bool = False
) -> None:
    command = list(command)
    forbidden = find_forbidden_token(command)
    if forbidden is not None:
        raise CommandPolicyViolation(f"command contains forbidden token {forbidden!r}: {command}")
    if not is_command_approved(command, spec, allow_confirmatory=allow_confirmatory):
        raise CommandPolicyViolation(
            f"command does not exactly match an approved command in the experiment specification: {command}"
        )


def validate_plan_commands(plan: "PlanResult", spec: "ExperimentSpec") -> list[str]:
    """Plan-policy-validation stage: every command an LLM plan proposes
    (informational only -- the deterministic runner never executes any of
    them) must independently pass the same approval gate. Returns a list of
    human-readable violation messages; empty means the plan is clean."""
    violations: list[str] = []
    commands = plan.proposed_commands

    if len(commands) > MAX_PROPOSED_COMMANDS:
        violations.append(
            f"too many proposed commands: {len(commands)} exceeds max_proposed_commands={MAX_PROPOSED_COMMANDS}"
        )

    seen: set[tuple[str, ...]] = set()
    for command in commands:
        key = tuple(command)
        if key in seen:
            violations.append(f"duplicate proposed command is not permitted: {command}")
            continue
        seen.add(key)

        forbidden = find_forbidden_token(command)
        if forbidden is not None:
            violations.append(f"proposed command contains forbidden token {forbidden!r}: {command}")
            continue
        if not is_command_approved(command, spec):
            violations.append(f"proposed command is not an approved command in the specification: {command}")
    return violations
