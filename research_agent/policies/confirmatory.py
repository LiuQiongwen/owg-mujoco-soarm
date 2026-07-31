"""Confirmatory approval gate.

The gate logic itself is implemented and testable, but confirmatory
execution is kept unreachable in this MVP: `CONFIRMATORY_EXECUTION_ENABLED`
is a hardcoded module constant, not a CLI flag or a spec field, so no YAML
content and no CLI argument can turn it on. Flipping it is a deliberate code
change, not a runtime decision. Nothing in research_agent.flow calls the
confirmatory command at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from research_agent.models import ExperimentSpec

CONFIRMATORY_EXECUTION_ENABLED = False


class ConfirmatoryBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class ConfirmatoryGateDecision:
    allowed: bool
    reason: str


def check_confirmatory_gate(
    spec: "ExperimentSpec", *, approved_by: Optional[str] = None
) -> ConfirmatoryGateDecision:
    """Evaluate (but never act on) whether a confirmatory run would be
    permitted. Always returns allowed=False in this MVP build."""
    if not CONFIRMATORY_EXECUTION_ENABLED:
        return ConfirmatoryGateDecision(False, "confirmatory execution is disabled in this MVP build")
    if spec.confirmatory_command is None:
        return ConfirmatoryGateDecision(False, "no confirmatory_command defined in the specification")
    if spec.approval.required_for_confirmatory and not approved_by:
        return ConfirmatoryGateDecision(False, "confirmatory run requires an approver but none was provided")
    return ConfirmatoryGateDecision(True, f"approved by {approved_by}" if approved_by else "no approval required")


def assert_confirmatory_disabled() -> None:
    if CONFIRMATORY_EXECUTION_ENABLED:
        raise ConfirmatoryBlocked(
            "CONFIRMATORY_EXECUTION_ENABLED is set, but no caller in this MVP invokes confirmatory execution"
        )
