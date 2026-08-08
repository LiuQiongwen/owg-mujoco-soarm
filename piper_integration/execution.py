"""Shared execution and transit-planning interfaces."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .contracts import Candidate, ExecutionConfig, ExecutionResult


class TransitPlanner(Protocol):
    def plan_transit(
        self, candidate: Candidate, constraints: Mapping[str, Any]
    ) -> Sequence[Mapping[str, Any]]: ...


class ExecutionBackend(Protocol):
    def execute(self, candidate: Candidate, execution_config: ExecutionConfig) -> ExecutionResult: ...


def require_explicit_transit_constraints(constraints: Mapping[str, Any]) -> None:
    """Reject hidden equivalents of the legacy global SAFE_TRANSIT_Z."""
    if "clearance_m" not in constraints and "minimum_world_z_m" not in constraints:
        raise ValueError("transit constraints require clearance_m or minimum_world_z_m")
