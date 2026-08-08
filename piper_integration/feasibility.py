"""Object-agnostic Piper feasibility adapter skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .contracts import Candidate, CandidateFeatures


@dataclass(frozen=True)
class IKResult:
    feasible: bool
    joints: Sequence[float] | None


class PiperFeasibilityAdapter:
    """Turns a candidate into provisional embodiment features.

    Solvers are injected so this contract does not freeze an IK or path planner.
    No object identifier is passed to either solver.
    """

    def __init__(self, joint_limits: Sequence[Sequence[float]], max_opening_m: float):
        self.joint_limits = tuple((float(lo), float(hi)) for lo, hi in joint_limits)
        self.max_opening_m = float(max_opening_m)

    def evaluate(
        self,
        candidate: Candidate,
        ik_solver: Callable[[Sequence[float], str, str], IKResult],
        required_opening_m: float | None = None,
        clearance_solver: Callable[[Sequence[float], str, str], float | None] | None = None,
    ) -> CandidateFeatures:
        ik = ik_solver(candidate.pose, candidate.pose_frame, candidate.pose_convention)
        margin = None
        if ik.feasible and ik.joints is not None:
            if len(ik.joints) != len(self.joint_limits):
                raise ValueError("IK joint count does not match embodiment")
            normalized = [
                min(float(q) - lo, hi - float(q)) / (hi - lo)
                for q, (lo, hi) in zip(ik.joints, self.joint_limits)
            ]
            margin = min(normalized)
        clearance = None
        if clearance_solver is not None:
            clearance = clearance_solver(candidate.pose, candidate.pose_frame, candidate.pose_convention)
        opening_ok = None if required_opening_m is None else required_opening_m <= self.max_opening_m
        return CandidateFeatures(
            ik_feasible=ik.feasible,
            joint_margin=margin,
            opening_feasible=opening_ok,
            path_clearance_m=clearance,
        )
