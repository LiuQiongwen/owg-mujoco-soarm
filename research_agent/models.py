"""Pydantic data model for the TANGO Experiment Agent.

Every model rejects unknown fields (extra="forbid") so a typo or an
unexpected LLM-authored field in a YAML spec or a JSON agent output fails
validation loudly instead of being silently ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── experiment specification ────────────────────────────────────────────────

class TimeoutSpec(StrictModel):
    planner_seconds: int = Field(120, gt=0, le=1800)
    executor_seconds: int = Field(300, gt=0, le=1800)
    smoke_seconds: int = Field(60, gt=0, le=1800)
    verifier_seconds: int = Field(60, gt=0, le=1800)
    reviewer_seconds: int = Field(120, gt=0, le=1800)


class ApprovalSpec(StrictModel):
    required_for_confirmatory: bool = True
    approved_by: Optional[str] = None


class RequiredMetric(StrictModel):
    name: str = Field(..., min_length=1)
    type: Literal["float", "int", "bool", "str"] = "float"
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ExperimentSpec(StrictModel):
    task_id: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")
    goal: str = Field(..., min_length=1)

    allowed_paths: list[str] = Field(..., min_length=1)
    forbidden_paths: list[str] = Field(default_factory=list)

    smoke_command: list[str] = Field(..., min_length=1)
    confirmatory_command: Optional[list[str]] = None

    seeds: list[int] = Field(default_factory=lambda: [0])
    timeouts: TimeoutSpec = Field(default_factory=TimeoutSpec)

    required_metrics: list[RequiredMetric] = Field(default_factory=list)
    approval: ApprovalSpec = Field(default_factory=ApprovalSpec)

    max_run_count: int = Field(1, ge=1, le=10)

    @field_validator("allowed_paths", "forbidden_paths")
    @classmethod
    def _repo_relative_paths_only(cls, value: list[str]) -> list[str]:
        for p in value:
            if not p or p.startswith("/") or ".." in Path(p).parts:
                raise ValueError(f"path must be a non-empty, repo-relative path with no '..': {p!r}")
        return value

    @field_validator("smoke_command", "confirmatory_command")
    @classmethod
    def _command_is_nonempty_string_list(cls, value):
        if value is None:
            return value
        if not value or any(not isinstance(tok, str) or not tok for tok in value):
            raise ValueError("command must be a non-empty list of non-empty strings (never a shell string)")
        return value

    @field_validator("seeds")
    @classmethod
    def _seeds_nonempty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("seeds must contain at least one seed")
        return value

    @model_validator(mode="after")
    def _allowed_forbidden_disjoint(self) -> "ExperimentSpec":
        overlap = set(self.allowed_paths) & set(self.forbidden_paths)
        if overlap:
            raise ValueError(f"paths cannot be both allowed and forbidden: {sorted(overlap)}")
        return self


# ── subprocess execution records ────────────────────────────────────────────

class CommandResult(StrictModel):
    name: str
    command: list[str]
    cwd: str
    returncode: Optional[int]
    timed_out: bool = False
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    started_at: str
    ended_at: str


# ── agent outputs ────────────────────────────────────────────────────────────

PlanVerdict = Literal["PLAN_PASS", "PLAN_REVISE", "PLAN_BLOCKED"]
ImplementationVerdict = Literal[
    "IMPLEMENTATION_READY_FOR_REVIEW", "IMPLEMENTATION_FAILED", "IMPLEMENTATION_BLOCKED"
]
ReviewVerdict = Literal["REVIEW_PASS", "REVIEW_REVISE", "REVIEW_BLOCKED"]


class AgentJudgement(StrictModel):
    schema_version: str = SCHEMA_VERSION
    task_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    issues: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class PlanResult(AgentJudgement):
    verdict: PlanVerdict
    # Informational only: the deterministic runner NEVER executes a command
    # from this list. It exists so plan-policy-validation has something
    # concrete to check every LLM-proposed command against (see
    # policies.commands.validate_plan_commands) before the executor stage
    # is ever reached.
    proposed_commands: list[list[str]] = Field(default_factory=list)


class ImplementationResult(AgentJudgement):
    verdict: ImplementationVerdict


class ReviewResult(AgentJudgement):
    verdict: ReviewVerdict


class VerificationResult(StrictModel):
    verdict: Literal["PASS", "FAIL"]
    required_metrics_ok: bool
    artifacts_ok: bool
    details: list[str] = Field(default_factory=list)


class FinalReport(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    task_id: str
    overall_status: Literal["PASS", "FAIL", "BLOCKED"]
    blocked_stage: Optional[str] = None
    blocked_reason: Optional[str] = None

    plan: Optional[PlanResult] = None
    implementation: Optional[ImplementationResult] = None
    static_checks: Optional[CommandResult] = None
    smoke_command: Optional[CommandResult] = None
    verification: Optional[VerificationResult] = None
    review: Optional[ReviewResult] = None

    run_dir: str
    created_at: str


# ── run filesystem layout (not a Pydantic model: holds Path objects) ───────

@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def spec_path(self) -> Path:
        return self.run_dir / "spec.yaml"

    @property
    def git_sha_path(self) -> Path:
        return self.run_dir / "git_sha.txt"

    @property
    def git_diff_path(self) -> Path:
        return self.run_dir / "git_diff.patch"

    @property
    def environment_path(self) -> Path:
        return self.run_dir / "environment.json"

    @property
    def commands_dir(self) -> Path:
        return self.run_dir / "commands"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def worktree_dir(self) -> Path:
        return self.run_dir / "worktree"

    @property
    def plan_path(self) -> Path:
        return self.run_dir / "plan.json"

    @property
    def implementation_path(self) -> Path:
        return self.run_dir / "implementation.json"

    @property
    def review_path(self) -> Path:
        return self.run_dir / "review.json"

    @property
    def verification_path(self) -> Path:
        return self.run_dir / "verification.json"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.json"
