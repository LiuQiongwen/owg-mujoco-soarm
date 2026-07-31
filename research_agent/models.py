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

    # ── MVP2 execute-flow (real Claude executor) policy fields ─────────────
    # All additive with defaults so existing MVP0/MVP1 specs keep validating
    # unchanged. Empty allowed_modify_paths means "fall back to
    # allowed_paths" -- see research_agent.policies.execution_policy
    # .effective_modify_paths -- so a spec that never mentions MVP2 at all
    # still has a sane (if identical-to-MVP1) modification scope.
    allowed_modify_paths: list[str] = Field(default_factory=list)
    max_changed_files: int = Field(20, ge=1, le=1000)
    max_changed_bytes: int = Field(200_000, ge=1)
    allowed_executor_commands: list[list[str]] = Field(default_factory=list)
    required_executor_checks: list[str] = Field(default_factory=list)

    @field_validator("allowed_paths", "forbidden_paths", "allowed_modify_paths")
    @classmethod
    def _repo_relative_paths_only(cls, value: list[str]) -> list[str]:
        for p in value:
            if not p or p.startswith("/") or ".." in Path(p).parts:
                raise ValueError(f"path must be a non-empty, repo-relative path with no '..': {p!r}")
        return value

    @field_validator("allowed_executor_commands")
    @classmethod
    def _executor_commands_are_argv_arrays(cls, value: list[list[str]]) -> list[list[str]]:
        for command in value:
            if not command or any(not isinstance(tok, str) or not tok for tok in command):
                raise ValueError(
                    f"allowed_executor_commands entries must be non-empty argv arrays, never a shell string: {command!r}"
                )
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
    expected_artifacts: list[str] = Field(default_factory=list)
    expected_metrics: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ImplementationResult(AgentJudgement):
    verdict: ImplementationVerdict


class ReviewResult(AgentJudgement):
    verdict: ReviewVerdict


# ── MVP2 execute-flow (real Claude executor) structured result ─────────────

ExecutorVerdict = Literal["IMPLEMENTATION_PASS", "IMPLEMENTATION_REVISE", "IMPLEMENTATION_BLOCKED"]

# The 21 deterministic terminal failure states an execute-flow run may end
# in (see the MVP2 task contract's "Failure states" section), plus the
# three non-failure terminal verdicts a run can otherwise reach.
EXECUTE_FAILURE_STATES = (
    "CLAUDE_EXECUTABLE_UNAVAILABLE",
    "CLAUDE_TIMEOUT",
    "CLAUDE_AUTHENTICATION_FAILED",
    "CLAUDE_NONZERO_EXIT",
    "CLAUDE_OUTPUT_MISSING",
    "CLAUDE_OUTPUT_MALFORMED",
    "CLAUDE_SCHEMA_INVALID",
    "CLAUDE_TASK_ID_MISMATCH",
    "CLAUDE_RUN_ID_MISMATCH",
    "CLAUDE_IMPLEMENTATION_REVISE",
    "CLAUDE_IMPLEMENTATION_BLOCKED",
    "CLAUDE_COMMAND_POLICY_FAILED",
    "CLAUDE_PATH_POLICY_FAILED",
    "CLAUDE_CHANGED_FILE_LIMIT_EXCEEDED",
    "CLAUDE_CHANGED_BYTES_LIMIT_EXCEEDED",
    "CLAUDE_MAIN_WORKTREE_CHANGED",
    "CLAUDE_EXECUTION_WORKTREE_INVALID",
    "CLAUDE_DIFF_MISMATCH",
    "CLAUDE_NESTED_GIT_DETECTED",
    "CLAUDE_SYMLINK_ESCAPE_DETECTED",
)
EXECUTE_TERMINAL_STATES = EXECUTE_FAILURE_STATES + (
    "EXECUTION_PASS", "EXECUTION_REVISE", "EXECUTION_BLOCKED", "CODEX_PLAN_NOT_PASS",
)


class ExecutorImplementationResult(StrictModel):
    """Structured final response required from the real (or fake, in tests)
    Claude executor subprocess. Strict schema: unknown fields, a missing
    field, or a shell-string command entry (list[list[str]] rejects a bare
    str) all fail Pydantic validation, which the caller maps to
    CLAUDE_SCHEMA_INVALID. This response is advisory only -- the
    deterministic Git inspection in execution_worktree.py is authoritative;
    see research_agent.execute_flow."""

    schema_version: str = SCHEMA_VERSION
    task_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    verdict: ExecutorVerdict
    summary: str = Field(..., min_length=1)
    changed_files: list[str] = Field(default_factory=list)
    commands_run: list[list[str]] = Field(default_factory=list)
    tests_run: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("commands_run")
    @classmethod
    def _commands_are_argv_arrays(cls, value: list[list[str]]) -> list[list[str]]:
        for command in value:
            if not command or any(not isinstance(tok, str) or not tok for tok in command):
                raise ValueError(f"commands_run entries must be non-empty argv arrays: {command!r}")
        return value


class ChangedFileRecord(StrictModel):
    path: str
    status: str  # e.g. "A", "M", "D", "??" (raw `git status --porcelain` code) or "untracked"
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None


class ManifestEntry(StrictModel):
    """One entry of the before/after ignored-and-untracked-file manifest --
    stronger than a plain git-status fingerprint because it also detects
    in-place edits to already-ignored files (MVP1 finding: repository
    fingerprints alone miss those)."""

    path: str
    file_type: str  # "file" | "dir" | "symlink" | "missing"
    size_bytes: Optional[int] = None
    mtime_ns: Optional[int] = None
    sha256: Optional[str] = None
    symlink_target: Optional[str] = None


class ExecutionWorktreeRecord(StrictModel):
    run_id: str
    base_commit: str
    branch_name: str
    worktree_path: str
    created_at: str
    initial_status: str
    initial_tracked_diff_empty: bool
    initial_untracked_files: list[str] = Field(default_factory=list)
    preserved: bool = True
    removed_at: Optional[str] = None


class ExecuteReport(StrictModel):
    """Deterministic final report for the MVP2 execute flow. Distinct from
    FinalReport (the MVP0/MVP1 smoke-flow report): this flow never runs a
    research experiment command, so there is no smoke_command/verification/
    review stage here -- only planning, isolated-worktree implementation,
    deterministic diff/path/command validation, and lightweight static
    checks."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    task_id: str
    overall_status: Literal["PASS", "FAIL", "BLOCKED"]
    terminal_state: str
    stage: Optional[str] = None
    reason: Optional[str] = None

    plan: Optional[PlanResult] = None
    implementation: Optional[ExecutorImplementationResult] = None
    claude_command: list[str] = Field(default_factory=list)
    static_checks: list[CommandResult] = Field(default_factory=list)

    execution_worktree: Optional[ExecutionWorktreeRecord] = None
    changed_files: list[ChangedFileRecord] = Field(default_factory=list)
    changed_file_count: int = 0
    changed_byte_count: int = 0

    main_worktree_unchanged: bool = True
    ignored_file_manifest_ok: bool = True

    run_dir: str
    created_at: str


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
    def prompts_dir(self) -> Path:
        return self.run_dir / "prompts"

    @property
    def plan_path(self) -> Path:
        return self.run_dir / "plan.json"

    @property
    def plan_raw_path(self) -> Path:
        return self.run_dir / "plan.raw.json"

    @property
    def plan_schema_path(self) -> Path:
        return self.run_dir / "plan.schema.json"

    @property
    def implementation_path(self) -> Path:
        return self.run_dir / "implementation.json"

    @property
    def implementation_raw_path(self) -> Path:
        return self.run_dir / "implementation.raw.json"

    @property
    def execution_worktree_record_path(self) -> Path:
        return self.run_dir / "execution_worktree.json"

    @property
    def ignored_file_manifest_before_path(self) -> Path:
        return self.run_dir / "ignored_file_manifest.before.json"

    @property
    def ignored_file_manifest_after_path(self) -> Path:
        return self.run_dir / "ignored_file_manifest.after.json"

    @property
    def review_path(self) -> Path:
        return self.run_dir / "review.json"

    @property
    def verification_path(self) -> Path:
        return self.run_dir / "verification.json"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.json"
