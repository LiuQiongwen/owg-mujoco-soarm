"""Prefect flow definitions for the TANGO Experiment Agent.

Pipeline (exactly the ten stages from the task contract):

    experiment YAML
    -> validate specification
    -> repository snapshot
    -> Codex read-only planner
    -> plan policy validation
    -> Claude Code executor in an isolated worktree
    -> static checks
    -> deterministic smoke experiment runner
    -> artifact verifier
    -> Codex read-only reviewer
    -> final report

Retries are wired ONLY on research_agent.subprocess_runner.InfrastructureError
(a launch failure or timeout). A completed command with a bad exit code, a
missing metric, or any policy violation is a normal FAIL/BLOCKED outcome and
is never retried -- see `_retry_only_infrastructure` below.

Neither Codex nor Claude Code ever executes an experiment command: the only
caller of tasks.experiment.run_smoke_command is `run_smoke_command_task`,
which takes the command verbatim from the validated ExperimentSpec.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from prefect import flow, task

from research_agent.agents.claude import ClaudeAgent, MockClaudeAgent
from research_agent.agents.codex import CodexAgent, CodexPlannerError, MockCodexAgent
from research_agent.models import (
    CommandResult,
    ExperimentSpec,
    FinalReport,
    ImplementationResult,
    PlanResult,
    ReviewResult,
    RunPaths,
    VerificationResult,
)
from research_agent.policies import commands as command_policy
from research_agent.policies import paths as path_policy
from research_agent.subprocess_runner import (
    CommandTimeoutError,
    ExecutableNotFoundError,
    InfrastructureError,
)
from research_agent.tasks import experiment as experiment_tasks
from research_agent.tasks import reporting as reporting_tasks
from research_agent.tasks import repository as repository_tasks
from research_agent.tasks import verification as verification_tasks


def _configure_prefect_local_mode() -> None:
    """Run entirely local and offline: no telemetry, no external API URL."""
    os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    os.environ.setdefault("PREFECT_LOGGING_TO_API_ENABLED", "false")
    os.environ.setdefault("PREFECT_API_URL", "")


_configure_prefect_local_mode()


class PipelineBlocked(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"[{stage}] {reason}")
        self.stage = stage
        self.reason = reason


def _retry_only_infrastructure(task_, task_run, state) -> bool:
    """Prefect retry_condition_fn: retry iff the task failed with
    InfrastructureError. Any other exception (policy violation, validation
    error, bad exit code surfaced as an exception) is never retried."""
    try:
        state.result(raise_on_failure=True)
    except InfrastructureError:
        return True
    except Exception:
        return False
    return False


# ── prompt builders (plain text; no LLM call happens here) ─────────────────

def _build_planner_prompt(spec: ExperimentSpec, *, run_paths: RunPaths, repo_root: Path) -> str:
    """Full planner prompt for both the mock and the real Codex adapter.
    Required content per the MVP1 task contract: experiment specification;
    task ID and run ID; repository context; allowed/forbidden paths;
    approved command arrays; required artifacts; required metrics;
    command-count limits; mode; and an explicit safety-restriction list."""
    approved = command_policy.approved_commands(spec, allow_confirmatory=False)
    required_metrics = [rm.model_dump() for rm in spec.required_metrics]
    return "\n".join([
        "You are a READ-ONLY planning assistant for the TANGO Experiment Agent.",
        "",
        "MODE: planning-only, non-confirmatory, read-only.",
        "",
        "SAFETY RESTRICTIONS (mandatory, no exceptions):",
        "- This step is planning only. Do not modify any file in the repository.",
        "- Treat the repository as read-only even if your own sandbox would otherwise allow writes.",
        "- Do not execute any experiment command yourself. A separate deterministic runner executes "
        "the one approved smoke command after this planning step -- you never run it.",
        "- Do not propose any command outside the approved command array(s) listed below.",
        "- Do not invoke Claude, claude-code, or any other coding/execution agent.",
        "- Do not run, or propose running, anything involving a GPU, CUDA, a physical robot, model "
        "training, Docker, sudo, `git push`, network downloads, or a confirmatory experiment.",
        "- Output only the single requested structured plan object; no other text.",
        "",
        f"task_id: {spec.task_id}",
        f"run_id: {run_paths.run_id}",
        f"repository_root: {repo_root}",
        f"goal: {spec.goal}",
        "",
        f"allowed_paths: {spec.allowed_paths}",
        f"forbidden_paths: {spec.forbidden_paths}",
        "",
        f"approved_command_arrays (the ONLY commands you may propose, byte-for-byte): {approved}",
        "confirmatory_command (if any) is intentionally excluded from the approved list above and "
        "must never be proposed.",
        "",
        f"required_artifacts: ['{run_paths.artifacts_dir.name}/metrics.json' under the run directory]",
        f"required_metrics: {required_metrics}",
        f"max_proposed_commands: {command_policy.MAX_PROPOSED_COMMANDS}",
        "",
        "Return exactly one JSON object matching the supplied PlanResult schema, with fields: "
        "schema_version, task_id, run_id, verdict, summary, issues, artifacts, proposed_commands, "
        "expected_artifacts, expected_metrics, risks, assumptions.",
        f"task_id must be exactly {spec.task_id!r}; run_id must be exactly {run_paths.run_id!r}.",
        "verdict must be exactly one of PLAN_PASS, PLAN_REVISE, PLAN_BLOCKED.",
        "proposed_commands must be a list of command arrays (list of lists of strings) -- never a "
        "shell string -- and each one must exactly match one of the approved_command_arrays above; "
        "leave it empty ([]) if you have nothing to propose.",
    ])


def _build_executor_prompt(spec: ExperimentSpec, plan: PlanResult) -> str:
    return (
        "You are the executor, confined to an isolated Git worktree. Only "
        "write inside the following allowed paths; never touch forbidden "
        "paths, and never run the experiment command yourself -- a "
        "deterministic runner executes it after you return.\n\n"
        f"task_id: {spec.task_id}\n"
        f"goal: {spec.goal}\n"
        f"allowed_paths: {spec.allowed_paths}\n"
        f"forbidden_paths: {spec.forbidden_paths}\n"
        f"plan summary: {plan.summary}\n\n"
        "Return exactly one ImplementationResult JSON object."
    )


def _build_reviewer_prompt(
    spec: ExperimentSpec,
    plan: PlanResult,
    implementation: ImplementationResult,
    verification: VerificationResult,
) -> str:
    return (
        "You are a read-only reviewer. Do not modify any files or run any "
        "command, and do not alter the verifier's verdict -- a verifier "
        "FAIL cannot be overridden by a reviewer PASS.\n\n"
        f"task_id: {spec.task_id}\n"
        f"plan verdict: {plan.verdict}\n"
        f"implementation verdict: {implementation.verdict}\n"
        f"verifier verdict: {verification.verdict}\n"
        f"verifier details: {verification.details}\n\n"
        "Return exactly one ReviewResult JSON object."
    )


# ── Prefect tasks (each wraps exactly one of the ten pipeline stages) ──────

@task(name="validate_specification")
def validate_spec_task(spec_path: Path) -> ExperimentSpec:
    return experiment_tasks.load_spec(spec_path)


@task(name="init_run")
def init_run_task(runs_root: Path, run_id: str, spec_path: Path) -> RunPaths:
    return experiment_tasks.init_run(runs_root=runs_root, run_id=run_id, spec_source_path=spec_path)


@task(name="repository_snapshot", retries=2, retry_delay_seconds=1, retry_condition_fn=_retry_only_infrastructure)
def snapshot_repository_task(repo_root: Path, run_paths: RunPaths) -> dict:
    return repository_tasks.snapshot_repository(repo_root, run_paths)


@task(name="codex_planner", retries=2, retry_delay_seconds=1, retry_condition_fn=_retry_only_infrastructure)
def codex_plan_task(codex_agent: CodexAgent, *, spec: ExperimentSpec, run_paths: RunPaths, repo_root: Path) -> PlanResult:
    prompt = _build_planner_prompt(spec, run_paths=run_paths, repo_root=repo_root)

    before = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="before_codex_planning")
    plan = codex_agent.plan(
        prompt=prompt,
        run_dir=run_paths.run_dir,
        cwd=repo_root,
        timeout=spec.timeouts.planner_seconds,
        task_id=spec.task_id,
        run_id=run_paths.run_id,
    )
    # Independently re-verify (never merely trust Codex's own claim) that the
    # read-only planning subprocess did not modify anything outside the run
    # directory it was given.
    after = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="after_codex_planning")
    changes = repository_tasks.diff_fingerprints(before, after, run_dir=run_paths.run_dir, repo_root=repo_root)
    if changes:
        raise PipelineBlocked(
            "codex_planner", "REPOSITORY_CHANGED_DURING_CODEX_PLANNING: " + "; ".join(changes)
        )

    reporting_tasks.save_json_artifact(run_paths.plan_path, plan)
    return plan


def _synthetic_blocked_plan(spec: ExperimentSpec, run_paths: RunPaths, *, reason: str) -> PlanResult:
    """Build and persist a terminal PLAN_BLOCKED PlanResult representing a
    real-Codex planning failure that never produced (or never validated
    into) an actual plan -- e.g. the executable was unavailable, the
    process timed out, or its output failed validation. `reason` must
    already be prefixed with the exact deterministic code (e.g.
    "CODEX_TIMEOUT: ..."); the code token (everything before the first
    ": ") becomes plan.issues[0]. This is the single place plan.json is
    written for every non-PLAN_PASS/REVISE/BLOCKED-from-Codex-itself
    outcome, so a failure here never leaves the run without a terminal,
    persisted plan.json."""
    code = reason.split(":", 1)[0].strip()
    plan = PlanResult(
        task_id=spec.task_id,
        run_id=run_paths.run_id,
        verdict="PLAN_BLOCKED",
        summary=reason,
        issues=[code],
        artifacts=[],
    )
    reporting_tasks.save_json_artifact(run_paths.plan_path, plan)
    return plan


def _run_codex_planner(codex_agent: CodexAgent, *, spec: ExperimentSpec, run_paths: RunPaths, repo_root: Path) -> PlanResult:
    """Wraps codex_plan_task so every possible real-Codex failure mode
    (infrastructure or research/policy) becomes a terminal PLAN_BLOCKED
    PlanResult instead of an uncaught exception -- see CodexAgent's module
    docstring for which exceptions are retried vs. terminal."""
    try:
        return codex_plan_task(codex_agent, spec=spec, run_paths=run_paths, repo_root=repo_root)
    except PipelineBlocked as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=e.reason)
    except ExecutableNotFoundError as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=f"CODEX_EXECUTABLE_UNAVAILABLE: {e}")
    except CommandTimeoutError as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=f"CODEX_TIMEOUT: {e}")
    except InfrastructureError as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=f"CODEX_TIMEOUT: {e}")
    except CodexPlannerError as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=str(e))


@task(name="plan_policy_validation")
def plan_policy_validation_task(spec: ExperimentSpec, plan: PlanResult) -> None:
    violations = command_policy.validate_plan_commands(plan, spec)
    if violations:
        raise PipelineBlocked("plan_policy_validation", "CODEX_COMMAND_POLICY_FAILED: " + "; ".join(violations))


@task(name="create_isolated_worktree", retries=1, retry_delay_seconds=1, retry_condition_fn=_retry_only_infrastructure)
def create_worktree_task(repo_root: Path, run_paths: RunPaths, run_id: str) -> Path:
    return repository_tasks.create_isolated_worktree(repo_root, run_paths, run_id)


@task(name="claude_executor", retries=1, retry_delay_seconds=1, retry_condition_fn=_retry_only_infrastructure)
def claude_execute_task(
    claude_agent: ClaudeAgent, *, spec: ExperimentSpec, plan: PlanResult, worktree_dir: Path, run_paths: RunPaths
) -> ImplementationResult:
    prompt = _build_executor_prompt(spec, plan)
    implementation = claude_agent.execute(
        prompt=prompt,
        worktree_dir=worktree_dir,
        run_dir=run_paths.run_dir,
        timeout=spec.timeouts.executor_seconds,
        task_id=spec.task_id,
        run_id=run_paths.run_id,
        allowed_paths=spec.allowed_paths,
    )
    reporting_tasks.save_json_artifact(run_paths.implementation_path, implementation)

    path_policy.assert_within_worktree(worktree_dir, run_paths.run_dir)
    changed = repository_tasks.worktree_changed_paths(worktree_dir, run_paths)
    path_policy.assert_paths_allowed(changed, spec)

    return implementation


@task(name="static_checks", retries=1, retry_delay_seconds=1, retry_condition_fn=_retry_only_infrastructure)
def static_checks_task(worktree_dir: Path, spec: ExperimentSpec, run_paths: RunPaths) -> CommandResult:
    return verification_tasks.run_static_checks(worktree_dir=worktree_dir, spec=spec, run_paths=run_paths)


@task(
    name="deterministic_smoke_runner",
    retries=1,
    retry_delay_seconds=1,
    retry_condition_fn=_retry_only_infrastructure,
)
def run_smoke_command_task(spec: ExperimentSpec, run_paths: RunPaths, worktree_dir: Path) -> CommandResult:
    return experiment_tasks.run_smoke_command(spec, run_paths, cwd=worktree_dir)


@task(name="artifact_verifier")
def verify_artifacts_task(spec: ExperimentSpec, run_paths: RunPaths, smoke_result: CommandResult) -> VerificationResult:
    verification = verification_tasks.verify_artifacts(spec, run_paths, smoke_result)
    reporting_tasks.save_json_artifact(run_paths.verification_path, verification)
    return verification


@task(name="codex_reviewer", retries=2, retry_delay_seconds=1, retry_condition_fn=_retry_only_infrastructure)
def codex_review_task(
    codex_agent: CodexAgent,
    *,
    spec: ExperimentSpec,
    plan: PlanResult,
    implementation: ImplementationResult,
    verification: VerificationResult,
    run_paths: RunPaths,
    repo_root: Path,
) -> ReviewResult:
    prompt = _build_reviewer_prompt(spec, plan, implementation, verification)
    review = codex_agent.review(
        prompt=prompt,
        run_dir=run_paths.run_dir,
        cwd=repo_root,
        timeout=spec.timeouts.reviewer_seconds,
        task_id=spec.task_id,
        run_id=run_paths.run_id,
    )
    reporting_tasks.save_json_artifact(run_paths.review_path, review)
    return review


@task(name="cleanup_isolated_worktree")
def cleanup_worktree_task(repo_root: Path, run_paths: RunPaths, run_id: str) -> None:
    repository_tasks.remove_isolated_worktree(repo_root, run_paths, run_id)


def _blocked_report(*, run_id: str, task_id: str, run_dir: Path, stage: str, reason: str, plan: Optional[PlanResult]) -> FinalReport:
    return FinalReport(
        run_id=run_id,
        task_id=task_id,
        overall_status="BLOCKED",
        blocked_stage=stage,
        blocked_reason=reason,
        plan=plan,
        run_dir=str(run_dir),
        created_at=reporting_tasks.utcnow_iso(),
    )


# ── flows ────────────────────────────────────────────────────────────────

@flow(name="tango-experiment-agent-plan")
def plan_flow(
    spec_path: Path,
    *,
    repo_root: Path,
    runs_root: Path,
    run_id: Optional[str] = None,
    codex_agent: Optional[CodexAgent] = None,
) -> PlanResult:
    """Runs only: validate specification -> repository snapshot -> Codex
    read-only planner -> plan policy validation."""
    codex_agent = codex_agent or MockCodexAgent()
    repo_root = Path(repo_root).resolve()
    runs_root = Path(runs_root).resolve()
    spec_path = Path(spec_path).resolve()

    spec = validate_spec_task(spec_path)
    experiment_tasks.assert_run_count_within_limit(runs_root, spec.task_id, spec.max_run_count)
    resolved_run_id = run_id or experiment_tasks.generate_run_id(spec.task_id)
    run_paths = init_run_task(runs_root, resolved_run_id, spec_path)

    snapshot_repository_task(repo_root, run_paths)
    plan = _run_codex_planner(codex_agent, spec=spec, run_paths=run_paths, repo_root=repo_root)
    if plan.verdict != "PLAN_PASS":
        return plan

    try:
        plan_policy_validation_task(spec, plan)
    except PipelineBlocked as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=e.reason)
    return plan


@flow(name="tango-experiment-agent-smoke")
def smoke_flow(
    spec_path: Path,
    *,
    repo_root: Path,
    runs_root: Path,
    run_id: Optional[str] = None,
    codex_agent: Optional[CodexAgent] = None,
    codex_reviewer_agent: Optional[CodexAgent] = None,
    claude_agent: Optional[ClaudeAgent] = None,
) -> FinalReport:
    """Runs the full ten-stage pipeline described in the module docstring.

    MVP1 scope note: only the PLANNER may be a RealCodexAgent; the reviewer
    stays mock unless a reviewer agent is explicitly supplied. If
    `codex_agent` is a MockCodexAgent and no `codex_reviewer_agent` is
    given, the same mock instance is reused for both roles (preserving
    MVP0 behavior, including a caller-configured review_verdict on that one
    instance) -- but if `codex_agent` is a RealCodexAgent, the reviewer
    defaults to a fresh MockCodexAgent() rather than silently reusing the
    real one, since "Replace only the mock Codex planner" is this MVP's
    explicit scope; the CLI never passes real for the reviewer role."""
    codex_agent = codex_agent or MockCodexAgent()
    if codex_reviewer_agent is None:
        codex_reviewer_agent = codex_agent if isinstance(codex_agent, MockCodexAgent) else MockCodexAgent()
    claude_agent = claude_agent or MockClaudeAgent()
    repo_root = Path(repo_root).resolve()
    runs_root = Path(runs_root).resolve()
    spec_path = Path(spec_path).resolve()

    spec = validate_spec_task(spec_path)
    experiment_tasks.assert_run_count_within_limit(runs_root, spec.task_id, spec.max_run_count)
    resolved_run_id = run_id or experiment_tasks.generate_run_id(spec.task_id)
    run_paths = init_run_task(runs_root, resolved_run_id, spec_path)

    snapshot_repository_task(repo_root, run_paths)

    plan = _run_codex_planner(codex_agent, spec=spec, run_paths=run_paths, repo_root=repo_root)

    worktree_created = False
    try:
        if plan.verdict != "PLAN_PASS":
            stage_code = "CODEX_PLAN_REVISE" if plan.verdict == "PLAN_REVISE" else "CODEX_PLAN_BLOCKED"
            report = _blocked_report(
                run_id=resolved_run_id, task_id=spec.task_id, run_dir=run_paths.run_dir,
                stage="codex_planner", reason=f"{stage_code}: plan verdict={plan.verdict}: {plan.summary}",
                plan=plan,
            )
            reporting_tasks.write_final_report(run_paths, report)
            return report

        plan_policy_validation_task(spec, plan)

        worktree_dir = create_worktree_task(repo_root, run_paths, resolved_run_id)
        worktree_created = True

        implementation = claude_execute_task(
            claude_agent, spec=spec, plan=plan, worktree_dir=worktree_dir, run_paths=run_paths
        )
        if implementation.verdict != "IMPLEMENTATION_READY_FOR_REVIEW":
            report = _blocked_report(
                run_id=resolved_run_id, task_id=spec.task_id, run_dir=run_paths.run_dir,
                stage="claude_executor",
                reason=f"implementation verdict={implementation.verdict}: {implementation.summary}",
                plan=plan,
            )
            reporting_tasks.write_final_report(run_paths, report)
            return report

        static_result = static_checks_task(worktree_dir, spec, run_paths)
        smoke_result = run_smoke_command_task(spec, run_paths, worktree_dir)
        verification = verify_artifacts_task(spec, run_paths, smoke_result)

        review = codex_review_task(
            codex_reviewer_agent, spec=spec, plan=plan, implementation=implementation,
            verification=verification, run_paths=run_paths, repo_root=repo_root,
        )

        # A verifier FAIL can never be overridden by a reviewer PASS: it is
        # checked first and short-circuits the rest of the status logic.
        if verification.verdict != "PASS":
            overall = "FAIL"
        elif review.verdict == "REVIEW_PASS":
            overall = "PASS"
        elif review.verdict == "REVIEW_REVISE":
            overall = "FAIL"
        else:
            overall = "BLOCKED"

        report = FinalReport(
            run_id=resolved_run_id,
            task_id=spec.task_id,
            overall_status=overall,
            plan=plan,
            implementation=implementation,
            static_checks=static_result,
            smoke_command=smoke_result,
            verification=verification,
            review=review,
            run_dir=str(run_paths.run_dir),
            created_at=reporting_tasks.utcnow_iso(),
        )
    except (PipelineBlocked, path_policy.PathPolicyViolation, command_policy.CommandPolicyViolation) as exc:
        stage = getattr(exc, "stage", "policy_enforcement")
        reason = getattr(exc, "reason", str(exc))
        report = _blocked_report(
            run_id=resolved_run_id, task_id=spec.task_id, run_dir=run_paths.run_dir,
            stage=stage, reason=reason, plan=plan,
        )
    finally:
        if worktree_created:
            cleanup_worktree_task(repo_root, run_paths, resolved_run_id)

    reporting_tasks.write_final_report(run_paths, report)
    return report
