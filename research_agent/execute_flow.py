"""MVP2 Prefect flow: real Claude executor in an isolated execution worktree.

Pipeline (exactly the stages from the MVP2 task contract -- distinct from,
and never invoking, research_agent.flow.smoke_flow's deterministic
experiment runner):

    experiment YAML
    -> validate specification
    -> repository snapshot (pre-existing state)
    -> Codex read-only planner (mock by default, real with --codex real)
    -> plan policy validation
    -> isolated execution worktree (unique branch + worktree, outside the
       main repository worktree)
    -> real (or mock) Claude executor, confined to that worktree
    -> deterministic diff and path validation (never trusts Claude's own
       claims -- see policies/execution_policy.py)
    -> command-claim policy audit (advisory)
    -> main-worktree-unchanged check
    -> lightweight static checks (compileall / declared pytest targets)
    -> deterministic ExecuteReport

No stage in this module ever runs spec.smoke_command or any research
experiment: that is exclusively research_agent.tasks.experiment
.run_smoke_command, called only from research_agent.flow.smoke_flow. The
execution worktree this flow creates is NEVER removed automatically, on
success or failure -- see research_agent.cli's worktree-status/
worktree-cleanup commands, the only place research_agent.execution_worktree
.remove_execution_worktree is ever called.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from prefect import flow, task

from research_agent import execution_worktree, subprocess_runner
from research_agent.agents.claude_executor import (
    ClaudeExecutorAgent,
    ClaudeExecutorError,
    MockClaudeExecutorAgent,
    build_executor_prompt,
)
from research_agent.agents.codex import CodexAgent, CodexPlannerError, MockCodexAgent
from research_agent.execution_worktree import ExecutionWorktreeError
from research_agent.flow import _retry_only_infrastructure  # reuse the same infra-only retry policy
from research_agent.models import (
    ChangedFileRecord,
    CommandResult,
    ExecuteReport,
    ExecutionWorktreeRecord,
    ExecutorImplementationResult,
    ExperimentSpec,
    PlanResult,
    RunPaths,
)
from research_agent.policies import commands as command_policy
from research_agent.policies import execution_policy
from research_agent.subprocess_runner import CommandTimeoutError, ExecutableNotFoundError, InfrastructureError
from research_agent.tasks import experiment as experiment_tasks
from research_agent.tasks import reporting as reporting_tasks
from research_agent.tasks import repository as repository_tasks
from research_agent.tasks import verification as verification_tasks

_BLOCKED_TERMINAL_STATES = {
    "CLAUDE_EXECUTABLE_UNAVAILABLE",
    "CLAUDE_AUTHENTICATION_FAILED",
    "CLAUDE_OUTPUT_MISSING",
    "CLAUDE_OUTPUT_MALFORMED",
    "CLAUDE_SCHEMA_INVALID",
    "CLAUDE_TASK_ID_MISMATCH",
    "CLAUDE_RUN_ID_MISMATCH",
    "CLAUDE_IMPLEMENTATION_BLOCKED",
    "CLAUDE_EXECUTION_WORKTREE_INVALID",
    "CLAUDE_MAIN_WORKTREE_CHANGED",
    "CLAUDE_NESTED_GIT_DETECTED",
    "CLAUDE_SYMLINK_ESCAPE_DETECTED",
    "CODEX_PLAN_NOT_PASS",
    "CODEX_COMMAND_POLICY_FAILED",
}


def _overall_status_for(terminal_state: str) -> str:
    if terminal_state == "EXECUTION_PASS":
        return "PASS"
    if terminal_state in _BLOCKED_TERMINAL_STATES:
        return "BLOCKED"
    return "FAIL"


# ── planner prompt (MVP2-specific: plans a source-code change, never an
#    experiment run) ────────────────────────────────────────────────────────

def _build_execute_planner_prompt(spec: ExperimentSpec, *, run_paths: RunPaths, repo_root: Path) -> str:
    allowed_commands = list(spec.allowed_executor_commands) + [
        ["python", "-m", "compileall", "..."],
        ["git", "status", "--short"],
        ["git", "diff", "--check"],
        ["git", "diff", "--name-status"],
    ]
    return "\n".join([
        "You are a READ-ONLY planning assistant for the TANGO Experiment Agent's MVP2 execute flow.",
        "",
        "MODE: planning-only, non-confirmatory, read-only. You are planning an APPROVED SOURCE-CODE",
        "MODIFICATION to be carried out later, by a separate executor, inside a disposable Git worktree.",
        "You never modify anything yourself, and no research experiment is ever run in this workflow.",
        "",
        "SAFETY RESTRICTIONS (mandatory, no exceptions):",
        "- This step is planning only. Do not modify any file in the repository.",
        "- Do not propose any command outside the approved command array(s) listed below.",
        "- Do not invoke Claude, claude-code, or any other coding/execution agent.",
        "- Do not run, or propose running, anything involving a GPU, CUDA, a physical robot, model",
        "  training, Docker, sudo, `git push`, `git commit`, network downloads, or a confirmatory or",
        "  otherwise research/experiment command of any kind -- MVP2 never runs a research experiment.",
        "- Output only the single requested structured plan object; no other text.",
        "",
        f"task_id: {spec.task_id}",
        f"run_id: {run_paths.run_id}",
        f"repository_root: {repo_root}",
        f"goal: {spec.goal}",
        "",
        f"allowed_modify_paths: {execution_policy.effective_modify_paths(spec)}",
        f"forbidden_paths: {spec.forbidden_paths}",
        "",
        f"approved_command_arrays (the ONLY commands you may propose, byte-for-byte): {allowed_commands}",
        "",
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


def _synthetic_blocked_plan(spec: ExperimentSpec, run_paths: RunPaths, *, reason: str) -> PlanResult:
    code = reason.split(":", 1)[0].strip()
    plan = PlanResult(task_id=spec.task_id, run_id=run_paths.run_id, verdict="PLAN_BLOCKED", summary=reason, issues=[code])
    reporting_tasks.save_json_artifact(run_paths.plan_path, plan)
    return plan


@task(name="execute_codex_planner", retries=2, retry_delay_seconds=1, retry_condition_fn=_retry_only_infrastructure)
def _codex_plan_task(codex_agent: CodexAgent, *, spec: ExperimentSpec, run_paths: RunPaths, repo_root: Path) -> PlanResult:
    prompt = _build_execute_planner_prompt(spec, run_paths=run_paths, repo_root=repo_root)
    before = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="before_execute_codex_planning")
    plan = codex_agent.plan(
        prompt=prompt, run_dir=run_paths.run_dir, cwd=repo_root, timeout=spec.timeouts.planner_seconds,
        task_id=spec.task_id, run_id=run_paths.run_id,
    )
    after = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="after_execute_codex_planning")
    changes = repository_tasks.diff_fingerprints(before, after, run_dir=run_paths.run_dir, repo_root=repo_root)
    if changes:
        raise RuntimeError("REPOSITORY_CHANGED_DURING_CODEX_PLANNING: " + "; ".join(changes))
    reporting_tasks.save_json_artifact(run_paths.plan_path, plan)
    return plan


def _run_codex_planner(codex_agent: CodexAgent, *, spec: ExperimentSpec, run_paths: RunPaths, repo_root: Path) -> PlanResult:
    try:
        return _codex_plan_task(codex_agent, spec=spec, run_paths=run_paths, repo_root=repo_root)
    except ExecutableNotFoundError as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=f"CODEX_EXECUTABLE_UNAVAILABLE: {e}")
    except (CommandTimeoutError, InfrastructureError) as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=f"CODEX_TIMEOUT: {e}")
    except CodexPlannerError as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=str(e))
    except RuntimeError as e:
        return _synthetic_blocked_plan(spec, run_paths, reason=str(e))


@task(name="execute_plan_policy_validation")
def _plan_policy_validation_task(spec: ExperimentSpec, plan: PlanResult) -> list[str]:
    violations: list[str] = []
    if len(plan.proposed_commands) > command_policy.MAX_PROPOSED_COMMANDS:
        violations.append(f"too many proposed commands: {len(plan.proposed_commands)}")
    for cmd in plan.proposed_commands:
        try:
            execution_policy.assert_executor_command_allowed(cmd, spec)
        except execution_policy.ExecutionPolicyViolation as e:
            violations.append(str(e))
    return violations


@task(name="create_execution_worktree", retries=1, retry_delay_seconds=1, retry_condition_fn=_retry_only_infrastructure)
def _create_execution_worktree_task(
    repo_root: Path, run_paths: RunPaths, run_id: str, execution_worktrees_root: Optional[Path]
) -> ExecutionWorktreeRecord:
    return execution_worktree.create_execution_worktree(
        repo_root, run_paths, run_id, execution_worktrees_root=execution_worktrees_root,
    )


@task(name="claude_executor", retries=0)
def _claude_execute_task(
    claude_agent: ClaudeExecutorAgent, *, prompt: str, worktree_dir: Path, run_paths: RunPaths,
    timeout: float, task_id: str, run_id: str,
) -> ExecutorImplementationResult:
    return claude_agent.execute(
        prompt=prompt, worktree_dir=worktree_dir, run_paths=run_paths, timeout=timeout, task_id=task_id, run_id=run_id,
    )


def _read_claude_command(run_paths: RunPaths) -> list[str]:
    path = run_paths.commands_dir / "claude_executor.command.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def _static_check_targets(spec: ExperimentSpec) -> list[str]:
    return spec.required_executor_checks or ["compileall"]


def _run_static_checks(worktree_dir: Path, spec: ExperimentSpec, run_paths: RunPaths) -> list[CommandResult]:
    results: list[CommandResult] = []
    for i, check in enumerate(_static_check_targets(spec)):
        if check == "compileall":
            results.append(verification_tasks.run_static_checks(worktree_dir=worktree_dir, spec=spec, run_paths=run_paths))
        elif check.startswith("pytest:"):
            target = check.split(":", 1)[1]
            command = [sys.executable, "-m", "pytest", "-q", target]
            execution_policy.assert_executor_command_allowed(command, spec)
            results.append(subprocess_runner.run_command(
                command, cwd=worktree_dir, run_dir=run_paths.commands_dir,
                name=f"static_check_pytest_{i}", timeout=spec.timeouts.verifier_seconds,
            ))
        else:
            raise ValueError(f"unknown required_executor_checks entry: {check!r}")
    return results


def _finalize(
    *, run_id: str, task_id: str, run_paths: RunPaths, terminal_state: str, stage: Optional[str] = None,
    reason: Optional[str] = None, plan: Optional[PlanResult] = None,
    implementation: Optional[ExecutorImplementationResult] = None,
    worktree_record: Optional[ExecutionWorktreeRecord] = None,
    changed_records: Optional[list[ChangedFileRecord]] = None,
    static_checks: Optional[list[CommandResult]] = None,
    main_worktree_unchanged: bool = True, ignored_file_manifest_ok: bool = True,
) -> ExecuteReport:
    changed_records = changed_records or []
    byte_count = execution_worktree.total_changed_bytes(changed_records)
    report = ExecuteReport(
        run_id=run_id, task_id=task_id, overall_status=_overall_status_for(terminal_state),
        terminal_state=terminal_state, stage=stage, reason=reason,
        plan=plan, implementation=implementation, claude_command=_read_claude_command(run_paths),
        static_checks=static_checks or [], execution_worktree=worktree_record,
        changed_files=changed_records, changed_file_count=len(changed_records), changed_byte_count=byte_count,
        main_worktree_unchanged=main_worktree_unchanged, ignored_file_manifest_ok=ignored_file_manifest_ok,
        run_dir=str(run_paths.run_dir), created_at=reporting_tasks.utcnow_iso(),
    )
    reporting_tasks.save_json_artifact(run_paths.report_path, report)
    return report


@flow(name="tango-experiment-agent-execute")
def execute_flow(
    spec_path: Path,
    *,
    repo_root: Path,
    runs_root: Path,
    run_id: Optional[str] = None,
    execution_worktrees_root: Optional[Path] = None,
    codex_agent: Optional[CodexAgent] = None,
    claude_agent: Optional[ClaudeExecutorAgent] = None,
) -> ExecuteReport:
    codex_agent = codex_agent or MockCodexAgent()
    claude_agent = claude_agent or MockClaudeExecutorAgent()
    repo_root = Path(repo_root).resolve()
    runs_root = Path(runs_root).resolve()
    spec_path = Path(spec_path).resolve()

    spec = experiment_tasks.load_spec(spec_path)
    experiment_tasks.assert_run_count_within_limit(runs_root, spec.task_id, spec.max_run_count)
    resolved_run_id = run_id or experiment_tasks.generate_run_id(spec.task_id)
    run_paths = experiment_tasks.init_run(runs_root=runs_root, run_id=resolved_run_id, spec_source_path=spec_path)

    repository_tasks.snapshot_repository(repo_root, run_paths)

    plan = _run_codex_planner(codex_agent, spec=spec, run_paths=run_paths, repo_root=repo_root)
    if plan.verdict != "PLAN_PASS":
        reason = f"plan verdict={plan.verdict}: {plan.summary}"
        code = plan.issues[0] if plan.issues else "CODEX_PLAN_NOT_PASS"
        return _finalize(
            run_id=resolved_run_id, task_id=spec.task_id, run_paths=run_paths,
            terminal_state=code, stage="codex_planner", reason=reason, plan=plan,
        )

    plan_violations = _plan_policy_validation_task(spec, plan)
    if plan_violations:
        return _finalize(
            run_id=resolved_run_id, task_id=spec.task_id, run_paths=run_paths,
            terminal_state="CODEX_COMMAND_POLICY_FAILED", stage="plan_policy_validation",
            reason="; ".join(plan_violations), plan=plan,
        )

    try:
        worktree_record = _create_execution_worktree_task(repo_root, run_paths, resolved_run_id, execution_worktrees_root)
    except ExecutionWorktreeError as e:
        return _finalize(
            run_id=resolved_run_id, task_id=spec.task_id, run_paths=run_paths,
            terminal_state=e.code, stage="create_execution_worktree", reason=e.message, plan=plan,
        )

    worktree_dir = Path(worktree_record.worktree_path)

    before_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="before_claude_execution")
    before_manifest = execution_worktree.build_ignored_and_untracked_manifest(worktree_dir, spec)
    execution_worktree.save_manifest(run_paths.ignored_file_manifest_before_path, before_manifest)

    prompt = build_executor_prompt(spec=spec, plan=plan, worktree_record=worktree_record, run_paths=run_paths)

    implementation: Optional[ExecutorImplementationResult] = None
    claude_error_code: Optional[str] = None
    claude_error_reason: Optional[str] = None
    try:
        implementation = _claude_execute_task(
            claude_agent, prompt=prompt, worktree_dir=worktree_dir, run_paths=run_paths,
            timeout=spec.timeouts.executor_seconds, task_id=spec.task_id, run_id=resolved_run_id,
        )
        reporting_tasks.save_json_artifact(run_paths.implementation_path, implementation)
    except ExecutableNotFoundError as e:
        claude_error_code, claude_error_reason = "CLAUDE_EXECUTABLE_UNAVAILABLE", str(e)
    except CommandTimeoutError as e:
        claude_error_code, claude_error_reason = "CLAUDE_TIMEOUT", str(e)
    except ClaudeExecutorError as e:
        claude_error_code, claude_error_reason = e.code, e.message

    # Post-checks always run, regardless of whether the Claude call itself
    # succeeded -- a failed or partially-completed call may still have
    # mutated files that must be inspected and recorded.
    after_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="after_claude_execution")
    main_changes = repository_tasks.diff_fingerprints(before_fp, after_fp, run_dir=run_paths.run_dir, repo_root=repo_root)
    main_worktree_unchanged = not main_changes

    after_manifest = execution_worktree.build_ignored_and_untracked_manifest(worktree_dir, spec)
    execution_worktree.save_manifest(run_paths.ignored_file_manifest_after_path, after_manifest)
    manifest_changes = execution_worktree.diff_manifests(before_manifest, after_manifest)

    status_entries = execution_worktree.collect_worktree_status(worktree_dir, run_paths)
    changed_paths = [p for _, p in status_entries]
    changed_records = execution_worktree.build_changed_file_records(worktree_dir, status_entries)

    # An in-place mutation of an already-ignored file never shows up in
    # `git status`; cross-check the manifest diff against the changed
    # paths git already reports, so only a genuinely *extra*, git-invisible
    # mutation flips ignored_file_manifest_ok to False.
    changed_path_set = {p.rstrip("/") for p in changed_paths}
    unexplained_manifest_changes = [
        c for c in manifest_changes if not any(c.startswith(p) for p in changed_path_set)
    ]
    ignored_file_manifest_ok = not unexplained_manifest_changes

    common_kwargs = dict(
        run_id=resolved_run_id, task_id=spec.task_id, run_paths=run_paths, plan=plan,
        implementation=implementation, worktree_record=worktree_record, changed_records=changed_records,
        main_worktree_unchanged=main_worktree_unchanged, ignored_file_manifest_ok=ignored_file_manifest_ok,
    )

    if not main_worktree_unchanged:
        return _finalize(
            terminal_state="CLAUDE_MAIN_WORKTREE_CHANGED", stage="main_worktree_protection",
            reason="; ".join(main_changes), **common_kwargs,
        )

    if not ignored_file_manifest_ok:
        return _finalize(
            terminal_state="CLAUDE_DIFF_MISMATCH", stage="ignored_file_manifest",
            reason="undeclared ignored/untracked-file mutation detected: " + "; ".join(unexplained_manifest_changes),
            **common_kwargs,
        )

    if claude_error_code:
        return _finalize(terminal_state=claude_error_code, stage="claude_executor", reason=claude_error_reason, **common_kwargs)

    try:
        execution_policy.validate_changed_paths(worktree_dir, changed_paths, spec)
        byte_count = execution_worktree.total_changed_bytes(changed_records)
        execution_policy.assert_change_limits(file_count=len(changed_records), byte_count=byte_count, spec=spec)
    except execution_policy.ExecutionPolicyViolation as e:
        return _finalize(terminal_state=e.code, stage="path_and_change_policy", reason=e.message, **common_kwargs)

    claimed_violations = execution_policy.validate_claimed_commands(implementation, spec)
    if claimed_violations:
        return _finalize(
            terminal_state="CLAUDE_COMMAND_POLICY_FAILED", stage="command_claim_audit",
            reason="; ".join(claimed_violations), **common_kwargs,
        )

    claimed_set = {Path(p).as_posix() for p in implementation.changed_files}
    actual_set = {Path(p.rstrip("/")).as_posix() for p in changed_paths}
    if claimed_set != actual_set:
        return _finalize(
            terminal_state="CLAUDE_DIFF_MISMATCH", stage="diff_match",
            reason=f"implementation.changed_files={sorted(claimed_set)} != actual git diff={sorted(actual_set)}",
            **common_kwargs,
        )

    if implementation.verdict == "IMPLEMENTATION_BLOCKED":
        return _finalize(terminal_state="CLAUDE_IMPLEMENTATION_BLOCKED", stage="claude_executor", reason=implementation.summary, **common_kwargs)
    if implementation.verdict == "IMPLEMENTATION_REVISE":
        return _finalize(terminal_state="CLAUDE_IMPLEMENTATION_REVISE", stage="claude_executor", reason=implementation.summary, **common_kwargs)

    try:
        static_results = _run_static_checks(worktree_dir, spec, run_paths)
    except InfrastructureError as e:
        return _finalize(terminal_state="CLAUDE_TIMEOUT", stage="static_checks", reason=str(e), **common_kwargs)
    except (execution_policy.ExecutionPolicyViolation, ValueError) as e:
        return _finalize(terminal_state="STATIC_CHECKS_FAILED", stage="static_checks", reason=str(e), **common_kwargs)

    if any((r.returncode != 0 or r.timed_out) for r in static_results):
        return _finalize(
            terminal_state="STATIC_CHECKS_FAILED", stage="static_checks",
            reason="one or more static checks failed", static_checks=static_results, **common_kwargs,
        )

    return _finalize(terminal_state="EXECUTION_PASS", stage=None, reason=None, static_checks=static_results, **common_kwargs)
