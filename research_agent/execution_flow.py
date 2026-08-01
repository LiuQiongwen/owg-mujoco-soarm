"""MVP4 restricted, pre-approved experiment execution flow.

Pipeline:

    experiment YAML
    -> validate specification (confirmatory/cpu_only/network/gpu/robot/
       training invariants are enforced by ExperimentSpec/ExecutionSpec's
       own Pydantic validators -- a spec that violates any of them never
       even successfully loads)
    -> Codex read-only planner (mock by default, real with --codex real)
    -> plan policy validation
    -> single shared, isolated execution worktree (created unconditionally,
       for isolation, even when no pre-execution implementation is needed)
    -> OPTIONAL single pre-execution implementation attempt (real or mock
       Claude executor + deterministic content verifier -- only when the
       spec configures expected_file_contents/required_artifacts/
       allowed_executor_commands; skipped entirely otherwise, which is the
       common case for a pure "run this fixed command" restricted spec)
    -> READY_FOR_EXECUTION / EXECUTION_NOT_REQUESTED gate: nothing below
       this point ever runs without an explicit --execute
    -> EXECUTING: every approved command is independently re-authorized
       (research_agent.policies.experiment_commands.authorize_execution)
       immediately before it is spawned via the restricted, resource-
       limited, process-group-isolated subprocess runner
       (research_agent.restricted_subprocess) -- Codex/Claude are advisors
       only and are NEVER consulted for this decision
    -> COLLECTING_ARTIFACTS: a fresh filesystem scan of the run's assigned
       artifacts directory (research_agent.policies.artifact_policy)
    -> VERIFYING_RESULTS: deterministic metric/artifact verification
       (research_agent.tasks.metric_verifier)
    -> on a retriable execution/verification failure (a nonzero exit
       plausibly caused by a code defect, or a metric/artifact mismatch)
       AND an implementation scope is configured, a BOUNDED number of
       (Codex diagnosis -> Claude repair -> re-verify implementation ->
       RETRYING_EXECUTION, re-running the exact same approved command)
       rounds follow
    -> terminal state: PASS, BLOCKED, RETRY_EXHAUSTED, INFRASTRUCTURE_FAILURE,
       POLICY_FAILURE, EXECUTION_FAILED, VERIFICATION_FAILED, or
       EXECUTION_NOT_REQUESTED -- state.json is NEVER left in a non-terminal
       state when this function returns, including after an unexpected
       internal exception (see the outer try/except at the bottom).

Never runs a GPU, robot, training, Docker, sudo, package-install, network-
download, git commit/push, or confirmatory command -- confirmatory execution
is rejected deterministically before planning even starts (see
_assert_no_confirmatory_indicators and ExecutionSpec's own validators).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import NamedTuple, Optional

from research_agent import execution_worktree
from research_agent.agents.claude_executor import (
    ClaudeExecutorAgent,
    ClaudeExecutorError,
    MockClaudeExecutorAgent,
    build_executor_prompt,
)
from research_agent.agents.codex import CodexAgent, CodexPlannerError, MockCodexAgent
from research_agent.execution_worktree import ExecutionWorktreeError
from research_agent.failure_taxonomy import build_failure_record, terminal_state_for
from research_agent.models import (
    AttemptRecord,
    ChangedFileRecord,
    ExecutionAttempt,
    ExecutionFinalReport,
    ExecutionLimits,
    ExecutionWorktreeRecord,
    ExecutorImplementationResult,
    ExperimentSpec,
    PlanResult,
)
from research_agent.policies import artifact_policy
from research_agent.policies import commands as command_policy
from research_agent.policies import environment_policy
from research_agent.policies import execution_policy
from research_agent.policies import experiment_commands
from research_agent.restricted_subprocess import (
    RestrictedExecutableNotFoundError,
    RestrictedSubprocessError,
    run_restricted_command,
)
from research_agent.subprocess_runner import CommandTimeoutError, ExecutableNotFoundError, InfrastructureError
from research_agent.tasks import experiment as experiment_tasks
from research_agent.tasks import experiment_execution as experiment_execution_tasks
from research_agent.tasks import reporting as reporting_tasks
from research_agent.tasks import repair as repair_tasks
from research_agent.tasks import repository as repository_tasks
from research_agent.tasks.metric_verifier import run_execution_verifier
from research_agent.tasks.repair_verification import primary_failure_class, run_content_verifier

_CLAUDE_ERROR_TO_FAILURE_CLASS = {
    "CLAUDE_AUTHENTICATION_FAILED": "CLAUDE_INFRASTRUCTURE_FAILURE",
    "CLAUDE_NONZERO_EXIT": "CLAUDE_INFRASTRUCTURE_FAILURE",
    "CLAUDE_OUTPUT_MISSING": "IMPLEMENTATION_SCHEMA_FAILURE",
    "CLAUDE_OUTPUT_MALFORMED": "IMPLEMENTATION_SCHEMA_FAILURE",
    "CLAUDE_SCHEMA_INVALID": "IMPLEMENTATION_SCHEMA_FAILURE",
    "CLAUDE_TASK_ID_MISMATCH": "IMPLEMENTATION_SCHEMA_FAILURE",
    "CLAUDE_RUN_ID_MISMATCH": "IMPLEMENTATION_SCHEMA_FAILURE",
    "CLAUDE_ATTEMPT_INDEX_MISMATCH": "IMPLEMENTATION_SCHEMA_FAILURE",
}
_CODEX_ERROR_TO_FAILURE_CLASS = {
    "CODEX_AUTHENTICATION_FAILED": "CODEX_INFRASTRUCTURE_FAILURE",
    "CODEX_NONZERO_EXIT": "CODEX_INFRASTRUCTURE_FAILURE",
    "CODEX_OUTPUT_MISSING": "PLAN_SCHEMA_FAILURE",
    "CODEX_OUTPUT_MALFORMED": "PLAN_SCHEMA_FAILURE",
    "CODEX_SCHEMA_INVALID": "PLAN_SCHEMA_FAILURE",
    "CODEX_TASK_ID_MISMATCH": "PLAN_SCHEMA_FAILURE",
    "CODEX_RUN_ID_MISMATCH": "PLAN_SCHEMA_FAILURE",
    "CODEX_ATTEMPT_INDEX_MISMATCH": "PLAN_SCHEMA_FAILURE",
}
_EXECUTION_POLICY_CODE_TO_FAILURE_CLASS = {
    "CLAUDE_PATH_POLICY_FAILED": "PATH_POLICY_FAILURE",
    "CLAUDE_SYMLINK_ESCAPE_DETECTED": "SYMLINK_ESCAPE",
    "CLAUDE_NESTED_GIT_DETECTED": "NESTED_GIT",
    "CLAUDE_CHANGED_FILE_LIMIT_EXCEEDED": "PATH_POLICY_FAILURE",
    "CLAUDE_CHANGED_BYTES_LIMIT_EXCEEDED": "PATH_POLICY_FAILURE",
    "CLAUDE_COMMAND_POLICY_FAILED": "COMMAND_POLICY_FAILURE",
}

_CONFIRMATORY_INDICATOR_SUBSTRINGS = (
    "confirmatory", "paper_final", "paper-final", "final_result", "final-result",
)
_HIGH_RISK_PATH_SUBSTRINGS = ("results/", "checkpoints/", "data/", "datasets/", "paperA_data/")


class ConfirmatoryRejected(RuntimeError):
    """MVP4 must never execute anything that looks like a confirmatory or
    final-result-targeting run -- see the MVP4 task contract's 'Confirmatory
    gate' section. This is checked independently of (in addition to)
    ExecutionSpec's own Pydantic validators, which already reject
    execution_mode='confirmatory'/confirmatory=true at spec-load time."""


def _assert_no_confirmatory_indicators(spec: ExperimentSpec) -> None:
    haystack = f"{spec.task_id} {spec.goal}".lower()
    for marker in _CONFIRMATORY_INDICATOR_SUBSTRINGS:
        if marker in haystack:
            raise ConfirmatoryRejected(
                f"CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL: task_id/goal contains {marker!r}"
            )
    execution = spec.execution
    for command in (execution.approved_commands if execution else []):
        for tok in command:
            low = tok.lower()
            for marker in _CONFIRMATORY_INDICATOR_SUBSTRINGS + _HIGH_RISK_PATH_SUBSTRINGS:
                if marker in low:
                    raise ConfirmatoryRejected(
                        f"CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL: approved command token contains "
                        f"{marker!r}: {tok!r}"
                    )


def _implementation_needed(spec: ExperimentSpec) -> bool:
    """True iff the spec configures a pre-execution implementation scope
    (MVP2/3-style fields). When False, IMPLEMENTING/VERIFYING_IMPLEMENTATION
    are skipped entirely -- the common case for a pure "run this fixed
    command" restricted spec, including the MVP4 live-validation spec."""
    return bool(spec.expected_file_contents) or bool(spec.required_artifacts) or bool(spec.allowed_executor_commands)


# ── policy gate for the (optional) implementation phase -- structurally
# identical to repair_flow._check_policy_gate, kept as an independent copy
# here rather than imported so MVP3's repair_flow.py is never touched by
# MVP4 work (protects the MVP3 regression suite) ───────────────────────────

class _PolicyGateResult(NamedTuple):
    failure_class: Optional[str]
    reason: Optional[str]
    evidence: list[str]
    changed_records: list[ChangedFileRecord]
    main_worktree_unchanged: bool


def _check_implementation_policy_gate(
    *, repo_root: Path, run_paths, worktree_dir: Path, spec: ExperimentSpec,
    before_fp: dict, before_manifest: list, base_commit: str,
    claimed_changed_files: Optional[list[str]], claimed_commands: list[list[str]],
    attempt_index: int, name_prefix: str,
) -> _PolicyGateResult:
    after_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label=f"after_{name_prefix}")
    main_changes = repository_tasks.diff_fingerprints(before_fp, after_fp, run_dir=run_paths.run_dir, repo_root=repo_root)
    after_manifest = execution_worktree.build_ignored_and_untracked_manifest(worktree_dir, spec)
    if main_changes:
        return _PolicyGateResult("MAIN_WORKTREE_CHANGED", "; ".join(main_changes), main_changes, [], False)

    manifest_changes = execution_worktree.diff_manifests(before_manifest, after_manifest)
    command_results_dir = run_paths.attempt_command_results_dir(attempt_index)
    status_entries = repair_tasks.collect_worktree_status(worktree_dir, command_results_dir=command_results_dir, name_prefix=name_prefix)
    changed_paths = [p for _, p in status_entries]
    changed_records = execution_worktree.build_changed_file_records(worktree_dir, status_entries)

    changed_path_set = {p.rstrip("/") for p in changed_paths}
    unexplained = [c for c in manifest_changes if not any(c.startswith(p) for p in changed_path_set)]
    if unexplained:
        if any(c.startswith(".git:") for c in unexplained):
            return _PolicyGateResult("GIT_METADATA_TAMPERING", "; ".join(unexplained), unexplained, changed_records, True)
        return _PolicyGateResult(
            "DIFF_MISMATCH", "undeclared ignored/untracked-file mutation detected: " + "; ".join(unexplained),
            unexplained, changed_records, True,
        )

    head = repair_tasks.worktree_head(worktree_dir, command_results_dir=command_results_dir, name_prefix=name_prefix)
    if head != base_commit:
        return _PolicyGateResult(
            "GIT_METADATA_TAMPERING", f"execution worktree HEAD moved from base_commit={base_commit} to {head}",
            [], changed_records, True,
        )

    try:
        execution_policy.validate_changed_paths(worktree_dir, changed_paths, spec)
    except execution_policy.ExecutionPolicyViolation as e:
        fc = _EXECUTION_POLICY_CODE_TO_FAILURE_CLASS.get(e.code, "PATH_POLICY_FAILURE")
        return _PolicyGateResult(fc, e.message, [e.message], changed_records, True)

    byte_count = execution_worktree.total_changed_bytes(changed_records)
    try:
        execution_policy.assert_change_limits(file_count=len(changed_records), byte_count=byte_count, spec=spec)
    except execution_policy.ExecutionPolicyViolation as e:
        return _PolicyGateResult("PATH_POLICY_FAILURE", e.message, [e.message], changed_records, True)

    command_violations: list[str] = []
    for cmd in claimed_commands:
        try:
            execution_policy.assert_executor_command_allowed(cmd, spec)
        except execution_policy.ExecutionPolicyViolation as e:
            command_violations.append(str(e))
    if command_violations:
        return _PolicyGateResult("COMMAND_POLICY_FAILURE", "; ".join(command_violations), command_violations, changed_records, True)

    claimed_set = None if claimed_changed_files is None else {Path(p).as_posix() for p in claimed_changed_files}
    actual_set = {Path(p.rstrip("/")).as_posix() for p in changed_paths}
    if claimed_set is not None and claimed_set != actual_set:
        return _PolicyGateResult(
            "DIFF_MISMATCH", f"claimed changed_files={sorted(claimed_set)} != actual git diff={sorted(actual_set)}",
            [], changed_records, True,
        )

    return _PolicyGateResult(None, None, [], changed_records, True)


# ── prompt builders (plain text; no LLM call happens here) ─────────────────

def _build_planner_prompt(spec: ExperimentSpec, *, run_paths, repo_root: Path, execute_requested: bool) -> str:
    execution = spec.execution
    approved = execution.approved_commands if execution else []
    return "\n".join([
        "You are a READ-ONLY planning assistant for the TANGO Experiment Agent's MVP4 restricted, ",
        "pre-approved experiment execution flow.",
        "",
        "MODE: planning-only, non-confirmatory, read-only. You never execute anything yourself. If your",
        "PLAN_PASS verdict is accepted, this harness may (only if --execute was explicitly passed on the",
        "command line, and only for execution_mode in {smoke, restricted}) run ONE OR MORE commands that",
        "already exist, byte-for-byte, in the specification's execution.approved_commands list below --",
        "never a command you or any other agent proposes.",
        "",
        "SAFETY RESTRICTIONS (mandatory, no exceptions):",
        "- Do not modify any file in the repository.",
        "- Do not propose any command that is not already listed verbatim in approved_commands below.",
        "- Do not invoke Claude, claude-code, or any other coding/execution agent.",
        "- Do not propose or endorse anything involving a GPU, CUDA, a physical robot, model training,",
        "  Docker, sudo, network access, `git push`, `git commit`, or a confirmatory experiment.",
        "",
        f"task_id: {spec.task_id}",
        f"run_id: {run_paths.run_id}",
        f"repository_root: {repo_root}",
        f"goal: {spec.goal}",
        f"execute_requested (informational only -- you cannot change this): {execute_requested}",
        "",
        f"execution.execution_mode: {execution.execution_mode if execution else None}",
        f"execution.approved_commands (the ONLY commands that may ever run): {approved}",
        f"execution.cpu_only={execution.cpu_only if execution else None} "
        f"network_allowed={execution.network_allowed if execution else None} "
        f"gpu_allowed={execution.gpu_allowed if execution else None} "
        f"robot_allowed={execution.robot_allowed if execution else None} "
        f"training_allowed={execution.training_allowed if execution else None}",
        "",
        f"max_proposed_commands: {command_policy.MAX_PROPOSED_COMMANDS}",
        "Return exactly one JSON object matching the supplied PlanResult schema, with fields: "
        "schema_version, task_id, run_id, verdict, summary, issues, artifacts, proposed_commands, "
        "expected_artifacts, expected_metrics, risks, assumptions.",
        f"task_id must be exactly {spec.task_id!r}; run_id must be exactly {run_paths.run_id!r}.",
        "verdict must be exactly one of PLAN_PASS, PLAN_REVISE, PLAN_BLOCKED.",
        "proposed_commands should normally be empty ([]); if you list anything, it must exactly match either an "
        "approved implementation command or one of execution.approved_commands above -- this field is purely "
        "informational and never causes anything to run.",
    ])


def _build_diagnosis_prompt(*, spec, run_paths, plan, failure, verifier_result, attempt_index, remaining_repair_rounds) -> str:
    modify_paths = execution_policy.effective_modify_paths(spec)
    return "\n".join([
        "You are a READ-ONLY diagnosis assistant for the TANGO Experiment Agent's MVP4 post-execution",
        "bounded repair integration. The APPROVED EXPERIMENT COMMAND itself ran and either exited nonzero",
        "or its results failed deterministic verification; you diagnose why and propose a SCOPE-RESTRICTED",
        "source-code repair for a following Claude repair step. You never modify anything yourself, and you",
        "can never change WHICH command will be re-run -- only the source code it depends on.",
        "",
        f"task_id: {spec.task_id}",
        f"run_id: {run_paths.run_id}",
        f"attempt_index: {attempt_index}",
        f"allowed_modify_paths (repair scope; can only be narrowed, never broadened): {modify_paths}",
        f"EXACT VERIFIER FAILURE: verdict={verifier_result.verdict} issues={verifier_result.issues}",
        f"STRUCTURED FAILURE RECORD: {failure.model_dump()}",
        f"remaining_repair_rounds: {remaining_repair_rounds}",
        "",
        "PROHIBITED: commit, push, merge, GPU, robot, Docker, sudo, package install, network download, any",
        "confirmatory/research experiment, and proposing a DIFFERENT command than the one already approved.",
        "",
        "Return exactly one JSON object matching the supplied DiagnosisResult schema.",
        f"task_id must be exactly {spec.task_id!r}; run_id must be exactly {run_paths.run_id!r}; "
        f"attempt_index must be exactly {attempt_index}.",
    ])


def _build_repair_prompt(*, spec, run_paths, diagnosis, verifier_result, attempt_index, remaining_repair_rounds) -> str:
    modify_paths = execution_policy.effective_modify_paths(spec)
    effective_files = diagnosis.files_allowed_to_touch or modify_paths
    return "\n".join([
        "You are the MVP4 post-execution REPAIR role: fix the SMALLEST NECESSARY source-code defect so the",
        "already-approved experiment command (unchanged) passes deterministic verification next time.",
        "You never change which command will be re-run, and you never run the experiment command yourself.",
        "",
        f"task_id: {spec.task_id}",
        f"run_id: {run_paths.run_id}",
        f"attempt_index: {attempt_index}",
        f"DIAGNOSIS: {diagnosis.model_dump()}",
        f"EXACT VERIFIER FAILURE: verdict={verifier_result.verdict} issues={verifier_result.issues}",
        f"ALLOWED MODIFICATION PATHS (never write outside this list): {effective_files}",
        f"remaining_repair_rounds (including this one): {remaining_repair_rounds}",
        "",
        "STRICT RULES: no git commit/push, no new/removed worktree, no nested Git, never touch the main",
        "repository worktree, no Codex/Claude/other-agent invocation, no sudo/Docker/GPU/robot/network, no",
        "research experiment command of any kind. If you cannot safely apply the diagnosis, report verdict=",
        "REPAIR_BLOCKED.",
        "",
        "Return exactly one JSON object matching the supplied RepairResult schema.",
        f"task_id must be exactly {spec.task_id!r}; run_id must be exactly {run_paths.run_id!r}; "
        f"attempt_index must be exactly {attempt_index}.",
    ])


# ── main flow ────────────────────────────────────────────────────────────

def run_experiment_flow(
    spec_path: Path,
    *,
    repo_root: Path,
    runs_root: Path,
    run_id: Optional[str] = None,
    execution_worktrees_root: Optional[Path] = None,
    codex_agent: Optional[CodexAgent] = None,
    claude_agent: Optional[ClaudeExecutorAgent] = None,
    execute: bool = False,
) -> ExecutionFinalReport:
    codex_agent = codex_agent or MockCodexAgent()
    claude_agent = claude_agent or MockClaudeExecutorAgent()
    repo_root = Path(repo_root).resolve()
    runs_root = Path(runs_root).resolve()
    spec_path = Path(spec_path).resolve()

    spec = experiment_tasks.load_spec(spec_path)
    _assert_no_confirmatory_indicators(spec)

    execution = spec.execution
    limits = execution.limits if execution else ExecutionLimits()

    experiment_tasks.assert_run_count_within_limit(runs_root, spec.task_id, spec.max_run_count)
    resolved_run_id = run_id or experiment_tasks.generate_run_id(spec.task_id)
    run_paths = experiment_execution_tasks.init_execution_run(runs_root=runs_root, run_id=resolved_run_id, spec_source_path=spec_path)

    start_time = time.monotonic()
    history: list[str] = []
    codex_invocations = 0
    claude_invocations = 0
    plan: Optional[PlanResult] = None
    implementation: Optional[ExecutorImplementationResult] = None
    implementation_attempts: list[AttemptRecord] = []
    worktree_record: Optional[ExecutionWorktreeRecord] = None
    main_worktree_unchanged_overall = True
    execution_attempts: list[ExecutionAttempt] = []
    artifact_manifest: list = []
    metrics: dict = {}
    verifier_result = None
    total_command_runtime = 0.0

    def _elapsed() -> float:
        return time.monotonic() - start_time

    def _transition(state: str, *, attempt_index: int = 0, detail: Optional[str] = None) -> None:
        nonlocal history
        record = experiment_execution_tasks.persist_execution_state(
            run_paths, run_id=resolved_run_id, task_id=spec.task_id, state=state,
            attempt_index=attempt_index, history=history, detail=detail,
        )
        history = record.history

    def _finalize(
        *, final_state: str, stage: Optional[str] = None, reason: Optional[str] = None,
        passing_attempt_index: Optional[int] = None, manual_action_required: bool = False,
    ) -> ExecutionFinalReport:
        last_attempt_index = execution_attempts[-1].attempt_index if execution_attempts else 0
        _transition(final_state, attempt_index=last_attempt_index, detail=reason)
        overall = "PASS" if final_state == "PASS" else ("FAIL" if final_state in ("RETRY_EXHAUSTED", "EXECUTION_FAILED", "VERIFICATION_FAILED") else "BLOCKED")
        report = ExecutionFinalReport(
            run_id=resolved_run_id, task_id=spec.task_id, overall_status=overall, final_state=final_state,
            stage=stage, reason=reason, execute_requested=execute,
            execution_mode=(execution.execution_mode if execution else None),
            plan=plan, implementation=implementation, implementation_attempts=implementation_attempts,
            execution_attempts=execution_attempts, passing_attempt_index=passing_attempt_index,
            artifact_manifest=artifact_manifest, metrics=metrics, verifier=verifier_result,
            execution_worktree=worktree_record, main_worktree_unchanged=main_worktree_unchanged_overall,
            total_codex_invocations=codex_invocations, total_claude_invocations=claude_invocations,
            total_command_runtime_seconds=round(total_command_runtime, 3), wall_clock_seconds=round(_elapsed(), 3),
            execution_limits=limits, manual_action_required=manual_action_required or final_state not in ("PASS", "EXECUTION_NOT_REQUESTED"),
            run_dir=str(run_paths.run_dir), created_at=reporting_tasks.utcnow_iso(),
        )
        reporting_tasks.save_json_artifact(run_paths.final_report_path, report)
        return report

    try:
        # ── PLANNING ─────────────────────────────────────────────────────
        _transition("PLANNING")
        plan_prompt = _build_planner_prompt(spec, run_paths=run_paths, repo_root=repo_root, execute_requested=execute)
        before_plan_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="before_planning")
        try:
            plan = codex_agent.plan(
                prompt=plan_prompt, run_dir=run_paths.run_dir, cwd=repo_root,
                timeout=spec.timeouts.planner_seconds, task_id=spec.task_id, run_id=resolved_run_id,
            )
            codex_invocations += 1
        except ExecutableNotFoundError as e:
            codex_invocations += 1
            return _finalize(final_state="INFRASTRUCTURE_FAILURE", stage="codex_planner", reason=f"CODEX_EXECUTABLE_UNAVAILABLE: {e}")
        except (CommandTimeoutError, InfrastructureError) as e:
            codex_invocations += 1
            return _finalize(final_state="INFRASTRUCTURE_FAILURE", stage="codex_planner", reason=f"CODEX_TIMEOUT: {e}")
        except CodexPlannerError as e:
            codex_invocations += 1
            fc = _CODEX_ERROR_TO_FAILURE_CLASS.get(e.code, "PLAN_SCHEMA_FAILURE")
            state = "INFRASTRUCTURE_FAILURE" if fc == "CODEX_INFRASTRUCTURE_FAILURE" else "BLOCKED"
            return _finalize(final_state=state, stage="codex_planner", reason=str(e))

        reporting_tasks.save_json_artifact(run_paths.plan_path, plan)
        after_plan_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="after_planning")
        plan_changes = repository_tasks.diff_fingerprints(before_plan_fp, after_plan_fp, run_dir=run_paths.run_dir, repo_root=repo_root)
        if plan_changes:
            main_worktree_unchanged_overall = False
            return _finalize(
                final_state="POLICY_FAILURE", stage="codex_planner",
                reason="REPOSITORY_CHANGED_DURING_CODEX_PLANNING (main worktree mutated): " + "; ".join(plan_changes),
            )

        if plan.verdict != "PLAN_PASS":
            return _finalize(final_state="BLOCKED", stage="codex_planner", reason=f"plan verdict={plan.verdict}: {plan.summary}")

        plan_violations: list[str] = []
        if len(plan.proposed_commands) > command_policy.MAX_PROPOSED_COMMANDS:
            plan_violations.append(f"too many proposed commands: {len(plan.proposed_commands)}")
        approved_experiment_commands = [list(c) for c in (execution.approved_commands if execution else [])]
        for cmd in plan.proposed_commands:
            # A proposed command is clean if it is EITHER an allowed MVP2/3-style
            # pre-execution implementation command (compileall/pytest/git status)
            # OR exactly one of this spec's MVP4 approved_commands (Codex's plan
            # naturally echoes back the one command it expects the harness to
            # run -- that is not itself an authorization; only
            # experiment_commands.authorize_execution, called independently at
            # EXECUTING time, ever decides what actually runs).
            if list(cmd) in approved_experiment_commands:
                continue
            try:
                execution_policy.assert_executor_command_allowed(cmd, spec)
            except execution_policy.ExecutionPolicyViolation as e:
                plan_violations.append(str(e))
        if plan_violations:
            return _finalize(final_state="POLICY_FAILURE", stage="plan_policy_validation", reason="; ".join(plan_violations))

        _transition("PLAN_VALIDATED")

        # ── defense-in-depth: re-validate every approved experiment command
        # (not just the pre-execution implementation commands above) ──────
        cmd_violations = experiment_commands.validate_approved_commands(spec)
        if cmd_violations:
            return _finalize(final_state="POLICY_FAILURE", stage="approved_commands_validation", reason="; ".join(cmd_violations))

        # ── shared execution worktree (always created, for isolation) ─────
        try:
            worktree_record = execution_worktree.create_execution_worktree(
                repo_root, run_paths, resolved_run_id, execution_worktrees_root=execution_worktrees_root,
            )
        except ExecutionWorktreeError as e:
            return _finalize(final_state="POLICY_FAILURE", stage="create_execution_worktree", reason=e.message)
        worktree_dir = Path(worktree_record.worktree_path)

        # ── optional pre-execution implementation (single attempt) ────────
        if _implementation_needed(spec):
            if claude_invocations >= limits.max_total_claude_invocations:
                return _finalize(final_state="RETRY_EXHAUSTED", stage="claude_budget", reason="max_total_claude_invocations exhausted before implementation could run")

            _transition("IMPLEMENTING")
            attempt_dir = run_paths.attempt_dir(0)
            attempt_dir.mkdir(parents=True)
            implementation_prompt = build_executor_prompt(spec=spec, plan=plan, worktree_record=worktree_record, run_paths=run_paths)
            before_impl_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="before_implementation")
            before_impl_manifest = execution_worktree.build_ignored_and_untracked_manifest(worktree_dir, spec)

            claude_error: Optional[tuple[str, str]] = None
            try:
                implementation = claude_agent.execute(
                    prompt=implementation_prompt, worktree_dir=worktree_dir, run_paths=run_paths,
                    timeout=spec.timeouts.executor_seconds, task_id=spec.task_id, run_id=resolved_run_id,
                )
                claude_invocations += 1
                reporting_tasks.save_json_artifact(run_paths.implementation_path, implementation)
                reporting_tasks.save_json_artifact(attempt_dir / "implementation.json", implementation)
            except ExecutableNotFoundError as e:
                claude_invocations += 1
                claude_error = ("CLAUDE_INFRASTRUCTURE_FAILURE", f"CLAUDE_EXECUTABLE_UNAVAILABLE: {e}")
            except CommandTimeoutError as e:
                claude_invocations += 1
                claude_error = ("CLAUDE_INFRASTRUCTURE_FAILURE", f"CLAUDE_TIMEOUT: {e}")
            except ClaudeExecutorError as e:
                claude_invocations += 1
                claude_error = (_CLAUDE_ERROR_TO_FAILURE_CLASS.get(e.code, "IMPLEMENTATION_SCHEMA_FAILURE"), str(e))

            gate = _check_implementation_policy_gate(
                repo_root=repo_root, run_paths=run_paths, worktree_dir=worktree_dir, spec=spec,
                before_fp=before_impl_fp, before_manifest=before_impl_manifest, base_commit=worktree_record.base_commit,
                claimed_changed_files=(implementation.changed_files if implementation else None),
                claimed_commands=(implementation.commands_run if implementation else []),
                attempt_index=0, name_prefix="attempt00_implementation",
            )
            if not gate.main_worktree_unchanged:
                main_worktree_unchanged_overall = False

            started_at = reporting_tasks.utcnow_iso()
            repair_tasks.write_attempt_git_artifacts(worktree_dir, attempt_dir, name_prefix="attempt00_implementation")
            implementation_attempts.append(AttemptRecord(
                attempt_index=0, kind="implementation",
                verdict=(implementation.verdict if implementation else None), verifier_verdict=None,
                failure_class=gate.failure_class, retriable=False if gate.failure_class or claude_error else None,
                changed_file_count=len(gate.changed_records),
                changed_byte_count=execution_worktree.total_changed_bytes(gate.changed_records),
                started_at=started_at, ended_at=reporting_tasks.utcnow_iso(), duration_seconds=0.0,
            ))

            if gate.failure_class:
                return _finalize(final_state=terminal_state_for(gate.failure_class), stage="implementation_policy_gate", reason=gate.reason)
            if claude_error:
                fc, reason = claude_error
                return _finalize(final_state=terminal_state_for(fc), stage="claude_executor", reason=reason)
            if implementation.verdict == "IMPLEMENTATION_BLOCKED":
                return _finalize(final_state="BLOCKED", stage="claude_executor", reason=implementation.summary)

            _transition("VERIFYING_IMPLEMENTATION")
            impl_verifier, breakdown, _static = run_content_verifier(
                spec=spec, worktree_dir=worktree_dir, attempt_dir=attempt_dir,
                task_id=spec.task_id, run_id=resolved_run_id, attempt_index=0,
            )
            reporting_tasks.save_json_artifact(attempt_dir / "verifier.json", impl_verifier)
            implementation_attempts[-1] = implementation_attempts[-1].model_copy(update={"verifier_verdict": impl_verifier.verdict})
            if impl_verifier.verdict != "PASS":
                fc0 = primary_failure_class(breakdown)
                return _finalize(final_state=terminal_state_for(fc0), stage="implementation_verifier", reason="; ".join(impl_verifier.details))

        # ── READY_FOR_EXECUTION / EXECUTION_NOT_REQUESTED gate ─────────────
        _transition("READY_FOR_EXECUTION")
        if not execute:
            return _finalize(final_state="EXECUTION_NOT_REQUESTED", stage="execution_gate", reason="--execute was not passed; planning and validation completed successfully")

        if execution is None or not execution.approved_commands:
            return _finalize(final_state="BLOCKED", stage="execution_gate", reason="no execution.approved_commands configured -- nothing to execute")
        if execution.execution_mode not in ("smoke", "restricted"):
            return _finalize(final_state="POLICY_FAILURE", stage="execution_gate", reason=f"execution_mode={execution.execution_mode!r} is not permitted in this MVP4 build")

        before_exec_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="before_execution")
        current_commands = [list(c) for c in execution.approved_commands]

        # ── bounded EXECUTING / VERIFYING_RESULTS / repair loop ────────────
        attempt_index = 0
        repair_round = 0
        while True:
            if attempt_index >= limits.max_execution_attempts:
                return _finalize(final_state="RETRY_EXHAUSTED", stage="execution_budget", reason=f"max_execution_attempts={limits.max_execution_attempts} exhausted")
            if _elapsed() > limits.max_wall_clock_seconds:
                return _finalize(final_state="RETRY_EXHAUSTED", stage="wall_clock_budget", reason=f"max_wall_clock_seconds={limits.max_wall_clock_seconds} exceeded")

            attempt_started = reporting_tasks.utcnow_iso()
            attempt_t0 = time.monotonic()
            _transition("EXECUTING", attempt_index=attempt_index)

            exec_cwd = worktree_dir if execution.working_directory_policy == "execution_worktree" else run_paths.execution_cwd_dir
            env = environment_policy.build_child_environment(spec, run_dir=run_paths.run_dir, artifacts_dir=run_paths.artifacts_dir)

            command_results = []
            command_issues: list[str] = []
            command_authorization_violations: list[str] = []
            infra_failure: Optional[tuple[str, str]] = None
            runtime_this_attempt = 0.0

            for i, command in enumerate(current_commands):
                if i >= limits.max_commands:
                    command_issues.append(f"command index {i} exceeds limits.max_commands={limits.max_commands}")
                    break
                try:
                    experiment_commands.authorize_execution(command, spec)
                except experiment_commands.ExperimentCommandPolicyViolation as e:
                    command_authorization_violations.append(str(e))
                    break

                remaining_runtime_budget = max(1.0, limits.max_total_command_runtime_seconds - total_command_runtime - runtime_this_attempt)
                per_cmd_timeout = min(limits.per_command_timeout_seconds, remaining_runtime_budget)
                command_dir = run_paths.execution_command_dir(attempt_index, i)
                try:
                    result = run_restricted_command(
                        command, cwd=exec_cwd, command_dir=command_dir, name=f"command_{i:02d}", env=env,
                        timeout=per_cmd_timeout, approved_command=command,
                        working_directory_policy=execution.working_directory_policy,
                        max_output_bytes=limits.max_output_bytes,
                    )
                except RestrictedExecutableNotFoundError as e:
                    infra_failure = ("EXECUTABLE_UNAVAILABLE", str(e))
                    break
                except RestrictedSubprocessError as e:
                    infra_failure = ("LAUNCH_FAILED", str(e))
                    break

                command_results.append(result)
                runtime_this_attempt += result.duration_seconds

                if result.timed_out:
                    command_issues.append(f"command {i} ({command}) timed out after {per_cmd_timeout:.1f}s")
                    break
                if result.returncode != 0:
                    command_issues.append(f"command {i} ({command}) exited with nonzero returncode={result.returncode}")
                    break

            total_command_runtime += runtime_this_attempt

            after_exec_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label=f"after_execution_{attempt_index:02d}")
            exec_changes = repository_tasks.diff_fingerprints(before_exec_fp, after_exec_fp, run_dir=run_paths.run_dir, repo_root=repo_root)
            if exec_changes:
                main_worktree_unchanged_overall = False
                attempt = ExecutionAttempt(
                    attempt_index=attempt_index, commands=command_results, verifier=None,
                    failure_class="MAIN_WORKTREE_CHANGED", retriable=False,
                    started_at=attempt_started, ended_at=reporting_tasks.utcnow_iso(),
                    duration_seconds=round(time.monotonic() - attempt_t0, 3),
                )
                execution_attempts.append(attempt)
                return _finalize(final_state="POLICY_FAILURE", stage="main_worktree_check", reason="main worktree mutated during execution: " + "; ".join(exec_changes))
            before_exec_fp = after_exec_fp

            if command_authorization_violations:
                attempt = ExecutionAttempt(
                    attempt_index=attempt_index, commands=command_results, verifier=None,
                    failure_class="COMMAND_POLICY_FAILURE", retriable=False,
                    started_at=attempt_started, ended_at=reporting_tasks.utcnow_iso(),
                    duration_seconds=round(time.monotonic() - attempt_t0, 3),
                )
                execution_attempts.append(attempt)
                return _finalize(final_state="POLICY_FAILURE", stage="command_authorization", reason="; ".join(command_authorization_violations))

            if infra_failure:
                _code, msg = infra_failure
                attempt = ExecutionAttempt(
                    attempt_index=attempt_index, commands=command_results, verifier=None,
                    failure_class=None, retriable=False, started_at=attempt_started,
                    ended_at=reporting_tasks.utcnow_iso(), duration_seconds=round(time.monotonic() - attempt_t0, 3),
                )
                execution_attempts.append(attempt)
                return _finalize(final_state="INFRASTRUCTURE_FAILURE", stage="restricted_subprocess", reason=msg)

            # ── COLLECTING_ARTIFACTS ────────────────────────────────────
            _transition("COLLECTING_ARTIFACTS", attempt_index=attempt_index)
            artifact_manifest, artifact_violations = artifact_policy.scan_artifacts(run_paths.artifacts_dir, spec)
            run_paths.artifact_manifest_path.write_text(
                json.dumps([r.model_dump() for r in artifact_manifest], indent=2) + "\n"
            )

            # ── VERIFYING_RESULTS ────────────────────────────────────────
            _transition("VERIFYING_RESULTS", attempt_index=attempt_index)
            command_checks = [
                f"command_{i:02d}_exit_zero" for i, r in enumerate(command_results)
                if r.returncode == 0 and not r.timed_out
            ]
            verifier_result, metrics = run_execution_verifier(
                spec=spec, artifacts_dir=run_paths.artifacts_dir, task_id=spec.task_id, run_id=resolved_run_id,
                attempt_index=attempt_index, command_checks=command_checks, command_issues=command_issues,
                policy_checks=[], policy_issues=artifact_violations,
            )
            reporting_tasks.save_json_artifact(run_paths.execution_verifier_path, verifier_result)
            run_paths.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

            duration = round(time.monotonic() - attempt_t0, 3)
            ended_at = reporting_tasks.utcnow_iso()

            if verifier_result.verdict == "PASS":
                execution_attempts.append(ExecutionAttempt(
                    attempt_index=attempt_index, commands=command_results, verifier=verifier_result,
                    failure_class=None, retriable=None, started_at=attempt_started, ended_at=ended_at, duration_seconds=duration,
                ))
                return _finalize(final_state="PASS", passing_attempt_index=attempt_index)

            if artifact_violations:
                failure_class = "ARTIFACT_POLICY_FAILURE"
            elif any(r.timed_out for r in command_results):
                failure_class = "EXECUTION_TIMEOUT"
            elif any(r.returncode not in (0, None) for r in command_results):
                failure_class = "EXECUTION_NONZERO_EXIT"
            else:
                failure_class = "EXPECTED_CONTENT_MISMATCH"

            failure = build_failure_record(
                task_id=spec.task_id, run_id=resolved_run_id, attempt_index=attempt_index, failure_class=failure_class,
                summary="; ".join(verifier_result.issues) or "execution verification failed",
                evidence=verifier_result.evidence, failed_checks=verifier_result.issues,
                recommended_action=(
                    "diagnose and repair" if failure_class in ("EXECUTION_NONZERO_EXIT", "EXPECTED_CONTENT_MISMATCH") and _implementation_needed(spec)
                    else "manual review required"
                ),
            )
            run_paths.failure_path.write_text(failure.model_dump_json(indent=2) + "\n")

            execution_attempts.append(ExecutionAttempt(
                attempt_index=attempt_index, commands=command_results, verifier=verifier_result,
                failure_class=failure_class, retriable=failure.retriable,
                started_at=attempt_started, ended_at=ended_at, duration_seconds=duration,
            ))

            # ── one-time bounded timeout retry (separate, narrower mechanism
            # than the diagnose/repair loop below -- same command, remaining
            # budget only) ────────────────────────────────────────────────
            if (
                failure_class == "EXECUTION_TIMEOUT" and execution.retry_policy.allow_timeout_retry
                and repair_round == 0 and attempt_index + 1 < limits.max_execution_attempts
                and _elapsed() < limits.max_wall_clock_seconds
            ):
                attempt_index += 1
                continue

            if not failure.retriable:
                terminal = (
                    "POLICY_FAILURE" if failure_class == "ARTIFACT_POLICY_FAILURE"
                    else "EXECUTION_FAILED" if failure_class == "EXECUTION_TIMEOUT"
                    else "VERIFICATION_FAILED"
                )
                return _finalize(final_state=terminal, stage="verifier", reason=failure.summary)

            if not _implementation_needed(spec):
                terminal = "EXECUTION_FAILED" if failure_class == "EXECUTION_NONZERO_EXIT" else "VERIFICATION_FAILED"
                return _finalize(
                    final_state=terminal, stage="verifier",
                    reason=failure.summary + " (no repairable implementation scope configured for this spec)",
                )

            # ── bounded post-execution diagnose/repair round ───────────────
            if repair_round >= limits.max_repair_rounds:
                return _finalize(final_state="RETRY_EXHAUSTED", stage="repair_budget", reason=f"max_repair_rounds={limits.max_repair_rounds} exhausted")
            if codex_invocations >= limits.max_total_codex_invocations:
                return _finalize(final_state="RETRY_EXHAUSTED", stage="codex_budget", reason="max_total_codex_invocations exhausted")
            if claude_invocations >= limits.max_total_claude_invocations:
                return _finalize(final_state="RETRY_EXHAUSTED", stage="claude_budget", reason="max_total_claude_invocations exhausted")

            repair_round += 1
            repair_attempt_index = repair_round
            repair_attempt_dir = run_paths.attempt_dir(repair_attempt_index)
            repair_attempt_dir.mkdir(parents=True)

            _transition("DIAGNOSING", attempt_index=attempt_index)
            diag_prompt = _build_diagnosis_prompt(
                spec=spec, run_paths=run_paths, plan=plan, failure=failure, verifier_result=verifier_result,
                attempt_index=repair_attempt_index, remaining_repair_rounds=limits.max_repair_rounds - repair_round + 1,
            )
            before_diag_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label=f"before_diagnosis_{repair_attempt_index:02d}")
            try:
                diagnosis = codex_agent.diagnose(
                    prompt=diag_prompt, run_dir=run_paths.run_dir, cwd=repo_root,
                    timeout=spec.timeouts.planner_seconds, task_id=spec.task_id, run_id=resolved_run_id,
                    attempt_index=repair_attempt_index,
                )
                codex_invocations += 1
            except ExecutableNotFoundError as e:
                codex_invocations += 1
                return _finalize(final_state="INFRASTRUCTURE_FAILURE", stage="codex_diagnosis", reason=f"CODEX_EXECUTABLE_UNAVAILABLE: {e}")
            except (CommandTimeoutError, InfrastructureError) as e:
                codex_invocations += 1
                return _finalize(final_state="INFRASTRUCTURE_FAILURE", stage="codex_diagnosis", reason=f"CODEX_TIMEOUT: {e}")
            except CodexPlannerError as e:
                codex_invocations += 1
                fc = _CODEX_ERROR_TO_FAILURE_CLASS.get(e.code, "PLAN_SCHEMA_FAILURE")
                state = "INFRASTRUCTURE_FAILURE" if fc == "CODEX_INFRASTRUCTURE_FAILURE" else "BLOCKED"
                return _finalize(final_state=state, stage="codex_diagnosis", reason=str(e))

            after_diag_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label=f"after_diagnosis_{repair_attempt_index:02d}")
            diag_changes = repository_tasks.diff_fingerprints(before_diag_fp, after_diag_fp, run_dir=run_paths.run_dir, repo_root=repo_root)
            if diag_changes:
                main_worktree_unchanged_overall = False
                return _finalize(final_state="POLICY_FAILURE", stage="codex_diagnosis", reason="REPOSITORY_CHANGED_DURING_CODEX_DIAGNOSIS: " + "; ".join(diag_changes))

            reporting_tasks.save_json_artifact(run_paths.diagnosis_path(repair_attempt_index), diagnosis)
            reporting_tasks.save_json_artifact(repair_attempt_dir / "diagnosis.json", diagnosis)

            if diagnosis.verdict in ("DIAGNOSE_BLOCKED", "DIAGNOSE_NOT_REPRODUCIBLE"):
                return _finalize(final_state="BLOCKED", stage="codex_diagnosis", reason=diagnosis.root_cause)
            if diagnosis.verdict == "DIAGNOSE_POLICY_FAILURE":
                return _finalize(final_state="POLICY_FAILURE", stage="codex_diagnosis", reason=diagnosis.root_cause)
            if diagnosis.verdict == "DIAGNOSE_INFRASTRUCTURE_FAILURE":
                return _finalize(final_state="INFRASTRUCTURE_FAILURE", stage="codex_diagnosis", reason=diagnosis.root_cause)

            scope_violations = [p for p in diagnosis.files_allowed_to_touch if not execution_policy.is_modify_path_allowed(p, spec)]
            if scope_violations:
                return _finalize(final_state="POLICY_FAILURE", stage="diagnosis_scope_validation", reason=f"diagnosis proposed file(s) outside allowed scope: {scope_violations}")
            command_scope_violations: list[str] = []
            for cmd in diagnosis.commands_allowed_to_run:
                try:
                    execution_policy.assert_executor_command_allowed(cmd, spec)
                except execution_policy.ExecutionPolicyViolation as e:
                    command_scope_violations.append(str(e))
            if command_scope_violations:
                return _finalize(final_state="POLICY_FAILURE", stage="diagnosis_scope_validation", reason="; ".join(command_scope_violations))

            if claude_invocations >= limits.max_total_claude_invocations:
                return _finalize(final_state="RETRY_EXHAUSTED", stage="claude_budget", reason="max_total_claude_invocations exhausted before repair could run")

            _transition("REPAIRING", attempt_index=attempt_index)
            repair_prompt = _build_repair_prompt(
                spec=spec, run_paths=run_paths, diagnosis=diagnosis, verifier_result=verifier_result,
                attempt_index=repair_attempt_index, remaining_repair_rounds=limits.max_repair_rounds - repair_round + 1,
            )
            before_repair_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label=f"before_repair_{repair_attempt_index:02d}")
            before_repair_manifest = execution_worktree.build_ignored_and_untracked_manifest(worktree_dir, spec)

            repair_result = None
            repair_error: Optional[tuple[str, str]] = None
            try:
                repair_result = claude_agent.repair(
                    prompt=repair_prompt, worktree_dir=worktree_dir, run_paths=run_paths,
                    timeout=spec.timeouts.executor_seconds, task_id=spec.task_id, run_id=resolved_run_id,
                    attempt_index=repair_attempt_index,
                )
                claude_invocations += 1
                reporting_tasks.save_json_artifact(run_paths.repair_result_path(repair_attempt_index), repair_result)
                reporting_tasks.save_json_artifact(repair_attempt_dir / "repair.json", repair_result)
            except ExecutableNotFoundError as e:
                claude_invocations += 1
                repair_error = ("CLAUDE_INFRASTRUCTURE_FAILURE", f"CLAUDE_EXECUTABLE_UNAVAILABLE: {e}")
            except CommandTimeoutError as e:
                claude_invocations += 1
                repair_error = ("CLAUDE_INFRASTRUCTURE_FAILURE", f"CLAUDE_TIMEOUT: {e}")
            except ClaudeExecutorError as e:
                claude_invocations += 1
                repair_error = (_CLAUDE_ERROR_TO_FAILURE_CLASS.get(e.code, "IMPLEMENTATION_SCHEMA_FAILURE"), str(e))

            gate = _check_implementation_policy_gate(
                repo_root=repo_root, run_paths=run_paths, worktree_dir=worktree_dir, spec=spec,
                before_fp=before_repair_fp, before_manifest=before_repair_manifest, base_commit=worktree_record.base_commit,
                claimed_changed_files=(repair_result.changed_files if repair_result else None),
                claimed_commands=(repair_result.commands_run if repair_result else []),
                attempt_index=repair_attempt_index, name_prefix=f"attempt{repair_attempt_index:02d}_repair",
            )
            if not gate.main_worktree_unchanged:
                main_worktree_unchanged_overall = False

            repair_tasks.write_attempt_git_artifacts(worktree_dir, repair_attempt_dir, name_prefix=f"attempt{repair_attempt_index:02d}_repair")
            implementation_attempts.append(AttemptRecord(
                attempt_index=repair_attempt_index, kind="repair",
                verdict=(repair_result.verdict if repair_result else None), verifier_verdict=None,
                failure_class=gate.failure_class, retriable=False if gate.failure_class or repair_error else None,
                changed_file_count=len(gate.changed_records),
                changed_byte_count=execution_worktree.total_changed_bytes(gate.changed_records),
                started_at=reporting_tasks.utcnow_iso(), ended_at=reporting_tasks.utcnow_iso(), duration_seconds=0.0,
            ))

            if gate.failure_class:
                return _finalize(final_state=terminal_state_for(gate.failure_class), stage="repair_policy_gate", reason=gate.reason)
            if repair_error:
                fc, reason = repair_error
                return _finalize(final_state=terminal_state_for(fc), stage="claude_repair", reason=reason)
            if repair_result.verdict == "REPAIR_BLOCKED":
                return _finalize(final_state="BLOCKED", stage="claude_repair", reason=repair_result.summary)

            impl_verifier2, breakdown2, _static2 = run_content_verifier(
                spec=spec, worktree_dir=worktree_dir, attempt_dir=repair_attempt_dir,
                task_id=spec.task_id, run_id=resolved_run_id, attempt_index=repair_attempt_index,
            )
            reporting_tasks.save_json_artifact(repair_attempt_dir / "verifier.json", impl_verifier2)
            implementation_attempts[-1] = implementation_attempts[-1].model_copy(update={"verifier_verdict": impl_verifier2.verdict})
            if impl_verifier2.verdict != "PASS":
                fc2 = primary_failure_class(breakdown2)
                if fc2 in ("TEST_FAILURE", "STATIC_CHECK_FAILURE", "ARTIFACT_MISSING", "ARTIFACT_MALFORMED", "EXPECTED_CONTENT_MISMATCH"):
                    continue  # loop again: another diagnose/repair round, budget permitting
                return _finalize(final_state=terminal_state_for(fc2), stage="implementation_reverifier", reason="; ".join(impl_verifier2.details))

            # repaired implementation verifies -- re-run the SAME approved command(s)
            _transition("RETRYING_EXECUTION", attempt_index=attempt_index)
            attempt_index += 1

    except Exception as exc:  # noqa: BLE001 -- safety net: state.json must never stay active
        return _finalize(
            final_state="INFRASTRUCTURE_FAILURE", stage="unexpected_exception",
            reason=f"{type(exc).__name__}: {exc}", manual_action_required=True,
        )
