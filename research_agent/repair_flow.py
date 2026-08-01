"""MVP3 repair flow: bounded diagnose/repair/reverify loop.

Pipeline:

    experiment YAML
    -> validate specification
    -> Codex read-only planner (mock by default, real with --codex real)
    -> plan policy validation
    -> single shared, isolated execution worktree (reused for every round --
       never re-created per repair attempt, see research_agent.execution_worktree)
    -> attempt 0: real (or mock) Claude executor -- IMPLEMENTING
    -> deterministic policy gate (path/symlink/nested-git/command/main-
       worktree/`.git`-tampering/diff-claim -- always non-retriable) --
       VERIFYING
    -> deterministic content verifier (expected_file_contents/
       required_artifacts/static checks -- always retriable)
    -> on FAIL, retriable, budget remaining: bounded loop of
       DIAGNOSING (Codex, read-only, advisory, scope-restricted) ->
       REPAIRING (Claude, SAME shared worktree) ->
       policy gate -> REVERIFYING (content verifier again)
    -> terminal state: PASS, BLOCKED, RETRY_EXHAUSTED, INFRASTRUCTURE_FAILURE,
       or POLICY_FAILURE -- state.json is NEVER left in a non-terminal state
       when this function returns, including after an unexpected internal
       exception (see the outer try/except at the bottom of repair_flow).

Never runs a research experiment, training, GPU, robot, Docker, sudo,
package install, network download, commit, push, merge, or confirmatory
command -- see research_agent.policies.execution_policy (the same
deterministic gate execute_flow.py uses) and the explicit prohibitions
embedded in every prompt this module builds.
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
    DiagnosisResult,
    ExecutionWorktreeRecord,
    ExecutorImplementationResult,
    ExperimentSpec,
    PlanResult,
    RepairFinalReport,
    RepairLimits,
    RepairResult,
    RepairRunPaths,
    VerifierResult,
)
from research_agent.policies import commands as command_policy
from research_agent.policies import execution_policy
from research_agent.subprocess_runner import CommandTimeoutError, ExecutableNotFoundError, InfrastructureError
from research_agent.tasks import experiment as experiment_tasks
from research_agent.tasks import reporting as reporting_tasks
from research_agent.tasks import repair as repair_tasks
from research_agent.tasks import repository as repository_tasks
from research_agent.tasks.repair_verification import primary_failure_class, run_content_verifier

# ── error-code -> failure_class maps (see research_agent.models.FailureClass
# and failure_taxonomy.py; both target maps are entirely non-retriable) ────

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

_DIFF_TRUNCATE_CHARS = 20_000


# ── prompt builders (plain text; no LLM call happens here) ─────────────────

def _build_repair_planner_prompt(spec: ExperimentSpec, *, run_paths: RepairRunPaths, repo_root: Path) -> str:
    modify_paths = execution_policy.effective_modify_paths(spec)
    forbidden = list(spec.forbidden_paths) + list(execution_policy.HIGH_RISK_PREFIXES) + list(execution_policy.HIGH_RISK_EXACT)
    return "\n".join([
        "You are a READ-ONLY planning assistant for the TANGO Experiment Agent's MVP3 bounded repair flow.",
        "",
        "MODE: planning-only, non-confirmatory, read-only. If your PLAN_PASS verdict is accepted, a",
        "separate, disposable Git worktree will be created and a Claude implementer will attempt the",
        "change; if the deterministic verifier finds it incomplete or wrong, a BOUNDED number of",
        "diagnose/repair rounds may follow -- you never run any of that yourself, and no research",
        "experiment is ever run in this workflow.",
        "",
        "SAFETY RESTRICTIONS (mandatory, no exceptions):",
        "- This step is planning only. Do not modify any file in the repository.",
        "- Do not propose any command outside the approved command array(s) listed below.",
        "- Do not invoke Claude, claude-code, or any other coding/execution agent.",
        "- Do not run, or propose running, anything involving a GPU, CUDA, a physical robot, model",
        "  training, Docker, sudo, `git push`, `git commit`, network downloads, or a confirmatory or",
        "  otherwise research/experiment command of any kind.",
        "- Output only the single requested structured plan object; no other text.",
        "",
        f"task_id: {spec.task_id}",
        f"run_id: {run_paths.run_id}",
        f"repository_root: {repo_root}",
        f"goal: {spec.goal}",
        "",
        f"allowed_modify_paths: {modify_paths}",
        f"forbidden_paths (non-exhaustive): {sorted(set(forbidden))}",
        f"expected_file_contents (exact final content required): {spec.expected_file_contents}",
        f"required_artifacts: {spec.required_artifacts}",
        f"repair_limits: {spec.repair_limits.model_dump()}",
        "",
        f"max_proposed_commands: {command_policy.MAX_PROPOSED_COMMANDS}",
        "",
        "Return exactly one JSON object matching the supplied PlanResult schema, with fields: "
        "schema_version, task_id, run_id, verdict, summary, issues, artifacts, proposed_commands, "
        "expected_artifacts, expected_metrics, risks, assumptions.",
        f"task_id must be exactly {spec.task_id!r}; run_id must be exactly {run_paths.run_id!r}.",
        "verdict must be exactly one of PLAN_PASS, PLAN_REVISE, PLAN_BLOCKED.",
        "proposed_commands must be empty ([]) unless it exactly matches an approved command array.",
    ])


def _read_text_truncated(path: Path, *, limit: int = _DIFF_TRUNCATE_CHARS) -> str:
    if not path.exists():
        return "(no diff available)"
    text = path.read_text()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def _build_diagnosis_prompt(
    *,
    spec: ExperimentSpec,
    run_paths: RepairRunPaths,
    plan: PlanResult,
    prior_result_summary: str,
    failure,
    verifier_result: VerifierResult,
    attempt_dir: Path,
    prior_attempts: list[AttemptRecord],
    attempt_index: int,
    remaining_repair_rounds: int,
    remaining_claude_invocations: int,
    remaining_codex_invocations: int,
    remaining_wall_clock_seconds: float,
) -> str:
    modify_paths = execution_policy.effective_modify_paths(spec)
    forbidden = list(spec.forbidden_paths) + list(execution_policy.HIGH_RISK_PREFIXES) + list(execution_policy.HIGH_RISK_EXACT)
    allowed_commands = list(spec.allowed_executor_commands) + [
        ["python", "-m", "compileall", "..."], ["python", "-m", "pytest", "<specific test path>"],
        ["git", "status", "--short"], ["git", "diff", "--check"], ["git", "diff", "--name-status"],
    ]
    diff_text = _read_text_truncated(attempt_dir / "git_diff.patch")
    prior_summary = [
        f"attempt {a.attempt_index} ({a.kind}): verdict={a.verdict} verifier={a.verifier_verdict} "
        f"failure_class={a.failure_class} retriable={a.retriable}"
        for a in prior_attempts
    ]
    return "\n".join([
        "You are a READ-ONLY diagnosis assistant for the TANGO Experiment Agent's MVP3 bounded repair flow.",
        "This is a DIFFERENT ROLE from the original planner: you diagnose why the deterministic verifier",
        "found the current implementation incomplete or wrong, and propose a SCOPE-RESTRICTED repair plan",
        "for a separate Claude repair step to follow. You never write any file yourself.",
        "",
        "MODE: planning-only, non-confirmatory, read-only, advisory only. Your output can only NARROW the",
        "original experiment specification's allowed scope -- it can never broaden it. Any file or command",
        "you propose outside the original policy will be rejected and will BLOCK the repair round entirely.",
        "",
        "SAFETY RESTRICTIONS (mandatory, no exceptions):",
        "- Do not modify any file. Do not run any command yourself.",
        "- Do not invoke Claude, claude-code, or any other coding/execution agent.",
        "- Do not propose anything involving a GPU, CUDA, a physical robot, model training, Docker, sudo,",
        "  `git push`, `git commit`, `git reset`, `git clean`, `git worktree`, network downloads, or a",
        "  confirmatory/research-experiment command of any kind.",
        "",
        f"task_id: {spec.task_id}",
        f"run_id: {run_paths.run_id}",
        f"attempt_index (this diagnosis informs repair attempt {attempt_index}): {attempt_index}",
        f"goal: {spec.goal}",
        "",
        "ORIGINAL EXPERIMENT SPECIFICATION:",
        f"  allowed_modify_paths: {modify_paths}",
        f"  forbidden_paths (non-exhaustive): {sorted(set(forbidden))}",
        f"  expected_file_contents: {spec.expected_file_contents}",
        f"  required_artifacts: {spec.required_artifacts}",
        f"  allowed commands (the ONLY commands you may list in commands_allowed_to_run): {allowed_commands}",
        "",
        f"VALIDATED PLAN: verdict={plan.verdict} summary={plan.summary!r}",
        f"CURRENT IMPLEMENTATION/REPAIR RESULT: {prior_result_summary}",
        "",
        f"EXACT VERIFIER FAILURE: verdict={verifier_result.verdict} "
        f"checks_failed={verifier_result.checks_failed} details={verifier_result.details}",
        f"STRUCTURED FAILURE RECORD: {failure.model_dump()}",
        "",
        "ACTUAL GIT DIFF in the shared execution worktree (authoritative -- trust this, not any agent claim):",
        diff_text,
        "",
        "PRIOR ATTEMPTS (this run):",
        *(prior_summary or ["  (none -- this is the first repair round)"]),
        "",
        "REMAINING RETRY BUDGET:",
        f"  remaining_repair_rounds: {remaining_repair_rounds}",
        f"  remaining_claude_invocations: {remaining_claude_invocations}",
        f"  remaining_codex_invocations: {remaining_codex_invocations}",
        f"  remaining_wall_clock_seconds: {remaining_wall_clock_seconds:.0f}",
        "",
        "PROHIBITED ACTIONS (repeated for emphasis): commit, push, merge, modify main worktree, alter",
        "worktree/Git metadata, create a nested Git repository, escape the worktree via a symlink, GPU,",
        "robot, Docker, sudo, package install, network download, or any research/confirmatory experiment.",
        "",
        "Return exactly one JSON object matching the supplied DiagnosisResult schema, with fields: "
        "schema_version, task_id, run_id, attempt_index, verdict, failure_class, root_cause, evidence, "
        "repair_instructions, files_allowed_to_touch, commands_allowed_to_run, risks, assumptions.",
        f"task_id must be exactly {spec.task_id!r}; run_id must be exactly {run_paths.run_id!r}; "
        f"attempt_index must be exactly {attempt_index}.",
        "verdict must be exactly one of DIAGNOSE_REPAIRABLE, DIAGNOSE_BLOCKED, DIAGNOSE_NOT_REPRODUCIBLE, "
        "DIAGNOSE_POLICY_FAILURE, DIAGNOSE_INFRASTRUCTURE_FAILURE.",
        "files_allowed_to_touch must be a subset of allowed_modify_paths above (empty [] means 'no further "
        "narrowing beyond the original scope', NOT 'nothing allowed'). commands_allowed_to_run must each "
        "exactly match one of the allowed commands above (empty [] if none are needed).",
    ])


def _build_repair_prompt(
    *,
    spec: ExperimentSpec,
    run_paths: RepairRunPaths,
    diagnosis: DiagnosisResult,
    prior_result_summary: str,
    verifier_result: VerifierResult,
    attempt_dir: Path,
    attempt_index: int,
    remaining_repair_rounds: int,
    remaining_claude_invocations: int,
) -> str:
    modify_paths = execution_policy.effective_modify_paths(spec)
    forbidden = list(spec.forbidden_paths) + list(execution_policy.HIGH_RISK_PREFIXES) + list(execution_policy.HIGH_RISK_EXACT)
    schema = RepairResult.model_json_schema()
    diff_text = _read_text_truncated(attempt_dir / "git_diff.patch")
    effective_files = diagnosis.files_allowed_to_touch or modify_paths
    return "\n".join([
        "You are the MVP3 REPAIR role: a source-code implementation assistant confined to the SAME",
        "disposable Git worktree used for the initial implementation. You perform the SMALLEST NECESSARY",
        "REPAIR to satisfy the failed checks listed below -- nothing more.",
        "",
        f"task_id: {spec.task_id}",
        f"run_id: {run_paths.run_id}",
        f"attempt_index: {attempt_index}",
        "",
        f"EXACT VERIFIER FAILURE: verdict={verifier_result.verdict} "
        f"checks_failed={verifier_result.checks_failed} details={verifier_result.details}",
        "",
        "STRUCTURED CODEX DIAGNOSIS (advisory; you must still independently follow every rule below):",
        f"  verdict: {diagnosis.verdict}",
        f"  failure_class: {diagnosis.failure_class}",
        f"  root_cause: {diagnosis.root_cause}",
        f"  repair_instructions: {diagnosis.repair_instructions}",
        f"  files_allowed_to_touch: {diagnosis.files_allowed_to_touch}",
        f"  commands_allowed_to_run: {diagnosis.commands_allowed_to_run}",
        "",
        f"PREVIOUS IMPLEMENTATION/REPAIR REPORT: {prior_result_summary}",
        "",
        "ACTUAL CURRENT GIT DIFF in this worktree (authoritative):",
        diff_text,
        "",
        f"ALLOWED MODIFICATION PATHS for this repair (already narrowed by diagnosis; never write outside "
        f"this list): {effective_files}",
        f"FORBIDDEN PATHS (non-exhaustive, checked independently after you exit): {sorted(set(forbidden))}",
        f"EXACT TARGET CHECKS THAT MUST PASS: {verifier_result.checks_failed}",
        "",
        "REMAINING RETRY BUDGET:",
        f"  remaining_repair_rounds (including this one): {remaining_repair_rounds}",
        f"  remaining_claude_invocations (including this one): {remaining_claude_invocations}",
        "",
        "STRICT RULES (mandatory, no exceptions):",
        "- Make the SMALLEST necessary repair. Do not revert unrelated valid changes. Do not broaden scope",
        "  beyond files_allowed_to_touch above.",
        "- Do not run `git commit`. Do not run `git push`. Leave all changes uncommitted.",
        "- Do not create or remove any Git worktree; do not alter worktree or `.git` metadata in any way.",
        "- Do not create a nested Git repository.",
        "- Do not modify the main repository worktree (any path outside this worktree) under any circumstance.",
        "- Do not invoke Codex, `codex`, or any other planning/execution agent.",
        "- Do not invoke another Claude process, `claude`, or spawn any subprocess that itself launches an LLM CLI.",
        "- Do not run any command outside commands_allowed_to_run above.",
        "- Do not use sudo, Docker, GPU/CUDA, robot hardware, or the network. Do not run a research experiment.",
        "- If the diagnosis cannot be safely applied within these constraints, STOP and report verdict=",
        "  REPAIR_BLOCKED with a clear `summary` and `issues` entries -- never bypass a policy to make progress.",
        "",
        "REQUIRED STRUCTURED FINAL RESPONSE:",
        "Return exactly one JSON object matching this JSON Schema (no other text before or after it):",
        json.dumps(schema, indent=2),
        f"task_id must be exactly {spec.task_id!r}; run_id must be exactly {run_paths.run_id!r}; "
        f"attempt_index must be exactly {attempt_index}.",
        "verdict must be exactly one of REPAIR_PASS, REPAIR_REVISE, REPAIR_BLOCKED.",
        "changed_files must list every repo-relative path you actually created or modified (the FULL",
        "current diff, including anything already changed by the initial implementation or a prior repair).",
        "commands_run must be a list of argv arrays -- never a shell string -- empty ([]) if you ran none.",
    ])


# ── policy gate: the non-retriable checks, shared by attempt 0 and every
# repair round -- exactly mirrors execute_flow.py's structure ─────────────

class _PolicyGateResult(NamedTuple):
    failure_class: Optional[str]
    reason: Optional[str]
    evidence: list[str]
    changed_records: list[ChangedFileRecord]
    main_worktree_unchanged: bool
    ignored_manifest_ok: bool
    after_fp: dict
    after_manifest: list


def _check_policy_gate(
    *,
    repo_root: Path,
    run_paths: RepairRunPaths,
    worktree_dir: Path,
    spec: ExperimentSpec,
    repair_limits: RepairLimits,
    before_fp: dict,
    before_manifest: list,
    base_commit: str,
    claimed_changed_files: Optional[list[str]],
    claimed_commands: list[list[str]],
    attempt_index: int,
    name_prefix: str,
) -> _PolicyGateResult:
    """`claimed_changed_files=None` means "no claim available" (the agent
    call raised before ever returning a result) -- the diff-claim-vs-actual
    check is skipped in that case, since comparing an absent claim against
    the worktree's cumulative diff (which may already carry changes from an
    EARLIER attempt in this same shared worktree) would otherwise produce a
    spurious DIFF_MISMATCH. Pass `[]` (not None) when the agent DID return a
    result claiming zero changed files."""
    after_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label=f"after_{name_prefix}")
    main_changes = repository_tasks.diff_fingerprints(before_fp, after_fp, run_dir=run_paths.run_dir, repo_root=repo_root)
    after_manifest = execution_worktree.build_ignored_and_untracked_manifest(worktree_dir, spec)
    if main_changes:
        return _PolicyGateResult(
            "MAIN_WORKTREE_CHANGED", "; ".join(main_changes), main_changes, [], False, True, after_fp, after_manifest,
        )

    manifest_changes = execution_worktree.diff_manifests(before_manifest, after_manifest)
    command_results_dir = run_paths.attempt_command_results_dir(attempt_index)
    status_entries = repair_tasks.collect_worktree_status(
        worktree_dir, command_results_dir=command_results_dir, name_prefix=name_prefix,
    )
    changed_paths = [p for _, p in status_entries]
    changed_records = execution_worktree.build_changed_file_records(worktree_dir, status_entries)

    changed_path_set = {p.rstrip("/") for p in changed_paths}
    unexplained = [c for c in manifest_changes if not any(c.startswith(p) for p in changed_path_set)]
    if unexplained:
        if any(c.startswith(".git:") for c in unexplained):
            return _PolicyGateResult(
                "GIT_METADATA_TAMPERING", "; ".join(unexplained), unexplained, changed_records, True, False, after_fp, after_manifest,
            )
        return _PolicyGateResult(
            "DIFF_MISMATCH", "undeclared ignored/untracked-file mutation detected: " + "; ".join(unexplained),
            unexplained, changed_records, True, False, after_fp, after_manifest,
        )

    head = repair_tasks.worktree_head(worktree_dir, command_results_dir=command_results_dir, name_prefix=name_prefix)
    if head != base_commit:
        return _PolicyGateResult(
            "GIT_METADATA_TAMPERING",
            f"execution worktree HEAD moved from base_commit={base_commit} to {head} -- a commit was made",
            [], changed_records, True, True, after_fp, after_manifest,
        )

    try:
        execution_policy.validate_changed_paths(worktree_dir, changed_paths, spec)
    except execution_policy.ExecutionPolicyViolation as e:
        fc = _EXECUTION_POLICY_CODE_TO_FAILURE_CLASS.get(e.code, "PATH_POLICY_FAILURE")
        return _PolicyGateResult(fc, e.message, [e.message], changed_records, True, True, after_fp, after_manifest)

    byte_count = execution_worktree.total_changed_bytes(changed_records)
    try:
        execution_policy.assert_change_limits(file_count=len(changed_records), byte_count=byte_count, spec=spec)
    except execution_policy.ExecutionPolicyViolation as e:
        return _PolicyGateResult("PATH_POLICY_FAILURE", e.message, [e.message], changed_records, True, True, after_fp, after_manifest)

    if len(changed_records) > repair_limits.max_total_changed_files:
        return _PolicyGateResult(
            "PATH_POLICY_FAILURE",
            f"{len(changed_records)} changed files exceeds repair_limits.max_total_changed_files="
            f"{repair_limits.max_total_changed_files}",
            [], changed_records, True, True, after_fp, after_manifest,
        )
    if byte_count > repair_limits.max_total_changed_bytes:
        return _PolicyGateResult(
            "PATH_POLICY_FAILURE",
            f"{byte_count} changed bytes exceeds repair_limits.max_total_changed_bytes="
            f"{repair_limits.max_total_changed_bytes}",
            [], changed_records, True, True, after_fp, after_manifest,
        )

    command_violations: list[str] = []
    for cmd in claimed_commands:
        try:
            execution_policy.assert_executor_command_allowed(cmd, spec)
        except execution_policy.ExecutionPolicyViolation as e:
            command_violations.append(str(e))
    if command_violations:
        return _PolicyGateResult(
            "COMMAND_POLICY_FAILURE", "; ".join(command_violations), command_violations, changed_records, True, True, after_fp, after_manifest,
        )

    claimed_set = None if claimed_changed_files is None else {Path(p).as_posix() for p in claimed_changed_files}
    actual_set = {Path(p.rstrip("/")).as_posix() for p in changed_paths}
    if claimed_set is not None and claimed_set != actual_set:
        return _PolicyGateResult(
            "DIFF_MISMATCH", f"claimed changed_files={sorted(claimed_set)} != actual git diff={sorted(actual_set)}",
            [f"claimed={sorted(claimed_set)}", f"actual={sorted(actual_set)}"], changed_records, True, True, after_fp, after_manifest,
        )

    return _PolicyGateResult(None, None, [], changed_records, True, True, after_fp, after_manifest)


def _close_attempt(
    *,
    attempt_dir: Path,
    worktree_dir: Path,
    name_prefix: str,
    gate: _PolicyGateResult,
    before_fp: dict,
    kind: str,
    attempt_index: int,
    started_at: str,
    t0: float,
    agent_kind: str,
    verdict: Optional[str],
    verifier_verdict: Optional[str],
    failure_class: Optional[str],
    retriable: Optional[bool],
) -> AttemptRecord:
    """Writes every required per-attempt artifact (git_status.txt,
    git_diff.patch, git_diff_name_status.txt, changed_file_manifest.json,
    main_repo_before.json, main_repo_after.json, attempt_meta.json) and
    returns the immutable AttemptRecord summary for the final report. Called
    exactly once per attempt, at the point its outcome is known -- the
    attempt directory (and every file inside it) is never touched again
    after this returns."""
    ended_at = reporting_tasks.utcnow_iso()
    duration = round(time.monotonic() - t0, 3)
    repair_tasks.write_attempt_git_artifacts(worktree_dir, attempt_dir, name_prefix=name_prefix)
    (attempt_dir / "changed_file_manifest.json").write_text(
        json.dumps([r.model_dump() for r in gate.changed_records], indent=2) + "\n"
    )
    (attempt_dir / "main_repo_before.json").write_text(json.dumps(before_fp, indent=2) + "\n")
    (attempt_dir / "main_repo_after.json").write_text(json.dumps(gate.after_fp, indent=2) + "\n")
    (attempt_dir / "attempt_meta.json").write_text(json.dumps({
        "attempt_index": attempt_index, "kind": kind, "started_at": started_at, "ended_at": ended_at,
        "duration_seconds": duration, "agent_kind": agent_kind, "policy_verdict": gate.failure_class or "OK",
        "main_worktree_unchanged": gate.main_worktree_unchanged, "ignored_file_manifest_ok": gate.ignored_manifest_ok,
    }, indent=2) + "\n")
    byte_count = execution_worktree.total_changed_bytes(gate.changed_records)
    return AttemptRecord(
        attempt_index=attempt_index, kind=kind, verdict=verdict, verifier_verdict=verifier_verdict,
        failure_class=failure_class, retriable=retriable, changed_file_count=len(gate.changed_records),
        changed_byte_count=byte_count, started_at=started_at, ended_at=ended_at, duration_seconds=duration,
    )


# ── main flow ────────────────────────────────────────────────────────────

def repair_flow(
    spec_path: Path,
    *,
    repo_root: Path,
    runs_root: Path,
    run_id: Optional[str] = None,
    execution_worktrees_root: Optional[Path] = None,
    codex_agent: Optional[CodexAgent] = None,
    claude_agent: Optional[ClaudeExecutorAgent] = None,
    repair_limits: Optional[RepairLimits] = None,
) -> RepairFinalReport:
    codex_agent = codex_agent or MockCodexAgent()
    claude_agent = claude_agent or MockClaudeExecutorAgent()
    repo_root = Path(repo_root).resolve()
    runs_root = Path(runs_root).resolve()
    spec_path = Path(spec_path).resolve()

    spec = experiment_tasks.load_spec(spec_path)
    limits = repair_limits or spec.repair_limits
    experiment_tasks.assert_run_count_within_limit(runs_root, spec.task_id, spec.max_run_count)
    resolved_run_id = run_id or experiment_tasks.generate_run_id(spec.task_id)
    run_paths = repair_tasks.init_repair_run(runs_root=runs_root, run_id=resolved_run_id, spec_source_path=spec_path)

    start_time = time.monotonic()
    history: list[str] = []
    attempts: list[AttemptRecord] = []
    codex_invocations = 0
    claude_invocations = 0
    plan: Optional[PlanResult] = None
    worktree_record: Optional[ExecutionWorktreeRecord] = None
    main_worktree_unchanged_overall = True

    def _elapsed() -> float:
        return time.monotonic() - start_time

    def _transition(state: str, *, attempt_index: int = 0, detail: Optional[str] = None) -> None:
        nonlocal history
        record = repair_tasks.persist_state(
            run_paths, run_id=resolved_run_id, task_id=spec.task_id, state=state,
            attempt_index=attempt_index, history=history, detail=detail,
        )
        history = record.history

    def _finalize(
        *, final_state: str, stage: Optional[str] = None, reason: Optional[str] = None,
        passing_attempt_index: Optional[int] = None, manual_action_required: bool = False,
    ) -> RepairFinalReport:
        last_attempt_index = attempts[-1].attempt_index if attempts else 0
        _transition(final_state, attempt_index=last_attempt_index, detail=reason)
        overall = "PASS" if final_state == "PASS" else ("FAIL" if final_state == "RETRY_EXHAUSTED" else "BLOCKED")
        report = RepairFinalReport(
            run_id=resolved_run_id, task_id=spec.task_id, overall_status=overall, final_state=final_state,
            stage=stage, reason=reason, passing_attempt_index=passing_attempt_index, attempts=attempts,
            plan=plan, execution_worktree=worktree_record, main_worktree_unchanged=main_worktree_unchanged_overall,
            total_claude_invocations=claude_invocations, total_codex_invocations=codex_invocations,
            changed_file_count=(attempts[-1].changed_file_count if attempts else 0),
            changed_byte_count=(attempts[-1].changed_byte_count if attempts else 0),
            wall_clock_seconds=round(_elapsed(), 3), repair_limits=limits,
            manual_action_required=manual_action_required or final_state != "PASS",
            run_dir=str(run_paths.run_dir), created_at=reporting_tasks.utcnow_iso(),
        )
        reporting_tasks.save_json_artifact(run_paths.final_report_path, report)
        return report

    try:
        # ── PLANNING ─────────────────────────────────────────────────────
        _transition("PLANNING")
        plan_prompt = _build_repair_planner_prompt(spec, run_paths=run_paths, repo_root=repo_root)
        before_plan_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="before_repair_planning")
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
        after_plan_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="after_repair_planning")
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
        for cmd in plan.proposed_commands:
            try:
                execution_policy.assert_executor_command_allowed(cmd, spec)
            except execution_policy.ExecutionPolicyViolation as e:
                plan_violations.append(str(e))
        if plan_violations:
            return _finalize(final_state="POLICY_FAILURE", stage="plan_policy_validation", reason="; ".join(plan_violations))

        _transition("PLAN_VALIDATED")

        # ── shared execution worktree (created once, reused every round) ──
        try:
            worktree_record = execution_worktree.create_execution_worktree(
                repo_root, run_paths, resolved_run_id, execution_worktrees_root=execution_worktrees_root,
            )
        except ExecutionWorktreeError as e:
            return _finalize(final_state="POLICY_FAILURE", stage="create_execution_worktree", reason=e.message)
        worktree_dir = Path(worktree_record.worktree_path)

        # ── attempt 0: initial implementation ──────────────────────────
        attempt_index = 0
        attempt_dir = run_paths.attempt_dir(0)
        attempt_dir.mkdir(parents=True)
        started_at = reporting_tasks.utcnow_iso()
        t0 = time.monotonic()
        agent_kind = "mock" if isinstance(claude_agent, MockClaudeExecutorAgent) else "real"

        if claude_invocations >= limits.max_total_claude_invocations:
            return _finalize(
                final_state="RETRY_EXHAUSTED", stage="claude_budget",
                reason="max_total_claude_invocations exhausted before attempt 0 could run",
            )

        _transition("IMPLEMENTING", attempt_index=0)
        implementation_prompt = build_executor_prompt(spec=spec, plan=plan, worktree_record=worktree_record, run_paths=run_paths)
        before_impl_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label="before_attempt00_implementation")
        before_impl_manifest = execution_worktree.build_ignored_and_untracked_manifest(worktree_dir, spec)

        implementation: Optional[ExecutorImplementationResult] = None
        claude_error: Optional[tuple[str, str]] = None
        try:
            implementation = claude_agent.execute(
                prompt=implementation_prompt, worktree_dir=worktree_dir, run_paths=run_paths,
                timeout=spec.timeouts.executor_seconds, task_id=spec.task_id, run_id=resolved_run_id,
            )
            claude_invocations += 1
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

        gate = _check_policy_gate(
            repo_root=repo_root, run_paths=run_paths, worktree_dir=worktree_dir, spec=spec, repair_limits=limits,
            before_fp=before_impl_fp, before_manifest=before_impl_manifest, base_commit=worktree_record.base_commit,
            claimed_changed_files=(implementation.changed_files if implementation else None),
            claimed_commands=(implementation.commands_run if implementation else []),
            attempt_index=0, name_prefix="attempt00_implementation",
        )
        if not gate.main_worktree_unchanged:
            main_worktree_unchanged_overall = False

        if gate.failure_class:
            failure = build_failure_record(
                task_id=spec.task_id, run_id=resolved_run_id, attempt_index=0, failure_class=gate.failure_class,
                summary=gate.reason or gate.failure_class, evidence=gate.evidence, failed_checks=["policy_gate"],
                recommended_action="manual review required -- non-retriable policy violation",
            )
            reporting_tasks.save_json_artifact(attempt_dir / "failure.json", failure)
            record = _close_attempt(
                attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix="attempt00_implementation", gate=gate,
                before_fp=before_impl_fp, kind="implementation", attempt_index=0, started_at=started_at, t0=t0,
                agent_kind=agent_kind, verdict=(implementation.verdict if implementation else None),
                verifier_verdict=None, failure_class=gate.failure_class, retriable=False,
            )
            attempts.append(record)
            return _finalize(final_state=terminal_state_for(gate.failure_class), stage="policy_gate", reason=gate.reason)

        if claude_error:
            failure_class, reason = claude_error
            failure = build_failure_record(
                task_id=spec.task_id, run_id=resolved_run_id, attempt_index=0, failure_class=failure_class,
                summary=reason, evidence=[reason], failed_checks=["claude_executor"],
                recommended_action="manual review required -- non-retriable agent-infrastructure failure",
            )
            reporting_tasks.save_json_artifact(attempt_dir / "failure.json", failure)
            record = _close_attempt(
                attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix="attempt00_implementation", gate=gate,
                before_fp=before_impl_fp, kind="implementation", attempt_index=0, started_at=started_at, t0=t0,
                agent_kind=agent_kind, verdict=None, verifier_verdict=None, failure_class=failure_class, retriable=False,
            )
            attempts.append(record)
            return _finalize(final_state=terminal_state_for(failure_class), stage="claude_executor", reason=reason)

        assert implementation is not None
        if implementation.verdict == "IMPLEMENTATION_BLOCKED":
            record = _close_attempt(
                attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix="attempt00_implementation", gate=gate,
                before_fp=before_impl_fp, kind="implementation", attempt_index=0, started_at=started_at, t0=t0,
                agent_kind=agent_kind, verdict=implementation.verdict, verifier_verdict=None,
                failure_class=None, retriable=False,
            )
            attempts.append(record)
            return _finalize(final_state="BLOCKED", stage="claude_executor", reason=implementation.summary)

        # ── content verifier (attempt 0) ───────────────────────────────
        _transition("VERIFYING", attempt_index=0)
        verifier_result, breakdown, _static = run_content_verifier(
            spec=spec, worktree_dir=worktree_dir, attempt_dir=attempt_dir,
            task_id=spec.task_id, run_id=resolved_run_id, attempt_index=0,
        )
        reporting_tasks.save_json_artifact(attempt_dir / "verifier.json", verifier_result)

        if verifier_result.verdict == "PASS":
            record = _close_attempt(
                attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix="attempt00_implementation", gate=gate,
                before_fp=before_impl_fp, kind="implementation", attempt_index=0, started_at=started_at, t0=t0,
                agent_kind=agent_kind, verdict=implementation.verdict, verifier_verdict="PASS",
                failure_class=None, retriable=None,
            )
            attempts.append(record)
            return _finalize(final_state="PASS", passing_attempt_index=0)

        fc0 = primary_failure_class(breakdown)
        failure = build_failure_record(
            task_id=spec.task_id, run_id=resolved_run_id, attempt_index=0, failure_class=fc0,
            summary="; ".join(verifier_result.details) or "content verifier failed",
            evidence=verifier_result.checks_failed, failed_checks=verifier_result.checks_failed,
            recommended_action="diagnose and repair" if fc0 in ("TEST_FAILURE", "STATIC_CHECK_FAILURE", "ARTIFACT_MISSING", "ARTIFACT_MALFORMED", "EXPECTED_CONTENT_MISMATCH") else "manual review required",
        )
        reporting_tasks.save_json_artifact(attempt_dir / "failure.json", failure)
        record = _close_attempt(
            attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix="attempt00_implementation", gate=gate,
            before_fp=before_impl_fp, kind="implementation", attempt_index=0, started_at=started_at, t0=t0,
            agent_kind=agent_kind, verdict=implementation.verdict, verifier_verdict="FAIL",
            failure_class=fc0, retriable=failure.retriable,
        )
        attempts.append(record)

        if not failure.retriable:
            return _finalize(final_state=terminal_state_for(fc0), stage="verifier", reason=failure.summary)

        implementation_summary = implementation.summary
        current_failure = failure
        current_verifier = verifier_result
        current_attempt_dir = attempt_dir

        # ── bounded diagnose/repair/reverify loop ──────────────────────
        round_ = 1
        while True:
            if round_ > limits.max_repair_rounds:
                return _finalize(
                    final_state="RETRY_EXHAUSTED", stage="repair_budget",
                    reason=f"max_repair_rounds={limits.max_repair_rounds} exhausted while failure_class="
                    f"{current_failure.failure_class} was still retriable",
                )
            if codex_invocations >= limits.max_total_codex_invocations:
                return _finalize(final_state="RETRY_EXHAUSTED", stage="codex_budget", reason="max_total_codex_invocations exhausted")
            if claude_invocations >= limits.max_total_claude_invocations:
                return _finalize(final_state="RETRY_EXHAUSTED", stage="claude_budget", reason="max_total_claude_invocations exhausted")
            if _elapsed() > limits.max_wall_clock_seconds:
                return _finalize(
                    final_state="RETRY_EXHAUSTED", stage="wall_clock_budget",
                    reason=f"max_wall_clock_seconds={limits.max_wall_clock_seconds} exceeded",
                )

            attempt_index = round_
            attempt_dir = run_paths.attempt_dir(attempt_index)
            attempt_dir.mkdir(parents=True)

            # ── DIAGNOSING ───────────────────────────────────────────
            _transition("DIAGNOSING", attempt_index=attempt_index)
            diag_prompt = _build_diagnosis_prompt(
                spec=spec, run_paths=run_paths, plan=plan, prior_result_summary=implementation_summary,
                failure=current_failure, verifier_result=current_verifier, attempt_dir=current_attempt_dir,
                prior_attempts=list(attempts), attempt_index=attempt_index,
                remaining_repair_rounds=limits.max_repair_rounds - round_ + 1,
                remaining_claude_invocations=limits.max_total_claude_invocations - claude_invocations,
                remaining_codex_invocations=limits.max_total_codex_invocations - codex_invocations,
                remaining_wall_clock_seconds=limits.max_wall_clock_seconds - _elapsed(),
            )
            before_diag_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label=f"before_diagnosis_{attempt_index:02d}")

            diagnosis: Optional[DiagnosisResult] = None
            diag_error: Optional[tuple[str, str]] = None

            def _invoke_diagnose() -> DiagnosisResult:
                return codex_agent.diagnose(
                    prompt=diag_prompt, run_dir=run_paths.run_dir, cwd=repo_root,
                    timeout=spec.timeouts.planner_seconds, task_id=spec.task_id, run_id=resolved_run_id,
                    attempt_index=attempt_index,
                )

            try:
                diagnosis = _invoke_diagnose()
                codex_invocations += 1
            except ExecutableNotFoundError as e:
                codex_invocations += 1
                diag_error = ("CODEX_INFRASTRUCTURE_FAILURE", f"CODEX_EXECUTABLE_UNAVAILABLE: {e}")
            except (CommandTimeoutError, InfrastructureError) as e:
                codex_invocations += 1
                diag_error = ("CODEX_INFRASTRUCTURE_FAILURE", f"CODEX_TIMEOUT: {e}")
            except CodexPlannerError as e:
                codex_invocations += 1
                if e.code == "CODEX_OUTPUT_MALFORMED" and codex_invocations < limits.max_total_codex_invocations:
                    # Explicit, narrow one-time retry: pure JSON-parse failure only,
                    # once per round, still bounded by the same invocation budget.
                    try:
                        diagnosis = _invoke_diagnose()
                        codex_invocations += 1
                    except (ExecutableNotFoundError, CommandTimeoutError, InfrastructureError) as e2:
                        codex_invocations += 1
                        diag_error = ("CODEX_INFRASTRUCTURE_FAILURE", str(e2))
                    except CodexPlannerError as e2:
                        codex_invocations += 1
                        diag_error = (_CODEX_ERROR_TO_FAILURE_CLASS.get(e2.code, "PLAN_SCHEMA_FAILURE"), str(e2))
                else:
                    diag_error = (_CODEX_ERROR_TO_FAILURE_CLASS.get(e.code, "PLAN_SCHEMA_FAILURE"), str(e))

            after_diag_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label=f"after_diagnosis_{attempt_index:02d}")
            diag_changes = repository_tasks.diff_fingerprints(before_diag_fp, after_diag_fp, run_dir=run_paths.run_dir, repo_root=repo_root)
            if diag_changes:
                main_worktree_unchanged_overall = False
                return _finalize(
                    final_state="POLICY_FAILURE", stage="codex_diagnosis",
                    reason="REPOSITORY_CHANGED_DURING_CODEX_DIAGNOSIS (main worktree mutated): " + "; ".join(diag_changes),
                )

            if diag_error:
                fc, reason = diag_error
                state = "INFRASTRUCTURE_FAILURE" if fc == "CODEX_INFRASTRUCTURE_FAILURE" else "BLOCKED"
                return _finalize(final_state=state, stage="codex_diagnosis", reason=reason)

            assert diagnosis is not None
            reporting_tasks.save_json_artifact(run_paths.diagnosis_path(attempt_index), diagnosis)
            reporting_tasks.save_json_artifact(attempt_dir / "diagnosis.json", diagnosis)

            if diagnosis.verdict == "DIAGNOSE_BLOCKED":
                return _finalize(final_state="BLOCKED", stage="codex_diagnosis", reason=diagnosis.root_cause)
            if diagnosis.verdict == "DIAGNOSE_NOT_REPRODUCIBLE":
                return _finalize(final_state="BLOCKED", stage="codex_diagnosis", reason=diagnosis.root_cause)
            if diagnosis.verdict == "DIAGNOSE_POLICY_FAILURE":
                return _finalize(final_state="POLICY_FAILURE", stage="codex_diagnosis", reason=diagnosis.root_cause)
            if diagnosis.verdict == "DIAGNOSE_INFRASTRUCTURE_FAILURE":
                return _finalize(final_state="INFRASTRUCTURE_FAILURE", stage="codex_diagnosis", reason=diagnosis.root_cause)

            # scope validation: a diagnosis may narrow allowed scope, never broaden it.
            scope_violations = [p for p in diagnosis.files_allowed_to_touch if not execution_policy.is_modify_path_allowed(p, spec)]
            if scope_violations:
                return _finalize(
                    final_state="POLICY_FAILURE", stage="diagnosis_scope_validation",
                    reason=f"diagnosis proposed file(s) outside the original allowed_modify_paths: {scope_violations}",
                )
            command_scope_violations: list[str] = []
            for cmd in diagnosis.commands_allowed_to_run:
                try:
                    execution_policy.assert_executor_command_allowed(cmd, spec)
                except execution_policy.ExecutionPolicyViolation as e:
                    command_scope_violations.append(str(e))
            if command_scope_violations:
                return _finalize(
                    final_state="POLICY_FAILURE", stage="diagnosis_scope_validation", reason="; ".join(command_scope_violations),
                )

            # ── REPAIRING ────────────────────────────────────────────
            started_at = reporting_tasks.utcnow_iso()
            t0 = time.monotonic()
            _transition("REPAIRING", attempt_index=attempt_index)
            repair_prompt = _build_repair_prompt(
                spec=spec, run_paths=run_paths, diagnosis=diagnosis, prior_result_summary=implementation_summary,
                verifier_result=current_verifier, attempt_dir=current_attempt_dir, attempt_index=attempt_index,
                remaining_repair_rounds=limits.max_repair_rounds - round_ + 1,
                remaining_claude_invocations=limits.max_total_claude_invocations - claude_invocations,
            )
            before_repair_fp = repository_tasks.capture_repo_fingerprint(repo_root, run_paths, label=f"before_repair_{attempt_index:02d}")
            before_repair_manifest = execution_worktree.build_ignored_and_untracked_manifest(worktree_dir, spec)

            repair_result: Optional[RepairResult] = None
            repair_error: Optional[tuple[str, str]] = None
            try:
                repair_result = claude_agent.repair(
                    prompt=repair_prompt, worktree_dir=worktree_dir, run_paths=run_paths,
                    timeout=spec.timeouts.executor_seconds, task_id=spec.task_id, run_id=resolved_run_id,
                    attempt_index=attempt_index,
                )
                claude_invocations += 1
                reporting_tasks.save_json_artifact(run_paths.repair_result_path(attempt_index), repair_result)
                reporting_tasks.save_json_artifact(attempt_dir / "repair.json", repair_result)
            except ExecutableNotFoundError as e:
                claude_invocations += 1
                repair_error = ("CLAUDE_INFRASTRUCTURE_FAILURE", f"CLAUDE_EXECUTABLE_UNAVAILABLE: {e}")
            except CommandTimeoutError as e:
                claude_invocations += 1
                repair_error = ("CLAUDE_INFRASTRUCTURE_FAILURE", f"CLAUDE_TIMEOUT: {e}")
            except ClaudeExecutorError as e:
                claude_invocations += 1
                repair_error = (_CLAUDE_ERROR_TO_FAILURE_CLASS.get(e.code, "IMPLEMENTATION_SCHEMA_FAILURE"), str(e))

            gate = _check_policy_gate(
                repo_root=repo_root, run_paths=run_paths, worktree_dir=worktree_dir, spec=spec, repair_limits=limits,
                before_fp=before_repair_fp, before_manifest=before_repair_manifest, base_commit=worktree_record.base_commit,
                claimed_changed_files=(repair_result.changed_files if repair_result else None),
                claimed_commands=(repair_result.commands_run if repair_result else []),
                attempt_index=attempt_index, name_prefix=f"attempt{attempt_index:02d}_repair",
            )
            if not gate.main_worktree_unchanged:
                main_worktree_unchanged_overall = False

            agent_kind = "mock" if isinstance(claude_agent, MockClaudeExecutorAgent) else "real"

            if gate.failure_class:
                failure = build_failure_record(
                    task_id=spec.task_id, run_id=resolved_run_id, attempt_index=attempt_index, failure_class=gate.failure_class,
                    summary=gate.reason or gate.failure_class, evidence=gate.evidence, failed_checks=["policy_gate"],
                    recommended_action="manual review required -- non-retriable policy violation; repair worktree preserved",
                )
                reporting_tasks.save_json_artifact(attempt_dir / "failure.json", failure)
                record = _close_attempt(
                    attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix=f"attempt{attempt_index:02d}_repair",
                    gate=gate, before_fp=before_repair_fp, kind="repair", attempt_index=attempt_index,
                    started_at=started_at, t0=t0, agent_kind=agent_kind,
                    verdict=(repair_result.verdict if repair_result else None), verifier_verdict=None,
                    failure_class=gate.failure_class, retriable=False,
                )
                attempts.append(record)
                return _finalize(final_state=terminal_state_for(gate.failure_class), stage="policy_gate", reason=gate.reason)

            if repair_error:
                fc, reason = repair_error
                failure = build_failure_record(
                    task_id=spec.task_id, run_id=resolved_run_id, attempt_index=attempt_index, failure_class=fc,
                    summary=reason, evidence=[reason], failed_checks=["claude_repair"],
                    recommended_action="manual review required -- non-retriable agent-infrastructure failure",
                )
                reporting_tasks.save_json_artifact(attempt_dir / "failure.json", failure)
                record = _close_attempt(
                    attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix=f"attempt{attempt_index:02d}_repair",
                    gate=gate, before_fp=before_repair_fp, kind="repair", attempt_index=attempt_index,
                    started_at=started_at, t0=t0, agent_kind=agent_kind, verdict=None, verifier_verdict=None,
                    failure_class=fc, retriable=False,
                )
                attempts.append(record)
                return _finalize(final_state=terminal_state_for(fc), stage="claude_repair", reason=reason)

            assert repair_result is not None
            if repair_result.verdict == "REPAIR_BLOCKED":
                record = _close_attempt(
                    attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix=f"attempt{attempt_index:02d}_repair",
                    gate=gate, before_fp=before_repair_fp, kind="repair", attempt_index=attempt_index,
                    started_at=started_at, t0=t0, agent_kind=agent_kind, verdict=repair_result.verdict,
                    verifier_verdict=None, failure_class=None, retriable=False,
                )
                attempts.append(record)
                return _finalize(final_state="BLOCKED", stage="claude_repair", reason=repair_result.summary)

            # ── REVERIFYING ──────────────────────────────────────────
            _transition("REVERIFYING", attempt_index=attempt_index)
            verifier_result, breakdown, _static = run_content_verifier(
                spec=spec, worktree_dir=worktree_dir, attempt_dir=attempt_dir,
                task_id=spec.task_id, run_id=resolved_run_id, attempt_index=attempt_index,
            )
            reporting_tasks.save_json_artifact(attempt_dir / "verifier.json", verifier_result)

            if verifier_result.verdict == "PASS":
                record = _close_attempt(
                    attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix=f"attempt{attempt_index:02d}_repair",
                    gate=gate, before_fp=before_repair_fp, kind="repair", attempt_index=attempt_index,
                    started_at=started_at, t0=t0, agent_kind=agent_kind, verdict=repair_result.verdict,
                    verifier_verdict="PASS", failure_class=None, retriable=None,
                )
                attempts.append(record)
                return _finalize(final_state="PASS", passing_attempt_index=attempt_index)

            fc_round = primary_failure_class(breakdown)
            failure = build_failure_record(
                task_id=spec.task_id, run_id=resolved_run_id, attempt_index=attempt_index, failure_class=fc_round,
                summary="; ".join(verifier_result.details) or "content verifier failed",
                evidence=verifier_result.checks_failed, failed_checks=verifier_result.checks_failed,
                recommended_action="diagnose and repair" if fc_round in ("TEST_FAILURE", "STATIC_CHECK_FAILURE", "ARTIFACT_MISSING", "ARTIFACT_MALFORMED", "EXPECTED_CONTENT_MISMATCH") else "manual review required",
            )
            reporting_tasks.save_json_artifact(attempt_dir / "failure.json", failure)
            record = _close_attempt(
                attempt_dir=attempt_dir, worktree_dir=worktree_dir, name_prefix=f"attempt{attempt_index:02d}_repair",
                gate=gate, before_fp=before_repair_fp, kind="repair", attempt_index=attempt_index,
                started_at=started_at, t0=t0, agent_kind=agent_kind, verdict=repair_result.verdict,
                verifier_verdict="FAIL", failure_class=fc_round, retriable=failure.retriable,
            )
            attempts.append(record)

            if not failure.retriable:
                return _finalize(final_state=terminal_state_for(fc_round), stage="verifier", reason=failure.summary)

            implementation_summary = repair_result.summary
            current_failure = failure
            current_verifier = verifier_result
            current_attempt_dir = attempt_dir
            round_ += 1

    except Exception as exc:  # noqa: BLE001 -- safety net: state.json must never stay active
        return _finalize(
            final_state="INFRASTRUCTURE_FAILURE", stage="unexpected_exception",
            reason=f"{type(exc).__name__}: {exc}", manual_action_required=True,
        )
