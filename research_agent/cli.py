"""Command-line entry point.

    python -m research_agent.cli validate experiments/example_smoke.yaml
    python -m research_agent.cli plan experiments/example_smoke.yaml
    python -m research_agent.cli smoke experiments/example_smoke.yaml
    python -m research_agent.cli execute experiments/example_smoke.yaml
    python -m research_agent.cli status <run_id>
    python -m research_agent.cli worktree-status <run_id>
    python -m research_agent.cli worktree-cleanup <run_id> [--dry-run] [--delete-branch]

`plan` and `smoke` default to a mock, offline Codex adapter and (for
`smoke`) a mock, offline Claude adapter -- no CLI required. Pass
`--codex real` to plan with the actual `codex` CLI (read-only, ephemeral)
once it is installed and authenticated; Codex is only ever an advisor, and
every command it proposes must still independently pass the Python
command/path policy engine before anything runs.

The real Claude adapter is disabled for `smoke` (MVP1 scope, unchanged by
MVP2): `smoke --claude real` is still rejected deterministically
(REAL_CLAUDE_DISABLED_IN_MVP1). `smoke` always runs the deterministic
research-experiment command from the specification, so it deliberately
never gets a real, unreviewed code-writing agent in this build.

`execute` (MVP2) is the new command that DOES support a real Claude
executor: plan -> isolated execution worktree -> Claude executor (mock by
default; real requires the EXPLICIT `--claude real`, never a generic
`--agents real`) -> deterministic diff/path validation -> static checks ->
report. `execute` never runs spec.smoke_command or any research experiment,
regardless of which agents are selected -- see research_agent.execute_flow.
Codex and Claude are always independently selectable on `execute`
(`--codex {mock,real}` / `--claude {mock,real}`, each defaulting to mock).

`repair` (MVP3) adds a BOUNDED diagnose/repair/reverify loop on top of
`execute`'s isolated-worktree implementation: plan -> attempt 0
(implementation) -> deterministic verifier -> on a retriable failure, a
bounded number of (Codex diagnosis -> Claude repair -> re-verify) rounds in
the SAME shared execution worktree, governed by --max-repair-rounds and the
other --max-total-*/--max-wall-clock-seconds budgets (all default to
conservative values; see research_agent.models.RepairLimits). Exactly like
`execute`, Codex and Claude each default to mock and real requires the
explicit `--codex real` / `--claude real` -- see research_agent.repair_flow.
`repair-status <run_id>` and `repair-resume <run_id>` inspect/attempt to
resume a prior `repair` run; resume refuses whenever continuing would not be
safe (see cmd_repair_resume).

`worktree-status`/`worktree-cleanup` support BOTH an `execute` run
(report.json / ExecuteReport) and a `repair` run (final_report.json /
RepairFinalReport). The run type is always determined from which structured
report artifact is actually present and schema-valid -- never guessed from
`run_id` text -- see `_load_recorded_report`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from research_agent import execution_worktree
from research_agent.agents.claude import ClaudeAgent, MockClaudeAgent
from research_agent.agents.claude_executor import MockClaudeExecutorAgent, RealClaudeExecutorAgent
from research_agent.agents.codex import CodexAgent, MockCodexAgent, RealCodexAgent
from research_agent.execute_flow import execute_flow
from research_agent.execution_flow import ConfirmatoryRejected, run_experiment_flow
from research_agent.flow import plan_flow, smoke_flow
from research_agent.models import (
    EXECUTION_TERMINAL_STATES,
    REPAIR_TERMINAL_STATES,
    ExecuteReport,
    ExecutionFinalReport,
    RepairFinalReport,
    RepairLimits,
)
from research_agent.policies import execution_policy
from research_agent.repair_flow import repair_flow
from research_agent.tasks import experiment as experiment_tasks

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2
EXIT_ERROR = 3

_STATUS_EXIT_CODES = {"PASS": EXIT_OK, "FAIL": EXIT_FAIL, "BLOCKED": EXIT_BLOCKED}

REAL_CLAUDE_DISABLED_CODE = "REAL_CLAUDE_DISABLED_IN_MVP1"

_REAL_CODEX_WARNING = (
    "WARNING: --codex real selected. This spawns the actual `codex` CLI as a read-only, "
    "ephemeral (`--sandbox read-only --ephemeral`) planning subprocess. This path is "
    "EXPERIMENTAL: automated tests in this repository only exercise it against fake `codex` "
    "executables (see tests/test_real_codex_planner.py), never a real authenticated `codex` "
    "installation. Codex is only an advisor -- every command it proposes must still "
    "independently pass the deterministic command/path policy engine, and a PLAN_PASS verdict "
    "never causes anything to execute by itself."
)


class RealClaudeDisabledError(RuntimeError):
    """Raised when `smoke --claude real` is requested. Real Claude stays
    disabled for the `smoke` command -- see the module docstring; use
    `execute --claude real` instead."""


_REAL_CLAUDE_EXECUTOR_WARNING = (
    "WARNING: --claude real selected. This spawns the actual `claude` CLI as a subprocess, confined to "
    "a disposable, isolated Git execution worktree (never the main repository worktree), with "
    "--tools Read,Write,Edit (no Bash) and --strict-mcp-config (no MCP servers). This path is "
    "EXPERIMENTAL: automated tests in this repository only exercise it against fake `claude` "
    "executables (see tests/test_real_claude_executor.py), plus exactly one designated harmless live "
    "validation. Claude never commits, never pushes, and never runs a research experiment -- every "
    "claim in its structured response is independently re-checked against the actual Git diff, and the "
    "execution worktree is preserved (never auto-removed) regardless of outcome."
)


def _build_codex_agent(kind: str) -> CodexAgent:
    if kind == "real":
        print(_REAL_CODEX_WARNING, file=sys.stderr)
        return RealCodexAgent()
    return MockCodexAgent()


def _build_claude_agent(kind: str) -> ClaudeAgent:
    if kind == "real":
        raise RealClaudeDisabledError(
            f"{REAL_CLAUDE_DISABLED_CODE}: the real Claude adapter is disabled for `smoke`; pass "
            "--claude mock (the only supported value for `smoke`), or use `execute --claude real`."
        )
    return MockClaudeAgent()


def _build_claude_executor_agent(kind: str):
    if kind == "real":
        print(_REAL_CLAUDE_EXECUTOR_WARNING, file=sys.stderr)
        return RealClaudeExecutorAgent()
    return MockClaudeExecutorAgent()


def _runs_root(args) -> Path:
    repo_root = Path(args.repo_root).resolve()
    return Path(args.runs_root).resolve() if args.runs_root else repo_root / "research_agent_runs"


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        spec = experiment_tasks.load_spec(Path(args.spec_path))
    except (ValidationError, ValueError, FileNotFoundError) as e:
        print(json.dumps({"valid": False, "error": str(e)}, indent=2), file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps({"valid": True, "task_id": spec.task_id, "goal": spec.goal}, indent=2))
    return EXIT_OK


def cmd_plan(args: argparse.Namespace) -> int:
    codex_agent = _build_codex_agent(args.codex)
    result = plan_flow(
        Path(args.spec_path),
        repo_root=Path(args.repo_root).resolve(),
        runs_root=_runs_root(args),
        run_id=args.run_id,
        codex_agent=codex_agent,
    )
    print(result.model_dump_json(indent=2))
    return {"PLAN_PASS": EXIT_OK, "PLAN_REVISE": EXIT_FAIL, "PLAN_BLOCKED": EXIT_BLOCKED}[result.verdict]


def cmd_smoke(args: argparse.Namespace) -> int:
    try:
        claude_agent = _build_claude_agent(args.claude)
    except RealClaudeDisabledError as e:
        print(json.dumps({"error": str(e), "code": REAL_CLAUDE_DISABLED_CODE}, indent=2), file=sys.stderr)
        return EXIT_ERROR
    codex_agent = _build_codex_agent(args.codex)
    report = smoke_flow(
        Path(args.spec_path),
        repo_root=Path(args.repo_root).resolve(),
        runs_root=_runs_root(args),
        run_id=args.run_id,
        codex_agent=codex_agent,
        claude_agent=claude_agent,
    )
    print(report.model_dump_json(indent=2))
    return _STATUS_EXIT_CODES[report.overall_status]


def cmd_execute(args: argparse.Namespace) -> int:
    """MVP2: plan -> isolated execution worktree -> Claude executor ->
    deterministic diff/path validation -> static checks -> report. Never
    runs spec.smoke_command or any research experiment. Codex and Claude
    are always independently selected -- there is no combined `--agents`
    flag, and `--claude real` (never bundled behind anything generic) is
    the only way to activate the real Claude subprocess."""
    codex_agent = _build_codex_agent(args.codex)
    claude_agent = _build_claude_executor_agent(args.claude)
    execution_worktrees_root = (
        Path(args.execution_worktrees_root).resolve() if args.execution_worktrees_root else None
    )
    report = execute_flow(
        Path(args.spec_path),
        repo_root=Path(args.repo_root).resolve(),
        runs_root=_runs_root(args),
        run_id=args.run_id,
        execution_worktrees_root=execution_worktrees_root,
        codex_agent=codex_agent,
        claude_agent=claude_agent,
    )
    print(report.model_dump_json(indent=2))
    return _STATUS_EXIT_CODES.get(report.overall_status, EXIT_ERROR)


def cmd_run_experiment(args: argparse.Namespace) -> int:
    """MVP4: plan -> optional isolated pre-execution implementation ->
    READY_FOR_EXECUTION/EXECUTION_NOT_REQUESTED gate -> (only if --execute
    was passed) restricted, resource-limited, pre-approved-command-only
    experiment subprocess -> artifact collection -> deterministic metric
    verification -> bounded post-execution diagnose/repair-then-retry loop
    -> report. Exactly like `execute`/`repair`, Codex and Claude are always
    independently selected (`--codex real`/`--claude real`, each defaulting
    to mock) -- there is no generic `--agents real` flag. Execution itself
    additionally requires the separate, explicit `--execute` flag: without
    it, the command always stops at EXECUTION_NOT_REQUESTED, regardless of
    what Codex or Claude said. A confirmatory specification (execution_mode:
    confirmatory, confirmatory: true, or a task_id/goal/approved_command
    naming 'confirmatory'/'paper_final'/'final_result'/a final-results
    directory) is rejected deterministically as
    CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL before planning starts --
    there is no --confirmatory flag and no way to bypass this from the CLI."""
    try:
        codex_agent = _build_codex_agent(args.codex)
        claude_agent = _build_claude_executor_agent(args.claude)
    except RealClaudeDisabledError as e:
        print(json.dumps({"error": str(e), "code": REAL_CLAUDE_DISABLED_CODE}, indent=2), file=sys.stderr)
        return EXIT_ERROR
    execution_worktrees_root = (
        Path(args.execution_worktrees_root).resolve() if args.execution_worktrees_root else None
    )
    try:
        report = run_experiment_flow(
            Path(args.spec_path),
            repo_root=Path(args.repo_root).resolve(),
            runs_root=_runs_root(args),
            run_id=args.run_id,
            execution_worktrees_root=execution_worktrees_root,
            codex_agent=codex_agent,
            claude_agent=claude_agent,
            execute=args.execute,
        )
    except ConfirmatoryRejected as e:
        print(json.dumps({"error": str(e), "code": "CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL"}, indent=2), file=sys.stderr)
        return EXIT_ERROR
    except (ValidationError, ValueError, FileNotFoundError) as e:
        code = "CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL" if "CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL" in str(e) else "SPEC_VALIDATION_ERROR"
        print(json.dumps({"error": str(e), "code": code}, indent=2), file=sys.stderr)
        return EXIT_ERROR
    print(report.model_dump_json(indent=2))
    if report.final_state == "EXECUTION_NOT_REQUESTED":
        return EXIT_OK
    return _STATUS_EXIT_CODES.get(report.overall_status, EXIT_ERROR)


_REPAIR_LIMIT_FIELDS = (
    "max_repair_rounds",
    "max_total_claude_invocations",
    "max_total_codex_invocations",
    "max_total_changed_files",
    "max_total_changed_bytes",
    "max_wall_clock_seconds",
)


def _repair_limits_from_args(args: argparse.Namespace, spec_limits: RepairLimits) -> RepairLimits:
    """CLI --max-* flags override the spec's own repair_limits field-by-field
    (only fields explicitly passed on the command line, i.e. not None,
    override); with no flags passed, the spec's own (conservative-by-default)
    repair_limits apply unchanged."""
    overrides = {name: getattr(args, name) for name in _REPAIR_LIMIT_FIELDS if getattr(args, name) is not None}
    if not overrides:
        return spec_limits
    return spec_limits.model_copy(update=overrides)


def cmd_repair(args: argparse.Namespace) -> int:
    """MVP3: plan -> attempt 0 (isolated execution worktree implementation)
    -> deterministic verifier -> bounded (Codex diagnosis -> Claude repair ->
    re-verify) loop -> terminal report. Never runs spec.smoke_command or any
    research experiment. Codex and Claude are always independently selected,
    exactly like `execute` -- `--claude real`/`--codex real` are the only way
    to activate the real subprocesses; both default to mock."""
    try:
        spec = experiment_tasks.load_spec(Path(args.spec_path))
    except (ValidationError, ValueError, FileNotFoundError) as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        return EXIT_ERROR

    codex_agent = _build_codex_agent(args.codex)
    claude_agent = _build_claude_executor_agent(args.claude)
    execution_worktrees_root = (
        Path(args.execution_worktrees_root).resolve() if args.execution_worktrees_root else None
    )
    limits = _repair_limits_from_args(args, spec.repair_limits)
    report = repair_flow(
        Path(args.spec_path),
        repo_root=Path(args.repo_root).resolve(),
        runs_root=_runs_root(args),
        run_id=args.run_id,
        execution_worktrees_root=execution_worktrees_root,
        codex_agent=codex_agent,
        claude_agent=claude_agent,
        repair_limits=limits,
    )
    print(report.model_dump_json(indent=2))
    return _STATUS_EXIT_CODES.get(report.overall_status, EXIT_ERROR)


def cmd_repair_status(args: argparse.Namespace) -> int:
    """Shows current state, attempts used, retry budget remaining, last
    failure class, execution worktree path, whether main is unchanged, and
    whether manual action is required. A run with no final_report.json yet
    (still active, or interrupted mid-run) is always reported as INCOMPLETE
    -- never silently treated as PASS, per the MVP3 task contract's
    "interrupted run" requirement (see also cmd_repair_resume)."""
    run_dir = _runs_root(args) / args.run_id
    if not run_dir.exists():
        print(json.dumps({"error": f"no run directory at {run_dir}"}, indent=2), file=sys.stderr)
        return EXIT_ERROR

    state_path = run_dir / "state.json"
    final_report_path = run_dir / "final_report.json"
    state_data = json.loads(state_path.read_text()) if state_path.exists() else None

    if not final_report_path.exists():
        print(json.dumps({
            "run_id": args.run_id,
            "current_state": state_data["state"] if state_data else "UNKNOWN",
            "status": "INCOMPLETE",
            "attempts_used": None,
            "retry_budget_remaining": None,
            "last_failure_class": None,
            "execution_worktree_path": None,
            "main_worktree_unchanged": None,
            "manual_action_required": True,
            "note": "no final_report.json yet -- this run is either still active or was interrupted "
            "mid-run; never treat an incomplete run as PASS",
        }, indent=2))
        return EXIT_ERROR

    report = json.loads(final_report_path.read_text())
    attempts = report.get("attempts", [])
    last_failure_class = None
    for a in reversed(attempts):
        if a.get("failure_class"):
            last_failure_class = a["failure_class"]
            break
    limits = report.get("repair_limits", {})
    repair_rounds_used = max((a["attempt_index"] for a in attempts if a.get("kind") == "repair"), default=0)
    retry_budget_remaining = {
        "repair_rounds": max(0, limits.get("max_repair_rounds", 0) - repair_rounds_used),
        "claude_invocations": max(0, limits.get("max_total_claude_invocations", 0) - report.get("total_claude_invocations", 0)),
        "codex_invocations": max(0, limits.get("max_total_codex_invocations", 0) - report.get("total_codex_invocations", 0)),
    }
    worktree_path = (report.get("execution_worktree") or {}).get("worktree_path")

    print(json.dumps({
        "run_id": args.run_id,
        "current_state": report.get("final_state"),
        "status": report.get("overall_status"),
        "attempts_used": len(attempts),
        "retry_budget_remaining": retry_budget_remaining,
        "last_failure_class": last_failure_class,
        "execution_worktree_path": worktree_path,
        "main_worktree_unchanged": report.get("main_worktree_unchanged"),
        "manual_action_required": report.get("manual_action_required"),
    }, indent=2))
    return _STATUS_EXIT_CODES.get(report.get("overall_status"), EXIT_ERROR)


def cmd_repair_resume(args: argparse.Namespace) -> int:
    """Conservative resume: refuses whenever the run is already terminal,
    state.json is missing, or the run was interrupted mid-flight. MVP3 does
    not implement automatic mid-flight resume -- re-entering a bounded
    invocation-budget loop after an unknown external interruption (was the
    last agent call actually in flight? did it partially mutate the shared
    worktree?) cannot be judged safe from state.json alone. Every attempt's
    artifacts and the execution worktree (if created) are always preserved
    for manual inspection regardless."""
    run_dir = _runs_root(args) / args.run_id
    if not run_dir.exists():
        print(json.dumps({"error": f"no run directory at {run_dir}"}, indent=2), file=sys.stderr)
        return EXIT_ERROR

    state_path = run_dir / "state.json"
    final_report_path = run_dir / "final_report.json"
    if not state_path.exists():
        print(json.dumps({
            "error": "REFUSED_NO_STATE", "reason": "no state.json found under this run directory",
        }, indent=2), file=sys.stderr)
        return EXIT_ERROR
    state_data = json.loads(state_path.read_text())

    if final_report_path.exists():
        report_data = json.loads(final_report_path.read_text())
        print(json.dumps({
            "error": "REFUSED_TERMINAL_STATE",
            "reason": f"run already reached terminal state {report_data.get('final_state')!r}; resume is "
            "only meaningful for an interrupted (non-terminal) run",
        }, indent=2), file=sys.stderr)
        return EXIT_ERROR

    if state_data.get("state") in REPAIR_TERMINAL_STATES:
        print(json.dumps({
            "error": "REFUSED_TERMINAL_STATE",
            "reason": f"state.json is already terminal ({state_data.get('state')!r}) but final_report.json "
            "is missing -- inconsistent run, needs manual inspection rather than an automatic resume",
        }, indent=2), file=sys.stderr)
        return EXIT_ERROR

    print(json.dumps({
        "error": "REFUSED_UNSAFE_RESUME",
        "reason": f"run was interrupted in a non-terminal state ({state_data.get('state')!r}); MVP3 does "
        "not implement mid-flight resume. The execution worktree (if any) and every completed attempt's "
        "artifacts under attempts/ are preserved for manual inspection. Start a new `repair` run instead.",
        "state": state_data.get("state"),
        "run_dir": str(run_dir),
    }, indent=2), file=sys.stderr)
    return EXIT_ERROR


def _load_execution_worktree_record(run_dir: Path) -> dict:
    record_path = run_dir / "execution_worktree.json"
    if not record_path.exists():
        raise FileNotFoundError(f"no execution_worktree.json under {run_dir} (not an `execute`/`repair` run, or it never reached worktree creation)")
    return json.loads(record_path.read_text())


def _load_recorded_report(run_dir: Path) -> tuple[str, Optional[dict], Optional[str]]:
    """Determine which structured final-report artifact this run produced --
    `report.json` (ExecuteReport, from `execute`), or `final_report.json`,
    which BOTH `repair` (RepairFinalReport) and MVP4 `run-experiment`
    (ExecutionFinalReport) write -- and return (run_type,
    validated_report_dict_or_None, error_message_or_None).

    run_type is one of "execute", "repair", "run-experiment", "unknown" and
    is derived PURELY from which file exists on disk and which schema its
    content actually validates against -- never guessed from run_id text.
    Both RepairFinalReport and ExecutionFinalReport are StrictModel
    (extra="forbid"), so the two schemas are mutually exclusive: a
    RepairFinalReport-shaped dict always fails ExecutionFinalReport
    validation (missing `execution_attempts` etc.) and vice versa (missing
    `repair_limits`); RepairFinalReport is tried first, purely for
    backward-compatible priority, with no ambiguity either way. If the file
    exists but is not valid JSON or fails BOTH schemas, run_type is still
    "repair"/"run-experiment" is undecidable, so it is reported as such via
    the error message and the returned dict is None -- callers must treat
    such a run as untrustworthy (refuse rather than compare against a
    report that might be truncated/corrupted/hand-edited)."""
    final_report_path = run_dir / "final_report.json"
    if final_report_path.exists():
        try:
            raw_text = final_report_path.read_text()
            data = json.loads(raw_text)
        except (json.JSONDecodeError, OSError) as e:
            return "unknown-final-report", None, f"final_report.json is missing, unreadable, or not valid JSON: {e}"
        try:
            RepairFinalReport.model_validate(data)
            return "repair", data, None
        except ValidationError as repair_err:
            repair_err_text = str(repair_err)
        try:
            ExecutionFinalReport.model_validate(data)
            return "run-experiment", data, None
        except ValidationError as exec_err:
            # Malformed/ambiguous content that matches NEITHER schema: which
            # command actually produced this run cannot be recovered from the
            # content alone. Default to "repair" (this repository's original,
            # pre-MVP4 behavior for any final_report.json, valid or not) purely
            # for backward-compatible run_type labeling -- functionally this
            # makes no difference either way, since every caller already
            # refuses (REFUSED_MALFORMED_REPORT / INCOMPLETE) whenever
            # report_data is None, regardless of the run_type label.
            return (
                "repair", None,
                f"final_report.json matches neither RepairFinalReport ({repair_err_text}) nor "
                f"ExecutionFinalReport ({exec_err})",
            )

    report_path = run_dir / "report.json"
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text())
            ExecuteReport.model_validate(data)
        except (json.JSONDecodeError, OSError, ValidationError) as e:
            return "execute", None, f"report.json is missing, unreadable, or malformed: {e}"
        return "execute", data, None

    return "unknown", None, None


def _run_activity_state(run_dir: Path, run_type: str) -> Optional[str]:
    """Returns the non-terminal state name if this run is still active, else
    None. A terminal, schema-valid report (run_type not in ("unknown",
    "unknown-final-report")) always means the run already finished -- only
    MVP3 `repair` and MVP4 `run-experiment` runs write state.json at all, so
    this can only fire once no final_report.json/report.json exists yet."""
    if run_type not in ("unknown", "unknown-final-report"):
        return None
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        state_data = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    state = state_data.get("state")
    if state in REPAIR_TERMINAL_STATES or state in EXECUTION_TERMINAL_STATES:
        return None
    return state


def _report_summary(run_type: str, report_data: dict) -> dict:
    if run_type == "repair":
        return {
            "run_type": "repair",
            "overall_status": report_data.get("overall_status"),
            "final_state": report_data.get("final_state"),
            "passing_attempt_index": report_data.get("passing_attempt_index"),
            "changed_file_count": report_data.get("changed_file_count"),
        }
    if run_type == "run-experiment":
        return {
            "run_type": "run-experiment",
            "overall_status": report_data.get("overall_status"),
            "final_state": report_data.get("final_state"),
            "execute_requested": report_data.get("execute_requested"),
            "passing_attempt_index": report_data.get("passing_attempt_index"),
            "execution_attempts": len(report_data.get("execution_attempts") or []),
            "implementation_attempts": len(report_data.get("implementation_attempts") or []),
        }
    return {
        "run_type": "execute",
        "overall_status": report_data.get("overall_status"),
        "terminal_state": report_data.get("terminal_state"),
        "changed_file_count": report_data.get("changed_file_count"),
    }


def _recorded_execution_worktree_identity_conflict(record: dict, report_data: dict) -> Optional[str]:
    """Cross-checks execution_worktree.json's own record against the SAME
    execution_worktree block embedded in the final report (both are written
    once, at different times, from the same underlying
    ExecutionWorktreeRecord) -- a mismatch means one of the two artifacts
    was edited/corrupted/replaced after the fact, or points at a different
    worktree/branch entirely. Returns a human-readable reason, or None if
    consistent (or the report has no embedded worktree record to compare,
    e.g. a run that was blocked before the worktree was ever created)."""
    embedded = report_data.get("execution_worktree")
    if not embedded:
        return None
    if embedded.get("worktree_path") != record.get("worktree_path"):
        return (
            f"execution_worktree.json worktree_path={record.get('worktree_path')!r} does not match "
            f"the report's own recorded worktree_path={embedded.get('worktree_path')!r}"
        )
    if embedded.get("branch_name") != record.get("branch_name"):
        return (
            f"execution_worktree.json branch_name={record.get('branch_name')!r} does not match "
            f"the report's own recorded branch_name={embedded.get('branch_name')!r}"
        )
    return None


def _checked_out_branch(worktree_path: Path) -> Optional[str]:
    proc = subprocess.run(
        ["git", "-C", str(worktree_path), "branch", "--show-current"],
        shell=False, capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _current_worktree_fingerprints(worktree_path: Path) -> dict[str, dict]:
    """path -> {"size_bytes": .., "sha256": ..} for every changed (tracked or
    untracked) path in worktree_path, computed fresh from the filesystem --
    never trusted from any prior record. Large files are hashed the same way
    execution_worktree.build_changed_file_records does (skip above
    execution_policy.MAX_HASH_BYTES -- metadata only), so this stays cheap.
    Comparing size+hash (not just the path set) is required to catch an
    in-place content edit of an ALREADY-recorded path -- `git status`'s
    porcelain line for that path does not change, so a path-set-only
    comparison would silently miss it."""
    proc = subprocess.run(
        ["git", "-C", str(worktree_path), "status", "--porcelain=v1", "--untracked-files=all"],
        shell=False, capture_output=True, text=True, timeout=30,
    )
    rel_paths: list[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        rest = line[3:] if len(line) > 3 else line.strip()
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        rel_paths.append(rest.strip().strip('"'))

    fingerprints: dict[str, dict] = {}
    for rel in rel_paths:
        norm = Path(rel.rstrip("/")).as_posix()
        abs_path = worktree_path / rel
        size: Optional[int] = None
        sha: Optional[str] = None
        try:
            if abs_path.is_file() and not abs_path.is_symlink():
                size = abs_path.stat().st_size
                if size <= execution_policy.MAX_HASH_BYTES:
                    h = hashlib.sha256()
                    with open(abs_path, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            h.update(chunk)
                    sha = h.hexdigest()
        except OSError:
            pass
        fingerprints[norm] = {"size_bytes": size, "sha256": sha}
    return fingerprints


def _fingerprints_from_changed_file_records(records: list) -> dict[str, dict]:
    fingerprints: dict[str, dict] = {}
    for c in records:
        norm = Path(str(c.get("path", "")).rstrip("/")).as_posix()
        fingerprints[norm] = {"size_bytes": c.get("size_bytes"), "sha256": c.get("sha256")}
    return fingerprints


def _recorded_changed_fingerprints_for_run(run_dir: Path, run_type: str, report_data: dict) -> Optional[dict[str, dict]]:
    """The recorded "final, trustworthy" changed-file fingerprints for this
    run -- what worktree-cleanup compares the CURRENT worktree state
    against. Returns None if there is nothing trustworthy to compare
    against (caller must then refuse rather than guess).

    execute: report.json's own top-level `changed_files` list -- the single
    attempt `execute` ever makes.

    repair: repair_flow.py never stores the full changed-file list at the
    top level of final_report.json (only aggregate counts) -- the actual
    per-path list lives in the PASSING attempt's (if any; else the final/
    terminal attempt's) attempts/attempt_NN/changed_file_manifest.json,
    written once and never touched again after that attempt closes -- see
    research_agent.repair_flow._close_attempt."""
    if run_type == "execute":
        return _fingerprints_from_changed_file_records(report_data.get("changed_files") or [])

    if run_type == "repair":
        attempts = report_data.get("attempts") or []
        if not attempts:
            return {}
        passing_index = report_data.get("passing_attempt_index")
        target_index = passing_index if passing_index is not None else max(a["attempt_index"] for a in attempts)
        manifest_path = run_dir / "attempts" / f"attempt_{target_index:02d}" / "changed_file_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest_data = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(manifest_data, list):
            return None
        return _fingerprints_from_changed_file_records(manifest_data)

    if run_type == "run-experiment":
        # MVP4 shares the same attempts/attempt_NN/changed_file_manifest.json
        # layout as `repair` (research_agent.models.ExecutionRunPaths extends
        # RepairRunPaths unchanged) -- but its optional pre-execution
        # implementation/post-execution repair phase is recorded under
        # `implementation_attempts`, not `attempts`, and (unlike `repair`,
        # which always makes at least one implementation attempt) may be
        # entirely EMPTY for a spec with no implementation scope configured
        # -- in which case the execution worktree is legitimately untouched
        # and the recorded fingerprint set is correctly {}.
        attempts = report_data.get("implementation_attempts") or []
        if not attempts:
            return {}
        target_index = max(a["attempt_index"] for a in attempts)
        manifest_path = run_dir / "attempts" / f"attempt_{target_index:02d}" / "changed_file_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest_data = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(manifest_data, list):
            return None
        return _fingerprints_from_changed_file_records(manifest_data)

    return None


def cmd_worktree_status(args: argparse.Namespace) -> int:
    run_dir = _runs_root(args) / args.run_id
    try:
        record = _load_execution_worktree_record(run_dir)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        return EXIT_ERROR

    worktree_path = Path(record["worktree_path"])
    exists = worktree_path.exists()
    status_lines: list[str] = []
    if exists:
        proc = subprocess.run(
            ["git", "-C", str(worktree_path), "status", "--porcelain=v1", "--untracked-files=all"],
            shell=False, capture_output=True, text=True, timeout=30,
        )
        status_lines = [line for line in proc.stdout.splitlines() if line.strip()]

    run_type, report_data, report_error = _load_recorded_report(run_dir)
    report_summary = _report_summary(run_type, report_data) if report_data is not None else None
    active_state = _run_activity_state(run_dir, run_type)

    print(json.dumps({
        "run_id": record["run_id"],
        "run_type": run_type,
        "base_commit": record["base_commit"],
        "branch_name": record["branch_name"],
        "worktree_path": record["worktree_path"],
        "worktree_exists": exists,
        "current_status_lines": status_lines,
        "recorded_preserved": record.get("preserved", True),
        "recorded_removed_at": record.get("removed_at"),
        "report_summary": report_summary,
        "report_error": report_error,
        "active_state": active_state,
    }, indent=2))
    return EXIT_OK


def cmd_worktree_cleanup(args: argparse.Namespace) -> int:
    """Refuses to remove a worktree with changes not accounted for by a
    saved, schema-valid, terminal report (report.json for `execute`,
    final_report.json for `repair` -- see _load_recorded_report), never uses
    `git clean -fdx`, never touches main, and only deletes the throwaway
    branch when --delete-branch is explicitly passed. --dry-run runs every
    check below and reports what would happen, without removing anything."""
    run_dir = _runs_root(args) / args.run_id
    try:
        record = _load_execution_worktree_record(run_dir)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        return EXIT_ERROR

    worktree_path = Path(record["worktree_path"])
    repo_root = Path(args.repo_root).resolve()

    if worktree_path.resolve() == repo_root:
        print(json.dumps({
            "error": "REFUSED_MAIN_WORKTREE_TARGET",
            "reason": f"recorded worktree_path {worktree_path} resolves to the main repository worktree "
            f"{repo_root} -- refusing to touch it under any circumstance",
        }, indent=2), file=sys.stderr)
        return EXIT_ERROR

    if not worktree_path.exists():
        print(json.dumps({"status": "ALREADY_REMOVED", "worktree_path": str(worktree_path)}, indent=2))
        return EXIT_OK

    run_type, report_data, report_error = _load_recorded_report(run_dir)

    active_state = _run_activity_state(run_dir, run_type)
    if active_state is not None:
        print(json.dumps({
            "error": "REFUSED_RUN_ACTIVE",
            "reason": f"run is still active (state.json state={active_state!r}, no terminal report yet) -- "
            "never clean up a worktree a run may still be writing to",
        }, indent=2), file=sys.stderr)
        return EXIT_ERROR

    if report_data is not None:
        identity_conflict = _recorded_execution_worktree_identity_conflict(record, report_data)
        if identity_conflict is not None:
            print(json.dumps({"error": "REFUSED_INCONSISTENT_IDENTITY", "reason": identity_conflict}, indent=2), file=sys.stderr)
            return EXIT_ERROR

    checked_out_branch = _checked_out_branch(worktree_path)
    if checked_out_branch is not None and checked_out_branch != record.get("branch_name"):
        print(json.dumps({
            "error": "REFUSED_INCONSISTENT_IDENTITY",
            "reason": f"worktree at {worktree_path} currently has branch {checked_out_branch!r} checked out, "
            f"but execution_worktree.json records branch_name={record.get('branch_name')!r}",
        }, indent=2), file=sys.stderr)
        return EXIT_ERROR

    current_fp = _current_worktree_fingerprints(worktree_path)

    if run_type == "unknown":
        if current_fp:
            print(json.dumps({
                "error": "REFUSED_UNRECORDED_CHANGES",
                "reason": "no report.json or final_report.json to compare against, and the worktree has "
                "uncommitted changes",
                "current_changed_paths": sorted(current_fp),
            }, indent=2), file=sys.stderr)
            return EXIT_ERROR
        # no report at all, but the worktree is genuinely untouched -- safe (matches prior MVP2 behavior).
    else:
        if report_data is None:
            print(json.dumps({"error": "REFUSED_MALFORMED_REPORT", "reason": report_error}, indent=2), file=sys.stderr)
            return EXIT_ERROR

        recorded_fp = _recorded_changed_fingerprints_for_run(run_dir, run_type, report_data)
        if recorded_fp is None:
            print(json.dumps({
                "error": "REFUSED_UNTRUSTWORTHY_RECORD",
                "reason": f"{run_type} run's final report exists but its recorded changed-file evidence "
                "(the passing/terminal attempt's changed_file_manifest.json) is missing or malformed",
            }, indent=2), file=sys.stderr)
            return EXIT_ERROR

        if set(current_fp) != set(recorded_fp):
            print(json.dumps({
                "error": "REFUSED_UNRECORDED_CHANGES",
                "reason": f"worktree's current changed paths do not match the recorded {run_type} report",
                "current_changed_paths": sorted(current_fp),
                "recorded_changed_paths": sorted(recorded_fp),
            }, indent=2), file=sys.stderr)
            return EXIT_ERROR

        mismatched = sorted(
            p for p in recorded_fp
            if (recorded_fp[p].get("size_bytes"), recorded_fp[p].get("sha256"))
            != (current_fp[p].get("size_bytes"), current_fp[p].get("sha256"))
        )
        if mismatched:
            print(json.dumps({
                "error": "REFUSED_MODIFIED_AFTER_REPORT",
                "reason": "one or more recorded paths were modified in place after the report was written "
                "(same path, different content) -- the git status line alone does not change for this, so "
                "size/hash fingerprints are compared",
                "modified_paths": mismatched,
            }, indent=2), file=sys.stderr)
            return EXIT_ERROR

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN",
            "run_type": run_type,
            "would_remove_worktree": str(worktree_path),
            "would_delete_branch": bool(args.delete_branch),
            "branch_name": record["branch_name"],
        }, indent=2))
        return EXIT_OK

    execution_worktree.remove_execution_worktree(
        repo_root, worktree_path, record["branch_name"], delete_branch=args.delete_branch,
    )
    record["preserved"] = False
    record["removed_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "execution_worktree.json").write_text(json.dumps(record, indent=2) + "\n")

    print(json.dumps({
        "status": "REMOVED",
        "run_type": run_type,
        "worktree_path": str(worktree_path),
        "branch_deleted": bool(args.delete_branch),
    }, indent=2))
    return EXIT_OK


def cmd_experiment_status(args: argparse.Namespace) -> int:
    """Status for an MVP4 `run-experiment` run. A run with no
    final_report.json yet (still active, or interrupted mid-run) is always
    reported as INCOMPLETE -- never silently treated as PASS. If the run
    directory actually belongs to a different command (`repair` or
    `execute`), this refuses and points at the right status command instead
    of guessing."""
    run_dir = _runs_root(args) / args.run_id
    if not run_dir.exists():
        print(json.dumps({"error": f"no run directory at {run_dir}"}, indent=2), file=sys.stderr)
        return EXIT_ERROR

    run_type, report_data, report_error = _load_recorded_report(run_dir)

    if run_type in ("unknown", "unknown-final-report"):
        active_state = _run_activity_state(run_dir, run_type)
        print(json.dumps({
            "run_id": args.run_id,
            "run_type": "run-experiment",
            "current_state": active_state or "UNKNOWN",
            "status": "INCOMPLETE",
            "manual_action_required": True,
            "note": report_error or (
                "no final_report.json yet -- this run is either still active or was interrupted mid-run; "
                "never treat an incomplete run as PASS"
            ),
        }, indent=2))
        return EXIT_ERROR

    if run_type != "run-experiment":
        other_status_cmd = "repair-status" if run_type == "repair" else "status"
        print(json.dumps({
            "error": "REFUSED_WRONG_RUN_TYPE",
            "reason": f"run {args.run_id!r} was produced by `{run_type}`, not `run-experiment`; use "
            f"`{other_status_cmd}` instead",
        }, indent=2), file=sys.stderr)
        return EXIT_ERROR

    if report_data is None:
        print(json.dumps({"error": "REFUSED_MALFORMED_REPORT", "reason": report_error}, indent=2), file=sys.stderr)
        return EXIT_ERROR

    limits = report_data.get("execution_limits") or {}
    execution_attempts = report_data.get("execution_attempts") or []
    implementation_attempts = report_data.get("implementation_attempts") or []
    repair_rounds_used = max((a["attempt_index"] for a in implementation_attempts if a.get("kind") == "repair"), default=0)
    retry_budget_remaining = {
        "execution_attempts": max(0, limits.get("max_execution_attempts", 0) - len(execution_attempts)),
        "repair_rounds": max(0, limits.get("max_repair_rounds", 0) - repair_rounds_used),
        "claude_invocations": max(0, limits.get("max_total_claude_invocations", 0) - report_data.get("total_claude_invocations", 0)),
        "codex_invocations": max(0, limits.get("max_total_codex_invocations", 0) - report_data.get("total_codex_invocations", 0)),
    }

    print(json.dumps({
        "run_id": args.run_id,
        "run_type": "run-experiment",
        "current_state": report_data.get("final_state"),
        "status": report_data.get("overall_status"),
        "execute_requested": report_data.get("execute_requested"),
        "execution_mode": report_data.get("execution_mode"),
        "passing_attempt_index": report_data.get("passing_attempt_index"),
        "execution_attempts_used": len(execution_attempts),
        "implementation_attempts_used": len(implementation_attempts),
        "execution_worktree_path": (report_data.get("execution_worktree") or {}).get("worktree_path"),
        "main_worktree_unchanged": report_data.get("main_worktree_unchanged"),
        "retry_budget_remaining": retry_budget_remaining,
        "manual_action_required": report_data.get("manual_action_required"),
    }, indent=2))
    return _STATUS_EXIT_CODES.get(report_data.get("overall_status"), EXIT_ERROR)


def cmd_experiment_cleanup(args: argparse.Namespace) -> int:
    """MVP4 execution-worktree cleanup -- a thin alias for
    `worktree-cleanup`, which already understands `run-experiment` reports
    (see `_load_recorded_report`/`_recorded_changed_fingerprints_for_run`
    above). Kept as a distinct, discoverable command per the MVP4 CLI
    surface; the underlying safety checks (refuse unrecorded changes,
    refuse active runs, refuse malformed reports, refuse main, dry-run,
    explicit --delete-branch, never `git clean -fdx`) are identical."""
    return cmd_worktree_cleanup(args)


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = _runs_root(args) / args.run_id
    if not run_dir.exists():
        print(json.dumps({"error": f"no run directory at {run_dir}"}, indent=2), file=sys.stderr)
        return EXIT_ERROR
    report_path = run_dir / "report.json"
    if report_path.exists():
        data = json.loads(report_path.read_text())
        print(json.dumps(data, indent=2))
        return _STATUS_EXIT_CODES.get(data.get("overall_status"), EXIT_ERROR)
    present = sorted(p.name for p in run_dir.iterdir())
    print(json.dumps({"run_id": args.run_id, "status": "INCOMPLETE", "artifacts_present": present}, indent=2))
    return EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research_agent.cli")
    parser.add_argument("--repo-root", default=".", help="repository root (default: cwd)")
    parser.add_argument(
        "--runs-root", default=None, help="run directory root (default: <repo-root>/research_agent_runs)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="validate an experiment specification YAML")
    p_validate.add_argument("spec_path")
    p_validate.set_defaults(func=cmd_validate)

    p_plan = sub.add_parser("plan", help="run validate -> snapshot -> Codex planner -> plan policy validation")
    p_plan.add_argument("spec_path")
    p_plan.add_argument("--run-id", default=None)
    p_plan.add_argument(
        "--codex", choices=["mock", "real"], default="mock",
        help="mock (default, offline, no CLI required) or real (invokes the codex CLI, read-only)",
    )
    p_plan.set_defaults(func=cmd_plan)

    p_smoke = sub.add_parser("smoke", help="run the full ten-stage smoke-experiment pipeline")
    p_smoke.add_argument("spec_path")
    p_smoke.add_argument("--run-id", default=None)
    p_smoke.add_argument(
        "--codex", choices=["mock", "real"], default="mock",
        help="mock (default, offline, no CLI required) or real (invokes the codex CLI, read-only)",
    )
    p_smoke.add_argument(
        "--claude", choices=["mock", "real"], default="mock",
        help="mock (default, and the only supported value in this MVP1 build -- "
        "real is rejected with REAL_CLAUDE_DISABLED_IN_MVP1)",
    )
    p_smoke.set_defaults(func=cmd_smoke)

    p_execute = sub.add_parser(
        "execute",
        help="MVP2: plan -> isolated execution worktree -> Claude executor -> deterministic diff/path "
        "validation -> static checks -> report. Never runs a research experiment.",
    )
    p_execute.add_argument("spec_path")
    p_execute.add_argument("--run-id", default=None)
    p_execute.add_argument(
        "--execution-worktrees-root", default=None,
        help="root directory for execution worktrees (default: a sibling directory of --repo-root, "
        "always outside the main repository worktree)",
    )
    p_execute.add_argument(
        "--codex", choices=["mock", "real"], default="mock",
        help="mock (default, offline, no CLI required) or real (invokes the codex CLI, read-only)",
    )
    p_execute.add_argument(
        "--claude", choices=["mock", "real"], default="mock",
        help="mock (default, offline) or real (invokes the claude CLI inside an isolated, disposable "
        "execution worktree; requires this EXPLICIT flag -- there is no generic --agents real)",
    )
    p_execute.set_defaults(func=cmd_execute)

    p_run_experiment = sub.add_parser(
        "run-experiment",
        help="MVP4: plan -> optional implementation -> execution gate -> restricted CPU-only, "
        "pre-approved-command-only experiment subprocess -> artifact collection -> metric verification -> "
        "bounded post-execution repair-then-retry -> report. Confirmatory execution is always rejected. "
        "Real execution requires the separate, explicit --execute flag.",
    )
    p_run_experiment.add_argument("spec_path")
    p_run_experiment.add_argument("--run-id", default=None)
    p_run_experiment.add_argument(
        "--execution-worktrees-root", default=None,
        help="root directory for the execution worktree (default: a sibling directory of --repo-root, "
        "always outside the main repository worktree)",
    )
    p_run_experiment.add_argument(
        "--codex", choices=["mock", "real"], default="mock",
        help="mock (default, offline, no CLI required) or real (invokes the codex CLI, read-only) -- "
        "used for BOTH the planner and the post-execution diagnosis role; Codex is only ever an advisor "
        "and can never authorize a command execution",
    )
    p_run_experiment.add_argument(
        "--claude", choices=["mock", "real"], default="mock",
        help="mock (default, offline) or real (invokes the claude CLI inside the isolated execution "
        "worktree; requires this EXPLICIT flag) -- used for the optional pre-execution implementation "
        "and every post-execution repair round; Claude never runs or authorizes the approved experiment "
        "command itself",
    )
    p_run_experiment.add_argument(
        "--execute", action="store_true",
        help="explicitly request restricted execution of the specification's approved commands; without "
        "this flag the command always stops at EXECUTION_NOT_REQUESTED after planning/validation, "
        "regardless of what Codex or Claude said",
    )
    p_run_experiment.set_defaults(func=cmd_run_experiment)

    p_repair = sub.add_parser(
        "repair",
        help="MVP3: plan -> attempt 0 -> deterministic verifier -> bounded (Codex diagnosis -> Claude "
        "repair -> re-verify) loop -> report. Never runs a research experiment.",
    )
    p_repair.add_argument("spec_path")
    p_repair.add_argument("--run-id", default=None)
    p_repair.add_argument(
        "--execution-worktrees-root", default=None,
        help="root directory for the shared execution worktree (default: a sibling directory of "
        "--repo-root, always outside the main repository worktree)",
    )
    p_repair.add_argument(
        "--codex", choices=["mock", "real"], default="mock",
        help="mock (default, offline, no CLI required) or real (invokes the codex CLI, read-only) -- "
        "used for BOTH the planner and the diagnosis role",
    )
    p_repair.add_argument(
        "--claude", choices=["mock", "real"], default="mock",
        help="mock (default, offline) or real (invokes the claude CLI inside the shared isolated "
        "execution worktree; requires this EXPLICIT flag) -- used for BOTH the initial implementation "
        "and every repair round",
    )
    p_repair.add_argument(
        "--max-repair-rounds", type=int, default=None,
        help="override repair_limits.max_repair_rounds from the spec (default: the spec's own value, "
        "itself defaulting to 2)",
    )
    p_repair.add_argument("--max-total-claude-invocations", type=int, default=None)
    p_repair.add_argument("--max-total-codex-invocations", type=int, default=None)
    p_repair.add_argument("--max-total-changed-files", type=int, default=None)
    p_repair.add_argument("--max-total-changed-bytes", type=int, default=None)
    p_repair.add_argument("--max-wall-clock-seconds", type=int, default=None)
    p_repair.set_defaults(func=cmd_repair)

    p_repair_status = sub.add_parser("repair-status", help="show the status of a previous `repair` run")
    p_repair_status.add_argument("run_id")
    p_repair_status.set_defaults(func=cmd_repair_status)

    p_repair_resume = sub.add_parser(
        "repair-resume",
        help="conservatively attempt to resume an interrupted `repair` run; refuses whenever "
        "continuing cannot be judged safe (see cmd_repair_resume)",
    )
    p_repair_resume.add_argument("run_id")
    p_repair_resume.set_defaults(func=cmd_repair_resume)

    p_status = sub.add_parser("status", help="show the status of a previous run")
    p_status.add_argument("run_id")
    p_status.set_defaults(func=cmd_status)

    p_worktree_status = sub.add_parser("worktree-status", help="show the status of an execute run's execution worktree")
    p_worktree_status.add_argument("run_id")
    p_worktree_status.set_defaults(func=cmd_worktree_status)

    p_worktree_cleanup = sub.add_parser(
        "worktree-cleanup",
        help="remove an execute run's execution worktree (refuses if it has unrecorded changes; "
        "never touches main; branch deletion is opt-in)",
    )
    p_worktree_cleanup.add_argument("run_id")
    p_worktree_cleanup.add_argument("--dry-run", action="store_true", help="report what would happen, remove nothing")
    p_worktree_cleanup.add_argument(
        "--delete-branch", action="store_true", help="also delete the run's throwaway branch (opt-in, off by default)"
    )
    p_worktree_cleanup.set_defaults(func=cmd_worktree_cleanup)

    p_experiment_status = sub.add_parser("experiment-status", help="show the status of a previous `run-experiment` (MVP4) run")
    p_experiment_status.add_argument("run_id")
    p_experiment_status.set_defaults(func=cmd_experiment_status)

    p_experiment_cleanup = sub.add_parser(
        "experiment-cleanup",
        help="remove a `run-experiment` (MVP4) run's execution worktree (alias for worktree-cleanup, "
        "which already understands MVP4 reports; refuses if it has unrecorded changes; never touches "
        "main; branch deletion is opt-in)",
    )
    p_experiment_cleanup.add_argument("run_id")
    p_experiment_cleanup.add_argument("--dry-run", action="store_true", help="report what would happen, remove nothing")
    p_experiment_cleanup.add_argument(
        "--delete-branch", action="store_true", help="also delete the run's throwaway branch (opt-in, off by default)"
    )
    p_experiment_cleanup.set_defaults(func=cmd_experiment_cleanup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
