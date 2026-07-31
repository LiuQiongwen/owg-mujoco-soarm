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
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from research_agent import execution_worktree
from research_agent.agents.claude import ClaudeAgent, MockClaudeAgent
from research_agent.agents.claude_executor import MockClaudeExecutorAgent, RealClaudeExecutorAgent
from research_agent.agents.codex import CodexAgent, MockCodexAgent, RealCodexAgent
from research_agent.execute_flow import execute_flow
from research_agent.flow import plan_flow, smoke_flow
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


def _load_execution_worktree_record(run_dir: Path) -> dict:
    record_path = run_dir / "execution_worktree.json"
    if not record_path.exists():
        raise FileNotFoundError(f"no execution_worktree.json under {run_dir} (not an `execute` run, or it never reached worktree creation)")
    return json.loads(record_path.read_text())


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

    report_path = run_dir / "report.json"
    report_summary = None
    if report_path.exists():
        report_data = json.loads(report_path.read_text())
        report_summary = {
            "overall_status": report_data.get("overall_status"),
            "terminal_state": report_data.get("terminal_state"),
            "changed_file_count": report_data.get("changed_file_count"),
        }

    print(json.dumps({
        "run_id": record["run_id"],
        "base_commit": record["base_commit"],
        "branch_name": record["branch_name"],
        "worktree_path": record["worktree_path"],
        "worktree_exists": exists,
        "current_status_lines": status_lines,
        "recorded_preserved": record.get("preserved", True),
        "recorded_removed_at": record.get("removed_at"),
        "report_summary": report_summary,
    }, indent=2))
    return EXIT_OK


def cmd_worktree_cleanup(args: argparse.Namespace) -> int:
    """Refuses to remove a worktree with changes not accounted for by a
    saved report.json, never uses `git clean -fdx`, never touches main, and
    only deletes the throwaway branch when --delete-branch is explicitly
    passed. --dry-run reports what would happen without removing anything."""
    run_dir = _runs_root(args) / args.run_id
    try:
        record = _load_execution_worktree_record(run_dir)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        return EXIT_ERROR

    worktree_path = Path(record["worktree_path"])
    repo_root = Path(args.repo_root).resolve()

    if not worktree_path.exists():
        print(json.dumps({"status": "ALREADY_REMOVED", "worktree_path": str(worktree_path)}, indent=2))
        return EXIT_OK

    proc = subprocess.run(
        ["git", "-C", str(worktree_path), "status", "--porcelain=v1", "--untracked-files=all"],
        shell=False, capture_output=True, text=True, timeout=30,
    )
    current_paths = sorted(
        {(line[3:] if len(line) > 3 else line.strip()).split(" -> ", 1)[-1].strip().strip('"')
         for line in proc.stdout.splitlines() if line.strip()}
    )

    report_path = run_dir / "report.json"
    if not report_path.exists():
        if current_paths:
            print(json.dumps({
                "error": "REFUSED_UNRECORDED_CHANGES",
                "reason": "no report.json to compare against, and the worktree has uncommitted changes",
                "current_changed_paths": current_paths,
            }, indent=2), file=sys.stderr)
            return EXIT_ERROR
    else:
        report_data = json.loads(report_path.read_text())
        recorded_paths = sorted(
            {Path(c["path"].rstrip("/")).as_posix() for c in report_data.get("changed_files", [])}
        )
        normalized_current = sorted({Path(p.rstrip("/")).as_posix() for p in current_paths})
        if normalized_current != recorded_paths:
            print(json.dumps({
                "error": "REFUSED_UNRECORDED_CHANGES",
                "reason": "worktree's current changes do not match report.json's recorded changed_files",
                "current_changed_paths": normalized_current,
                "recorded_changed_paths": recorded_paths,
            }, indent=2), file=sys.stderr)
            return EXIT_ERROR

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN",
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
        "worktree_path": str(worktree_path),
        "branch_deleted": bool(args.delete_branch),
    }, indent=2))
    return EXIT_OK


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
