"""Command-line entry point.

    python -m research_agent.cli validate experiments/example_smoke.yaml
    python -m research_agent.cli plan experiments/example_smoke.yaml
    python -m research_agent.cli smoke experiments/example_smoke.yaml
    python -m research_agent.cli status <run_id>

`plan` and `smoke` default to mock Codex/Claude adapters (no CLI required,
fully offline). Pass `--agents real` to use the real `codex`/`claude` CLIs
once they are installed and authenticated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from research_agent.agents.claude import ClaudeAgent, MockClaudeAgent, RealClaudeAgent
from research_agent.agents.codex import CodexAgent, MockCodexAgent, RealCodexAgent
from research_agent.flow import plan_flow, smoke_flow
from research_agent.tasks import experiment as experiment_tasks

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2
EXIT_ERROR = 3

_STATUS_EXIT_CODES = {"PASS": EXIT_OK, "FAIL": EXIT_FAIL, "BLOCKED": EXIT_BLOCKED}


_REAL_AGENTS_WARNING = (
    "WARNING: --agents real selected. This spawns the actual `codex` and `claude` "
    "CLIs as subprocess agents. This path is EXPERIMENTAL and UNVERIFIED -- no "
    "automated test in this repository exercises it, and it has never been run "
    "end-to-end (see .agents/tasks/active/tango_agent_mvp0.md, Reconciliation "
    "note). Codex still runs read-only and Claude is still confined to an "
    "isolated worktree, but the CLI output parsing and error handling on this "
    "path are unverified. Proceed only if `codex` and `claude` are installed, "
    "authenticated, and you understand these guarantees."
)


def _build_agents(kind: str) -> tuple[CodexAgent, ClaudeAgent]:
    if kind == "real":
        print(_REAL_AGENTS_WARNING, file=sys.stderr)
        return RealCodexAgent(), RealClaudeAgent()
    return MockCodexAgent(), MockClaudeAgent()


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
    codex_agent, _ = _build_agents(args.agents)
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
    codex_agent, claude_agent = _build_agents(args.agents)
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
        "--agents", choices=["mock", "real"], default="mock",
        help="mock (default, offline, no CLI required) or real (invokes the codex/claude CLIs)",
    )
    p_plan.set_defaults(func=cmd_plan)

    p_smoke = sub.add_parser("smoke", help="run the full ten-stage smoke-experiment pipeline")
    p_smoke.add_argument("spec_path")
    p_smoke.add_argument("--run-id", default=None)
    p_smoke.add_argument("--agents", choices=["mock", "real"], default="mock")
    p_smoke.set_defaults(func=cmd_smoke)

    p_status = sub.add_parser("status", help="show the status of a previous run")
    p_status.add_argument("run_id")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
