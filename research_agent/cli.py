"""Command-line entry point.

    python -m research_agent.cli validate experiments/example_smoke.yaml
    python -m research_agent.cli plan experiments/example_smoke.yaml
    python -m research_agent.cli smoke experiments/example_smoke.yaml
    python -m research_agent.cli status <run_id>

`plan` and `smoke` default to a mock, offline Codex adapter and (for
`smoke`) a mock, offline Claude adapter -- no CLI required. Pass
`--codex real` to plan with the actual `codex` CLI (read-only, ephemeral)
once it is installed and authenticated; Codex is only ever an advisor, and
every command it proposes must still independently pass the Python
command/path policy engine before anything runs.

The real Claude adapter is disabled in this MVP1 build: `--claude real` is
rejected deterministically (REAL_CLAUDE_DISABLED_IN_MVP1) rather than
silently falling back to mock or silently invoking a real agent. `mock` is
the only supported value for `--claude` right now.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from research_agent.agents.claude import ClaudeAgent, MockClaudeAgent
from research_agent.agents.codex import CodexAgent, MockCodexAgent, RealCodexAgent
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
    """Raised when --claude real is requested. Real Claude is not connected
    in this MVP1 build -- see the module docstring."""


def _build_codex_agent(kind: str) -> CodexAgent:
    if kind == "real":
        print(_REAL_CODEX_WARNING, file=sys.stderr)
        return RealCodexAgent()
    return MockCodexAgent()


def _build_claude_agent(kind: str) -> ClaudeAgent:
    if kind == "real":
        raise RealClaudeDisabledError(
            f"{REAL_CLAUDE_DISABLED_CODE}: the real Claude adapter is disabled in this MVP1 "
            "build; pass --claude mock (the only supported value)."
        )
    return MockClaudeAgent()


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
