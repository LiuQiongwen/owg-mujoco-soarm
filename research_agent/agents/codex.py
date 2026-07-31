"""Codex CLI adapter -- read-only planner and reviewer roles.

Real invocation mirrors the pattern already used by this repository's Bash
scaffold (.agents/scripts/orchestrate.sh):

    codex exec --sandbox read-only \\
        --output-schema <run_dir>/<role>.schema.json \\
        --output-last-message <run_dir>/<role>.raw.json \\
        - < <run_dir>/<role>_prompt.md

The JSON schema is generated from research_agent.models at call time
(PlanResult.model_json_schema() / ReviewResult.model_json_schema()) so it is
always in sync with the Pydantic models, rather than a hand-maintained
schema file living somewhere else.
"""
from __future__ import annotations

import abc
import json
from pathlib import Path

from research_agent import subprocess_runner
from research_agent.models import PlanResult, ReviewResult


class CodexAgent(abc.ABC):
    """Contract for a read-only Codex planner/reviewer. Implementations must
    never modify the repository: the real adapter enforces this with
    `--sandbox read-only`; the mock adapter never spawns a process at all."""

    @abc.abstractmethod
    def plan(
        self, *, prompt: str, run_dir: Path, cwd: Path, timeout: float, task_id: str, run_id: str
    ) -> PlanResult:
        ...

    @abc.abstractmethod
    def review(
        self, *, prompt: str, run_dir: Path, cwd: Path, timeout: float, task_id: str, run_id: str
    ) -> ReviewResult:
        ...


class RealCodexAgent(CodexAgent):
    """Spawns the real `codex` CLI. Not exercised by any automated test in
    this repository -- enable with `--agents real` on the CLI once `codex`
    is installed and authenticated in the target environment."""

    def __init__(self, binary: str = "codex"):
        self._binary = binary

    def _invoke(self, *, role: str, model_cls, prompt: str, run_dir: Path, cwd: Path, timeout: float):
        run_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = run_dir / f"{role}_prompt.md"
        prompt_path.write_text(prompt)
        schema_path = run_dir / f"{role}.schema.json"
        schema_path.write_text(json.dumps(model_cls.model_json_schema(), indent=2) + "\n")
        out_path = run_dir / f"{role}.raw.json"
        command = [
            self._binary,
            "exec",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(out_path),
            "-",
        ]
        subprocess_runner.run_command(
            command,
            cwd=cwd,
            run_dir=run_dir,
            name=f"{role}_codex",
            timeout=timeout,
            input_path=prompt_path,
        )
        if not out_path.exists():
            raise RuntimeError(f"codex {role} produced no output at {out_path}")
        data = json.loads(out_path.read_text())
        return model_cls.model_validate(data)

    def plan(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> PlanResult:
        return self._invoke(
            role="plan", model_cls=PlanResult, prompt=prompt, run_dir=run_dir, cwd=cwd, timeout=timeout
        )

    def review(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> ReviewResult:
        return self._invoke(
            role="review", model_cls=ReviewResult, prompt=prompt, run_dir=run_dir, cwd=cwd, timeout=timeout
        )


class MockCodexAgent(CodexAgent):
    """Deterministic, offline stand-in. Never spawns a subprocess and never
    touches the network -- the smoke flow's default agent, and the only
    Codex adapter any automated test in this repository uses."""

    def __init__(self, plan_verdict: str = "PLAN_PASS", review_verdict: str = "REVIEW_PASS"):
        self._plan_verdict = plan_verdict
        self._review_verdict = review_verdict

    def plan(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> PlanResult:
        return PlanResult(
            task_id=task_id,
            run_id=run_id,
            verdict=self._plan_verdict,
            summary="mock planner: inspected the specification only; proposed no ad hoc commands",
            issues=[],
            artifacts=[],
            proposed_commands=[],
        )

    def review(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> ReviewResult:
        return ReviewResult(
            task_id=task_id,
            run_id=run_id,
            verdict=self._review_verdict,
            summary="mock reviewer: inspected saved verification artifacts; no issues raised",
            issues=[],
            artifacts=[],
        )
