"""Claude Code CLI adapter -- executor role, confined to an isolated worktree.

Real invocation mirrors the pattern already used by this repository's Bash
scaffold (.agents/scripts/orchestrate.sh):

    claude --print --output-format json --permission-mode acceptEdits \\
        --tools Read,Write,Edit,Bash < <run_dir>/executor_prompt.md

`cwd` is always fixed to the isolated worktree directory, never the main
checkout, so any writes the model makes land only inside that worktree. The
caller (research_agent.flow.claude_execute_task) additionally checks the
worktree's changed paths against the specification's allowed/forbidden
paths after every call, for both the real and the mock adapter.
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Sequence

from research_agent import subprocess_runner
from research_agent.models import ImplementationResult


class ClaudeAgent(abc.ABC):
    """Contract for the isolated-worktree executor."""

    @abc.abstractmethod
    def execute(
        self,
        *,
        prompt: str,
        worktree_dir: Path,
        run_dir: Path,
        timeout: float,
        task_id: str,
        run_id: str,
        allowed_paths: Sequence[str],
    ) -> ImplementationResult:
        ...


class RealClaudeAgent(ClaudeAgent):
    """Spawns the real `claude` CLI. Not exercised by any automated test in
    this repository -- enable with `--agents real` on the CLI once `claude`
    is installed and authenticated in the target environment."""

    def __init__(self, binary: str = "claude"):
        self._binary = binary

    def execute(
        self, *, prompt, worktree_dir, run_dir, timeout, task_id, run_id, allowed_paths
    ) -> ImplementationResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = run_dir / "executor_prompt.md"
        prompt_path.write_text(prompt)
        command = [
            self._binary,
            "--print",
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",
            "--tools",
            "Read,Write,Edit,Bash",
        ]
        result = subprocess_runner.run_command(
            command,
            cwd=worktree_dir,
            run_dir=run_dir,
            name="executor_claude",
            timeout=timeout,
            input_path=prompt_path,
        )
        if result.returncode != 0:
            return ImplementationResult(
                task_id=task_id,
                run_id=run_id,
                verdict="IMPLEMENTATION_FAILED",
                summary=f"claude CLI exited with code {result.returncode}",
                issues=["see executor_claude.stderr"],
                artifacts=[],
            )
        try:
            outer = json.loads(Path(result.stdout_path).read_text())
            if isinstance(outer, dict) and isinstance(outer.get("result"), str):
                inner = json.loads(outer["result"])
            else:
                inner = outer
            return ImplementationResult.model_validate(inner)
        except Exception as e:  # malformed CLI output is a failed implementation, not a crash
            return ImplementationResult(
                task_id=task_id,
                run_id=run_id,
                verdict="IMPLEMENTATION_FAILED",
                summary=f"could not parse claude output as ImplementationResult: {e}",
                issues=[str(e)],
                artifacts=[],
            )


class MockClaudeAgent(ClaudeAgent):
    """Deterministic, offline stand-in. Never spawns a subprocess and makes
    no filesystem writes of its own -- per the workflow's core invariant,
    the deterministic smoke-command runner (not this agent, and not any
    LLM) is what ever executes experiment commands."""

    def __init__(self, verdict: str = "IMPLEMENTATION_READY_FOR_REVIEW"):
        self._verdict = verdict

    def execute(
        self, *, prompt, worktree_dir, run_dir, timeout, task_id, run_id, allowed_paths
    ) -> ImplementationResult:
        return ImplementationResult(
            task_id=task_id,
            run_id=run_id,
            verdict=self._verdict,
            summary="mock executor: no source changes required for this smoke specification",
            issues=[],
            artifacts=[],
        )
