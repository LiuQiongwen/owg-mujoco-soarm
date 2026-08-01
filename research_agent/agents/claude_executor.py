"""Real Claude Code CLI adapter -- MVP2 execute-flow executor role, confined
to an isolated execution worktree (research_agent.execution_worktree).

Real invocation contract (discovered from `claude --help` on the installed
2.1.220 build, per the MVP2 task's "Inspect the installed Claude CLI
contract first" instruction -- see the module-level comment below for the
exact command array):

    claude -p --output-format json --permission-mode acceptEdits \\
        --tools Read,Write,Edit --strict-mcp-config --no-session-persistence \\
        --json-schema <ExecutorImplementationResult JSON Schema>
        < <run_dir>/prompts/claude_executor_prompt.md

`-p`/`--print` with no positional prompt argument reads the prompt from
stdin (mirrors the existing RealCodexAgent pattern in agents/codex.py,
which already passes prompts via subprocess_runner's `input_path`).
`--tools Read,Write,Edit` is the safety-critical flag: Claude is never
given Bash, so it cannot literally invoke a forbidden command (sudo,
docker, git push, training, ...) no matter what the prompt says --
research_agent.policies.execution_policy's command-policy gate still exists
and is exercised (against whatever `commands_run` Claude *claims* in its
structured response) purely as defense in depth / an audit trail, per the
task's "do not trust commands merely because Claude reports them" rule.
`--strict-mcp-config` (no `--mcp-config` given) additionally guarantees zero
MCP servers are loaded, so Claude cannot reach an MCP-exposed Codex/Claude
tool either. `--no-session-persistence` avoids writing a session transcript
under the user's global `~/.claude` state.

Every terminal failure is raised as one of two exception kinds, exactly
mirroring agents/codex.py's RealCodexAgent (see that module's docstring for
the retry-vs-terminal rationale, unchanged here):

  * subprocess_runner.ExecutableNotFoundError / CommandTimeoutError --
    infrastructure failures, mapped by the caller (execute_flow.py) to
    CLAUDE_EXECUTABLE_UNAVAILABLE / CLAUDE_TIMEOUT.
  * ClaudeExecutorError -- everything else (nonzero exit, authentication
    failure, missing/malformed/schema-invalid output, task_id/run_id
    mismatch). Never retried: these are research/policy outcomes, not
    infrastructure flakiness.

research_agent.execute_flow is what turns either kind of exception into a
terminal ExecuteReport (never an uncaught exception, never a state left
implicitly "running").
"""
from __future__ import annotations

import abc
import json
import shutil
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from research_agent import subprocess_runner
from research_agent.models import CommandResult, ExecutionWorktreeRecord, ExecutorImplementationResult, PlanResult, RunPaths
from research_agent.policies import execution_policy


def build_executor_prompt(
    *,
    spec,
    plan: PlanResult,
    worktree_record: ExecutionWorktreeRecord,
    run_paths: RunPaths,
) -> str:
    """Full MVP2 executor prompt: every field required by the task contract
    (task/run ID, the validated Codex plan, the experiment specification,
    allowed/forbidden paths, the change-count/byte budget, required tests,
    the exact command allow/deny lists, the execution worktree path and
    base commit, and the required structured final-response schema), plus
    the full explicit do-not list. This prompt is advisory context for the
    model; every constraint it describes is independently, deterministically
    re-checked afterwards by execution_policy.py and execution_worktree.py
    regardless of what Claude does or claims."""
    modify_paths = execution_policy.effective_modify_paths(spec)
    forbidden = list(spec.forbidden_paths) + list(execution_policy.HIGH_RISK_PREFIXES) + list(
        execution_policy.HIGH_RISK_EXACT
    )
    allowed_commands = [list(c) for c in spec.allowed_executor_commands] + [
        ["python", "-m", "compileall", "..."],
        ["python", "-m", "pytest", "<specific test path>"],
        ["git", "status", "--short"],
        ["git", "diff", "--check"],
        ["git", "diff", "--name-status"],
    ]
    schema = ExecutorImplementationResult.model_json_schema()

    return "\n".join([
        "You are the MVP2 executor: a source-code implementation assistant confined to a single",
        "disposable Git worktree. You perform APPROVED SOURCE-CODE MODIFICATION ONLY.",
        "You never run a research experiment, and you are not the deterministic experiment runner.",
        "",
        f"task_id: {spec.task_id}",
        f"run_id: {run_paths.run_id}",
        "",
        "APPROVED GOAL (from the experiment specification):",
        f"  {spec.goal}",
        "",
        "VALIDATED CODEX PLAN (advisory context; you must still independently follow every rule below):",
        f"  plan.verdict: {plan.verdict}",
        f"  plan.summary: {plan.summary}",
        f"  plan.risks: {plan.risks}",
        f"  plan.assumptions: {plan.assumptions}",
        "",
        "EXPERIMENT SPECIFICATION (relevant fields):",
        f"  allowed_paths: {spec.allowed_paths}",
        f"  allowed_modify_paths (your effective write scope): {modify_paths}",
        f"  forbidden_paths (never write here, non-exhaustive -- see the full forbidden list below): {spec.forbidden_paths}",
        f"  max_changed_files: {spec.max_changed_files}",
        f"  max_changed_bytes: {spec.max_changed_bytes}",
        f"  required_executor_checks: {spec.required_executor_checks}",
        "",
        "EXECUTION WORKTREE (the ONLY place you may write; your current working directory is already set to it):",
        f"  execution_worktree_path: {worktree_record.worktree_path}",
        f"  base_commit: {worktree_record.base_commit}",
        f"  branch_name: {worktree_record.branch_name}",
        "",
        f"ALLOWED MODIFICATION PATHS (write only inside these, resolved relative to the worktree root): {modify_paths}",
        f"FORBIDDEN PATHS (never touch, checked independently after you exit -- a violation blocks this run): {sorted(set(forbidden))}",
        "",
        f"COMMANDS YOU MAY RUN (only if you have Bash access at all; you likely do not in this build -- Read/Write/Edit only): {allowed_commands}",
        "COMMANDS YOU MAY NOT RUN (non-exhaustive; also see 'STOP' rules below): "
        "bash -c, sh -c, sudo <anything>, docker/docker-compose, curl, wget, pip install, conda install, "
        "git push, git commit, git reset, git clean, git worktree, rm -rf, any training entry point "
        "(e.g. lerobot_train.py, train_lggsn_pairwise.py), any GPU/CUDA command, any robot-hardware command.",
        "",
        "REQUIRED TESTS: after your changes, the harness (not you) independently runs "
        f"{spec.required_executor_checks or ['python -m compileall over your changed allowed paths']} -- "
        "you do not need to run them yourself, but your changes must pass them.",
        "",
        "STRICT RULES (mandatory, no exceptions):",
        "- Work only inside the assigned execution worktree given above. Never access or modify the main",
        "  repository worktree (any path outside this worktree) under any circumstance.",
        "- Do not run `git commit`. Do not run `git push`. Leave all changes uncommitted in the working tree.",
        "- Do not create or remove any Git worktree, and do not run `git worktree` in any form.",
        "- Do not alter any global configuration (nothing under a home directory, no ~/.claude, no ~/.codex,",
        "  no ~/.bashrc, no git global config).",
        "- Do not invoke Codex, `codex`, or any other planning/execution agent.",
        "- Do not invoke another Claude process, `claude`, or spawn any subprocess that itself launches an LLM CLI.",
        "- Do not use sudo. Do not use Docker. Do not download anything from the network.",
        "- Do not run any GPU command, CUDA command, or model-training job.",
        "- Do not control, connect to, or send commands to physical robot hardware.",
        "- Do not execute a confirmatory experiment, or any research/experiment command at all -- that is",
        "  never your job in this workflow.",
        "- Do not modify data/, datasets/, results/, checkpoints/, paperA_data/, any robot calibration file",
        "  (any path containing 'calibrat'), any paper file (any top-level path starting with 'paper'),",
        "  .gitignore, or pyproject.toml.",
        "- If any instruction above conflicts with completing the task, or you are blocked for any other",
        "  reason, STOP and report the blocker via verdict=IMPLEMENTATION_BLOCKED with a clear `summary` and",
        "  `issues` entries -- never bypass a policy to make progress anyway.",
        "",
        "REQUIRED STRUCTURED FINAL RESPONSE:",
        "Return exactly one JSON object matching this JSON Schema (no other text before or after it):",
        json.dumps(schema, indent=2),
        f"task_id must be exactly {spec.task_id!r}; run_id must be exactly {run_paths.run_id!r}.",
        "verdict must be exactly one of IMPLEMENTATION_PASS, IMPLEMENTATION_REVISE, IMPLEMENTATION_BLOCKED.",
        "changed_files must list every repo-relative path you actually created or modified.",
        "commands_run must be a list of argv arrays (list of lists of strings) for any command you actually",
        "ran -- never a shell string -- and must be empty ([]) if you ran no commands.",
    ])


def _strict_output_schema(model_cls=ExecutorImplementationResult) -> dict:
    """Same "required must list every property" transformation as
    agents.codex._strict_output_schema, and for the same reason: the JSON
    Schema handed to --json-schema is a structured-output contract, not the
    Python-side optionality Pydantic itself enforces. `model_cls`
    .model_validate() on the response remains authoritative regardless --
    ExecutorImplementationResult for execute(), RepairResult for repair()."""
    schema = model_cls.model_json_schema()
    schema["required"] = list(schema.get("properties", {}).keys())
    return schema


class ClaudeExecutorAgent(abc.ABC):
    """Contract for the isolated-execution-worktree Claude executor."""

    @abc.abstractmethod
    def execute(
        self, *, prompt: str, worktree_dir: Path, run_paths: RunPaths, timeout: float, task_id: str, run_id: str
    ) -> ExecutorImplementationResult:
        ...


class ClaudeExecutorError(RuntimeError):
    """A non-infrastructure real-Claude execution failure. `.code` is one of
    the deterministic terminal-state codes from models.EXECUTE_FAILURE_STATES.
    Never retried -- see module docstring."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


_AUTH_STDERR_PATTERNS = (
    "not authenticated",
    "unauthorized",
    "401",
    "login required",
    "please run `claude login`",
    "please log in",
    "invalid api key",
    "authentication failed",
    "auth token",
    "no credentials",
    "please run /login",
)


def _looks_like_auth_failure(returncode: Optional[int], stderr_text: str) -> bool:
    if not returncode:
        return False
    lowered = stderr_text.lower()
    return any(pattern in lowered for pattern in _AUTH_STDERR_PATTERNS)


def _extract_structured_payload(stdout_text: str) -> dict:
    """Claude's --output-format json response is an outer envelope object;
    the final assistant message (matching our --json-schema contract) is
    normally its "result" field, either as a JSON string (parse it again)
    or already as an object. Falls back to treating the outer object itself
    as the payload if it already looks like one (has a "verdict" key) --
    defensive against a CLI version that returns the schema-validated
    object directly instead of wrapping it."""
    outer = json.loads(stdout_text)
    if isinstance(outer, dict):
        result = outer.get("result")
        if isinstance(result, str):
            return json.loads(result)
        if isinstance(result, dict):
            return result
        if "verdict" in outer:
            return outer
    raise ValueError("no structured result payload found in claude output")


class RealClaudeExecutorAgent(ClaudeExecutorAgent):
    """Spawns the real `claude` CLI. Enabled with `--claude real` on the
    CLI once `claude` is installed and authenticated; every scenario below
    is covered offline by fake `claude` executables in
    tests/test_real_claude_executor.py, and exactly one real invocation is
    ever made in this repository's own validation (the single harmless live
    Claude task described in the MVP2 task contract)."""

    def __init__(self, binary: str = "claude"):
        self._binary = binary

    def execute(
        self, *, prompt: str, worktree_dir: Path, run_paths: RunPaths, timeout: float, task_id: str, run_id: str
    ) -> ExecutorImplementationResult:
        run_paths.prompts_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = run_paths.prompts_dir / "claude_executor_prompt.md"
        prompt_path.write_text(prompt)

        resolved_binary = shutil.which(self._binary)
        command = [
            resolved_binary or self._binary,
            "-p",
            "--output-format", "json",
            "--permission-mode", "acceptEdits",
            "--tools", "Read,Write,Edit",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--json-schema", json.dumps(_strict_output_schema()),
        ]

        commands_dir = run_paths.commands_dir
        commands_dir.mkdir(parents=True, exist_ok=True)
        (commands_dir / "claude_executor.command.json").write_text(json.dumps(command, indent=2) + "\n")

        try:
            result = subprocess_runner.run_command(
                command, cwd=worktree_dir, run_dir=commands_dir, name="claude_executor", timeout=timeout,
                input_path=prompt_path,
            )
        except subprocess_runner.ExecutableNotFoundError as e:
            self._write_launch_failure_artifacts(commands_dir, command, worktree_dir, resolved_binary, message=str(e))
            raise
        except subprocess_runner.CommandTimeoutError:
            self._write_completed_or_timeout_artifacts(commands_dir, resolved_binary)
            raise

        self._write_completed_or_timeout_artifacts(commands_dir, resolved_binary, result=result)

        if result.returncode != 0:
            stderr_text = Path(result.stderr_path).read_text()
            if _looks_like_auth_failure(result.returncode, stderr_text):
                raise ClaudeExecutorError(
                    "CLAUDE_AUTHENTICATION_FAILED",
                    f"claude exited {result.returncode}; see commands/claude_executor.stderr",
                )
            raise ClaudeExecutorError(
                "CLAUDE_NONZERO_EXIT", f"claude exited {result.returncode}; see commands/claude_executor.stderr",
            )

        stdout_text = Path(result.stdout_path).read_text()
        if not stdout_text.strip():
            raise ClaudeExecutorError("CLAUDE_OUTPUT_MISSING", "claude produced no stdout output")

        try:
            payload = _extract_structured_payload(stdout_text)
        except (json.JSONDecodeError, ValueError) as e:
            # best-effort raw artifact even on failure: save exactly what
            # claude printed, so a human can inspect why extraction failed
            run_paths.implementation_raw_path.write_text(stdout_text)
            raise ClaudeExecutorError("CLAUDE_OUTPUT_MALFORMED", f"could not parse claude output as JSON: {e}") from e

        run_paths.implementation_raw_path.write_text(json.dumps(payload, indent=2) + "\n")

        try:
            implementation = ExecutorImplementationResult.model_validate(payload)
        except ValidationError as e:
            raise ClaudeExecutorError("CLAUDE_SCHEMA_INVALID", str(e)) from e

        if implementation.task_id != task_id:
            raise ClaudeExecutorError(
                "CLAUDE_TASK_ID_MISMATCH", f"expected task_id={task_id!r}, got {implementation.task_id!r}"
            )
        if implementation.run_id != run_id:
            raise ClaudeExecutorError(
                "CLAUDE_RUN_ID_MISMATCH", f"expected run_id={run_id!r}, got {implementation.run_id!r}"
            )

        return implementation

    def repair(
        self, *, prompt: str, worktree_dir: Path, run_paths: RunPaths, timeout: float, task_id: str, run_id: str,
        attempt_index: int,
    ) -> "RepairResult":
        """MVP3 repair role: runs in the SAME shared execution worktree
        already created for attempt 0 (never a fresh worktree per round --
        `worktree_dir`/`cwd` here is always the caller's one persistent
        execution worktree). Produces RepairResult, not
        ExecutorImplementationResult. The actual Git diff in that shared
        worktree remains authoritative regardless of what this returns --
        see research_agent.repair_flow."""
        from research_agent.models import RepairResult  # local import: avoid a cycle with models importing agents

        run_paths.prompts_dir.mkdir(parents=True, exist_ok=True)
        name = f"claude_repair_{attempt_index:02d}"
        prompt_path = run_paths.prompts_dir / f"{name}_prompt.md"
        prompt_path.write_text(prompt)

        resolved_binary = shutil.which(self._binary)
        command = [
            resolved_binary or self._binary,
            "-p",
            "--output-format", "json",
            "--permission-mode", "acceptEdits",
            "--tools", "Read,Write,Edit",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--json-schema", json.dumps(_strict_output_schema(RepairResult)),
        ]

        commands_dir = run_paths.commands_dir
        commands_dir.mkdir(parents=True, exist_ok=True)
        (commands_dir / f"{name}.command.json").write_text(json.dumps(command, indent=2) + "\n")

        try:
            result = subprocess_runner.run_command(
                command, cwd=worktree_dir, run_dir=commands_dir, name=name, timeout=timeout, input_path=prompt_path,
            )
        except subprocess_runner.ExecutableNotFoundError as e:
            self._write_launch_failure_artifacts(commands_dir, command, worktree_dir, resolved_binary, message=str(e), name=name)
            raise
        except subprocess_runner.CommandTimeoutError:
            self._write_completed_or_timeout_artifacts(commands_dir, resolved_binary, name=name)
            raise

        self._write_completed_or_timeout_artifacts(commands_dir, resolved_binary, result=result, name=name)

        if result.returncode != 0:
            stderr_text = Path(result.stderr_path).read_text()
            if _looks_like_auth_failure(result.returncode, stderr_text):
                raise ClaudeExecutorError(
                    "CLAUDE_AUTHENTICATION_FAILED", f"claude exited {result.returncode}; see commands/{name}.stderr",
                )
            raise ClaudeExecutorError("CLAUDE_NONZERO_EXIT", f"claude exited {result.returncode}; see commands/{name}.stderr")

        stdout_text = Path(result.stdout_path).read_text()
        if not stdout_text.strip():
            raise ClaudeExecutorError("CLAUDE_OUTPUT_MISSING", "claude produced no stdout output")

        raw_path = run_paths.run_dir / f"{name}.raw.json"
        try:
            payload = _extract_structured_payload(stdout_text)
        except (json.JSONDecodeError, ValueError) as e:
            raw_path.write_text(stdout_text)
            raise ClaudeExecutorError("CLAUDE_OUTPUT_MALFORMED", f"could not parse claude output as JSON: {e}") from e

        raw_path.write_text(json.dumps(payload, indent=2) + "\n")

        try:
            repair_result = RepairResult.model_validate(payload)
        except ValidationError as e:
            raise ClaudeExecutorError("CLAUDE_SCHEMA_INVALID", str(e)) from e

        if repair_result.task_id != task_id:
            raise ClaudeExecutorError(
                "CLAUDE_TASK_ID_MISMATCH", f"expected task_id={task_id!r}, got {repair_result.task_id!r}"
            )
        if repair_result.run_id != run_id:
            raise ClaudeExecutorError(
                "CLAUDE_RUN_ID_MISMATCH", f"expected run_id={run_id!r}, got {repair_result.run_id!r}"
            )
        if repair_result.attempt_index != attempt_index:
            raise ClaudeExecutorError(
                "CLAUDE_ATTEMPT_INDEX_MISMATCH",
                f"expected attempt_index={attempt_index!r}, got {repair_result.attempt_index!r}",
            )

        return repair_result

    @staticmethod
    def _write_launch_failure_artifacts(
        commands_dir: Path, command, cwd, resolved_binary, *, message: str, name: str = "claude_executor"
    ) -> None:
        (commands_dir / f"{name}.stdout").write_text("")
        (commands_dir / f"{name}.stderr").write_text(message)
        (commands_dir / f"{name}.exit_code").write_text("")
        payload = {
            "name": name,
            "command": [str(tok) for tok in command],
            "cwd": str(cwd),
            "returncode": None,
            "timed_out": False,
            "duration_seconds": 0.0,
            "started_at": None,
            "ended_at": None,
            "resolved_executable": resolved_binary,
            "error": message,
        }
        (commands_dir / f"{name}.result.json").write_text(json.dumps(payload, indent=2) + "\n")

    @staticmethod
    def _write_completed_or_timeout_artifacts(
        commands_dir: Path, resolved_binary, *, result: Optional[CommandResult] = None, name: str = "claude_executor"
    ) -> None:
        if result is not None:
            exit_code_text = "" if result.returncode is None else str(result.returncode)
            payload = {
                "name": result.name,
                "command": result.command,
                "cwd": result.cwd,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "duration_seconds": result.duration_seconds,
                "started_at": result.started_at,
                "ended_at": result.ended_at,
                "resolved_executable": resolved_binary,
            }
        else:
            meta_path = commands_dir / f"{name}.meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            exit_code_text = ""
            payload = {**meta, "resolved_executable": resolved_binary}
        (commands_dir / f"{name}.exit_code").write_text(exit_code_text)
        (commands_dir / f"{name}.result.json").write_text(json.dumps(payload, indent=2) + "\n")


class MockClaudeExecutorAgent(ClaudeExecutorAgent):
    """Deterministic, offline stand-in -- the execute flow's default
    (`--claude mock`). With no MVP3 write parameters given (the MVP2
    default, unchanged), never spawns a subprocess and makes no filesystem
    writes of its own inside the execution worktree, so a mock-executor run
    always has zero changed files -- existing MVP2 tests rely on exactly
    this behavior. `execute_write_*`/`repair_write_*` are new, additive,
    opt-in MVP3 parameters used only by research_agent.repair_flow's fake-
    agent tests to deterministically simulate "attempt 0 writes wrong
    content, the repair round writes correct content" without a real CLI."""

    def __init__(
        self,
        verdict: str = "IMPLEMENTATION_PASS",
        *,
        execute_write_relpath: Optional[str] = None,
        execute_write_content: Optional[str] = None,
        repair_verdict: str = "REPAIR_PASS",
        repair_write_relpath: Optional[str] = None,
        repair_write_content: Optional[str] = None,
    ):
        self._verdict = verdict
        self._execute_write_relpath = execute_write_relpath
        self._execute_write_content = execute_write_content
        self._repair_verdict = repair_verdict
        self._repair_write_relpath = repair_write_relpath
        self._repair_write_content = repair_write_content

    def execute(
        self, *, prompt, worktree_dir, run_paths, timeout, task_id, run_id
    ) -> ExecutorImplementationResult:
        changed_files: list[str] = []
        if self._execute_write_relpath is not None:
            target = Path(worktree_dir) / self._execute_write_relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self._execute_write_content or "")
            changed_files = [self._execute_write_relpath]
        return ExecutorImplementationResult(
            task_id=task_id,
            run_id=run_id,
            verdict=self._verdict,
            summary="mock executor: no source changes required for this execute-flow specification"
            if not changed_files else f"mock executor: wrote {self._execute_write_relpath}",
            changed_files=changed_files,
        )

    def repair(
        self, *, prompt, worktree_dir, run_paths, timeout, task_id, run_id, attempt_index
    ) -> "RepairResult":
        from research_agent.models import RepairResult  # local import: avoid a cycle with models importing agents

        changed_files: list[str] = []
        if self._repair_write_relpath is not None:
            target = Path(worktree_dir) / self._repair_write_relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self._repair_write_content or "")
            changed_files = [self._repair_write_relpath]
        return RepairResult(
            task_id=task_id,
            run_id=run_id,
            attempt_index=attempt_index,
            verdict=self._repair_verdict,
            summary="mock repair: no changes made" if not changed_files else f"mock repair: wrote {self._repair_write_relpath}",
            changed_files=changed_files,
        )
