"""Codex CLI adapter -- read-only planner and reviewer roles.

Real planner invocation:

    codex exec --sandbox read-only --ephemeral \\
        --output-schema <run_dir>/plan.schema.json \\
        --output-last-message <run_dir>/plan.raw.json \\
        - < <run_dir>/prompts/codex_planner_prompt.md

Codex is only an advisor: a PLAN_PASS verdict never causes anything to
execute by itself. Every command it proposes is re-checked from scratch by
research_agent.policies.commands.validate_plan_commands before the executor
stage is ever reached, and the deterministic smoke runner only ever executes
the exact command array taken verbatim from the experiment specification.

The JSON schema is generated from research_agent.models at call time
(PlanResult.model_json_schema()) so it is always in sync with the Pydantic
model, rather than a hand-maintained schema file living somewhere else.
Because PlanResult inherits StrictModel (extra="forbid"), the generated
schema also sets "additionalProperties": false, so an unknown field in
Codex's output is rejected by both the JSON schema Codex was given AND by
Pydantic validation on the Python side (belt and suspenders -- the Python
side is authoritative; a real `codex exec` invocation is not guaranteed to
enforce --output-schema itself).

Every terminal failure below is raised as one of two kinds of exception, and
callers must not conflate them:

  * subprocess_runner.ExecutableNotFoundError / CommandTimeoutError -- both
    InfrastructureError subclasses. These are the ONLY failures the flow's
    existing retry_condition_fn (research_agent.flow._retry_only_infrastructure)
    retries; nothing here changes that policy, it only lets RealCodexAgent
    raise the already-existing, already-retried exception types.
  * CodexPlannerError -- everything else (nonzero exit, authentication
    failure, missing/malformed/invalid output, task_id/run_id mismatch).
    A plain RuntimeError subclass, so the existing retry_condition_fn never
    retries it: these are research/policy outcomes, not infrastructure
    flakiness, exactly like a nonzero smoke-command exit code is a FAIL, not
    a retry trigger (see subprocess_runner's module docstring).

research_agent.flow._run_codex_planner is what turns either kind of
exception into a terminal PLAN_BLOCKED PlanResult (never an uncaught
exception, never a state left implicitly "running") -- see that function
for the CODEX_* code mapping.
"""
from __future__ import annotations

import abc
import json
import shutil
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from research_agent import subprocess_runner
from research_agent.models import CommandResult, DiagnosisResult, PlanResult, ReviewResult


def _strict_output_schema(model_cls) -> dict:
    """PlanResult/ReviewResult.model_json_schema() only lists fields
    WITHOUT a Python-side default in "required" (e.g. issues/artifacts/
    proposed_commands are optional in Pydantic because they default to
    []). The real `codex exec` CLI forwards --output-schema verbatim as an
    OpenAI structured-output response_format, which enforces its own
    stricter rule: "required" must list every key in "properties" (a
    schema is invalid otherwise -- confirmed against a live `codex`
    invocation, which returns HTTP 400 invalid_json_schema without this).
    This does not change what Python considers optional -- it only changes
    the JSON Schema file handed to Codex so its own request is accepted;
    PlanResult.model_validate() on the response is still authoritative."""
    schema = model_cls.model_json_schema()
    schema["required"] = list(schema.get("properties", {}).keys())
    return schema


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


class CodexPlannerError(RuntimeError):
    """A non-infrastructure real-Codex planning failure: nonzero exit,
    authentication failure, missing/malformed output, schema violation, or
    task_id/run_id mismatch. Never retried -- see module docstring."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


_AUTH_STDERR_PATTERNS = (
    "not authenticated",
    "unauthorized",
    "401",
    "login required",
    "please run `codex login`",
    "please log in",
    "invalid api key",
    "authentication failed",
    "auth token",
    "no credentials",
)


def _looks_like_auth_failure(returncode: Optional[int], stderr_text: str) -> bool:
    if not returncode:
        return False
    lowered = stderr_text.lower()
    return any(pattern in lowered for pattern in _AUTH_STDERR_PATTERNS)


class RealCodexAgent(CodexAgent):
    """Spawns the real `codex` CLI as a read-only, ephemeral planning
    subprocess. Enabled with `--codex real` on the CLI once `codex` is
    installed and authenticated in the target environment; every scenario
    below is covered offline by fake `codex` executables in
    tests/test_real_codex_planner.py, never a real Codex installation."""

    def __init__(self, binary: str = "codex"):
        self._binary = binary

    def plan(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> PlanResult:
        run_dir = Path(run_dir)
        prompts_dir = run_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompts_dir / "codex_planner_prompt.md"
        prompt_path.write_text(prompt)

        schema_path = run_dir / "plan.schema.json"
        schema_path.write_text(json.dumps(_strict_output_schema(PlanResult), indent=2) + "\n")

        plan_raw_path = run_dir / "plan.raw.json"
        if plan_raw_path.exists():
            plan_raw_path.unlink()

        resolved_binary = shutil.which(self._binary)
        command = [
            resolved_binary or self._binary,
            "exec",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--output-schema",
            str(schema_path.resolve()),
            "--output-last-message",
            str(plan_raw_path.resolve()),
            "-",
        ]

        commands_dir = run_dir / "commands"
        commands_dir.mkdir(parents=True, exist_ok=True)
        (commands_dir / "codex_planner.command.json").write_text(json.dumps(command, indent=2) + "\n")

        try:
            result = subprocess_runner.run_command(
                command, cwd=cwd, run_dir=commands_dir, name="codex_planner", timeout=timeout,
                input_path=prompt_path,
            )
        except subprocess_runner.ExecutableNotFoundError as e:
            self._write_launch_failure_artifacts(commands_dir, command, cwd, resolved_binary, message=str(e))
            raise
        except subprocess_runner.CommandTimeoutError:
            self._write_completed_or_timeout_artifacts(commands_dir, resolved_binary)
            raise

        self._write_completed_or_timeout_artifacts(commands_dir, resolved_binary, result=result)

        if result.returncode != 0:
            stderr_text = Path(result.stderr_path).read_text()
            if _looks_like_auth_failure(result.returncode, stderr_text):
                raise CodexPlannerError(
                    "CODEX_AUTHENTICATION_FAILED",
                    f"codex exec exited {result.returncode}; see commands/codex_planner.stderr",
                )
            raise CodexPlannerError(
                "CODEX_NONZERO_EXIT",
                f"codex exec exited {result.returncode}; see commands/codex_planner.stderr",
            )

        if not plan_raw_path.exists():
            raise CodexPlannerError("CODEX_OUTPUT_MISSING", f"no output written to {plan_raw_path}")

        raw_text = plan_raw_path.read_text()
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise CodexPlannerError("CODEX_OUTPUT_MALFORMED", f"plan.raw.json is not valid JSON: {e}") from e

        try:
            plan = PlanResult.model_validate(data)
        except ValidationError as e:
            raise CodexPlannerError("CODEX_SCHEMA_INVALID", str(e)) from e

        if plan.task_id != task_id:
            raise CodexPlannerError(
                "CODEX_TASK_ID_MISMATCH", f"expected task_id={task_id!r}, got {plan.task_id!r}"
            )
        if plan.run_id != run_id:
            raise CodexPlannerError(
                "CODEX_RUN_ID_MISMATCH", f"expected run_id={run_id!r}, got {plan.run_id!r}"
            )

        return plan

    def review(self, *, prompt, run_dir, cwd, timeout, task_id, run_id) -> ReviewResult:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = run_dir / "reviewer_prompt.md"
        prompt_path.write_text(prompt)
        schema_path = run_dir / "review.schema.json"
        schema_path.write_text(json.dumps(_strict_output_schema(ReviewResult), indent=2) + "\n")
        out_path = run_dir / "review.raw.json"
        resolved_binary = shutil.which(self._binary)
        command = [
            resolved_binary or self._binary,
            "exec",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--output-schema",
            str(schema_path.resolve()),
            "--output-last-message",
            str(out_path.resolve()),
            "-",
        ]
        subprocess_runner.run_command(
            command, cwd=cwd, run_dir=run_dir, name="review_codex", timeout=timeout, input_path=prompt_path,
        )
        if not out_path.exists():
            raise RuntimeError(f"codex review produced no output at {out_path}")
        data = json.loads(out_path.read_text())
        return ReviewResult.model_validate(data)

    def diagnose(
        self, *, prompt: str, run_dir: Path, cwd: Path, timeout: float, task_id: str, run_id: str, attempt_index: int
    ) -> DiagnosisResult:
        """MVP3 diagnosis role: a read-only `codex exec` invocation, exactly
        as read-only/ephemeral as `plan` -- a separate ROLE (different
        prompt, different schema, produces DiagnosisResult not PlanResult),
        not a separate adapter class. Never retried by the caller (see
        research_agent.repair_flow) for anything except the narrow,
        explicit "malformed JSON, first occurrence" case the MVP3 task
        contract permits."""
        run_dir = Path(run_dir)
        prompts_dir = run_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompts_dir / f"codex_diagnosis_{attempt_index:02d}_prompt.md"
        prompt_path.write_text(prompt)

        schema_path = run_dir / f"diagnosis_{attempt_index:02d}.schema.json"
        schema_path.write_text(json.dumps(_strict_output_schema(DiagnosisResult), indent=2) + "\n")

        raw_path = run_dir / f"diagnosis_{attempt_index:02d}.raw.json"
        if raw_path.exists():
            raw_path.unlink()

        resolved_binary = shutil.which(self._binary)
        command = [
            resolved_binary or self._binary,
            "exec",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--output-schema",
            str(schema_path.resolve()),
            "--output-last-message",
            str(raw_path.resolve()),
            "-",
        ]

        commands_dir = run_dir / "commands"
        commands_dir.mkdir(parents=True, exist_ok=True)
        name = f"codex_diagnosis_{attempt_index:02d}"
        (commands_dir / f"{name}.command.json").write_text(json.dumps(command, indent=2) + "\n")

        try:
            result = subprocess_runner.run_command(
                command, cwd=cwd, run_dir=commands_dir, name=name, timeout=timeout, input_path=prompt_path,
            )
        except subprocess_runner.ExecutableNotFoundError as e:
            self._write_launch_failure_artifacts(commands_dir, command, cwd, resolved_binary, message=str(e), name=name)
            raise
        except subprocess_runner.CommandTimeoutError:
            self._write_completed_or_timeout_artifacts(commands_dir, resolved_binary, name=name)
            raise

        self._write_completed_or_timeout_artifacts(commands_dir, resolved_binary, result=result, name=name)

        if result.returncode != 0:
            stderr_text = Path(result.stderr_path).read_text()
            if _looks_like_auth_failure(result.returncode, stderr_text):
                raise CodexPlannerError(
                    "CODEX_AUTHENTICATION_FAILED", f"codex exec exited {result.returncode}; see commands/{name}.stderr",
                )
            raise CodexPlannerError(
                "CODEX_NONZERO_EXIT", f"codex exec exited {result.returncode}; see commands/{name}.stderr",
            )

        if not raw_path.exists():
            raise CodexPlannerError("CODEX_OUTPUT_MISSING", f"no output written to {raw_path}")

        raw_text = raw_path.read_text()
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise CodexPlannerError("CODEX_OUTPUT_MALFORMED", f"{raw_path.name} is not valid JSON: {e}") from e

        try:
            diagnosis = DiagnosisResult.model_validate(data)
        except ValidationError as e:
            raise CodexPlannerError("CODEX_SCHEMA_INVALID", str(e)) from e

        if diagnosis.task_id != task_id:
            raise CodexPlannerError("CODEX_TASK_ID_MISMATCH", f"expected task_id={task_id!r}, got {diagnosis.task_id!r}")
        if diagnosis.run_id != run_id:
            raise CodexPlannerError("CODEX_RUN_ID_MISMATCH", f"expected run_id={run_id!r}, got {diagnosis.run_id!r}")
        if diagnosis.attempt_index != attempt_index:
            raise CodexPlannerError(
                "CODEX_ATTEMPT_INDEX_MISMATCH",
                f"expected attempt_index={attempt_index!r}, got {diagnosis.attempt_index!r}",
            )

        return diagnosis

    @staticmethod
    def _write_launch_failure_artifacts(
        commands_dir: Path, command, cwd, resolved_binary, *, message: str, name: str = "codex_planner"
    ) -> None:
        """The executable could not even be launched: subprocess_runner
        raises before writing any of its usual sidecar files, so we write
        the required artifacts ourselves here."""
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
        commands_dir: Path, resolved_binary, *, result: Optional[CommandResult] = None, name: str = "codex_planner"
    ) -> None:
        """Normal completion (any exit code) always has a CommandResult; a
        timeout does not (subprocess_runner raises instead of returning),
        but it has already written commands/{name}.meta.json to disk
        before raising -- read that back to populate result.json."""
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


class MockCodexAgent(CodexAgent):
    """Deterministic, offline stand-in. Never spawns a subprocess and never
    touches the network -- the smoke flow's default agent, and the only
    Codex adapter any automated test in this repository uses."""

    def __init__(
        self,
        plan_verdict: str = "PLAN_PASS",
        review_verdict: str = "REVIEW_PASS",
        *,
        diagnose_verdict: str = "DIAGNOSE_REPAIRABLE",
        diagnose_failure_class: str = "EXPECTED_CONTENT_MISMATCH",
        diagnose_files_allowed_to_touch: Optional[list[str]] = None,
        diagnose_commands_allowed_to_run: Optional[list[list[str]]] = None,
    ):
        self._plan_verdict = plan_verdict
        self._review_verdict = review_verdict
        # MVP3 diagnosis-role configuration (additive; unused by MVP0/1/2
        # callers, which never call .diagnose()). An empty/None
        # files_allowed_to_touch means "do not narrow the original spec's
        # allowed_modify_paths any further" -- see repair_flow.py's scope
        # validation, which always intersects this against the ORIGINAL
        # spec and never trusts it to expand scope.
        self._diagnose_verdict = diagnose_verdict
        self._diagnose_failure_class = diagnose_failure_class
        self._diagnose_files_allowed_to_touch = diagnose_files_allowed_to_touch
        self._diagnose_commands_allowed_to_run = diagnose_commands_allowed_to_run or []

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

    def diagnose(
        self, *, prompt, run_dir, cwd, timeout, task_id, run_id, attempt_index
    ) -> DiagnosisResult:
        return DiagnosisResult(
            task_id=task_id,
            run_id=run_id,
            attempt_index=attempt_index,
            verdict=self._diagnose_verdict,
            failure_class=self._diagnose_failure_class,
            root_cause="mock diagnosis: canned, deterministic, offline -- see MockCodexAgent(diagnose_*=...)",
            evidence=[],
            repair_instructions=["apply the smallest change that satisfies the verifier's failed checks"],
            files_allowed_to_touch=list(self._diagnose_files_allowed_to_touch or []),
            commands_allowed_to_run=[list(c) for c in self._diagnose_commands_allowed_to_run],
            risks=[],
            assumptions=[],
        )
