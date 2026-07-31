"""Deterministic subprocess execution used by every command the workflow runs.

Every command is an argv array; `subprocess.run` is always called with
`shell=False`. Two distinct failure modes are modeled:

  * InfrastructureError -- the process could not be launched, or it exceeded
    its timeout. This is the ONLY failure the workflow ever retries (see
    research_agent.flow._retry_only_infrastructure).
  * A completed process with a nonzero return code -- returned normally as a
    CommandResult, never raised. A bad exit code (a poor metric, a
    legitimately failing smoke command, ...) is a workflow-level FAIL/BLOCKED
    outcome to be handled by the caller, never a retry trigger.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from research_agent.models import CommandResult

_SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


class InfrastructureError(RuntimeError):
    """Launch failure or timeout -- the only condition the flow retries."""


class ExecutableNotFoundError(InfrastructureError):
    """The command's executable could not be found (FileNotFoundError). A
    distinct subclass so callers (e.g. RealCodexAgent) can map it to a
    specific terminal code without string-sniffing the message -- still an
    InfrastructureError, so existing isinstance/pytest.raises(InfrastructureError)
    checks and the flow's retry_condition_fn are unaffected."""


class CommandTimeoutError(InfrastructureError):
    """The command exceeded its timeout and was killed. A distinct subclass
    for the same reason as ExecutableNotFoundError above."""


def redact_environment(env: Mapping[str, str]) -> dict:
    """Return a copy of env with likely-sensitive values masked, for saving
    an environment snapshot alongside a run without leaking secrets."""
    redacted: dict = {}
    for k in sorted(env):
        v = env[k]
        redacted[k] = "***REDACTED***" if any(m in k.upper() for m in _SENSITIVE_ENV_MARKERS) else v
    return redacted


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    run_dir: Path,
    name: str,
    timeout: float,
    env: Optional[Mapping[str, str]] = None,
    input_path: Optional[Path] = None,
) -> CommandResult:
    """Run `command` (an argv array) with shell=False, saving stdout, stderr,
    and a JSON metadata sidecar under run_dir/{name}.*.

    Raises InfrastructureError if the executable cannot be launched or the
    command times out. Otherwise always returns a CommandResult, even when
    the command's own return code is nonzero.
    """
    if isinstance(command, (str, bytes)):
        raise TypeError("command must be a list of argv tokens, never a shell string")
    command = [str(tok) for tok in command]
    if not command:
        raise ValueError("command must not be empty")

    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / f"{name}.stdout"
    stderr_path = run_dir / f"{name}.stderr"
    meta_path = run_dir / f"{name}.meta.json"

    full_env = dict(os.environ)
    if env:
        full_env.update({str(k): str(v) for k, v in env.items()})

    stdin_text: Optional[str] = None
    if input_path is not None:
        stdin_text = Path(input_path).read_text()

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    timed_out = False
    returncode: Optional[int] = None
    stdout_text = ""
    stderr_text = ""
    try:
        proc = subprocess.run(
            command,
            shell=False,
            cwd=str(cwd),
            env=full_env,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        returncode = proc.returncode
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout_text = (e.stdout or "") if isinstance(e.stdout, str) else ""
        stderr_text = ((e.stderr or "") if isinstance(e.stderr, str) else "") + (
            f"\n[research_agent] command timed out after {timeout}s\n"
        )
    except FileNotFoundError as e:
        raise ExecutableNotFoundError(f"executable not found: {command[0]!r}: {e}") from e
    except OSError as e:
        raise InfrastructureError(f"failed to launch command {command!r}: {e}") from e

    duration = round(time.monotonic() - t0, 4)
    ended = datetime.now(timezone.utc)

    stdout_path.write_text(stdout_text)
    stderr_path.write_text(stderr_text)

    result = CommandResult(
        name=name,
        command=command,
        cwd=str(cwd),
        returncode=returncode,
        timed_out=timed_out,
        duration_seconds=duration,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
    )
    meta_path.write_text(result.model_dump_json(indent=2) + "\n")

    if timed_out:
        raise CommandTimeoutError(f"command '{name}' timed out after {timeout}s: {command}")

    return result
