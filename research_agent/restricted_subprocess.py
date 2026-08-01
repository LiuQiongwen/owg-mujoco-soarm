"""MVP4 restricted CPU-only subprocess runner.

Distinct from research_agent.subprocess_runner (used for the harness's OWN
deterministic commands -- git plumbing, static checks -- which inherits the
full parent environment): this module is the ONLY place an approved
experiment command is ever actually spawned, and it is deliberately more
restrictive:

  * argv array, shell=False (same invariant as subprocess_runner).
  * its own process group (`start_new_session=True`), so a timeout always
    reaches every child the command itself may have spawned, not just the
    immediate process.
  * a controlled environment -- exactly what the caller passes via `env`
    (see research_agent.policies.environment_policy), never a copy of
    os.environ.
  * stdin always disabled (`subprocess.DEVNULL`) -- no MVP4 approved
    command is ever given stdin input.
  * on Linux, practical rlimits (CPU time, address space, file size,
    process count, open-file count) applied in a preexec_fn -- best-effort,
    process-level resource limits, NOT container-level isolation. This
    module makes no claim of container/namespace/cgroup isolation; see the
    MVP4 task contract's "Do not claim container-level isolation" note.
  * terminate-then-kill escalation on timeout, always targeting the whole
    process group via os.killpg -- never just the immediate PID.
  * captured stdout/stderr are truncated at `max_output_bytes`, with the
    truncation flag recorded rather than silently growing an unbounded
    file.

Never raises for a nonzero exit code or a timeout -- both are recorded on
the returned ExecutionCommandResult (`.returncode` / `.timed_out`); only a
genuine launch failure (missing executable, or another OSError) raises.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from research_agent.models import ExecutionCommandResult

# Conservative defaults: a harmless `python -c` one-liner completes in well
# under one CPU-second and a few MB of memory; these leave generous headroom
# while still being a real, enforced ceiling -- see the MVP4 task contract's
# "practical resource limits where possible without sudo" section.
DEFAULT_LIMITS = {
    "cpu_seconds": 10,
    "address_space_bytes": 1_000_000_000,  # 1 GB
    "file_size_bytes": 50_000_000,  # 50 MB
    "max_processes": 32,
    "max_open_files": 256,
}

_TERMINATE_GRACE_SECONDS = 5.0


class RestrictedSubprocessError(RuntimeError):
    """Launch failure -- the process could not be started at all."""


class RestrictedExecutableNotFoundError(RestrictedSubprocessError):
    """A distinct subclass so callers can map it to a specific terminal
    code without string-sniffing the message (mirrors
    subprocess_runner.ExecutableNotFoundError)."""


def _resolve_executable(token: str) -> Optional[str]:
    return shutil.which(token)


def _build_preexec_fn(limits: dict):
    """Linux only -- see the module docstring. Returns None on any other
    platform, in which case no rlimits are applied (process-group isolation
    and terminate/kill escalation still apply everywhere)."""
    if platform.system() != "Linux":
        return None
    import resource

    def _apply() -> None:
        try:
            os.setsid()
        except OSError:
            pass
        cpu = limits.get("cpu_seconds")
        if cpu:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            except (ValueError, OSError):
                pass
        address_space = limits.get("address_space_bytes")
        if address_space:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
            except (ValueError, OSError):
                pass
        file_size = limits.get("file_size_bytes")
        if file_size:
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
            except (ValueError, OSError):
                pass
        max_processes = limits.get("max_processes")
        if max_processes and hasattr(resource, "RLIMIT_NPROC"):
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (max_processes, max_processes))
            except (ValueError, OSError):
                pass
        max_open_files = limits.get("max_open_files")
        if max_open_files:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (max_open_files, max_open_files))
            except (ValueError, OSError):
                pass

    return _apply


def _truncate(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + f"\n... [truncated, output exceeded max_output_bytes={max_bytes}]\n", True


def _terminate_process_group(proc: "subprocess.Popen") -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _kill_process_group(proc: "subprocess.Popen") -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_restricted_command(
    command: Sequence[str],
    *,
    cwd: Path,
    command_dir: Path,
    name: str,
    env: Mapping[str, str],
    timeout: float,
    approved_command: Optional[Sequence[str]] = None,
    working_directory_policy: str = "isolated_run_directory",
    limits: Optional[Mapping[str, int]] = None,
    max_output_bytes: int = 1_000_000,
) -> ExecutionCommandResult:
    if isinstance(command, (str, bytes)):
        raise TypeError("command must be a list of argv tokens, never a shell string")
    command = [str(tok) for tok in command]
    if not command:
        raise ValueError("command must not be empty")

    command_dir = Path(command_dir)
    command_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    effective_limits = dict(DEFAULT_LIMITS)
    if limits:
        effective_limits.update(limits)

    resolved_executable = _resolve_executable(command[0])
    full_env = {str(k): str(v) for k, v in env.items()}
    approved = list(approved_command) if approved_command is not None else list(command)

    (command_dir / "command.json").write_text(json.dumps({
        "approved_command": approved, "executed_command": command, "resolved_executable": resolved_executable,
    }, indent=2) + "\n")
    (command_dir / "environment.json").write_text(json.dumps({"names": sorted(full_env)}, indent=2) + "\n")
    (command_dir / "limits.json").write_text(json.dumps(effective_limits, indent=2) + "\n")

    stdout_path = command_dir / "stdout"
    stderr_path = command_dir / "stderr"
    exit_code_path = command_dir / "exit_code"

    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    timed_out = False
    returncode: Optional[int] = None

    popen_kwargs: dict = dict(
        cwd=str(cwd), env=full_env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    preexec_fn = _build_preexec_fn(effective_limits)
    if preexec_fn is not None:
        popen_kwargs["preexec_fn"] = preexec_fn

    try:
        proc = subprocess.Popen(command, **popen_kwargs)
    except FileNotFoundError as e:
        stdout_path.write_text("")
        stderr_path.write_text(f"executable not found: {command[0]!r}: {e}")
        exit_code_path.write_text("")
        raise RestrictedExecutableNotFoundError(f"executable not found: {command[0]!r}: {e}") from e
    except OSError as e:
        stdout_path.write_text("")
        stderr_path.write_text(f"failed to launch command {command!r}: {e}")
        exit_code_path.write_text("")
        raise RestrictedSubprocessError(f"failed to launch command {command!r}: {e}") from e

    try:
        stdout_text, stderr_text = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(proc)
        try:
            stdout_text, stderr_text = proc.communicate(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            stdout_text, stderr_text = proc.communicate()
        stderr_text = (stderr_text or "") + f"\n[research_agent] command timed out after {timeout}s and was terminated\n"
    else:
        returncode = proc.returncode

    duration = round(time.monotonic() - t0, 4)
    ended = datetime.now(timezone.utc)

    stdout_text, stdout_truncated = _truncate(stdout_text or "", max_output_bytes)
    stderr_text, stderr_truncated = _truncate(stderr_text or "", max_output_bytes)

    stdout_path.write_text(stdout_text)
    stderr_path.write_text(stderr_text)
    exit_code_path.write_text("" if returncode is None else str(returncode))

    result = ExecutionCommandResult(
        name=name, approved_command=approved, executed_command=command, resolved_executable=resolved_executable,
        cwd=str(cwd), working_directory_policy=working_directory_policy, env_names=sorted(full_env),
        returncode=returncode, timed_out=timed_out, duration_seconds=duration,
        stdout_path=str(stdout_path), stderr_path=str(stderr_path),
        stdout_truncated=stdout_truncated, stderr_truncated=stderr_truncated,
        started_at=started.isoformat(), ended_at=ended.isoformat(), limits=effective_limits,
    )
    (command_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result
