"""MVP4 restricted-subprocess-runner tests: research_agent.restricted_subprocess.

Covers (see the MVP4 task contract's "Fake execution tests" list):
  12. timeout
  13. nonzero exit
  14. stdout capture
  15. stderr capture
  16. output truncation limit
  50. interrupted execution reported safely (timeout -> terminate/kill,
      never raises, always returns a well-formed result)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.restricted_subprocess import (
    RestrictedExecutableNotFoundError,
    run_restricted_command,
)

_ENV = {"PATH": "/usr/bin:/bin"}


def test_normal_command_captures_stdout_and_exit_zero(tmp_path):
    result = run_restricted_command(
        [sys.executable, "-c", "print('hello-mvp4')"],
        cwd=tmp_path, command_dir=tmp_path / "cmd", name="c0", env=_ENV, timeout=5,
    )
    assert result.returncode == 0
    assert not result.timed_out
    assert "hello-mvp4" in Path(result.stdout_path).read_text()


def test_stderr_is_captured(tmp_path):
    result = run_restricted_command(
        [sys.executable, "-c", "import sys; sys.stderr.write('oops\\n')"],
        cwd=tmp_path, command_dir=tmp_path / "cmd", name="c0", env=_ENV, timeout=5,
    )
    assert "oops" in Path(result.stderr_path).read_text()


def test_nonzero_exit_recorded_not_raised(tmp_path):
    result = run_restricted_command(
        [sys.executable, "-c", "import sys; sys.exit(5)"],
        cwd=tmp_path, command_dir=tmp_path / "cmd", name="c0", env=_ENV, timeout=5,
    )
    assert result.returncode == 5
    assert not result.timed_out


def test_timeout_terminates_process_and_is_reported(tmp_path):
    result = run_restricted_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path, command_dir=tmp_path / "cmd", name="c0", env=_ENV, timeout=1,
    )
    assert result.timed_out is True
    assert result.returncode is None
    assert result.duration_seconds < 10


def test_timed_out_process_and_children_are_actually_killed(tmp_path):
    """A timed-out command must not leave the process (or its process
    group) running after this function returns."""
    marker = tmp_path / "still_running.marker"
    script = (
        f"import time, pathlib\n"
        f"p = pathlib.Path({str(marker)!r})\n"
        f"while True:\n"
        f"    p.write_text('alive')\n"
        f"    time.sleep(0.2)\n"
    )
    run_restricted_command(
        [sys.executable, "-c", script], cwd=tmp_path, command_dir=tmp_path / "cmd",
        name="c0", env=_ENV, timeout=1,
    )
    import time as _time
    _time.sleep(1.0)
    mtime_after_kill = marker.stat().st_mtime if marker.exists() else 0
    _time.sleep(1.0)
    mtime_later = marker.stat().st_mtime if marker.exists() else 0
    assert mtime_after_kill == mtime_later, "process kept writing after the runner returned -- it was not actually killed"


def test_output_truncated_at_max_output_bytes(tmp_path):
    result = run_restricted_command(
        [sys.executable, "-c", "print('x' * 1000)"],
        cwd=tmp_path, command_dir=tmp_path / "cmd", name="c0", env=_ENV, timeout=5, max_output_bytes=20,
    )
    assert result.stdout_truncated is True
    assert len(Path(result.stdout_path).read_text().encode()) < 1000


def test_missing_executable_raises_specific_error(tmp_path):
    with pytest.raises(RestrictedExecutableNotFoundError):
        run_restricted_command(
            ["/no/such/executable/at/all"], cwd=tmp_path, command_dir=tmp_path / "cmd",
            name="c0", env=_ENV, timeout=5,
        )


def test_stdin_is_disabled(tmp_path):
    """stdin=DEVNULL: a command that tries to read stdin gets EOF
    immediately rather than hanging on input never approved for this run."""
    result = run_restricted_command(
        [sys.executable, "-c", "import sys; data = sys.stdin.read(); print(repr(data))"],
        cwd=tmp_path, command_dir=tmp_path / "cmd", name="c0", env=_ENV, timeout=5,
    )
    assert result.returncode == 0
    assert "''" in Path(result.stdout_path).read_text()


def test_environment_is_exactly_what_was_passed(tmp_path):
    result = run_restricted_command(
        [sys.executable, "-c", "import os; print(sorted(os.environ.keys()))"],
        cwd=tmp_path, command_dir=tmp_path / "cmd", name="c0",
        env={"PATH": "/usr/bin:/bin", "MY_ONLY_VAR": "x"}, timeout=5,
    )
    stdout = Path(result.stdout_path).read_text()
    assert "MY_ONLY_VAR" in stdout
    assert "HOME" not in stdout  # never inherits the caller's HOME etc.


def test_limits_persisted_to_disk(tmp_path):
    command_dir = tmp_path / "cmd"
    run_restricted_command(
        [sys.executable, "-c", "print(1)"], cwd=tmp_path, command_dir=command_dir,
        name="c0", env=_ENV, timeout=5, limits={"cpu_seconds": 3},
    )
    import json

    limits_on_disk = json.loads((command_dir / "limits.json").read_text())
    assert limits_on_disk["cpu_seconds"] == 3
    assert (command_dir / "command.json").exists()
    assert (command_dir / "environment.json").exists()
    assert (command_dir / "result.json").exists()
    env_names = json.loads((command_dir / "environment.json").read_text())["names"]
    assert "PATH" in env_names


@pytest.mark.skipif(sys.platform != "linux", reason="rlimits are Linux-only in this MVP4 build")
def test_cpu_time_limit_terminates_a_busy_loop(tmp_path):
    result = run_restricted_command(
        [sys.executable, "-c", "x = 0\nwhile True:\n    x += 1"],
        cwd=tmp_path, command_dir=tmp_path / "cmd", name="c0", env=_ENV, timeout=15,
        limits={"cpu_seconds": 1},
    )
    assert result.duration_seconds < 5
    assert not result.timed_out  # killed by RLIMIT_CPU (SIGKILL), not the wall-clock timeout
