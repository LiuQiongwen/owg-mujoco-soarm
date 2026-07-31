import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.subprocess_runner import InfrastructureError, redact_environment, run_command


def test_run_command_executes_and_captures_stdout(tmp_path):
    result = run_command(
        [sys.executable, "-c", "print('hello-from-subprocess')"],
        cwd=tmp_path,
        run_dir=tmp_path,
        name="hello",
        timeout=10,
    )
    assert result.returncode == 0
    assert not result.timed_out
    assert "hello-from-subprocess" in Path(result.stdout_path).read_text()
    assert Path(f"{tmp_path}/hello.meta.json").exists()


def test_run_command_never_uses_shell(tmp_path):
    """Commands must be arrays and subprocess.run must always be called with
    shell=False (requirement: never a shell string)."""
    with patch("research_agent.subprocess_runner.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        run_command([sys.executable, "-c", "pass"], cwd=tmp_path, run_dir=tmp_path, name="x", timeout=5)
    _, kwargs = mock_run.call_args
    assert kwargs["shell"] is False
    positional_command = mock_run.call_args[0][0]
    assert isinstance(positional_command, list)


def test_run_command_rejects_shell_string(tmp_path):
    with pytest.raises(TypeError):
        run_command("python -c \"print('hi')\"", cwd=tmp_path, run_dir=tmp_path, name="bad", timeout=5)


def test_run_command_rejects_empty_command(tmp_path):
    with pytest.raises(ValueError):
        run_command([], cwd=tmp_path, run_dir=tmp_path, name="empty", timeout=5)


def test_run_command_nonzero_exit_does_not_raise(tmp_path):
    """A completed process with a bad exit code is a normal result, never
    an exception -- this is what makes it non-retryable (see flow.py's
    _retry_only_infrastructure, which only retries InfrastructureError)."""
    result = run_command(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        cwd=tmp_path, run_dir=tmp_path, name="bad_exit", timeout=10,
    )
    assert result.returncode == 3
    assert not result.timed_out


def test_run_command_missing_executable_raises_infrastructure_error(tmp_path):
    with pytest.raises(InfrastructureError):
        run_command(
            ["definitely_not_a_real_executable_xyz"],
            cwd=tmp_path, run_dir=tmp_path, name="missing", timeout=5,
        )


def test_run_command_timeout_raises_infrastructure_error_and_saves_artifacts(tmp_path):
    with pytest.raises(InfrastructureError):
        run_command(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path, run_dir=tmp_path, name="slow", timeout=0.2,
        )
    meta = json.loads((tmp_path / "slow.meta.json").read_text())
    assert meta["timed_out"] is True
    assert (tmp_path / "slow.stdout").exists()
    assert (tmp_path / "slow.stderr").exists()


def test_run_command_passes_stdin_from_input_path(tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("hello-stdin\n")
    result = run_command(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
        cwd=tmp_path, run_dir=tmp_path, name="stdin_echo", timeout=10, input_path=input_path,
    )
    assert Path(result.stdout_path).read_text().strip() == "hello-stdin"


def test_run_command_passes_extra_env(tmp_path):
    result = run_command(
        [sys.executable, "-c", "import os; print(os.environ.get('RESEARCH_AGENT_TEST_VAR', 'missing'))"],
        cwd=tmp_path, run_dir=tmp_path, name="env_check", timeout=10,
        env={"RESEARCH_AGENT_TEST_VAR": "present"},
    )
    assert Path(result.stdout_path).read_text().strip() == "present"


def test_redact_environment_masks_sensitive_keys():
    env = {"AWS_SECRET_KEY": "topsecret", "API_TOKEN": "abc123", "HOME": "/home/user", "PATH": "/usr/bin"}
    redacted = redact_environment(env)
    assert redacted["AWS_SECRET_KEY"] == "***REDACTED***"
    assert redacted["API_TOKEN"] == "***REDACTED***"
    assert redacted["HOME"] == "/home/user"
    assert redacted["PATH"] == "/usr/bin"
