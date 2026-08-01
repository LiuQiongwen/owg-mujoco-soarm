# -*- coding: utf-8 -*-
"""Regression test for a real testability defect found while building the
LGGSN Tier-3 IK pilot (experiments/lggsn_tier3_ik_pilot.yaml): env_soarm.py
used to `import mujoco.viewer` at module level, unconditionally. That
transitively imports glfw, which runs a subprocess-based version probe as an
import-time side effect (tango_robot/env_soarm.py, tango_robot/headless_ik.py).

Under the MVP4 restricted-execution harness's rlimits
(research_agent/restricted_subprocess.py), that probe's fork() failed with
`BlockingIOError: [Errno 11] Resource temporarily unavailable` -- even
though the pilot only imports HOME_QPOS (a plain constant) via
tango_robot.headless_ik and never constructs a viewer (vis=False always).
Fixed by moving `import mujoco.viewer` into the two vis=True call sites in
EnvironmentSoArm, aliased (`import mujoco.viewer as _mj_viewer`) so it
doesn't shadow the module-level `mujoco` name for the rest of that method
(a bare `import mujoco.viewer` inside a function makes `mujoco` local to the
WHOLE function, which broke `_rebuild_model`'s own earlier
`mujoco.MjModel.from_xml_string(...)` call the first time this was fixed).

Needs mujoco -- skips cleanly under the research-agent venv.

Run: conda run -n tango python -m pytest -q tests/test_env_soarm_lazy_viewer_import.py
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("mujoco")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_snippet(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = _REPO_ROOT
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30,
    )


def test_importing_env_soarm_does_not_pull_in_glfw():
    proc = _run_snippet(
        "import sys, tango_robot.env_soarm; "
        "print('glfw' in sys.modules)"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_importing_headless_ik_does_not_pull_in_glfw():
    """headless_ik.py imports HOME_QPOS/ARM_JOINTS/... from env_soarm at
    module level specifically to stay usable in GL-less environments -- the
    exact case this regression test guards."""
    proc = _run_snippet(
        "import sys, tango_robot.headless_ik; "
        "print('glfw' in sys.modules)"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_environment_so_arm_vis_false_construction_still_works():
    """The lazy-import fix must not change vis=False behavior at all."""
    proc = _run_snippet(
        "from tango_robot.env_soarm import EnvironmentSoArm; "
        "env = EnvironmentSoArm(vis=False); "
        "print('OK')"
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_rebuild_model_does_not_shadow_module_level_mujoco_name():
    """Regression for the UnboundLocalError caught while fixing this:
    a bare `import mujoco.viewer` inside _rebuild_model (which also calls
    `mujoco.MjModel.from_xml_string(...)` earlier in the same function) made
    `mujoco` local to the whole method, breaking that earlier call. Calling
    reset_robot() (which re-invokes the reattach-viewer code path with
    vis=False, i.e. it's a no-op there, but exercises the same function body)
    after construction is enough to prove no shadowing bug remains."""
    proc = _run_snippet(
        "from tango_robot.env_soarm import EnvironmentSoArm; "
        "env = EnvironmentSoArm(vis=False); "
        "env.reset_robot(); "
        "print('OK')"
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
