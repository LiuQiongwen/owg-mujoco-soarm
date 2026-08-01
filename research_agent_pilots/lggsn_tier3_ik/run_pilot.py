# -*- coding: utf-8 -*-
"""LGGSN Tier-3 per-candidate IK reachability pilot -- adapter script.

Exercises the real, committed Tier-3 IK feature code
(EnvironmentSoArm.compute_ik_reachability_per_candidate's underlying solve,
tango_robot/env_soarm.py::_solve_ik_jaw_topdown) via
tango_robot.headless_ik.HeadlessIKSolver.solve_ik_jaw_topdown -- a verbatim
extraction proven bit-for-bit equivalent by
tests/test_headless_ik_topdown_parity.py. HeadlessIKSolver is used instead of
EnvironmentSoArm directly because EnvironmentSoArm._rebuild_model
unconditionally constructs a GL/EGL renderer (mujoco.Renderer(...)) even with
vis=False, which needs >1GB of virtual address space -- more than the MVP4
restricted-execution harness's fixed RLIMIT_AS. HeadlessIKSolver never
touches the renderer at all, so it fits.

This script contains NO IK algorithm of its own and NO fabricated metrics --
it only (a) reads a fixed, committed fixture of candidate poses, (b) calls
the real solver once per candidate, and (c) hands the raw results to
pilot_core (pure stdlib, no mujoco dependency) for schema validation,
aggregation, and the deterministic digest. See experiments/lggsn_tier3_ik_pilot.yaml
for the approved-command spec this runs under.

Writes exactly two artifacts under RESEARCH_AGENT_ARTIFACTS_DIR:
  metrics.json             -- aggregate, objectively-verifiable metrics
  candidate_features.json  -- one record per candidate (input pose + raw IK result)
"""
from __future__ import annotations

import json
import os
import sys
import time

# Must be set before any repository module (tango_robot.*) is imported --
# otherwise Python writes .pyc bytecode-cache files into the repo tree
# (e.g. tango_robot/__pycache__/), which the MVP4 harness's mutation
# detection (research_agent.tasks.repository.capture_repo_fingerprint)
# correctly flags as a POLICY_FAILURE ("main worktree mutated during
# execution") even though __pycache__/ is gitignored -- it checks for ANY
# tree mutation, not just trackable ones. environment_overrides in
# experiments/lggsn_tier3_ik_pilot.yaml also sets PYTHONDONTWRITEBYTECODE=1
# at the process-launch level (belt and suspenders, and the one that
# actually matters if this script is ever invoked without importing this
# module first) -- unlike MALLOC_ARENA_MAX, sys.dont_write_bytecode IS
# re-checked by the import system on every import, not just at interpreter
# startup, so setting it here is not "too late" the way that was.
sys.dont_write_bytecode = True

# "disable": this pilot only calls HeadlessIKSolver (mj_forward/mj_jacSite --
# pure physics, no rendering at all), so no GL backend is needed. Skipping
# GL/EGL initialization entirely avoids a driver thread that EGL/osmesa
# otherwise spin up at import/first-use time -- which, empirically, was
# enough additional thread creation to exceed the MVP4 restricted harness's
# fixed RLIMIT_NPROC=32 on a shared host that already runs >32 processes
# under this user account system-wide (RLIMIT_NPROC is a per-UID ceiling,
# not scoped to this subprocess's own tree).
os.environ.setdefault("MUJOCO_GL", "disable")
# Defense in depth only -- see experiments/lggsn_tier3_ik_pilot.yaml's
# environment_overrides for the mechanism that actually matters. glibc's
# malloc creates its per-thread arenas (each a large mmap reservation) at
# the FIRST nontrivial allocation, which happens during Python/mujoco
# startup, before this script's own top-level code runs -- so
# MALLOC_ARENA_MAX must be set in the process's environment at exec time
# (empirically confirmed: os.environ.setdefault() here alone is too late
# and has no effect). Left here anyway for the direct/manual-run case,
# since it's harmless when redundant with the YAML's environment_overrides.
# Without it, numpy+mujoco's default multi-arena VSZ exceeds 1GB even
# though actual RSS stays under 550MB, which trips the MVP4 restricted
# harness's fixed RLIMIT_AS (research_agent/restricted_subprocess.py
# DEFAULT_LIMITS -- not spec-configurable, so this must be solved on the
# command's own side, not the harness's).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "1")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
# Inserted explicitly (not relying on PYTHONPATH surviving the restricted
# subprocess's environment allowlist) so this script is self-sufficient.
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pilot_core  # noqa: E402  (path setup must run first)

FIXTURE_PATH = os.path.join(_THIS_DIR, "fixtures", "candidate_poses.json")


def main() -> int:
    artifacts_dir = os.environ.get("RESEARCH_AGENT_ARTIFACTS_DIR")
    if not artifacts_dir:
        raise RuntimeError("RESEARCH_AGENT_ARTIFACTS_DIR is not set")
    os.makedirs(artifacts_dir, exist_ok=True)

    t_start = time.monotonic()

    candidates, fixture_digest = pilot_core.load_fixture(FIXTURE_PATH)

    import numpy as np
    from tango_robot.env_soarm import HOME_QPOS
    from tango_robot.headless_ik import HeadlessIKSolver

    solver = HeadlessIKSolver()
    home = np.asarray(HOME_QPOS, dtype=float)

    raw_results = []
    for cand in candidates:
        target = np.array([cand["x"], cand["y"], cand["z"]], dtype=float)
        # reset_to_home=True (default): each candidate is solved
        # independently from HOME_QPOS, exactly as
        # EnvironmentSoArm.compute_ik_reachability_per_candidate does on its
        # scratch MjData -- no candidate's solve depends on another's.
        converged, pe, _oe = solver.solve_ik_jaw_topdown(target, yaw=cand["yaw"])
        solved_q = solver.get_arm_qpos()
        max_joint_delta = float(np.max(np.abs(solved_q - home)))
        raw_results.append({
            "ik_converged": bool(converged),
            "ik_residual": float(pe),
            "max_joint_delta": max_joint_delta,
        })

    candidate_features = pilot_core.build_candidate_features(candidates, raw_results)
    metrics = pilot_core.compute_metrics(candidate_features, fixture_digest)
    metrics["duration_seconds"] = round(time.monotonic() - t_start, 4)

    with open(os.path.join(artifacts_dir, "candidate_features.json"), "w") as f:
        json.dump(candidate_features, f, indent=2)
        f.write("\n")
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(f"[lggsn_tier3_ik_pilot] candidate_count={metrics['candidate_count']} "
          f"converged_count={metrics['converged_count']} "
          f"convergence_rate={metrics['convergence_rate']:.3f} "
          f"pilot_ok={metrics['pilot_ok']} "
          f"duration_seconds={metrics['duration_seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
