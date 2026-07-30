#!/usr/bin/env python3
"""
Phase 3 (real-robot validation, see /home/lena/.claude/plans/floating-crunching-yeti.md):
records the full joint-space trajectory of a CONSENSUS-selected grasp in
simulation, for later replay on the physical SO-ARM101.

Design note (revised 2026-07-10): an earlier version of this script tried to
independently re-derive ui.py's CONSENSUS_N candidate-selection logic
(ensemble draw + weighted-median-nearest selection) standalone. The CFM
ensemble/selection state matched exactly (same winning yaw), but the object
spawn position did not -- tango_robot.ui.RobotEnvUI.__init__ consumes numpy
global-RNG state in ways not fully reproducible from outside (CFM/EBM model
loading, YcbObjects setup, etc., between np.random.seed(orient_seed) and the
actual env.load_isolated_obj() call), so a standalone re-derivation is
fragile. This version instead drives the REAL RobotEnvUI/demo.py pipeline
directly (100% correct by construction, zero re-derived logic) and attaches
robots.TrajectoryRecorder via env._step_hook (env_soarm.py's EnvironmentSoArm
._steps() already checks and calls self._step_hook() if set -- the same
mechanism robots.MujocoBackend.execute_grasp() uses, exposed here without
needing MujocoBackend itself since RobotEnvUI uses EnvironmentSoArm directly,
not the robots/ abstraction layer).

Usage:
    conda run -n tango python paperA_data/scripts/record_consensus_grasp.py \
        --obj Pear --orient-seed 5 --gen-seed-base 1 \
        --out trajs/pear_consensus_orient5_gen1.json
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from robots.trajectory import TrajectoryRecorder
from robots.mujoco_backend import GRIP_CLOSED, GRIP_OPEN


class _EnvAsBackend:
    """Minimal duck-typed adapter so TrajectoryRecorder.snap() can read
    EnvironmentSoArm state directly -- identical logic to
    robots.mujoco_backend.MujocoBackend's get_joint_positions/
    get_gripper_opening/get_eef_pos, without wrapping env in the full
    robots/ abstraction layer (which RobotEnvUI does not use)."""

    _GRIP_TRAVEL_M = 0.10

    def __init__(self, env):
        self._env = env

    def get_joint_positions(self) -> np.ndarray:
        env = self._env
        return np.array([env.data.qpos[adr] for adr in env._arm_qpos_adr], dtype=np.float32)

    def get_gripper_opening(self) -> float:
        env = self._env
        angle = float(env.data.qpos[env._grip_qpos_adr])
        t = np.clip((angle - GRIP_CLOSED) / (GRIP_OPEN - GRIP_CLOSED), 0.0, 1.0)
        return float(t * self._GRIP_TRAVEL_M)

    def get_eef_pos(self) -> np.ndarray:
        return self._env._get_eef_pos().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj", default="Pear")
    ap.add_argument("--orient-seed", type=int, default=5)
    ap.add_argument("--gen-seed-base", type=int, default=1)
    ap.add_argument("--strategy", choices=["consensus", "ikmargin"], default="consensus")
    ap.add_argument("--ensemble-n", type=int, default=10)
    ap.add_argument("--cfm-ckpt", default="grasp_6dof/models/cfm_allobj_ot.pt")
    ap.add_argument("--spawn-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                     help="Override the object's spawn xy after ui.py's default single-object "
                          "placement (0, -0.30) -- see paperA_data/README.md, Phase 3 round 4/5: "
                          "the default spawn direction requires ~90-100 deg of shoulder_pan from "
                          "HOME to reach, which exceeded this project's physical workspace's safe "
                          "rotation range (~60 deg). Use this to record a trajectory whose grasp "
                          "point stays within a physically safe pan angle -- e.g. (0.20, -0.20) "
                          "empirically gives ~50 deg. Real-hardware-pilot-only override, does not "
                          "affect any paper-reported evaluation protocol.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path(
        f"trajs/{args.obj.lower()}_{args.strategy}_orient{args.orient_seed}_gen{args.gen_seed_base}.json")

    # Exactly demo.py's own env-var / seeding sequence for --consensus-n /
    # --ikmargin-n / --gen-seed, applied BEFORE RobotEnvUI construction
    # (matches demo.py's documented requirement: "Must run before
    # RobotEnvUI(...) is constructed (which spawns objects in __init__)").
    os.environ["CFM_CKPT"] = args.cfm_ckpt
    if args.strategy == "consensus":
        os.environ["CONSENSUS_N"] = str(args.ensemble_n)
    else:
        os.environ["IKMARGIN_N"] = str(args.ensemble_n)
    os.environ["GEN_SEED"] = str(args.gen_seed_base)
    os.environ["OWG_NO_SEMANTIC"] = "1"
    os.environ["OWG_GATE_DELTA"] = "0.0"
    os.environ["OWG_MC_GATE_DELTA"] = "0.0"
    np.random.seed(args.orient_seed)

    from tango.utils.config import load_config
    from tango_robot.ui import RobotEnvUI

    cfg = load_config("./config/mujoco/env.yaml")  # demo.py's own default config path

    cfg.seed = args.orient_seed
    cfg.prompt = args.obj
    cfg.backend = "mujoco"
    if hasattr(cfg.policy, "verbose"):
        cfg.policy.verbose = 1

    demo = RobotEnvUI(cfg, backend="mujoco")
    demo.user_input = cfg.prompt

    if args.spawn_xy is not None:
        import mujoco
        sx, sy = args.spawn_xy
        oid = demo.env.obj_ids[0]
        slot = demo.env._obj_pool_slot(oid)
        jnt = demo.env.model.joint(f"obj_joint_{slot}")
        adr, vadr = jnt.qposadr[0], jnt.dofadr[0]
        z = demo.env.data.qpos[adr + 2]  # keep current height, only override xy
        demo.env.data.qpos[adr:adr + 2] = [sx, sy]
        demo.env.data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(demo.env.model, demo.env.data)
        demo.env.dummy_simulation_steps(100)  # same settle convention as ui.py's own spawn
        settled = demo.env.get_obj_pos(oid)
        print(f"[record-consensus] spawn override -> requested=({sx},{sy}), "
              f"settled CoM={settled}")

    recorder = TrajectoryRecorder()
    backend_adapter = _EnvAsBackend(demo.env)
    recorder.begin(metadata={
        "backend": "mujoco", "obj_name": args.obj, "strategy": args.strategy,
        "orient_seed": args.orient_seed, "gen_seed_base": args.gen_seed_base,
    })
    demo.env._step_hook = lambda: recorder.snap(backend_adapter)

    # Tee stdout so we can detect "Done pick" (same success signal every
    # other script in this session greps for) while still printing live.
    import io

    class _Tee(io.TextIOBase):
        def __init__(self, *streams): self._streams = streams
        def write(self, s):
            for st in self._streams:
                st.write(s)
            return len(s)
        def flush(self):
            for st in self._streams:
                st.flush()

    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, buf)
    try:
        demo.run(initial_query=cfg.prompt, once=True)
    finally:
        sys.stdout = real_stdout
        demo.env._step_hook = None

    success = "Done pick" in buf.getvalue()
    traj = recorder.end(success=success)
    demo.env.close()
    print(f"[record-consensus] success={success}")

    if traj.n_points == 0:
        print("[record-consensus] no steps captured (recorder never active during a _steps() call) -- nothing saved.")
        return

    saved_path = traj.save(out_path)
    print(f"[record-consensus] saved -> {saved_path}  "
          f"({traj.n_points} points, {traj.duration:.2f}s)")


if __name__ == "__main__":
    main()
