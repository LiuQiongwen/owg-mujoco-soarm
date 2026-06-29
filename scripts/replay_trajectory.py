#!/usr/bin/env python3
"""Replay a recorded grasp trajectory on MuJoCo sim or the physical SO-ARM101.

Loads a Trajectory JSON saved by record_trajectory.py and replays it via
TrajectoryReplayer.  Works with both MujocoBackend (sim verification) and
SOARMRealBackend (hardware deployment).

Usage
-----
    # Verify replay in simulation (default)
    conda run -n owg-mujoco python scripts/replay_trajectory.py \
        --traj trajs/banana_42.json

    # Slow down playback for visual inspection
    conda run -n owg-mujoco python scripts/replay_trajectory.py \
        --traj trajs/banana_42.json --speed 0.5 --vis

    # Replay on real hardware
    conda run -n owg-mujoco python scripts/replay_trajectory.py \
        --traj trajs/banana_42.json --backend real --port /dev/ttyUSB0

    # Real hardware with safety clamp (max 30° per command)
    conda run -n owg-mujoco python scripts/replay_trajectory.py \
        --traj trajs/banana_42.json --backend real --port /dev/ttyUSB0 \
        --max-delta 30
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from robots import MujocoBackend, SOARMRealBackend, Trajectory, TrajectoryReplayer
from robots.trajectory import TrajectoryPoint


# ── Step callback helpers ─────────────────────────────────────────────────────

def _make_step_logger(verbose: bool):
    """Returns a per-step callback that optionally prints joint/eef state."""
    start = [None]

    def on_step(idx: int, pt: TrajectoryPoint) -> None:
        if start[0] is None:
            start[0] = time.time()
        if verbose:
            elapsed = time.time() - start[0]
            print(
                f"  step {idx:4d}  t={pt.t:.3f}s  "
                f"eef=({pt.eef_pos[0]:.3f},{pt.eef_pos[1]:.3f},{pt.eef_pos[2]:.3f})  "
                f"grip={pt.gripper:.3f}m  "
                f"[wall {elapsed:.2f}s]"
            )

    return on_step


def _make_real_step_callback(backend: "SOARMRealBackend", verbose: bool):
    """Step callback for hardware: updates the EEF cache and optionally logs."""
    log = _make_step_logger(verbose)

    def on_step(idx: int, pt: TrajectoryPoint) -> None:
        backend.set_eef_pos(pt.eef_pos)
        log(idx, pt)

    return on_step


# ── Mujoco replay ─────────────────────────────────────────────────────────────

def _replay_mujoco(traj: Trajectory, args: argparse.Namespace) -> None:
    from tango_robot.env_soarm import EnvironmentSoArm

    grasp_mode = traj.metadata.get("grasp_mode", "physics_weld_after_bilateral")
    obj_name   = traj.metadata.get("obj_name")
    ycb_name   = traj.metadata.get("ycb_name")

    print(f"[replay] backend=mujoco  grasp_mode={grasp_mode}")

    env     = EnvironmentSoArm(vis=args.vis, grasp_mode=grasp_mode)
    backend = MujocoBackend(env)

    if ycb_name and obj_name:
        from tango_robot.env_soarm import TABLE_TOP_Z
        obj_id = env.load_obj(ycb_name, name=obj_name,
                              pos=[0.0, -0.40, TABLE_TOP_Z + 0.12])
        env._steps(300)
        obj_pos = env.get_obj_pos(obj_id)
        print(f"[replay] object '{obj_name}' settled at "
              f"({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")

    replayer = TrajectoryReplayer(speed=args.speed)
    on_step  = _make_step_logger(args.verbose)

    t0 = time.time()
    print(f"[replay] starting  ({traj.n_points} points, {traj.duration:.2f}s recorded)")
    replayer.replay(traj, backend, on_step=on_step)
    wall = time.time() - t0
    print(f"[replay] done  wall={wall:.2f}s")

    env.close()


# ── Real hardware replay ──────────────────────────────────────────────────────

def _replay_real(traj: Trajectory, args: argparse.Namespace) -> None:
    try:
        from cameras.realsense_stub import RealSenseCamera
    except ImportError:
        # fall back to a null camera if realsense is unavailable
        class RealSenseCamera:   # type: ignore[no-redef]
            def capture(self):
                raise RuntimeError("RealSense not available; install pyrealsense2")

    port = args.port or "/dev/ttyUSB0"
    print(f"[replay] backend=real  port={port}  speed={args.speed}")

    backend = SOARMRealBackend(
        port=port,
        camera=RealSenseCamera(),
        calibration=args.calibration or None,
        max_relative_target=args.max_delta,
    )
    backend.connect()
    print("[replay] connected to hardware")

    replayer = TrajectoryReplayer(speed=args.speed)
    on_step  = _make_real_step_callback(backend, args.verbose)

    t0 = time.time()
    print(f"[replay] starting  ({traj.n_points} points, {traj.duration:.2f}s recorded)")
    replayer.replay(traj, backend, on_step=on_step)
    wall = time.time() - t0
    print(f"[replay] done  wall={wall:.2f}s")

    backend.close()
    print("[replay] hardware disconnected")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay a recorded grasp trajectory in sim or on hardware."
    )
    p.add_argument("--traj",        required=True,
                   help="Path to trajectory JSON saved by record_trajectory.py")
    p.add_argument("--backend",     default="mujoco", choices=["mujoco", "real"],
                   help="Replay backend: 'mujoco' (sim) or 'real' (hardware)")
    p.add_argument("--speed",       type=float, default=1.0,
                   help="Playback speed multiplier (0.5 = half speed, 2.0 = double)")
    p.add_argument("--vis",         action="store_true",
                   help="Enable MuJoCo viewer (mujoco backend only)")
    p.add_argument("--verbose",     action="store_true",
                   help="Print per-step state during replay")
    # real-hardware options
    p.add_argument("--port",        default=None,
                   help="Serial port for real backend (e.g. /dev/ttyUSB0)")
    p.add_argument("--calibration", default=None,
                   help="Path to lerobot calibration JSON (real backend)")
    p.add_argument("--max-delta",   type=float, default=None, dest="max_delta",
                   help="Safety clamp: max degrees per joint per command (real backend)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    traj_path = Path(args.traj)
    if not traj_path.exists():
        print(f"[replay] ERROR: trajectory file not found: {traj_path}")
        sys.exit(1)

    traj = Trajectory.load(traj_path)
    print(
        f"[replay] loaded  {traj_path}\n"
        f"         {traj}"
    )

    if args.backend == "mujoco":
        _replay_mujoco(traj, args)
    else:
        _replay_real(traj, args)


if __name__ == "__main__":
    main()
