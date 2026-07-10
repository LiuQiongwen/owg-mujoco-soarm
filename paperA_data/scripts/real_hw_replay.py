"""
Phase 3 (real-robot validation): replay a recorded grasp trajectory on the
physical SO-ARM101.

Reuses real_hw_connect.connect_backend() (same environment workarounds:
datasets/lerobot naming collision, scservo_sdk, calibration file path) --
does not use scripts/replay_trajectory.py's --backend real path, whose
camera fallback (cameras/realsense_stub.RealSenseCamera) raises ImportError
at INSTANTIATION time (not import time) when pyrealsense2 isn't installed,
so its try/except ImportError around the import statement doesn't actually
catch it. TrajectoryReplayer.replay() never calls camera methods (pure
joint-position/gripper replay, no live perception), so the _NullCamera
stub already validated in real_hw_connect.py is sufficient here too.

Usage:
    conda run -n tango python paperA_data/scripts/real_hw_replay.py \
        --traj trajs/pear_consensus_orient5_gen1.json --speed 0.5
"""
import argparse
import sys
sys.path.insert(0, "/lena/projects/OWG-main/paperA_data/scripts")
from real_hw_connect import connect_backend, PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--speed", type=float, default=0.5,
                     help="Playback speed multiplier (default 0.5 = half speed, slower/safer)")
    ap.add_argument("--max-relative-target", type=float, default=30.0)
    args = ap.parse_args()

    # connect_backend() must run BEFORE robots.trajectory is imported --
    # it internally imports lerobot first (while only site-packages is on
    # sys.path) to avoid this project's own datasets/episode.py shadowing
    # the HuggingFace datasets package lerobot needs. Importing robots.*
    # at module top-level here would add the project root to sys.path too
    # early, exactly like the collision documented in real_hw_connect.py.
    backend = connect_backend(max_relative_target=args.max_relative_target)

    sys.path.insert(0, PROJECT_ROOT)
    from robots.trajectory import Trajectory, TrajectoryReplayer

    traj = Trajectory.load(args.traj)
    print(f"[replay] loaded {args.traj}: {traj}")
    print(f"[replay] {traj.n_points} points, {traj.duration:.2f}s recorded, "
          f"speed={args.speed} -> ~{traj.duration/args.speed:.2f}s wall-clock")
    try:
        print("[replay] connected. Starting replay in 3 seconds -- watch the arm...")
        import time
        time.sleep(3)

        replayer = TrajectoryReplayer(speed=args.speed)

        def on_step(idx, pt):
            if idx % 200 == 0:
                print(f"[replay] step {idx}/{traj.n_points}  t={pt.t:.2f}s")

        replayer.replay(traj, backend, on_step=on_step)
        print("[replay] done.")
    finally:
        backend.close()
        print("[replay] disconnected cleanly (torque disabled).")


if __name__ == "__main__":
    main()
