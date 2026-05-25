#!/usr/bin/env python3
"""Record replay videos for failed grasp trials.

Loads scene JSON files produced by DiverseBenchmarkRunner, re-executes
the grasp in headless MuJoCo with EGL rendering, and saves MP4 / PNG
frames to results/<run_id>/videos/.

Requires:
  - MUJOCO_GL=egl (set automatically)
  - cv2 (opencv-python) for video encoding

Usage
-----
    MUJOCO_GL=egl python scripts/record_failure_videos.py \\
        --run-dir results/diverse_medium \\
        --max-videos 20

    # All runs, only failures, limit 10 per run:
    MUJOCO_GL=egl python scripts/record_failure_videos.py \\
        --all --max-videos 10 --failures-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


# ── scene recording ───────────────────────────────────────────────────────────

def _try_import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def record_scene(
    scene_path:  Path,
    out_dir:     Path,
    trial_rec:   Optional[dict] = None,
    fps:         int = 30,
    width:       int = 640,
    height:      int = 480,
) -> Optional[Path]:
    """Re-execute a scene and record it.  Returns path to saved video, or None."""
    from owg_robot.env_soarm import (
        EnvironmentSoArm, GRASP_MODE_PHYSICS_WELD,
        TABLE_TOP_Z, GRASP_Z_TABLE_MARGIN,
    )
    from benchmark.runner import OBJECT_REGISTRY

    with open(scene_path) as f:
        state = json.load(f)

    obj_name  = state["object"]
    ycb_name  = state.get("ycb_name") or OBJECT_REGISTRY.get(obj_name, obj_name)
    obj_pos   = state["obj_pos"]
    spawn_yaw = state.get("spawn_yaw", 0.0)

    env = EnvironmentSoArm(
        obj_names  = [ycb_name],
        vis        = False,
        grasp_mode = GRASP_MODE_PHYSICS_WELD,
    )

    frames = []
    try:
        obj_id = env.load_obj(ycb_name, name=obj_name,
                              pos=[obj_pos[0], obj_pos[1], obj_pos[2] + 0.10],
                              yaw=spawn_yaw)
        env._steps(100)

        # restore physics state
        qpos = np.array(state["qpos"], dtype=np.float64)
        qvel = np.array(state["qvel"], dtype=np.float64)
        env.data.qpos[:len(qpos)] = qpos[:len(env.data.qpos)]
        env.data.qvel[:len(qvel)] = qvel[:len(env.data.qvel)]
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        env._steps(20)

        # build grasp parameters from saved state or use CoM-based defaults
        com = env.get_obj_com_pos(obj_id)
        grasp_z  = float(com[2]) + GRASP_Z_TABLE_MARGIN
        pos      = (float(obj_pos[0]), float(obj_pos[1]), grasp_z)
        roll     = 0.0
        opening  = 0.07
        obj_height = max(0.05, float(obj_pos[2]) - TABLE_TOP_Z)

        # renderer setup
        renderer = mujoco.Renderer(env.model, height=height, width=width)

        def _capture():
            renderer.update_scene(env.data, camera="fixed_cam" if "fixed_cam" in
                                   [env.model.cam(i).name for i in range(env.model.ncam)]
                                   else 0)
            img = renderer.render()
            return img[:, :, ::-1].copy()  # RGB → BGR for cv2

        # pre-grasp frame burst
        for _ in range(15):
            env._steps(2)
            frames.append(_capture())

        # execute and capture
        ok, _ = env._execute_grasp(
            pos=pos, roll=roll,
            gripper_opening_length=opening, obj_height=obj_height,
        )

        # post-grasp frame burst
        for _ in range(30):
            env._steps(5)
            frames.append(_capture())

        renderer.close()

    except Exception as e:
        print(f"    [WARN] render failed for {scene_path.name}: {e}")
        return None
    finally:
        env.close()

    if not frames:
        return None

    # save video
    cv2 = _try_import_cv2()
    stem = scene_path.stem
    if cv2 is not None:
        out_path = out_dir / f"{stem}.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frames[0].shape[:2]
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
    else:
        # fallback: save first and last frame as PNG
        out_path = out_dir / f"{stem}_frame0.png"
        try:
            from PIL import Image
            Image.fromarray(frames[0][:, :, ::-1]).save(out_dir / f"{stem}_first.png")
            Image.fromarray(frames[-1][:, :, ::-1]).save(out_dir / f"{stem}_last.png")
        except Exception:
            pass
        out_path = out_dir / f"{stem}_last.png"

    return out_path


# ── trial matching ────────────────────────────────────────────────────────────

def load_failure_scene_paths(run_dir: Path, failures_only: bool = True) -> List[Path]:
    """Return scene paths for failed trials, sorted by object+seed."""
    trials_path = run_dir / "trials.jsonl"
    scenes_dir  = run_dir / "scenes"

    if not scenes_dir.exists():
        return []

    # build set of (object, seed, method) for failures
    failure_keys = set()
    if trials_path.exists() and failures_only:
        with open(trials_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("stability_valid") and not rec.get("success"):
                        failure_keys.add((rec["object"], rec["seed"], rec["method"]))
                except Exception:
                    pass

    all_scenes = sorted(scenes_dir.glob("*.json"))

    if not failures_only or not failure_keys:
        return all_scenes

    # filter to failures only
    matched = []
    for sp in all_scenes:
        try:
            with open(sp) as f:
                s = json.load(f)
            key = (s["object"], s["seed"], s["method"])
            if key in failure_keys:
                matched.append(sp)
        except Exception:
            pass
    return matched


# ── main ──────────────────────────────────────────────────────────────────────

def record_run(run_dir: Path, max_videos: int, failures_only: bool) -> None:
    out_dir = run_dir / "videos"
    out_dir.mkdir(exist_ok=True)

    scenes = load_failure_scene_paths(run_dir, failures_only=failures_only)
    if not scenes:
        print(f"  [skip] no {'failure ' if failures_only else ''}scenes in {run_dir.name}")
        return

    print(f"  {len(scenes)} {'failure ' if failures_only else ''}scenes → recording up to {max_videos}")
    scenes = scenes[:max_videos]

    for i, sp in enumerate(scenes):
        print(f"  [{i+1}/{len(scenes)}] {sp.name} ...", end=" ", flush=True)
        t0 = time.time()
        out = record_scene(sp, out_dir)
        if out:
            print(f"saved → {out.name}  ({time.time()-t0:.1f}s)")
        else:
            print("skipped")

    print(f"  videos → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Record replay videos for failed trials")
    ap.add_argument("--run-dir",      default=None)
    ap.add_argument("--all",          action="store_true",
                    help="Record all results/diverse_* runs")
    ap.add_argument("--max-videos",   type=int, default=20,
                    help="Max videos per run (default: 20)")
    ap.add_argument("--failures-only", action="store_true", default=True,
                    help="Only record failure scenes (default: True)")
    ap.add_argument("--all-trials",   action="store_true",
                    help="Record all trials (overrides --failures-only)")
    args = ap.parse_args()

    failures_only = not args.all_trials

    if args.all:
        results_root = ROOT / "results"
        run_dirs = sorted(results_root.glob("diverse_*"))
        for rd in run_dirs:
            if (rd / "scenes").exists():
                print(f"\n[video] {rd.name}")
                record_run(rd, args.max_videos, failures_only)
    elif args.run_dir:
        rd = Path(args.run_dir)
        print(f"\n[video] {rd.name}")
        record_run(rd, args.max_videos, failures_only)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
