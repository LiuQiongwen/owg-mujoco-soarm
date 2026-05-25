"""Scene replay tool for the SO-ARM101 grasp benchmark.

Loads a saved pre-grasp scene state (JSON produced by BenchmarkRunner) and
re-executes a grasp so the failure can be inspected visually or re-evaluated
with different parameters.

Usage
-----
    # CLI — replay a single scene with the viewer open
    python -m benchmark.replay scenes/banana_seed0042_random.json

    # Programmatic
    from benchmark.replay import SceneReplayer
    replayer = SceneReplayer("results/run_001")
    replayer.replay("scenes/banana_seed0042_random.json", vis=True)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


class SceneReplayer:
    """Load saved scene states and replay grasps for inspection.

    Parameters
    ----------
    run_dir  : base run directory (contains scenes/ and trials.jsonl)
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)

    def list_scenes(self) -> list[Path]:
        scenes_dir = self.run_dir / "scenes"
        if not scenes_dir.exists():
            return []
        return sorted(scenes_dir.glob("*.json"))

    def load_state(self, scene_path: str | Path) -> dict:
        with open(scene_path) as f:
            return json.load(f)

    def replay(
        self,
        scene_path:             str | Path,
        vis:                    bool = True,
        grasp_override:         Optional[dict] = None,
        settle_steps:           int  = 50,
    ) -> bool:
        """Replay a saved scene.

        Parameters
        ----------
        scene_path      : path to the scene JSON file
        vis             : open the MuJoCo viewer
        grasp_override  : dict with optional keys: pos, roll, yaw,
                          gripper_opening_length, obj_height
                          (if None, a default centre-object grasp is used)
        settle_steps    : physics steps to run after restoring state

        Returns
        -------
        success : bool
        """
        from owg_robot.env_soarm import (
            EnvironmentSoArm, GRASP_MODE_PHYSICS, TABLE_TOP_Z,
        )

        state = self.load_state(scene_path)
        obj_name = state["object"]
        obj_pos  = state["obj_pos"]

        from benchmark.runner import OBJECT_REGISTRY
        ycb_name = OBJECT_REGISTRY.get(obj_name, obj_name)

        env = EnvironmentSoArm(
            obj_names  = [ycb_name],
            vis        = vis,
            grasp_mode = GRASP_MODE_PHYSICS,
        )
        try:
            obj_id = env.load_obj(ycb_name, name=obj_name,
                                  pos=[obj_pos[0], obj_pos[1], obj_pos[2] + 0.08])
            env._steps(settle_steps)

            # restore exact physics state
            qpos = np.array(state["qpos"], dtype=np.float64)
            qvel = np.array(state["qvel"], dtype=np.float64)
            env.data.qpos[:len(qpos)] = qpos[:len(env.data.qpos)]
            env.data.qvel[:len(qvel)] = qvel[:len(env.data.qvel)]
            import mujoco
            mujoco.mj_forward(env.model, env.data)
            env._steps(10)

            # build grasp parameters
            if grasp_override:
                pos  = grasp_override.get("pos",  (obj_pos[0], obj_pos[1], obj_pos[2] + 0.01))
                roll = grasp_override.get("roll", 0.0)
                opening = grasp_override.get("gripper_opening_length", 0.08)
                height  = grasp_override.get("obj_height", max(0.05, obj_pos[2] - TABLE_TOP_Z))
            else:
                pos     = (float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]) + 0.01)
                roll    = 0.0
                opening = 0.08
                height  = max(0.05, float(obj_pos[2]) - TABLE_TOP_Z)

            print(f"[replay] scene={Path(scene_path).name}  obj={obj_name}")
            print(f"[replay] pos={pos}  roll={roll:.3f}  opening={opening:.3f}")

            ok, _ = env._execute_grasp(
                pos=pos, roll=roll,
                gripper_opening_length=opening, obj_height=height,
            )

            print(f"[replay] result={'SUCCESS' if ok else 'FAIL'}")

            if vis:
                print("[replay] viewer open — press Ctrl-C to close")
                try:
                    while env._viewer and env._viewer.is_running():
                        env._viewer.sync()
                        time.sleep(0.05)
                except KeyboardInterrupt:
                    pass

            return ok
        finally:
            env.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    ap = argparse.ArgumentParser(description="Replay a saved benchmark scene")
    ap.add_argument("scene", help="Path to scene JSON file")
    ap.add_argument("--run-dir", default=None,
                    help="Run directory (default: parent of scene file's scenes/ folder)")
    ap.add_argument("--no-vis", action="store_true", help="Headless mode")
    ap.add_argument("--pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="Override grasp position")
    ap.add_argument("--roll", type=float, default=None)
    ap.add_argument("--opening", type=float, default=None)
    args = ap.parse_args()

    scene_path = Path(args.scene)
    run_dir    = args.run_dir or scene_path.parent.parent

    override = {}
    if args.pos:
        override["pos"] = tuple(args.pos)
    if args.roll is not None:
        override["roll"] = args.roll
    if args.opening is not None:
        override["gripper_opening_length"] = args.opening

    replayer = SceneReplayer(run_dir)
    replayer.replay(scene_path, vis=not args.no_vis,
                    grasp_override=override or None)


if __name__ == "__main__":
    _main()
