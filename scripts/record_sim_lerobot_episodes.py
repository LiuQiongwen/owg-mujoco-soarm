#!/usr/bin/env python3
"""
Stage 1 of the sim-only BC-pretrain + RWR-fine-tune plan
(paperA_data/new_method_affordance_auxiliary_proposal.md's sibling RL-skill
thread, discussed 2026-07-12): record SUCCESSFUL grasp episodes entirely in
MuJoCo sim as a LeRobot-format dataset (same schema as the real
paperA_data/lerobot_datasets/pear_grasp/, so the existing lerobot_train.py
wrapper works unchanged) -- used later to BC-pretrain an ACT policy, which
is then RWR-fine-tuned using cheap, fast sim rollouts (no real-hardware time,
no real/sim visual-domain mismatch since both stages stay in sim).

Uses the existing Baseline pipeline (random-CoM candidates + LGGSN v5d
reranker, physics_weld_after_bilateral execution) to generate successes --
only SUCCESSFUL episodes are kept as demonstrations, matching standard BC
practice (learn from what worked).

Observation: single "overhead" camera (the only camera the sim currently
has -- no wrist camera in MuJoCo, unlike the real robot's teleop dataset;
fine since this whole pipeline stays sim-internal, no cross-domain transfer
claimed).
State/action: 6-dim (5 arm joints + gripper), same names/order as the real
pear_grasp dataset, but in this env's own native units (radians for the arm
joints, matching env_soarm.py's qpos convention) -- internally consistent,
not claimed to match the real robot's numeric convention.

Usage:
  conda run -n tango python scripts/record_sim_lerobot_episodes.py \
    --object pear --n-episodes 5 --out paperA_data/lerobot_datasets/pear_grasp_sim --smoke
  conda run -n tango python scripts/record_sim_lerobot_episodes.py \
    --object pear --n-episodes 100 --out paperA_data/lerobot_datasets/pear_grasp_sim
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from tango_robot.env_soarm import (
    EnvironmentSoArm, TABLE_TOP_Z, GRASP_MODE_PHYSICS_WELD,
    ARM_JOINTS, GRIP_OPEN, GRASP_Z_TABLE_MARGIN,
)

OBJECTS = {
    "banana": "YcbBanana", "pear": "YcbPear", "mustard": "YcbMustardBottle",
    "cracker": "YcbCrackerBox", "drill": "YcbPowerDrill",
    "can": "YcbTomatoSoupCan", "cylinder": "YcbMediumClamp",
}
JOINT_NAMES = [f"{j}.pos" for j in ARM_JOINTS] + ["gripper.pos"]
FPS = 20
IMG_KEY = "observation.images.overhead"


def build_features():
    return {
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        IMG_KEY: {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"]},
    }


def get_state(env):
    arm = env.data.qpos[env._arm_qpos_adr].copy()
    grip = env.data.qpos[env._grip_qpos_adr]
    return np.concatenate([arm, [grip]]).astype(np.float32)


def get_action(env):
    arm_ctrl = []
    for j in ARM_JOINTS:
        act_id = env.model.actuator(f"{j}_act").id if _has_actuator(env, f"{j}_act") else None
        arm_ctrl.append(env.data.ctrl[act_id] if act_id is not None else 0.0)
    grip_ctrl = env.data.ctrl[env._grip_act_id]
    return np.array(arm_ctrl + [grip_ctrl], dtype=np.float32)


def _has_actuator(env, name):
    try:
        env.model.actuator(name)
        return True
    except Exception:
        return False


def run_one_episode(env, dataset, obj_key, obj_class, seed, task_str):
    """Run one baseline (random-CoM + LGGSN rerank) grasp attempt, recording
    every simulation step. Only commits the episode to the dataset if the
    grasp succeeds. Returns True if kept."""
    import mujoco
    rng = np.random.default_rng(seed)

    env.reset_robot()
    env.remove_all_obj()
    cx = float(rng.uniform(-0.06, 0.06))
    cy = -0.40 + float(rng.uniform(-0.04, 0.04))
    obj_id = env.load_obj(obj_class, name=obj_key, pos=[cx, cy, TABLE_TOP_Z + 0.12])
    env._steps(300)

    frames = []

    def hook():
        obs_img = env.get_obs(pointcloud=False)["image"]
        frames.append({
            "observation.state": get_state(env),
            "action": get_action(env),
            IMG_KEY: obs_img,
        })

    env._step_hook = hook
    try:
        com = env.get_obj_com_pos(obj_id)
    except Exception:
        com = env.get_obj_pos(obj_id)
    gx, gy = float(com[0]), float(com[1])
    gz = float(com[2]) + GRASP_Z_TABLE_MARGIN

    cand = (
        float(gx + rng.uniform(-0.06, 0.06)),
        float(gy + rng.uniform(-0.06, 0.06)),
        float(rng.uniform(-np.pi / 2, np.pi / 2)),
        float(rng.uniform(0.04, 0.09)),
    )
    grasp = {"position": [cand[0], cand[1], gz], "rpy": [np.pi, 0.0, cand[2]], "width": cand[3]}
    env.set_obj_grasps(obj_id, [np.array([cand[0], cand[1], gz, cand[2], cand[3],
                                           0.05], dtype=np.float32)], grasp_rects=[])
    success, _, _ = env.pick_obj_by_id(obj_id, grasp_indices=[0])
    env._step_hook = None

    if not success or len(frames) < 5:
        return False

    for f in frames:
        dataset.add_frame({**f, "task": task_str})
    dataset.save_episode()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="pear")
    ap.add_argument("--n-episodes", type=int, default=100, help="target number of SUCCESSFUL episodes to keep")
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = n-episodes*6 (rough success-rate margin)")
    ap.add_argument("--out", default="paperA_data/lerobot_datasets/pear_grasp_sim")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_episodes = 2

    sys.path.insert(0, "paperA_data/scripts")
    import _lerobot_groot_patch  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    obj_class = OBJECTS[args.object]
    task_str = f"Pick up the {args.object} and place it in the tray"

    out_root = Path(args.out)
    if out_root.exists():
        print(f"[WARN] {out_root} already exists -- lerobot will refuse to overwrite. "
              f"Remove it first or pick a new --out path.")
        return

    dataset = LeRobotDataset.create(
        repo_id=f"local/{args.object}_grasp_sim",
        fps=FPS,
        features=build_features(),
        root=str(out_root),
        robot_type="mujoco_soarm",
        use_videos=True,
    )

    env = EnvironmentSoArm(vis=False, debug=False, grasp_mode=GRASP_MODE_PHYSICS_WELD)
    env.preload_pool([obj_class.replace("Ycb", "")])

    n_kept, n_attempts = 0, 0
    max_attempts = args.max_attempts or args.n_episodes * 6
    seed = 1000  # separate seed range from the lggsn_candidates_v9 dataset's 1-200
    try:
        while n_kept < args.n_episodes and n_attempts < max_attempts:
            n_attempts += 1
            seed += 1
            kept = run_one_episode(env, dataset, args.object, obj_class, seed, task_str)
            if kept:
                n_kept += 1
                print(f"  [{n_kept}/{args.n_episodes}] kept (seed={seed}, attempt {n_attempts})")
    finally:
        env.close()

    print(f"Done. {n_kept} successful episodes kept out of {n_attempts} attempts "
          f"({n_kept/max(1,n_attempts):.1%} success rate) -> {out_root}")


if __name__ == "__main__":
    main()
