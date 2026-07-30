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
import contextlib
import io
import json
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
from data.transition_logger import compute_pc_stats
from scripts.eval_wm_reranking_full import _sample_grasp
from world_model.train_counterfactual_critic import load_ensemble, score_candidates

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
        "observation.action_perturbation": {
            "dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        IMG_KEY: {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"]},
    }


def get_state(env):
    arm = env.data.qpos[env._arm_qpos_adr].copy()
    grip = env.data.qpos[env._grip_qpos_adr]
    return np.concatenate([arm, [grip]]).astype(np.float32)


def get_action(env):
    # Actuator names are identical to ARM_JOINTS in EnvironmentSoArm (there is
    # no "_act" suffix). The old lookup silently recorded five zeros, yielding
    # a dataset in which ACT could only learn the gripper dimension.
    arm_ctrl = [env.data.ctrl[act_id] for act_id in env._arm_act_ids]
    grip_ctrl = env.data.ctrl[env._grip_act_id]
    return np.array(arm_ctrl + [grip_ctrl], dtype=np.float32)


def _has_actuator(env, name):
    try:
        env.model.actuator(name)
        return True
    except Exception:
        return False


def run_one_episode(env, dataset, obj_key, obj_class, seed, task_str,
                    critic_ensemble=None, k_candidates=10, quiet_grasp=False,
                    perturb_std=0.0, perturb_prob=0.0):
    """Run one baseline (random-CoM + LGGSN rerank) grasp attempt, recording
    every simulation step. Only commits the episode to the dataset if the
    grasp succeeds. Returns True if kept."""
    import mujoco
    rng = np.random.default_rng(seed)

    env.reset_robot()
    env.remove_all_obj()
    cx = float(rng.uniform(-0.06, 0.06))
    # Current -90 degree robot mount workspace; -0.40 was the obsolete mount
    # centre and produced centimetres of descent-IK error.
    cy = -0.30 + float(rng.uniform(-0.04, 0.04))
    obj_id = env.load_obj(obj_class, name=obj_key, pos=[cx, cy, TABLE_TOP_Z + 0.12])
    env._steps(300)

    frames = []
    # MuJoCo runs at 500 Hz (0.002 s timestep), while the dataset declares
    # 20 FPS. Rendering in every physics step was both ~25x too slow and made
    # ACT learn the wrong time scale. Sample at the declared dataset rate.
    sim_dt = float(env.model.opt.timestep)
    render_stride = max(1, int(round(1.0 / (FPS * sim_dt))))
    hook_steps = 0

    def hook():
        nonlocal hook_steps
        hook_steps += 1
        if hook_steps != 1 and hook_steps % render_stride != 0:
            return
        perturb = np.zeros(6, dtype=np.float32)
        if perturb_std > 0.0 and rng.random() < perturb_prob:
            perturb[:5] = rng.normal(0.0, perturb_std, size=5).astype(np.float32)
            for act_id, delta in zip(env._arm_act_ids, perturb[:5]):
                lo, hi = env.model.actuator_ctrlrange[act_id]
                env.data.ctrl[act_id] = np.clip(env.data.ctrl[act_id] + delta, lo, hi)
            perturb[5] = float(rng.normal(0.0, perturb_std * 0.25))
            lo, hi = env.model.actuator_ctrlrange[env._grip_act_id]
            env.data.ctrl[env._grip_act_id] = np.clip(
                env.data.ctrl[env._grip_act_id] + perturb[5], lo, hi)
        obs_img = env.get_obs(pointcloud=False)["image"]
        frames.append({
            "observation.state": get_state(env),
            "action": get_action(env),
            "observation.action_perturbation": perturb,
            IMG_KEY: obs_img,
        })

    env._step_hook = hook
    try:
        com = env.get_obj_com_pos(obj_id)
    except Exception:
        com = env.get_obj_pos(obj_id)
    gx, gy = float(com[0]), float(com[1])
    gz = float(com[2]) + GRASP_Z_TABLE_MARGIN

    if critic_ensemble is None:
        cand6 = np.array([
            float(gx + rng.uniform(-0.06, 0.06)),
            float(gy + rng.uniform(-0.06, 0.06)), gz,
            float(rng.uniform(-np.pi / 2, np.pi / 2)),
            float(rng.uniform(0.04, 0.09)), 0.05], dtype=np.float32)
    else:
        obj_pos = env.get_obj_pos(obj_id).copy()
        candidates = np.stack([
            _sample_grasp(obj_pos, rng) for _ in range(k_candidates)])
        scene = {"object": obj_key, "obj_pos_before": obj_pos,
                 "obj_quat_before": env.get_obj_pose(obj_id)["quaternion"].copy(),
                 "pc_stats_before": compute_pc_stats(
                     env.get_obs(pointcloud=True), obj_id)}
        score, _ = score_candidates(
            scene, [{"candidate_pose": c} for c in candidates],
            critic_ensemble, relative=True)
        cand6 = candidates[int(np.argmax(score))]
    grasp = {"position": cand6[:3].tolist(),
             "rpy": [np.pi, 0.0, float(cand6[3])],
             "width": float(cand6[4])}
    env.set_obj_grasps(obj_id, [cand6.astype(np.float32)], grasp_rects=[])
    try:
        if quiet_grasp:
            with contextlib.redirect_stdout(io.StringIO()):
                success, _, _ = env.pick_obj_by_id(obj_id, grasp_indices=[0])
        else:
            success, _, _ = env.pick_obj_by_id(obj_id, grasp_indices=[0])
    finally:
        env._step_hook = None

    if not success or len(frames) < 5:
        return False

    for f in frames:
        dataset.add_frame({**f, "task": task_str})
    dataset.save_episode()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="pear",
                    help="single object (backward compatible)")
    ap.add_argument("--objects", default="",
                    help="comma-separated objects for one multi-task dataset")
    ap.add_argument("--n-episodes", type=int, default=100,
                    help="target successful episodes PER object")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="maximum attempts PER object; 0 = n-episodes*6")
    ap.add_argument("--out", default="paperA_data/lerobot_datasets/pear_grasp_sim")
    ap.add_argument("--critic-model-dir", default="",
                    help="optional counterfactual critic ensemble for efficient demos")
    ap.add_argument("--critic-variant", default="object_counterfactual")
    ap.add_argument("--k-candidates", type=int, default=10)
    ap.add_argument("--quiet-grasp", action="store_true",
                    help="suppress verbose primitive/contact diagnostics")
    ap.add_argument("--perturb-std", type=float, default=0.0,
                    help="controller-space Gaussian std for recovery augmentation")
    ap.add_argument("--perturb-prob", type=float, default=0.0,
                    help="per-frame probability of applying perturbation")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_episodes = 2

    sys.path.insert(0, "paperA_data/scripts")
    import _lerobot_groot_patch  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    obj_keys = ([x.strip() for x in args.objects.split(",") if x.strip()]
                if args.objects else [args.object])
    unknown = [x for x in obj_keys if x not in OBJECTS]
    if unknown:
        ap.error(f"unknown objects {unknown}; valid={sorted(OBJECTS)}")

    out_root = Path(args.out)
    if out_root.exists():
        print(f"[WARN] {out_root} already exists -- lerobot will refuse to overwrite. "
              f"Remove it first or pick a new --out path.")
        return

    dataset = LeRobotDataset.create(
        repo_id=f"local/{'-'.join(obj_keys)}_grasp_sim",
        fps=FPS,
        features=build_features(),
        root=str(out_root),
        robot_type="mujoco_soarm",
        use_videos=True,
    )

    env = EnvironmentSoArm(vis=False, debug=False, grasp_mode=GRASP_MODE_PHYSICS_WELD)
    env.preload_pool([OBJECTS[k].replace("Ycb", "") for k in obj_keys])
    critic_ensemble = (load_ensemble(Path(args.critic_model_dir), args.critic_variant)
                       if args.critic_model_dir else None)
    run_config = {
        "objects": obj_keys, "episodes_per_object": args.n_episodes,
        "max_attempts_per_object": args.max_attempts,
        "critic_model_dir": args.critic_model_dir,
        "critic_variant": args.critic_variant,
        "k_candidates": args.k_candidates,
        "perturb_std": args.perturb_std, "perturb_prob": args.perturb_prob,
        "ik_topdown_bias": os.environ.get("IK_TOPDOWN_BIAS", "0.0"),
        "fps": FPS, "sim_timestep": float(env.model.opt.timestep),
        "action_source": "env.data.ctrl via cached actuator ids",
    }
    (out_root / "collection_config.json").write_text(json.dumps(run_config, indent=2))

    totals = {}
    seed = 1000  # separate seed range from the lggsn_candidates_v9 dataset's 1-200
    try:
        for obj_key in obj_keys:
            obj_class = OBJECTS[obj_key]
            task_str = f"Pick up the {obj_key} and place it in the tray"
            n_kept, n_attempts = 0, 0
            max_attempts = args.max_attempts or args.n_episodes * 6
            while n_kept < args.n_episodes and n_attempts < max_attempts:
                n_attempts += 1
                seed += 1
                kept = run_one_episode(
                    env, dataset, obj_key, obj_class, seed, task_str,
                    critic_ensemble, args.k_candidates, args.quiet_grasp,
                    args.perturb_std, args.perturb_prob)
                if kept:
                    n_kept += 1
                    print(f"  [{obj_key} {n_kept}/{args.n_episodes}] kept "
                          f"(seed={seed}, attempt {n_attempts})")
            totals[obj_key] = {"kept": n_kept, "attempts": n_attempts}
    finally:
        env.close()

    print(f"Done -> {out_root}")
    for obj_key, row in totals.items():
        print(f"  {obj_key}: {row['kept']} successful episodes / "
              f"{row['attempts']} attempts "
              f"({row['kept']/max(1,row['attempts']):.1%})")


if __name__ == "__main__":
    main()
