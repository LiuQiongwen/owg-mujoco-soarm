#!/usr/bin/env python3
"""
Iteration 3 diagnostic — feature-augmented ranking.

Tests whether adding dist_to_centroid (grasp x,y distance from episode centroid)
and z_rel (grasp height relative to episode mean) improves candidate separation.

Strategy:
  augmented_score = lggsn_score + alpha * (1 - dist_norm) + beta * z_rel_norm
  where dist_norm and z_rel_norm are min-max normalised within the episode.

Runs only 9 trials: seeds [1,2,3] x [MustardBottle, Scissors, TomatoSoupCan].
Reports:
  - per-episode feature spread (before/after augmentation)
  - ranking_improvement / ranking_regression / net vs Stage-3 baseline
"""

import cv2, json, os, time
import numpy as np
from PIL import Image

from owg.policy import OwgPolicy
from owg.utils.config import load_config
from owg.utils.grasp import Grasp2D
from owg_robot.camera import Camera
from owg_robot.env import Environment
from owg_robot.objects import YcbObjects
from third_party.grconvnet import load_grasp_generator

# ── config ────────────────────────────────────────────────────────────────────
SEEDS   = [1, 2, 3]
PROMPTS = ["MustardBottle", "Scissors", "TomatoSoupCan"]
N_OBJ   = 8
CFG_PATH = "config/pyb/env.yaml"
OUT_PATH = "logs/diag_iter3.jsonl"

# Augmentation weights — start conservative
ALPHA = 0.3   # weight for (1 - dist_to_centroid_norm): prefer grasps near centroid
BETA  = 0.2   # weight for z_rel_norm: prefer higher grasps
# ─────────────────────────────────────────────────────────────────────────────


def make_camera(cfg):
    return Camera(
        (cfg.camera.center_x, cfg.camera.center_y, cfg.camera.center_z),
        (cfg.camera.target_x, cfg.camera.target_y, cfg.camera.target_z),
        cfg.camera.znear, cfg.camera.zfar,
        (cfg.camera.img_size, cfg.camera.img_size), cfg.camera.fov,
    )


def make_scene(cfg, seed):
    camera = make_camera(cfg)
    env = Environment(camera, vis=False, asset_root="./owg_robot/assets",
                      finger_length=cfg.finger_length,
                      n_grasp_attempts=cfg.n_grasp_attempts)
    objects = YcbObjects("./owg_robot/assets/ycb_objects",
                         mod_orn=["ChipsCan", "MustardBottle", "TomatoSoupCan"],
                         mod_stiffness=["Strawberry"], seed=seed)
    objects.shuffle_objects()
    pinned = [n for n in PROMPTS if n in objects.obj_names]
    rest   = [n for n in objects.obj_names if n not in PROMPTS]
    objects.obj_names = (pinned + rest)[:N_OBJ]
    for obj_name in objects.obj_names:
        path, mod_orn, mod_stiff = objects.get_obj_info(obj_name)
        env.load_isolated_obj(path, obj_name, mod_orn, mod_stiff)
        env.dummy_simulation_steps(30)
    env.dummy_simulation_steps(10)
    init_state = env.get_obj_states()
    obs = env.get_obs()
    return env, init_state, obs


def setup_grasps(env, grasp_gen, n_grasps, obs):
    rgb, depth, seg = obs["image"], obs["depth"], obs["seg"]
    img_size = grasp_gen.IMG_WIDTH
    w = env.camera.width
    for obj_id in env.obj_ids:
        mask = seg == obj_id
        if img_size != w:
            rgb_r   = cv2.resize(rgb,   (img_size, img_size))
            depth_r = cv2.resize(depth, (img_size, img_size))
            mask_r  = np.array(Image.fromarray(mask).resize(
                                (img_size, img_size), Image.LANCZOS))
        else:
            rgb_r, depth_r, mask_r = rgb, depth, mask
        grasps, grasp_rects = grasp_gen.predict_grasp_from_mask(
            rgb_r, depth_r, mask_r, n_grasps=n_grasps, show_output=False)
        if img_size != w:
            for j, gr in enumerate(grasp_rects):
                grasp_rects[j][0] = int(gr[0] / img_size * w)
                grasp_rects[j][1] = int(gr[1] / img_size * w)
                grasp_rects[j][4] = int(gr[4] / img_size * w)
                grasp_rects[j][3] = int(gr[3] / img_size * w)
        grasp_rects = [Grasp2D.from_vector(
            x=g[1], y=g[0], w=g[4], h=g[3], theta=g[2],
            W=w, H=w, normalized=False, line_offset=5) for g in grasp_rects]
        env.set_obj_grasps(obj_id, grasps, grasp_rects)


def refresh(env, init_state, grasp_gen, n_grasps):
    env.reset_robot()
    env.set_obj_state(init_state)
    env.dummy_simulation_steps(30)
    env.update_obj_states()
    obs = env.get_obs()
    setup_grasps(env, grasp_gen, n_grasps, obs)
    env.dummy_simulation_steps(10)
    return obs


def _norm(arr):
    """Min-max normalise; return zeros if constant."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def augmented_rank(lggsn_scores, grasp_xyz):
    """
    Combine LGGSN score with two new spatial features:
      1. proximity to centroid (prefer grasps near the object centre)
      2. relative height z_rel (prefer higher grasps — more clearance)

    Returns augmented score array and a feature-spread dict for diagnostics.
    """
    lggsn = np.array(lggsn_scores, dtype=float)
    xyz   = np.array(grasp_xyz, dtype=float)   # [N, 3]

    # feature 1: dist to xy centroid
    cent   = xyz[:, :2].mean(axis=0)
    dists  = np.linalg.norm(xyz[:, :2] - cent, axis=1)
    prox   = _norm(1.0 - dists)                # higher = closer to centroid

    # feature 2: normalised z (height)
    z_rel  = _norm(xyz[:, 2])

    aug    = lggsn + ALPHA * prox + BETA * z_rel

    spreads = {
        "lggsn_spread":  float(lggsn.max() - lggsn.min()),
        "dist_spread":   float(dists.max() - dists.min()),
        "z_spread":      float(xyz[:, 2].max() - xyz[:, 2].min()),
        "aug_spread":    float(aug.max() - aug.min()),
    }
    return aug, spreads


def run_trial(env, policy, obs, prompt, stage, seed, use_augmented=False):
    target_present = prompt in env.obj_names
    all_grasps = {int(k): env.get_obj_grasps(k) for k in env.obj_ids}

    obj_grasps        = []
    n_grasps          = None
    lggsn_scores      = None
    feature_spreads   = None
    ranking_changed   = False

    if target_present:
        try:
            tid        = env.get_obj_id_by_name(prompt)
            obj_grasps = all_grasps.get(tid, [])
            n_grasps   = len(obj_grasps)
            if stage == 4 and getattr(policy, "grasp_ranker", None) and n_grasps > 0:
                order, scores = policy.grasp_ranker.rank(obj_grasps, verbose=False)
                lggsn_scores  = scores.tolist()
        except Exception as e:
            print(f"      [WARN] rank failed: {e}")

    try:
        action = policy.predict(obs, prompt, all_grasps,
                                obj_names=getattr(env, "obj_names", None),
                                env_obj_ids=getattr(env, "obj_ids", None))
    except Exception as e:
        print(f"      [WARN] predict raised: {e}")
        action = {"action": "fail"}

    if stage == 4 and lggsn_scores and n_grasps:
        if use_augmented:
            grasp_xyz = []
            for g in obj_grasps:
                if isinstance(g, (list, tuple)) and len(g) >= 3:
                    grasp_xyz.append([float(g[0]), float(g[1]), float(g[2])])
                else:
                    grasp_xyz.append([0.0, 0.0, 0.0])
            aug_scores, feature_spreads = augmented_rank(lggsn_scores, grasp_xyz)
            final_order = np.argsort(-aug_scores).tolist()
        else:
            final_order = np.argsort(-np.array(lggsn_scores)).tolist()
            # compute spreads for comparison even in non-augmented mode
            grasp_xyz = []
            for g in obj_grasps:
                if isinstance(g, (list, tuple)) and len(g) >= 3:
                    grasp_xyz.append([float(g[0]), float(g[1]), float(g[2])])
                else:
                    grasp_xyz.append([0.0, 0.0, 0.0])
            _, feature_spreads = augmented_rank(lggsn_scores, grasp_xyz)

        action["grasps"] = final_order
        ranking_changed  = (final_order[0] != 0) if final_order else False

    if action["action"] == "fail":
        root_cause = "target_not_in_scene" if not target_present else "grounding_failed"
        success = False
    else:
        ok_grasp, ok_target = env.put_obj_in_tray(
            action["input"], grasp_indices=action.get("grasps", []))
        for _ in range(30):
            env.step_simulation()
        success = bool(ok_grasp and ok_target)
        root_cause = "success" if success else ("grasp_failed" if not ok_grasp else "place_failed")

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": stage, "seed": seed, "prompt": prompt,
        "use_augmented": use_augmented,
        "n_grasps": n_grasps,
        "lggsn_scores": lggsn_scores,
        "feature_spreads": feature_spreads,
        "ranking_changed": ranking_changed,
        "success": success, "root_cause": root_cause,
    }


def main():
    open(OUT_PATH, "w").close()
    cfg = load_config(CFG_PATH)

    shared_camera = make_camera(cfg)
    grasp_gen     = load_grasp_generator(shared_camera)

    policy_s3 = OwgPolicy(cfg.policy.config_path, verbose=False, vis=False,
                           use_grasp_ranker=False)
    policy_s4 = OwgPolicy(cfg.policy.config_path, verbose=False, vis=False,
                           use_grasp_ranker=True)

    results = []

    for seed in SEEDS:
        print(f"\n{'='*56}\n  seed={seed}")
        env, init_state, obs = make_scene(cfg, seed)
        setup_grasps(env, grasp_gen, cfg.n_grasp_attempts, obs)
        print(f"  scene: {env.obj_names}")

        for prompt in PROMPTS:
            # Stage 3 baseline
            obs = refresh(env, init_state, grasp_gen, cfg.n_grasp_attempts)
            r3  = run_trial(env, policy_s3, obs, prompt, stage=3, seed=seed)
            r3["scene_id"] = seed
            results.append(r3)
            with open(OUT_PATH, "a") as f:
                f.write(json.dumps(r3) + "\n")
            flag = "✓" if r3["success"] else "✗"
            print(f"  [S3] {prompt:<16} {flag}")

            # Stage 4 — LGGSN only (no augmentation)
            obs  = refresh(env, init_state, grasp_gen, cfg.n_grasp_attempts)
            r4   = run_trial(env, policy_s4, obs, prompt, stage=4, seed=seed,
                             use_augmented=False)
            r4["scene_id"] = seed
            results.append(r4)
            with open(OUT_PATH, "a") as f:
                f.write(json.dumps(r4) + "\n")
            flag = "✓" if r4["success"] else "✗"
            fs   = r4.get("feature_spreads") or {}
            print(f"  [S4] {prompt:<16} {flag}  "
                  f"lggsn_spread={fs.get('lggsn_spread',0):.3f}  "
                  f"dist_spread={fs.get('dist_spread',0):.4f}  "
                  f"z_spread={fs.get('z_spread',0):.4f}")

            # Stage 4 — augmented ranking
            obs  = refresh(env, init_state, grasp_gen, cfg.n_grasp_attempts)
            r4a  = run_trial(env, policy_s4, obs, prompt, stage=4, seed=seed,
                             use_augmented=True)
            r4a["scene_id"] = seed
            results.append(r4a)
            with open(OUT_PATH, "a") as f:
                f.write(json.dumps(r4a) + "\n")
            flag = "✓" if r4a["success"] else "✗"
            fs   = r4a.get("feature_spreads") or {}
            print(f"  [S4+aug] {prompt:<12} {flag}  "
                  f"aug_spread={fs.get('aug_spread',0):.3f}")

        env.close()

    _print_summary(results)
    print(f"\nDone. {OUT_PATH}")


def _classify(r3, r4):
    changed = r4.get("ranking_changed", False)
    if r3["success"] and r4["success"]:       return "consistent_success"
    if changed and r3["success"] and not r4["success"]: return "ranking_regression"
    if changed and not r3["success"] and r4["success"]: return "ranking_improvement"
    if not changed and r3["success"] != r4["success"]:  return "non_determinism"
    return "consistent_failure"


def _print_summary(results):
    from collections import defaultdict
    print(f"\n{'='*60}")
    print("SUMMARY")

    s3_idx = {(r["scene_id"], r["prompt"]): r
              for r in results if r["stage"] == 3}

    for aug_flag, label in [(False, "S4 (LGGSN only)"), (True, "S4 (augmented)")]:
        s4_recs = [r for r in results
                   if r["stage"] == 4 and r.get("use_augmented") == aug_flag]
        counts = defaultdict(int)
        for r in s4_recs:
            r3 = s3_idx.get((r["scene_id"], r["prompt"]))
            if r3:
                counts[_classify(r3, r)] += 1
        succ = sum(r["success"] for r in s4_recs)
        s3_succ = sum(r["success"] for r in results if r["stage"] == 3)
        print(f"\n  {label}:  {succ}/{len(s4_recs)} success")
        for k in ["consistent_success","ranking_improvement","non_determinism",
                  "consistent_failure","ranking_regression"]:
            print(f"    {k:<26}: {counts[k]}")
        net = counts["ranking_improvement"] - counts["ranking_regression"]
        print(f"    {'net_improvement':<26}: {net:+d}")

    s3_succ = sum(r["success"] for r in results if r["stage"] == 3)
    s3_total = sum(1 for r in results if r["stage"] == 3)
    print(f"\n  Stage 3 baseline:  {s3_succ}/{s3_total} success")

    # Feature spread comparison
    print(f"\n  Feature spread (S4 non-augmented vs augmented):")
    print(f"  {'prompt':<16} {'seed':>4}  {'lggsn':>6}  {'dist':>6}  {'z':>6}  {'aug':>6}")
    for r in results:
        if r["stage"] != 4 or not r.get("feature_spreads"): continue
        fs = r["feature_spreads"]
        tag = "aug" if r.get("use_augmented") else "   "
        print(f"  {r['prompt']:<16} {r['seed']:>4}  "
              f"{fs.get('lggsn_spread',0):>6.3f}  "
              f"{fs.get('dist_spread',0):>6.4f}  "
              f"{fs.get('z_spread',0):>6.4f}  "
              f"{fs.get('aug_spread',0):>6.3f}  {tag}")


if __name__ == "__main__":
    main()
