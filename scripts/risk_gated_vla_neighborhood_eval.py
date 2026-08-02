"""Live MuJoCo evaluation of local grasp-pose neighborhood robustness.

This script is intentionally separate from the frozen Phase-1 harness. It
constructs one shared candidate pool per scene, evaluates a compact local
neighborhood for every candidate, and executes every nominal/perturbed pose
against a fresh scene reset. Existing result directories are never reused.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from tango_robot.env_soarm import EnvironmentSoArm
from world_model.train_counterfactual_critic import load_ensemble, score_candidates
from scripts.risk_gated_vla_phase1_eval import (
    OBJECTS,
    _DROP_Z,
    _SPREAD_XY,
    EVAL_CENTRE_Y,
    _load_scene,
    _sample_grasp,
    execute_candidate,
    geo_score,
)

PERTURBATIONS = (
    ("nominal", np.zeros(4, dtype=float)),
    ("x_plus_2mm", np.array([0.002, 0.0, 0.0, 0.0])),
    ("x_minus_2mm", np.array([-0.002, 0.0, 0.0, 0.0])),
    ("yaw_plus_2deg", np.array([0.0, 0.0, 0.0, np.deg2rad(2.0)])),
    ("yaw_minus_2deg", np.array([0.0, 0.0, 0.0, -np.deg2rad(2.0)])),
)
SELECTION_MODES = ("point", "mean", "worst_case")


def git_sha(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def build_local_perturbations(candidate_pose: Sequence[float]) -> list[dict]:
    pose = np.asarray(candidate_pose, dtype=float)
    if pose.shape != (6,):
        raise ValueError("candidate_pose must contain 6 values")
    rows = []
    for name, delta in PERTURBATIONS:
        out = pose.copy()
        out[:3] += delta[:3]
        out[3] += delta[3]
        rows.append({"perturbation_type": name, "delta": delta.tolist(), "candidate_pose": out.tolist()})
    return rows


def score_neighborhood(scene: dict, candidate_pose: Sequence[float], bundles, relative: bool) -> dict:
    perturbations = build_local_perturbations(candidate_pose)
    candidates = [{"candidate_pose": row["candidate_pose"]} for row in perturbations]
    scores, uncertainty = score_candidates(scene, candidates, bundles, relative=relative)
    return {
        "perturbations": perturbations,
        "critic_scores": [float(x) for x in scores],
        "uncertainties": [float(x) for x in uncertainty],
        "point_score": float(scores[0]),
        "neighborhood_mean": float(np.mean(scores)),
        "neighborhood_worst_case": float(np.min(scores)),
    }


def select_candidate(scored: list[dict], mode: Literal["point", "mean", "worst_case"]) -> int:
    key = {"point": "point_score", "mean": "neighborhood_mean", "worst_case": "neighborhood_worst_case"}[mode]
    return int(np.argmax([row[key] for row in scored]))


def _build_pool(env, obj_key: str, obj_idx: int, scene_idx: int, base_seed: int, k: int) -> dict:
    seed = (base_seed * 10_000_000 + obj_idx * 100_000 + scene_idx) % (2**32)
    rng = np.random.default_rng(seed)
    cx = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
    cy = EVAL_CENTRE_Y + float(rng.uniform(-0.04, 0.04))
    oid = _load_scene(env, OBJECTS[obj_key], obj_key, cx, cy)
    obj_pos = env.get_obj_pos(oid).copy()
    obj_quat = env.get_obj_pose(oid)["quaternion"].copy()
    obs = env.get_obs(pointcloud=True)
    from data.transition_logger import compute_pc_stats
    pc_stats = compute_pc_stats(obs, oid)
    candidates = np.stack([_sample_grasp(obj_pos, rng) for _ in range(k)])
    return {"seed": int(seed), "cx": cx, "cy": cy, "obj_name": OBJECTS[obj_key],
            "obj_pos": obj_pos, "obj_quat": obj_quat, "pc_stats": pc_stats,
            "candidates": candidates}


def run_scene(env, obj_key: str, obj_idx: int, scene_idx: int, base_seed: int,
              bundles, k: int, relative: bool) -> dict:
    pool = _build_pool(env, obj_key, obj_idx, scene_idx, base_seed, k)
    scene = {"object": obj_key, "obj_pos_before": pool["obj_pos"],
             "obj_quat_before": pool["obj_quat"], "pc_stats_before": pool["pc_stats"]}
    scored = []
    for candidate in pool["candidates"]:
        row = score_neighborhood(scene, candidate, bundles, relative)
        row["geometry_score"] = float(geo_score(candidate, pool["obj_pos"], pool["pc_stats"]))
        scored.append(row)
    selected = {"geometry": int(np.argmax([r["geometry_score"] for r in scored])),
                "point": select_candidate(scored, "point"),
                "mean": select_candidate(scored, "mean"),
                "worst_case": select_candidate(scored, "worst_case")}
    outcomes = []
    for method, idx in selected.items():
        for local in scored[idx]["perturbations"]:
            result = execute_candidate(env, obj_key, pool, np.asarray(local["candidate_pose"]), grasp_debug=False)
            outcomes.append({"method": method, "candidate_idx": idx, **local, **result})
    return {"object": obj_key, "scene_idx": scene_idx, "seed": pool["seed"],
            "selected": selected, "candidate_scores": scored, "outcomes": outcomes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="cracker,mustard,drill")
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=600)
    ap.add_argument("--k-grasps", type=int, default=10)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--variant", default="object_bce")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent.parent
    config = vars(args) | {"git_sha": git_sha(root), "perturbations": [x[0] for x in PERTURBATIONS]}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = EnvironmentSoArm(grasp_mode="physics_weld_after_bilateral", visual=False)
    bundles = load_ensemble(Path(args.model_dir), args.variant)
    records = []
    objects = [x.strip() for x in args.objects.split(",") if x.strip()]
    unknown = sorted(set(objects) - set(OBJECTS))
    if unknown:
        raise SystemExit(f"unknown objects: {unknown}")
    for obj_idx, obj_key in enumerate(objects):
        for scene_idx in range(args.scenes):
            records.append(run_scene(env, obj_key, obj_idx, scene_idx, args.base_seed,
                                     bundles, args.k_grasps, relative=True))
    (out / "scenes.jsonl").write_text("\n".join(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else v) for x in records) + "\n")
    (out / "trial_manifest.json").write_text(json.dumps({"n_scenes": len(records), "git_sha": config["git_sha"]}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
