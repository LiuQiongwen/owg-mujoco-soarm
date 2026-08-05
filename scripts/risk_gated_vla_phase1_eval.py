#!/usr/bin/env python3
"""
Risk-Gated VLA -- Phase 1: paired candidate-selection evaluation.

Fixes the pairing bug found in results/risk_gated_vla/audit.md Section 4
(scripts/eval_wm_reranking_full.py::trial_seed() encoded `method` into the
RNG seed, so "geometry" and "world_model" never shared a candidate pool).
Here the scene seed depends ONLY on (base_seed, object, scene_idx) -- every
compared method sees the exact same object placement and the exact same
K-candidate pool, built once per scene before any method is chosen.

Design
------
Per scene:
  1. Build the candidate pool once (object placement + K sampled grasp
     poses + geo/wm scores for every candidate) -- nothing executed yet.
  2. For each of {random, geometry, world_critic}: reset the environment to
     a FRESH, identically-placed copy of the same scene, execute that
     method's chosen candidate, record the outcome.
  3. Oracle: execute EVERY candidate in the pool (each against its own
     fresh identical scene reset) -- oracle_success = any candidate
     succeeded. This also gives real per-candidate ground truth for every
     scored candidate, used for the critic's AUROC/AUPRC/ECE/risk-coverage
     metrics (not just top-1).

Cost: K + 3 physics executions per scene (K=10 default -> 13/scene).

Causal validity: gated at import time via
causal_validity_audit.provenance.audit_feature_set() against the exact
field names this script feeds the critic (see WORLD_MODEL_FIELDS in
provenance.py, registered 2026-07-30 during this study's Phase 0 audit).

Usage
-----
  # Smoke test: 2 objects x 3 seeds, verifies pool identity + determinism
  conda run -n tango python scripts/risk_gated_vla_phase1_eval.py \\
      --objects cracker,mustard --seeds 3 --out-dir results/risk_gated_vla/smoke --smoke-check

  # Formal run
  conda run -n tango python scripts/risk_gated_vla_phase1_eval.py \\
      --objects cracker,mustard,drill --seeds 50 --out-dir results/risk_gated_vla/phase1
"""

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from causal_validity_audit.provenance import audit_feature_set

audit_feature_set(
    ["grasp_pose", "obj_pos_before", "obj_quat_before", "pc_stats_before", "pc_stats_local"],
    context="risk_gated_vla_phase1_eval.py candidate-selection features",
)

from tango_robot.env_soarm import EnvironmentSoArm, TABLE_TOP_Z
from data.transition_logger import compute_pc_stats, compute_pc_stats_local
from world_model.train_mlp_predictor import load_model, MODEL_PATH
from world_model.rerank_grasps import score_grasps
from world_model.train_counterfactual_critic import load_ensemble, score_candidates
from scripts.eval_wm_reranking_full import (
    OBJECTS, _SPREAD_XY, _DROP_Z, _SETTLE_STEPS, _FELL_OFF_Z,
    geo_score, _sample_grasp,
)

DEFAULT_K = 10
METHODS = ["random", "geometry", "world_critic"]
# The legacy WM evaluation used y=-0.40 before the robot mount was rotated
# -90 degrees.  Under the current mount that centre gives centimetres of
# descent-IK error.  y=-0.30 is the current validated tabletop target zone.
EVAL_CENTRE_Y = -0.30


# ── Reproducibility metadata ───────────────────────────────────────────────

def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent
        ).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"], cwd=Path(__file__).resolve().parent.parent
        ).decode()
        return bool(out.strip())
    except Exception:
        return True


# ── Deterministic seeding (method-independent, unlike the buggy predecessor) ─

def scene_seed(base_seed: int, obj_key: str, obj_idx: int, scene_idx: int) -> int:
    """Seed depends ONLY on (base_seed, object, scene_idx) -- shared by every
    method compared on this scene. This is the fix for the bug documented in
    results/risk_gated_vla/audit.md Section 4."""
    return (base_seed * 10_000_000 + obj_idx * 100_000 + scene_idx) % (2 ** 32)


def random_choice_seed(base_seed: int, obj_key: str, obj_idx: int, scene_idx: int) -> int:
    """Separate RNG stream for the 'random' method's candidate index, salted
    so it does not perturb the pool-construction RNG sequence above."""
    return (scene_seed(base_seed, obj_key, obj_idx, scene_idx) * 7 + 3) % (2 ** 32)


# ── Scene / pool construction (nothing executed yet) ───────────────────────

def _load_scene(env: EnvironmentSoArm, obj_name: str, obj_key: str, cx: float, cy: float) -> int:
    env.reset_robot()
    env.remove_all_obj()
    # remove_all_obj() clears obj_ids/obj_positions/etc. but NOT _welded_obj_id --
    # a weld triggered by a PRIOR execute_candidate() call in this same env
    # instance (env.grasp() -> _attach_obj()) otherwise survives into the next
    # candidate's "fresh" scene, breaking determinism (found empirically: two
    # identical calls to run_scene() disagreed on outcomes even though the
    # scene/candidate selection itself was reproducible). Explicit unconditional
    # detach before every reload -- _detach_obj() is safe to call even when
    # nothing is welded (self._welded_obj_id is None) or when the welded id no
    # longer resolves (already-cleared obj_ids -> caught internally).
    env._detach_obj()
    oid = env.load_obj(obj_name, name=obj_key, pos=[cx, cy, _DROP_Z])
    env._steps(_SETTLE_STEPS)
    return oid


def build_pool(env: EnvironmentSoArm, obj_key: str, seed: int, k_grasps: int) -> dict:
    obj_name = OBJECTS[obj_key]
    rng = np.random.default_rng(seed)
    cx = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
    cy = EVAL_CENTRE_Y + float(rng.uniform(-0.04, 0.04))

    oid = _load_scene(env, obj_name, obj_key, cx, cy)
    obj_pos = env.get_obj_pos(oid).copy()
    obj_quat = env.get_obj_pose(oid)["quaternion"].copy()
    obs = env.get_obs(pointcloud=True)
    pc_stats = compute_pc_stats(obs, oid)

    candidates = np.stack([_sample_grasp(obj_pos, rng) for _ in range(k_grasps)])

    return {
        "cx": cx, "cy": cy, "obj_name": obj_name, "oid": oid,
        "obj_pos": obj_pos, "obj_quat": obj_quat, "pc_stats": pc_stats,
        "candidates": candidates,
    }


def score_pool(pool: dict, model: dict, obj_key: str | None = None,
               critic_ensemble=None, critic_relative: bool = True) -> dict:
    geo_scores = np.array([geo_score(g, pool["obj_pos"], pool["pc_stats"])
                            for g in pool["candidates"]])
    wm_scores, preds = score_grasps(
        pool["candidates"], pool["obj_pos"], pool["obj_quat"], pool["pc_stats"], model
    )
    if critic_ensemble is not None:
        rec = {
            "object": obj_key,
            "obj_pos_before": pool["obj_pos"],
            "obj_quat_before": pool["obj_quat"],
            "pc_stats_before": pool["pc_stats"],
        }
        candidates = [{"candidate_pose": c} for c in pool["candidates"]]
        wm_scores, uncertainty = score_candidates(
            rec, candidates, critic_ensemble, relative=critic_relative
        )
        preds = {"success_prob": wm_scores, "uncertainty": uncertainty}
    return {"geo_scores": geo_scores, "wm_scores": wm_scores, "preds": preds}


# ── Execution against a fresh, identically-placed copy of the scene ────────

def execute_candidate(env: EnvironmentSoArm, obj_key: str, pool: dict,
                      candidate: np.ndarray, grasp_debug: bool = False) -> dict:
    oid = _load_scene(env, pool["obj_name"], obj_key, pool["cx"], pool["cy"])
    pos_before = env.get_obj_pos(oid).copy()
    grasp_args = (
        tuple(float(v) for v in candidate[:3]),
        float(candidate[3]),
        float(candidate[4]),
        float(candidate[5]),
    )
    if grasp_debug:
        success, grasped_id = env.grasp(*grasp_args)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            success, grasped_id = env.grasp(*grasp_args)
    env._steps(40)
    pos_after = env.get_obj_pos(oid).copy()
    dz = float(pos_after[2] - pos_before[2])
    fell_off = bool(pos_after[2] < _FELL_OFF_Z)
    metrics = env.last_grasp_metrics or {}
    return {
        "success": bool(success),
        "grasped_id": int(grasped_id) if grasped_id is not None else None,
        "bilateral_contact": bool(metrics.get("bilateral_contact", False)),
        "weld_triggered": bool(metrics.get("weld_triggered", False)),
        "table_contact": bool(metrics.get("table_contact", False)),
        "lifted": bool(metrics.get("lifted", False)),
        "jaw_obj_xy_gap": float(metrics.get("jaw_obj_xy_gap", float("nan"))),
        "dz": dz,
        "fell_off": fell_off,
    }


# ── One full scene: pool + all methods + oracle sweep ───────────────────────

def run_scene(env, obj_key: str, obj_idx: int, scene_idx: int, base_seed: int,
              model: dict, k_grasps: int, grasp_debug: bool = False,
              critic_ensemble=None, critic_relative: bool = True) -> dict:
    seed = scene_seed(base_seed, obj_key, obj_idx, scene_idx)
    pool = build_pool(env, obj_key, seed, k_grasps)
    scores = score_pool(pool, model, obj_key, critic_ensemble, critic_relative)

    # Captured once per scene, immediately after build_pool() and before any
    # execute_candidate() call moves the scene -- the SAME pre-execution
    # observation build_pool() itself used for pool["pc_stats"], just kept
    # around so each candidate can be cropped against it individually below.
    # No new physics execution, no extra scene reload: the object hasn't
    # moved since build_pool()'s own internal capture.
    obs_for_local_crop = env.get_obs(pointcloud=True)

    rand_rng = np.random.default_rng(random_choice_seed(base_seed, obj_key, obj_idx, scene_idx))
    idx_by_method = {
        "random": int(rand_rng.integers(0, k_grasps)),
        "geometry": int(np.argmax(scores["geo_scores"])),
        "world_critic": int(np.argmax(scores["wm_scores"])),
    }

    outcomes = {}
    for method, idx in idx_by_method.items():
        outcomes[method] = execute_candidate(
            env, obj_key, pool, pool["candidates"][idx], grasp_debug=grasp_debug
        )
        outcomes[method]["candidate_idx"] = idx

    oracle_per_candidate = []
    for i in range(k_grasps):
        res = execute_candidate(
            env, obj_key, pool, pool["candidates"][i], grasp_debug=grasp_debug
        )
        res["candidate_idx"] = i
        res["candidate_pose"] = [float(v) for v in pool["candidates"][i]]
        res["geo_score"] = float(scores["geo_scores"][i])
        res["wm_score"] = float(scores["wm_scores"][i])
        res["success_prob"] = float(scores["preds"]["success_prob"][i])
        local_stats = compute_pc_stats_local(
            obs_for_local_crop, pool["oid"], pool["candidates"][i][:3]
        )
        res["pc_stats_local"] = [float(v) for v in local_stats]
        oracle_per_candidate.append(res)
    oracle_success = any(c["success"] for c in oracle_per_candidate)

    return {
        "object": obj_key, "obj_idx": obj_idx, "scene_idx": scene_idx,
        "seed": seed, "cx": pool["cx"], "cy": pool["cy"],
        "obj_pos_before": [float(v) for v in pool["obj_pos"]],
        "obj_quat_before": [float(v) for v in pool["obj_quat"]],
        "pc_stats_before": [float(v) for v in pool["pc_stats"]],
        "k_grasps": k_grasps,
        "idx_by_method": idx_by_method,
        "outcomes": outcomes,
        "oracle_success": bool(oracle_success),
        "oracle_per_candidate": oracle_per_candidate,
        "timestamp": time.time(),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", default="cracker,mustard,drill")
    ap.add_argument("--seeds", type=int, default=50, help="scenes per object")
    ap.add_argument("--k-grasps", type=int, default=DEFAULT_K)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--model-path", default=str(MODEL_PATH))
    ap.add_argument("--critic-model-dir", default="")
    ap.add_argument("--critic-variant", default="object_counterfactual",
                    choices=["global_bce", "object_bce", "object_counterfactual"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke-check", action="store_true",
                     help="also re-run scene 0 of the first object twice and diff results "
                          "(determinism check) + verify seed 0 vs seed 1 pools differ "
                          "(scene diversity check)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--grasp-debug", action="store_true",
                    help="show verbose diagnostics from the production grasp primitive")
    args = ap.parse_args()

    obj_keys = [k.strip() for k in args.objects.split(",")]
    for k in obj_keys:
        if k not in OBJECTS:
            ap.error(f"Unknown object '{k}'. Valid: {sorted(OBJECTS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "scenes.jsonl"
    config_path = out_dir / "config.json"

    config = {
        "objects": obj_keys, "seeds": args.seeds, "k_grasps": args.k_grasps,
        "base_seed": args.base_seed, "model_path": args.model_path,
        "git_hash": git_hash(), "git_dirty": git_dirty(),
        "ik_topdown_bias": os.environ.get("IK_TOPDOWN_BIAS", "0.0"),
        "eval_centre_y": EVAL_CENTRE_Y,
        "critic_model_dir": args.critic_model_dir,
        "critic_variant": args.critic_variant,
        "script": "scripts/risk_gated_vla_phase1_eval.py",
        "started_at": time.time(),
    }
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[phase1] config -> {config_path}")
    print(f"[phase1] git_hash={config['git_hash']}  dirty={config['git_dirty']}")

    model = load_model(Path(args.model_path))
    critic_ensemble = None
    if args.critic_model_dir:
        critic_ensemble = load_ensemble(Path(args.critic_model_dir), args.critic_variant)
        print(f"[phase1] loaded {len(critic_ensemble)} critic models: {args.critic_variant}")
    critic_relative = args.critic_variant != "global_bce"
    env = EnvironmentSoArm(vis=False, debug=False)

    try:
        if args.smoke_check:
            print("\n[smoke-check] determinism: running (obj0, scene0) twice ...")
            r1 = run_scene(env, obj_keys[0], 0, 0, args.base_seed, model,
                           args.k_grasps, args.grasp_debug,
                           critic_ensemble, critic_relative)
            r2 = run_scene(env, obj_keys[0], 0, 0, args.base_seed, model,
                           args.k_grasps, args.grasp_debug,
                           critic_ensemble, critic_relative)
            # dz tolerance is 1e-3, not 1e-6: MuJoCo's iterative contact solver is
            # not bit-reproducible run-to-run even with identical seeds/inputs
            # (observed ~1e-4 level noise on dz, ~2e-4 on jaw_obj_xy_gap, with the
            # boolean `success` outcome always exactly matching) -- the primary
            # paired-comparison label is `success`, which we require exact.
            det_ok = (
                r1["idx_by_method"] == r2["idx_by_method"]
                and all(r1["outcomes"][m]["success"] == r2["outcomes"][m]["success"] for m in METHODS)
                and all(abs(r1["outcomes"][m]["dz"] - r2["outcomes"][m]["dz"]) < 1e-3 for m in METHODS)
                and r1["oracle_success"] == r2["oracle_success"]
            )
            print(f"[smoke-check] determinism {'PASS' if det_ok else 'FAIL'}  "
                  f"run1_idx={r1['idx_by_method']}  run2_idx={r2['idx_by_method']}")
            if not det_ok:
                print("[smoke-check] ABORT: environment is not deterministic under fresh "
                      "reset+reload -- Phase 1's paired design assumption is violated.")
                sys.exit(1)

            print("\n[smoke-check] scene diversity: scene0 vs scene1 pools must differ ...")
            r_s1 = run_scene(env, obj_keys[0], 0, 1, args.base_seed, model,
                             args.k_grasps, args.grasp_debug,
                             critic_ensemble, critic_relative)
            diversity_ok = (r1["cx"] != r_s1["cx"] or r1["cy"] != r_s1["cy"])
            print(f"[smoke-check] diversity {'PASS' if diversity_ok else 'FAIL'}  "
                  f"scene0=({r1['cx']:.4f},{r1['cy']:.4f})  scene1=({r_s1['cx']:.4f},{r_s1['cy']:.4f})")
            if not diversity_ok:
                print("[smoke-check] ABORT: different scene_idx produced identical placement.")
                sys.exit(1)

            print("\n[smoke-check] pool identity across methods: verified structurally "
                  "(single build_pool() call feeds all methods + oracle in run_scene(); "
                  "no per-method reseeding exists in this script).\n")

        n_total = len(obj_keys) * args.seeds
        n_done = 0
        t0 = time.time()
        with open(jsonl_path, "a") as f:
            for obj_idx, obj_key in enumerate(obj_keys):
                n_succ = {m: 0 for m in METHODS}
                n_succ["oracle"] = 0
                for scene_idx in range(args.seeds):
                    rec = run_scene(env, obj_key, obj_idx, scene_idx, args.base_seed,
                                     model, args.k_grasps, args.grasp_debug,
                                     critic_ensemble, critic_relative)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    n_done += 1
                    for m in METHODS:
                        n_succ[m] += int(rec["outcomes"][m]["success"])
                    n_succ["oracle"] += int(rec["oracle_success"])
                    if not args.quiet:
                        elapsed = time.time() - t0
                        eta = elapsed / n_done * (n_total - n_done)
                        print(f"  [{n_done:04d}/{n_total}] {obj_key:<10} scene={scene_idx:03d}  "
                              f"rand={rec['outcomes']['random']['success']}  "
                              f"geo={rec['outcomes']['geometry']['success']}  "
                              f"wm={rec['outcomes']['world_critic']['success']}  "
                              f"oracle={rec['oracle_success']}  ETA={eta/60:.1f}m")
                sr = {k: v / args.seeds for k, v in n_succ.items()}
                print(f"\n  {obj_key}: n={args.seeds}  " +
                      "  ".join(f"{k}={v:.3f}" for k, v in sr.items()) + "\n")
    finally:
        env.close()

    print(f"\n[phase1] wrote {jsonl_path}")


if __name__ == "__main__":
    main()
