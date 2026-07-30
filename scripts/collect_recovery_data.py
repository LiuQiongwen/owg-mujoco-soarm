#!/usr/bin/env python3
"""
C.7 -- DAgger-style recovery-data collection (formal run, base_seed=500).

Per-trial protocol ("deviate then recover", matching the WM-DAgger pattern
found in results/risk_gated_vla/LITERATURE_AND_NOVELTY_PLAN.md item #13,
adapted to this project's own execution primitive rather than importing that
paper's world-model-synthesis machinery):

  1. Build a candidate pool (shared with the Phase 1/C.3 harness's own
     build_pool/score_pool -- same object placement, same causally-admissible
     features), pick the nominal candidate by the trained object_counterfactual
     critic's own top-1 score (reusing counterfactual_models_20260730/,
     unmodified, read-only).
  2. Execute the nominal candidate WITH ctrl-level perturbation injected via
     env._step_hook, throttled to ~20Hz (matching
     scripts/record_sim_lerobot_episodes.py's render_stride convention --
     see _PERTURB_FPS). Built on env.grasp() (matching
     scripts/risk_gated_vla_phase1_eval.py's execute_candidate()), not
     pick_obj_by_id(), so bilateral_contact/lifted/weld_triggered/fell_off
     populate correctly for world_model.multihead_labels.failure_type_3class()
     reuse -- same label definitions as C.3.
  3. If the perturbed attempt fails: apply a "return to nominal" recovery
     action (re-execute the SAME nominal candidate, no perturbation) and
     record whether that recovers success. No recovery step is attempted if
     the perturbed attempt already succeeded.

Safety circuit breakers (per-trial, checked during the collection loop, not
just at the end):
  - perturbation_count outside an expected sane range aborts the run
    immediately (catches the ~500-application throttle bug found during the
    n=10 sanity check before it silently produces 60 garbage trials/object).
  - N consecutive no_contact perturbed outcomes pauses the run for manual
    review rather than continuing to burn compute on a possibly-broken setup.

Output: NEW directory only (--out-dir must not already exist and be
non-empty) -- never overwrites existing scenes.jsonl-format data from
Phase 1/2/C.3 or an earlier recovery-data run.
"""
import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from tango_robot.env_soarm import EnvironmentSoArm
from data.transition_logger import compute_pc_stats
from world_model.train_counterfactual_critic import load_ensemble, score_candidates
from world_model.multihead_labels import failure_type_3class
from scripts.eval_wm_reranking_full import OBJECTS, _SPREAD_XY, _DROP_Z, _SETTLE_STEPS, _FELL_OFF_Z, _sample_grasp
from scripts.risk_gated_vla_phase1_eval import EVAL_CENTRE_Y

DEFAULT_CRITIC_DIR = "results/risk_gated_vla/counterfactual_models_20260730"
DEFAULT_CRITIC_VARIANT = "object_counterfactual"
_PERTURB_FPS = 20  # matches scripts/record_sim_lerobot_episodes.py's FPS

# Circuit-breaker thresholds (sanity-check-derived: n=10 cracker run saw
# 12-28 applications/trial at perturb_std=0.05/prob=0.3 after the throttle
# fix; a wide band with real margin, not a tight statistical bound).
_PERTURB_COUNT_MIN, _PERTURB_COUNT_MAX = 3, 100
_CONSECUTIVE_NO_CONTACT_LIMIT = 10


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_scene(env, obj_name, obj_key, cx, cy):
    env.reset_robot()
    env.remove_all_obj()
    env._detach_obj()
    oid = env.load_obj(obj_name, name=obj_key, pos=[cx, cy, _DROP_Z])
    env._steps(_SETTLE_STEPS)
    return oid


def build_pool_recovery(env, obj_key, seed, k_grasps):
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
    return {"cx": cx, "cy": cy, "obj_name": obj_name, "obj_pos": obj_pos,
            "obj_quat": obj_quat, "pc_stats": pc_stats, "candidates": candidates}


def pick_nominal_candidate(pool, obj_key, critic_ensemble):
    rec = {"object": obj_key, "obj_pos_before": pool["obj_pos"],
           "obj_quat_before": pool["obj_quat"], "pc_stats_before": pool["pc_stats"]}
    cands = [{"candidate_pose": c} for c in pool["candidates"]]
    scores, _ = score_candidates(rec, cands, critic_ensemble, relative=True)
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])


def execute_with_perturbation(env, obj_key, pool, candidate, perturb_std, perturb_prob, rng,
                               quiet=True):
    """Returns (outcome, applied, trajectory_length, abnormal_termination_info)."""
    oid = _load_scene(env, pool["obj_name"], obj_key, pool["cx"], pool["cy"])
    pos_before = env.get_obj_pos(oid).copy()

    applied = {"vec": np.zeros(6, dtype=np.float32), "n_applications": 0}
    sim_dt = float(env.model.opt.timestep)
    stride = max(1, int(round(1.0 / (_PERTURB_FPS * sim_dt))))
    step_count = {"n": 0}

    def hook():
        step_count["n"] += 1
        if step_count["n"] != 1 and step_count["n"] % stride != 0:
            return
        if perturb_std > 0.0 and rng.random() < perturb_prob:
            pert = np.zeros(6, dtype=np.float32)
            pert[:5] = rng.normal(0.0, perturb_std, size=5).astype(np.float32)
            for act_id, delta in zip(env._arm_act_ids, pert[:5]):
                lo, hi = env.model.actuator_ctrlrange[act_id]
                env.data.ctrl[act_id] = np.clip(env.data.ctrl[act_id] + delta, lo, hi)
            pert[5] = float(rng.normal(0.0, perturb_std * 0.25))
            lo, hi = env.model.actuator_ctrlrange[env._grip_act_id]
            env.data.ctrl[env._grip_act_id] = np.clip(env.data.ctrl[env._grip_act_id] + pert[5], lo, hi)
            applied["vec"] += pert
            applied["n_applications"] += 1

    grasp_args = (
        tuple(float(v) for v in candidate[:3]), float(candidate[3]),
        float(candidate[4]), float(candidate[5]),
    )
    env._step_hook = hook
    abnormal = None
    success, grasped_id = False, None
    try:
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                success, grasped_id = env.grasp(*grasp_args)
        else:
            success, grasped_id = env.grasp(*grasp_args)
    except Exception as exc:
        abnormal = {"error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        env._step_hook = None

    drop_step_idx = None
    if abnormal is None:
        for i in range(0, 40, 4):
            env._steps(4)
            h = env.get_obj_pos(oid)[2]
            if h < _FELL_OFF_Z and drop_step_idx is None:
                drop_step_idx = i
    pos_after = env.get_obj_pos(oid).copy()
    dz = float(pos_after[2] - pos_before[2])
    fell_off = bool(pos_after[2] < _FELL_OFF_Z)
    metrics = env.last_grasp_metrics or {}

    outcome = {
        "success": bool(success) if abnormal is None else False,
        "bilateral_contact": bool(metrics.get("bilateral_contact", False)),
        "weld_triggered": bool(metrics.get("weld_triggered", False)),
        "lifted": bool(metrics.get("lifted", False)),
        "fell_off": fell_off,
        "dz": dz,
        "drop_step_idx_proxy": drop_step_idx,
    }
    outcome["failure_type"] = failure_type_3class(outcome) if abnormal is None else "abnormal_termination"
    trajectory_length = step_count["n"]
    return outcome, applied, trajectory_length, abnormal


def run_trial(env, obj_key, obj_idx, trial_idx, base_seed, critic_ensemble,
              perturb_std, perturb_prob, k_grasps=10, quiet=True):
    seed = (base_seed * 10_000_000 + obj_idx * 100_000 + trial_idx) % (2 ** 32)
    pool = build_pool_recovery(env, obj_key, seed, k_grasps)
    nominal_idx, nominal_critic_score = pick_nominal_candidate(pool, obj_key, critic_ensemble)
    nominal_candidate = pool["candidates"][nominal_idx]

    pert_rng = np.random.default_rng((seed * 7 + 11) % (2 ** 32))
    perturbed_outcome, applied, traj_len, abnormal = execute_with_perturbation(
        env, obj_key, pool, nominal_candidate, perturb_std, perturb_prob, pert_rng, quiet=quiet)

    recovery = None
    recovery_success = None
    if abnormal is None and not perturbed_outcome["success"]:
        recovery_outcome, _, recovery_traj_len, recovery_abnormal = execute_with_perturbation(
            env, obj_key, pool, nominal_candidate, 0.0, 0.0, pert_rng, quiet=quiet)
        recovery = {
            "action": "return_to_nominal_no_perturbation",
            "outcome": recovery_outcome,
            "recovery_success": bool(recovery_outcome["success"]) if recovery_abnormal is None else False,
            "trajectory_length": recovery_traj_len,
            "abnormal_termination": recovery_abnormal,
        }
        recovery_success = recovery["recovery_success"]

    return {
        # required top-level fields, exact names
        "object": obj_key, "seed": seed,
        "applied_perturbation": applied["vec"].tolist(),
        "perturbation_count": applied["n_applications"],
        "perturb_std": perturb_std, "perturb_prob": perturb_prob,
        "failure_type": perturbed_outcome["failure_type"],
        "perturbed_success": bool(perturbed_outcome["success"]),
        "recovery_success": recovery_success,
        "abnormal_termination": abnormal,
        "trajectory_length": traj_len,
        # additional context, kept from the sanity-check-validated schema
        "obj_idx": obj_idx, "trial_idx": trial_idx, "cx": pool["cx"], "cy": pool["cy"],
        "nominal_candidate_pose": [float(v) for v in nominal_candidate],
        "nominal_critic_score": nominal_critic_score,
        "perturbed_outcome": perturbed_outcome,
        "recovery": recovery,
        "recovery_triggered": recovery is not None,
        "timestamp": time.time(),
    }


def write_intermediate_stats(out_dir, obj_key, rows):
    obj_rows = [r for r in rows if r["object"] == obj_key]
    ft_counts = Counter(r["failure_type"] for r in obj_rows)
    n_recov_triggered = sum(1 for r in obj_rows if r["recovery_triggered"])
    n_recov_success = sum(1 for r in obj_rows if r["recovery_success"])
    n_abnormal = sum(1 for r in obj_rows if r["abnormal_termination"] is not None)
    stats = {
        "object": obj_key, "n_trials": len(obj_rows),
        "failure_type_distribution": dict(ft_counts),
        "n_perturbed_success": sum(1 for r in obj_rows if r["perturbed_success"]),
        "n_recovery_triggered": n_recov_triggered, "n_recovery_success": n_recov_success,
        "recovery_success_rate": n_recov_success / max(n_recov_triggered, 1),
        "n_abnormal_termination": n_abnormal,
        "perturbation_count_stats": {
            "min": min((r["perturbation_count"] for r in obj_rows), default=None),
            "max": max((r["perturbation_count"] for r in obj_rows), default=None),
            "mean": float(np.mean([r["perturbation_count"] for r in obj_rows])) if obj_rows else None,
        },
    }
    path = out_dir / f"intermediate_stats_{obj_key}.json"
    path.write_text(json.dumps(stats, indent=2))
    print(f"\n[intermediate] {obj_key}: {json.dumps(stats, indent=2)}\n")
    return stats


def write_final_report(out_dir, rows, config):
    objs = sorted(set(r["object"] for r in rows))
    lines = [
        "# C.7 Recovery-Data Collection Report",
        "",
        f"**base_seed={config['base_seed']}**, objects={config['objects']}, "
        f"target={config['n_trials_per_object']}/object. "
        f"git commit: `{config['git_hash']}`. Started: {config['started_at']}.",
        "",
        "| Object | Valid trials | Perturbed successes | Recovery triggered | Recovery successes | Abnormal terminations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for obj in objs:
        obj_rows = [r for r in rows if r["object"] == obj]
        n_valid = sum(1 for r in obj_rows if r["abnormal_termination"] is None)
        n_pert_succ = sum(1 for r in obj_rows if r["perturbed_success"])
        n_recov_trig = sum(1 for r in obj_rows if r["recovery_triggered"])
        n_recov_succ = sum(1 for r in obj_rows if r["recovery_success"])
        n_abnormal = sum(1 for r in obj_rows if r["abnormal_termination"] is not None)
        lines.append(f"| {obj} | {n_valid}/{len(obj_rows)} | {n_pert_succ} | {n_recov_trig} | "
                     f"{n_recov_succ} | {n_abnormal} |")

    lines += ["", "## Failure-type distribution (perturbed attempts)", "",
              "| Object | success | no_contact | weld_no_lift | abnormal_termination |",
              "|---|---:|---:|---:|---:|"]
    for obj in objs:
        obj_rows = [r for r in rows if r["object"] == obj]
        c = Counter(r["failure_type"] for r in obj_rows)
        lines.append(f"| {obj} | {c.get('success',0)} | {c.get('no_contact',0)} | "
                     f"{c.get('weld_no_lift',0)} | {c.get('abnormal_termination',0)} |")

    lines += [
        "", "## Notes",
        "- Recovery = re-executing the SAME nominal candidate with perturbation turned off, "
        "only attempted when the perturbed attempt failed.",
        "- `recovery_success=null` means no recovery was triggered (perturbed attempt already "
        "succeeded), not that recovery failed.",
        "- This data has NOT been used to train anything -- collection only, per this round's scope.",
    ]
    (out_dir / "DATA_COLLECTION_REPORT.md").write_text("\n".join(lines))
    print(f"[report] wrote {out_dir / 'DATA_COLLECTION_REPORT.md'}")


def _load_existing_rows(jsonl_path: Path) -> list:
    if not jsonl_path.exists():
        return []
    return [json.loads(l) for l in open(jsonl_path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="cracker,mustard,drill")
    ap.add_argument("--n-trials", type=int, default=60, help="trials per object")
    ap.add_argument("--base-seed", type=int, default=500)
    ap.add_argument("--perturb-std", type=float, default=0.05)
    ap.add_argument("--perturb-prob", type=float, default=0.3)
    ap.add_argument("--critic-model-dir", default=DEFAULT_CRITIC_DIR)
    ap.add_argument("--critic-variant", default=DEFAULT_CRITIC_VARIANT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--consecutive-no-contact-limit", type=int, default=_CONSECUTIVE_NO_CONTACT_LIMIT,
                     help="circuit-breaker threshold, override per object's known base rate -- "
                          "e.g. cracker's measured pool-wide bilateral_contact rate is 14.2%% "
                          "(confirmatory-300), so P(10 consecutive no_contact | independent "
                          "trials) = 0.86^10 ~= 22%% [conservative: using 1-0.142] happens by "
                          "chance alone; raise this for objects with a known low base rate "
                          "rather than treating every trip as an anomaly. Document the derivation "
                          "in the run's report, do not just raise it to stop pausing.")
    ap.add_argument("--allow-append", action="store_true",
                     help="intentionally add more objects/trials to an existing --out-dir "
                          "(e.g. mustard/drill into a directory cracker already partially "
                          "populated). Refuses if any requested object already has "
                          ">=--n-trials rows recorded, to prevent accidental duplication.")
    args = ap.parse_args()

    obj_keys = [k.strip() for k in args.objects.split(",")]
    out_dir = Path(args.out_dir)
    jsonl_path = out_dir / "recovery_trials.jsonl"
    existing_rows = []

    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.allow_append:
            raise SystemExit(f"[collect_recovery_data] {out_dir} already exists and is non-empty "
                              f"-- refusing to overwrite. Use a fresh --out-dir or --allow-append "
                              f"if this is intentional.")
        existing_rows = _load_existing_rows(jsonl_path)
        existing_counts = Counter(r["object"] for r in existing_rows)
        already_done = [o for o in obj_keys if existing_counts.get(o, 0) >= args.n_trials]
        if already_done:
            raise SystemExit(f"[collect_recovery_data] {out_dir} already has "
                              f">={args.n_trials} rows for {already_done} -- refusing to "
                              f"duplicate. Remove them from --objects or use a fresh directory.")
        print(f"[collect_recovery_data] --allow-append: {out_dir} has {len(existing_rows)} "
              f"existing rows ({dict(existing_counts)}), appending {obj_keys}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "git_hash": git_hash(),
        "command": " ".join(sys.argv),
        "objects": obj_keys, "n_trials_per_object": args.n_trials, "base_seed": args.base_seed,
        "perturb_std": args.perturb_std, "perturb_prob": args.perturb_prob,
        "perturb_throttle_fps": _PERTURB_FPS,
        "critic_model_dir": args.critic_model_dir, "critic_variant": args.critic_variant,
        "eval_centre_y": EVAL_CENTRE_Y, "ik_topdown_bias": os.environ.get("IK_TOPDOWN_BIAS", "0.0"),
        "perturbation_count_expected_range": [_PERTURB_COUNT_MIN, _PERTURB_COUNT_MAX],
        "consecutive_no_contact_limit": args.consecutive_no_contact_limit,
        "started_at": time.time(),
    }
    config_path = out_dir / "frozen_config.json"
    if config_path.exists():
        # Never overwrite the original frozen_config.json -- append-invocation
        # configs get their own numbered file so the original record survives.
        n = 1
        while (out_dir / f"frozen_config_append_{n}.json").exists():
            n += 1
        config_path = out_dir / f"frozen_config_append_{n}.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[collect_recovery_data] wrote {config_path}")

    critic_ensemble = load_ensemble(Path(args.critic_model_dir), args.critic_variant)
    env = EnvironmentSoArm(vis=False, debug=False)

    all_rows = list(existing_rows)  # carry forward for reporting; new rows appended below
    consecutive_no_contact = 0
    try:
        with open(jsonl_path, "a") as f:
            for obj_idx, obj_key in enumerate(obj_keys):
                for trial_idx in range(args.n_trials):
                    rec = run_trial(env, obj_key, obj_idx, trial_idx, args.base_seed,
                                     critic_ensemble, args.perturb_std, args.perturb_prob,
                                     quiet=args.quiet)
                    all_rows.append(rec)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()

                    pc = rec["perturbation_count"]
                    if not (_PERTURB_COUNT_MIN <= pc <= _PERTURB_COUNT_MAX):
                        raise SystemExit(
                            f"\n*** CIRCUIT BREAKER: perturbation_count={pc} outside expected "
                            f"range [{_PERTURB_COUNT_MIN},{_PERTURB_COUNT_MAX}] at "
                            f"{obj_key} trial {trial_idx} -- stopping immediately for manual "
                            f"review, matching the exact failure mode found in the n=10 sanity "
                            f"check. Partial data in {jsonl_path} is preserved. ***\n")

                    if rec["failure_type"] == "no_contact":
                        consecutive_no_contact += 1
                    else:
                        consecutive_no_contact = 0
                    if consecutive_no_contact >= args.consecutive_no_contact_limit:
                        raise SystemExit(
                            f"\n*** CIRCUIT BREAKER: {consecutive_no_contact} consecutive "
                            f"no_contact perturbed outcomes at {obj_key} trial {trial_idx} -- "
                            f"pausing for manual review. Partial data in {jsonl_path} is "
                            f"preserved. ***\n")

                    print(f"  [{obj_key} {trial_idx+1}/{args.n_trials}] "
                          f"perturbation_count={pc} "
                          f"perturbed_success={rec['perturbed_success']} "
                          f"failure_type={rec['failure_type']} "
                          f"recovery_success={rec['recovery_success']} "
                          f"abnormal={rec['abnormal_termination'] is not None}")
                write_intermediate_stats(out_dir, obj_key, all_rows)
    finally:
        env.close()

    write_final_report(out_dir, all_rows, config)
    print(f"\n[collect_recovery_data] done. {len(all_rows)} total trials in {jsonl_path}")


if __name__ == "__main__":
    main()
