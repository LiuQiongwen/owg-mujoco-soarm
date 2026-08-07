#!/usr/bin/env python3
"""Paired evaluation of attached-object lift and optional fallback recovery."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paperA_data" / "scripts"))
os.environ.setdefault("MUJOCO_GL", "egl")
import _lerobot_groot_patch  # noqa: E402,F401

from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from scripts.collect_act_rac_episodes import run_policy_until_intervention  # noqa: E402
from scripts.eval_act_rf_recovery_mujoco import fit_frozen_rf, recover  # noqa: E402
from scripts.record_sim_lerobot_episodes import OBJECTS  # noqa: E402
from tango_robot.env_soarm import (  # noqa: E402
    EnvironmentSoArm, GRASP_MODE_PHYSICS_WELD, JAW_CONTACT_PROXY_SPHERES,
    TABLE_TOP_Z, _JAW_CONTACT_MODELS,
)

def summarize(rows):
    n = len(rows)
    result = {
        "n": n,
        "baseline_act_successes": sum(r["baseline_act_success"] for r in rows),
        "intervention_reasons": dict(Counter(r["intervention_reason"] for r in rows)),
        "attached_insufficient_lift": sum(r["attached_failure"] for r in rows),
        "variants": {},
    }
    for variant in rows[0]["variants"]:
        successes = sum(r["variants"][variant]["system_success"] for r in rows)
        result["variants"][variant] = {
            "successes": successes, "success_rate": successes / max(1, n)}
    result["attached_lift_attempts"] = sum(
        r["attached_lift"] is not None for r in rows)
    result["attached_lift_successes"] = sum(
        bool(r["attached_lift"] and r["attached_lift"]["success"]) for r in rows)
    result["fallback_attempts"] = sum(r["fallback_regrasp"] is not None for r in rows)
    result["fallback_successes"] = sum(
        bool(r["fallback_regrasp"] and r["fallback_regrasp"]["success"]) for r in rows)
    return result


def attached_lift(env, obj_id, initial_obj_pos, z_offset, success_threshold):
    eef_before = np.asarray(env.get_ee_pos()[0], dtype=float).copy()
    obj_before = np.asarray(env.get_obj_pos(obj_id), dtype=float).copy()
    target = eef_before.copy(); target[2] = min(target[2] + z_offset, 1.15)
    ik_ok, _ = env.move_ee([*target, None], max_step=240)
    env._steps(40)
    obj_after = np.asarray(env.get_obj_pos(obj_id), dtype=float).copy()
    still_attached = env._welded_obj_id == obj_id
    total_dz = float(obj_after[2] - initial_obj_pos[2])
    success = bool(still_attached and total_dz > success_threshold)
    metrics = env.get_grasp_debug_metrics(obj_id)
    return success, {
        "success": success, "ik_converged": bool(ik_ok),
        "still_attached": bool(still_attached),
        "eef_before": eef_before.tolist(), "eef_target": target.tolist(),
        "object_before": obj_before.tolist(), "object_after": obj_after.tolist(),
        "incremental_dz": float(obj_after[2] - obj_before[2]),
        "total_dz": total_dz,
        "table_contact_after": bool(metrics.get("table_contact", False)),
    }


def continue_attached_lift(env, obj_id, initial_obj_pos, step_offset,
                           max_extensions, success_threshold,
                           min_progress=0.002):
    """Continue a failed fixed lift in measured increments while attachment holds."""
    segments = []
    success = False
    stop_reason = "extension_budget_exhausted"
    for segment_index in range(max_extensions):
        if env._welded_obj_id != obj_id:
            stop_reason = "attachment_lost"
            break
        eef_before = np.asarray(env.get_ee_pos()[0], dtype=float).copy()
        obj_before = np.asarray(env.get_obj_pos(obj_id), dtype=float).copy()
        target = eef_before.copy()
        target[2] = min(target[2] + step_offset, 1.15)
        ik_ok, _ = env.move_ee([*target, None], max_step=180)
        env._steps(40)
        obj_after = np.asarray(env.get_obj_pos(obj_id), dtype=float).copy()
        total_dz = float(obj_after[2] - initial_obj_pos[2])
        progress = float(obj_after[2] - obj_before[2])
        attached = env._welded_obj_id == obj_id
        segment = {
            "segment": segment_index + 1, "ik_converged": bool(ik_ok),
            "still_attached": bool(attached), "eef_before": eef_before.tolist(),
            "eef_target": target.tolist(), "object_before": obj_before.tolist(),
            "object_after": obj_after.tolist(), "incremental_dz": progress,
            "total_dz": total_dz,
        }
        segments.append(segment)
        if attached and total_dz > success_threshold:
            success = True
            stop_reason = "success_threshold_reached"
            break
        if not attached:
            stop_reason = "attachment_lost"
            break
        if abs(progress) < min_progress:
            stop_reason = "insufficient_object_progress"
            break
    return success, {
        "success": success, "segments": segments,
        "extensions_executed": len(segments), "stop_reason": stop_reason,
        "final_total_dz": (segments[-1]["total_dz"] if segments else None),
    }


def recover_with_replanning(env, obj_id, obj_key, seed, templates, rf, quiet,
                            intervention_eef, attempts, require_ik_valid=False,
                            phase_ref=None):
    """Retry after returning home and regenerating candidates from current state."""
    details = []
    success = False
    for attempt in range(1, attempts + 1):
        if phase_ref is not None:
            # Phase 1 covers retreat and returning to a safe planning state.
            phase_ref["value"] = 1
        success, detail = recover(
            env, obj_id, obj_key, seed + 7919 * (attempt - 1), templates,
            "rf", rf, None, quiet, intervention_eef, 1, require_ik_valid,
            phase_ref=phase_ref)
        details.append(detail)
        if success:
            break
        if detail.get("skipped_no_ik_valid_candidate", False):
            break
    aggregate = dict(details[-1])
    aggregate["replan_mode"] = "current_state_after_each_failure"
    aggregate["replan_attempts_executed"] = len(details)
    aggregate["successful_attempt"] = len(details) if success else None
    aggregate["attempt_details"] = details
    return success, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--object", required=True,
                    choices=("cracker", "mustard", "drill"))
    ap.add_argument("--scenes", type=int, required=True)
    ap.add_argument("--base-seed", type=int, default=1031)
    ap.add_argument("--max-policy-steps", type=int, default=80)
    ap.add_argument("--displacement-threshold", type=float, default=0.02)
    ap.add_argument("--lift-threshold", type=float, default=0.07)
    ap.add_argument("--lift-offset", type=float, default=0.08)
    ap.add_argument("--adaptive-lift-extensions", type=int, default=0)
    ap.add_argument("--adaptive-lift-step", type=float, default=0.03)
    ap.add_argument("--adaptive-lift-min-progress", type=float, default=0.002)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--candidate-count", type=int, default=15)
    ap.add_argument("--regrasp-attempts", type=int, default=1,
                    help="Number of RF-ranked candidates tried for unattached recovery.")
    ap.add_argument("--regrasp-mode", choices=("static-ranked", "replan"),
                    default="static-ranked")
    ap.add_argument("--require-ik-valid", action="store_true",
                    help="Stop instead of executing fallback poses when all candidates fail IK.")
    ap.add_argument("--contact-servo-shift", type=float, default=0.0,
                    help="One-shot lateral shift in metres after unilateral contact during closure.")
    ap.add_argument("--ranker-train-results", required=True)
    ap.add_argument(
        "--enable-fallback", action="store_true",
        help="Run the development-only release-and-regrasp R2 after a failed R1 lift.")
    ap.add_argument("--jaw-contact-model", choices=_JAW_CONTACT_MODELS,
                    default=JAW_CONTACT_PROXY_SPHERES,
                    help="jaw contact geometry; the default is the legacy "
                         "buried-sphere collider every recorded result used. "
                         "See docs/JAW_CONTACT_MODEL_AB_20260807.md.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        ap.error(f"output already exists: {out}")
    templates = json.loads(Path(args.templates).read_text())["templates"][:args.candidate_count]
    rf = fit_frozen_rf(args.ranker_train_results)
    policy = ACTPolicy.from_pretrained(args.checkpoint); policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})
    env = EnvironmentSoArm(vis=False, debug=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           contact_servo_shift_m=args.contact_servo_shift,
                           jaw_contact_model=args.jaw_contact_model)
    env.preload_pool([OBJECTS[args.object].replace("Ycb", "")])
    rows = []
    try:
        for index in range(args.scenes):
            seed = args.base_seed * 10000 + index
            rng = np.random.default_rng(seed)
            env.reset_robot(); env.remove_all_obj(); env._detach_obj()
            obj_id = env.load_obj(
                OBJECTS[args.object], name=args.object,
                pos=[float(rng.uniform(-0.06, 0.06)),
                     -0.30 + float(rng.uniform(-0.04, 0.04)),
                     TABLE_TOP_Z + 0.12])
            env._steps(300)
            pos0 = np.asarray(env.get_obj_pos(obj_id), dtype=float).copy()
            reason, _, steps, attached, max_planar = run_policy_until_intervention(
                env, policy, pre, post, obj_id, args.object,
                args.max_policy_steps, args.displacement_threshold)
            pos_intervention = np.asarray(env.get_obj_pos(obj_id), dtype=float).copy()
            intervention_eef = np.asarray(env.get_ee_pos()[0], dtype=float).copy()
            baseline = reason == "policy_success"
            unattached_recoverable = bool(
                not baseline and not attached and pos_intervention[2] - pos0[2] > -0.05)
            attached_failure = bool(not baseline and attached)
            regrasp_success = False; regrasp_row = None
            lift_success = False; lift_row = None
            adaptive_lift_success = False; adaptive_lift_row = None
            fallback_success = False; fallback_row = None
            if unattached_recoverable:
                if args.regrasp_mode == "replan":
                    regrasp_success, regrasp_row = recover_with_replanning(
                        env, obj_id, args.object, seed, templates, rf, args.quiet,
                        intervention_eef, args.regrasp_attempts,
                        args.require_ik_valid)
                else:
                    regrasp_success, regrasp_row = recover(
                        env, obj_id, args.object, seed, templates, "rf", rf, None,
                        args.quiet, intervention_eef, args.regrasp_attempts,
                        args.require_ik_valid)
            elif attached_failure:
                lift_success, lift_row = attached_lift(
                    env, obj_id, pos0, args.lift_offset, args.lift_threshold)
                if (not lift_success and args.adaptive_lift_extensions > 0
                        and env._welded_obj_id == obj_id):
                    adaptive_lift_success, adaptive_lift_row = continue_attached_lift(
                        env, obj_id, pos0, args.adaptive_lift_step,
                        args.adaptive_lift_extensions, args.lift_threshold,
                        args.adaptive_lift_min_progress)
                if args.enable_fallback and not lift_success:
                    env.move_gripper(0.085, step=80)
                    env.wait_until_still(obj_id, max_wait_epochs=300)
                    released_pos = np.asarray(env.get_obj_pos(obj_id), dtype=float)
                    if released_pos[2] - pos0[2] > -0.05:
                        fallback_success, fallback_row = recover(
                            env, obj_id, args.object, seed, templates, "rf", rf,
                            None, args.quiet, intervention_eef, 1)
            top1_regrasp_success = bool(
                regrasp_success and regrasp_row
                and regrasp_row.get("successful_attempt") == 1)
            r0 = bool(baseline or top1_regrasp_success)
            r1 = bool(r0 or lift_success)
            r2 = bool(r1 or fallback_success)
            r_topk = bool(baseline or regrasp_success or lift_success)
            r_adaptive = bool(r_topk or adaptive_lift_success)
            row = {
                "object": args.object, "seed": seed,
                "intervention_reason": reason, "policy_steps": steps,
                "max_planar_displacement": max_planar,
                "object_dz_at_intervention": float(pos_intervention[2] - pos0[2]),
                "baseline_act_success": baseline,
                "attached_failure": attached_failure,
                "unattached_recoverable": unattached_recoverable,
                "unattached_regrasp": ({"success": regrasp_success, "detail": regrasp_row}
                                       if regrasp_row is not None else None),
                "attached_lift": lift_row,
                "adaptive_lift": adaptive_lift_row,
                "fallback_regrasp": ({"success": fallback_success, "detail": fallback_row}
                                     if fallback_row is not None else None),
                "variants": {
                    "r0_regrasp_only": {"system_success": r0},
                    "r1_plus_attached_lift": {"system_success": r1},
                    "r2_plus_release_regrasp_fallback": {"system_success": r2},
                    f"r1_{args.regrasp_mode}_top{args.regrasp_attempts}_plus_attached_lift": {
                        "system_success": r_topk},
                    f"r1_{args.regrasp_mode}_top{args.regrasp_attempts}_plus_adaptive_lift": {
                        "system_success": r_adaptive},
                },
            }
            rows.append(row); print(json.dumps(row), flush=True)
    finally:
        env.close()
    result = {"config": vars(args), "rows": rows, "summary": summarize(rows)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
