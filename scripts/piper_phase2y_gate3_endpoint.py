"""Gate 3 endpoint dynamic qualification: corrected dY=0 vs +15mm.

Diagnostics only. Runs corrected descend-refresh, close, and lift from one
reconstructed root. It does not run the five-level sweep and writes only the
conditional close/lift estimand.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import yaml

from scripts.piper_phase2y_driver import (
    gate3_step_record, restore_reconstructed_root,
)
from scripts.piper_phase2y_handoff import Capture, make_env
from scripts.piper_phase2y_variants import body_axes, load, make_variant, register_variant
from tango_robot.piper_robosuite import piper_pick_and_place as ppp

CONFIG = ROOT / "configs/piper/phase2y_clearance_corrected.yaml"
OUT = ROOT / "outputs/phase2y_gate3_endpoint.json"
ENDPOINTS = (0.0, 15.0)
DIVERGENCE_FIELDS = ("qpos", "qvel", "ctrl", "qfrc_actuator", "qacc", "eef_pos",
                     "object_pos", "object_quat")


def corrected_qpos(ik, qpos, delta_z_m):
    saved = ik._get_qpos()
    ik._set_qpos(qpos)
    target_pos = ik.data.site_xpos[ik.eef_site_id].copy() + [0.0, 0.0, delta_z_m]
    target_mat = ik.data.site_xmat[ik.eef_site_id].reshape(3, 3).copy()
    solution, converged, residual, source = ik.solve_multi_seed(
        target_pos, primary_seed=qpos, target_mat=target_mat)
    ik._set_qpos(saved)
    if not converged:
        raise RuntimeError(f"corrected action IK failed: residual={residual}")
    return solution, {"residual_m": residual, "source": source}


def build_corrected_segment(env, bundle, actions, phases, delta_z_m):
    restore_reconstructed_root(env, bundle)
    ik = ppp.ArmIK(env)
    root_qpos, root_ik = corrected_qpos(ik, ik._get_qpos(), delta_z_m)
    selected = [(action, phase) for action, phase in zip(actions, phases)
                if phase in ("descend_refresh", "lift")]
    cache, rows, diagnostics = {}, [], []
    for action, phase in selected:
        key = np.asarray(action[:6]).tobytes()
        if key not in cache:
            cache[key] = corrected_qpos(ik, np.asarray(action[:6]), delta_z_m)
        qpos, diag = cache[key]
        rows.append(np.concatenate([qpos, action[6:]]))
        diagnostics.append({"phase": phase, **diag})
    return root_qpos, rows, diagnostics


def apply_corrected_root(env, bundle, corrected_root_qpos):
    restore_reconstructed_root(env, bundle)
    model, data = env.sim.model._model, env.sim.data._data
    for name, value in zip(ppp.JOINTS, corrected_root_qpos):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    composite = env.robots[0].composite_controller
    composite.update_state()
    for controller in composite.part_controllers.values():
        controller.update(force=True)


def run_branch(env, bundle, corrected_root_qpos, actions):
    apply_corrected_root(env, bundle, corrected_root_qpos)
    object_id = env.object_body_ids["pear"]
    initial_object_z = float(env.sim.data._data.xpos[object_id, 2])
    records = [gate3_step_record(env, "pear", -1)]
    target_onset = None
    for step, action in enumerate(actions):
        env.step(action)
        record = gate3_step_record(env, "pear", step)
        records.append(record)
        if target_onset is None and record["target_finger_contact"]:
            target_onset = step
    final_z = float(env.sim.data._data.xpos[object_id, 2])
    return {
        "records": records, "target_contact_step": target_onset,
        "object_delta_z_m": final_z - initial_object_z,
        "conditional_lift_success": bool(final_z - initial_object_z > 0.07),
    }


def first_divergence(left, right, eps=1e-12):
    for index, (a, b) in enumerate(zip(left, right)):
        maxima = {field: float(np.max(np.abs(np.asarray(a[field]) - np.asarray(b[field]))))
                  for field in DIVERGENCE_FIELDS}
        if max(maxima.values()) > eps:
            return int(a["step"]), maxima
    return None, {}


def non_target_signature(record):
    return sorted((pair["geom1"], pair["geom2"], pair["distance_m"])
                  for pair in record["contact_pairs"]
                  if not pair["target_finger"] and not pair["finger_table"])


def main():
    config = yaml.safe_load(CONFIG.read_text())
    delta_z_m = float(config["correction"]["applied_vertical_mm"]) / 1000.0
    baseline = make_env(5001)
    capture = Capture(baseline)
    try:
        ppp.run_pick_and_place(baseline, "pear", use_oriented_grasp=True, verbose=False,
                               candidate_selection=None, wrist_friendly_orientation=True,
                               step_hook=capture)
        capture._armed = False
        baseline.step = capture._orig_step
        root_qpos, corrected_actions, ik_diagnostics = build_corrected_segment(
            baseline, capture.bundle, tuple(capture.actions), tuple(capture.action_phases),
            delta_z_m)
    finally:
        baseline.close()

    axes, branches, variant_names = body_axes(), {}, {}
    for level in ENDPOINTS:
        path, tag = make_variant(level, axes)
        variant_names[level] = register_variant(path, tag)
        env = load(variant_names[level])
        try:
            branches[str(level)] = run_branch(
                env, capture.bundle, root_qpos, corrected_actions)
        finally:
            env.close()

    replica_env = load(variant_names[0.0])
    try:
        zero_replica = run_branch(replica_env, capture.bundle, root_qpos, corrected_actions)
    finally:
        replica_env.close()

    zero, plus = branches["0.0"], branches["15.0"]
    divergence_step, divergence_magnitudes = first_divergence(zero["records"], plus["records"])
    reconstruction_divergence, _ = first_divergence(zero["records"], zero_replica["records"], eps=0.0)
    onsets = [step for step in (zero["target_contact_step"], plus["target_contact_step"])
              if step is not None]
    earliest_onset = min(onsets) if onsets else None
    precontact = [record for record in zero["records"] if earliest_onset is None or record["step"] < earliest_onset]
    precontact_plus = [record for record in plus["records"] if earliest_onset is None or record["step"] < earliest_onset]
    checks = {
        "cross_instance_reconstruction_replay_exact": reconstruction_divergence is None,
        "both_target_contact_onsets_observed": len(onsets) == 2,
        "first_divergence_not_before_target_contact": (
            divergence_step is not None and earliest_onset is not None
            and divergence_step >= earliest_onset),
        "finger_table_zero_before_target_contact": all(
            not record["finger_table_contact"] for record in precontact + precontact_plus),
        "non_target_contacts_identical_before_target_contact": all(
            non_target_signature(a) == non_target_signature(b)
            for a, b in zip(precontact, precontact_plus)),
        "conditional_lift_success_generated": all(
            isinstance(branch["conditional_lift_success"], bool) for branch in branches.values()),
    }
    report = {
        "audit": "P2Y-Gate3-endpoint-dynamic-qualification",
        "endpoints_dY_mm": list(ENDPOINTS), "delta_z_applied_mm": delta_z_m * 1000,
        "n_corrected_actions": len(corrected_actions),
        "first_divergence_step": divergence_step,
        "reconstruction_replica_first_divergence_step": reconstruction_divergence,
        "first_divergence_magnitudes": divergence_magnitudes,
        "earliest_target_contact_step": earliest_onset,
        "checks": checks, "pass": all(checks.values()),
        "branches": branches, "ik_diagnostics": ik_diagnostics,
        "formal_five_level_sweep_performed": False,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(OUT.relative_to(ROOT)), "pass": report["pass"],
        "first_divergence_step": divergence_step,
        "target_contact_steps": {key: value["target_contact_step"] for key, value in branches.items()},
        "conditional_lift_success": {
            key: value["conditional_lift_success"] for key, value in branches.items()},
        "checks": checks,
    }, indent=2))
    if not report["pass"]:
        raise SystemExit("Gate 3 endpoint qualification failed; five-level sweep remains blocked")


if __name__ == "__main__":
    main()
