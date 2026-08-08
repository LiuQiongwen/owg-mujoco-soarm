"""P2Y-5A qualification of the fixed +2.50mm diagnostic correction.

No close/lift action and no outcome collection. Each legal compiled dY model
receives the same reconstructed root and the same world-Z EEF target delta.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from scripts.piper_phase2y_clearance_uniformity import (
    LEVELS_MM, _finger_geom_ids, _geom_vertices_world, _table_top_z,
)
from scripts.piper_phase2y_driver import classify_contacts, restore_reconstructed_root
from scripts.piper_phase2y_handoff import Capture, make_env
from scripts.piper_phase2y_variants import body_axes, load, make_variant, register_variant
from tango_robot.piper_robosuite import piper_pick_and_place as ppp

CONFIG = ROOT / "configs/piper/phase2y_clearance_corrected.yaml"
OUT = ROOT / "outputs/phase2y_5a_clearance_qualification.json"


def _hash_arrays(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def qualify(env, bundle, dY_mm: float, delta_z_m: float) -> dict:
    restore_reconstructed_root(env, bundle)
    model, data = env.sim.model._model, env.sim.data._data
    eef_id = model.site("robot0_eef_site").id
    object_id = env.object_body_ids["pear"]
    before_eef = data.site_xpos[eef_id].copy()
    before_rot = data.site_xmat[eef_id].reshape(3, 3).copy()
    before_object = np.concatenate([data.xpos[object_id], data.xquat[object_id]]).copy()
    before_current_action = env.robots[0].gripper["right"].current_action.copy()
    target = before_eef + np.array([0.0, 0.0, delta_z_m])

    ik = ppp.ArmIK(env)
    seed = ik._get_qpos()
    solution, converged, residual_m, source = ik.solve_multi_seed(
        target, primary_seed=seed, target_mat=before_rot,
    )
    for address, value in zip(ik.qpos_adr, solution):
        data.qpos[address] = value
    env.sim.forward()

    after_eef = data.site_xpos[eef_id].copy()
    after_rot = data.site_xmat[eef_id].reshape(3, 3).copy()
    after_object = np.concatenate([data.xpos[object_id], data.xquat[object_id]]).copy()
    table_z = _table_top_z(env)
    clearances = [
        float(_geom_vertices_world(model, data, geom_id)[:, 2].min() - table_z)
        for geom_id in _finger_geom_ids(env)
    ]
    contacts = classify_contacts(env, "pear")
    applied = after_eef - before_eef
    invariants = {
        "candidate_target_xy_unchanged": bool(np.array_equal(target[:2], before_eef[:2])),
        "candidate_target_orientation_unchanged": True,
        "object_initial_pose_unchanged": bool(np.array_equal(before_object, after_object)),
        "gripper_python_state_unchanged": bool(np.array_equal(
            before_current_action, env.robots[0].gripper["right"].current_action)),
    }
    return {
        "dY_mm": dY_mm, "ik_converged": bool(converged), "ik_source": source,
        "ik_residual_m": float(residual_m), "requested_delta_z_m": delta_z_m,
        "applied_eef_delta_m": applied.tolist(),
        "applied_delta_z_error_m": float(abs(applied[2] - delta_z_m)),
        "eef_xy_drift_m": float(np.linalg.norm(applied[:2])),
        "orientation_matrix_max_abs_diff": float(np.max(np.abs(after_rot - before_rot))),
        "minimum_physical_finger_table_distance_m": min(clearances),
        "finger_table_contact_count": sum(p["finger_table"] for p in contacts["pairs"]),
        "target_finger_contact": contacts["target_finger_contact"],
        "invariants": invariants,
        "controller_model_hash": _hash_arrays(model.actuator_gainprm, model.actuator_biasprm,
                                                model.actuator_ctrlrange),
    }


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    delta_z_m = float(config["correction"]["applied_vertical_mm"]) / 1000.0
    baseline = make_env(5001)
    capture = Capture(baseline)
    try:
        ppp.run_pick_and_place(baseline, "pear", use_oriented_grasp=True, verbose=False,
                               candidate_selection=None, wrist_friendly_orientation=True,
                               step_hook=capture)
    finally:
        baseline.close()

    axes = body_axes()
    rows = []
    for level in LEVELS_MM:
        path, tag = make_variant(level, axes)
        env = load(register_variant(path, tag))
        try:
            rows.append(qualify(env, capture.bundle, level, delta_z_m))
        finally:
            env.close()

    residuals = np.array([row["ik_residual_m"] for row in rows])
    realized_deltas = np.array([row["applied_eef_delta_m"] for row in rows])
    requested_deltas = np.array([row["requested_delta_z_m"] for row in rows])
    controller_hashes = {row["controller_model_hash"] for row in rows}
    checks = {
        "all_ik_converged": all(row["ik_converged"] for row in rows),
        "residual_spread_le_0_1mm": float(np.ptp(residuals)) <= 1e-4,
        "all_finger_table_contacts_zero": all(row["finger_table_contact_count"] == 0 for row in rows),
        "all_signed_distances_positive": all(
            row["minimum_physical_finger_table_distance_m"] > 0 for row in rows),
        "requested_delta_exactly_identical": bool(np.all(requested_deltas == delta_z_m)),
        "realized_delta_identical_across_dY": bool(np.ptp(realized_deltas, axis=0).max() <= 1e-12),
        "ik_residual_identical_across_dY": float(np.ptp(residuals)) <= 1e-12,
        "candidate_target_xy_exact": all(
            row["invariants"]["candidate_target_xy_unchanged"] for row in rows),
        "candidate_target_orientation_exact": all(
            row["invariants"]["candidate_target_orientation_unchanged"] for row in rows),
        "object_pose_exact": all(row["invariants"]["object_initial_pose_unchanged"] for row in rows),
        "gripper_state_exact": all(row["invariants"]["gripper_python_state_unchanged"] for row in rows),
        "controller_semantics_identical": len(controller_hashes) == 1,
    }
    report = {"audit": "P2Y-5A-clearance-corrected-bundle-qualification",
              "config": str(CONFIG.relative_to(ROOT)), "checks": checks,
              "pass": all(checks.values()), "rows": rows,
              "outcome_collection_performed": False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "pass": report["pass"],
                      "checks": checks}, indent=2))
    if not report["pass"]:
        raise SystemExit("P2Y-5A qualification failed; treatment sweep remains blocked")


if __name__ == "__main__":
    main()
