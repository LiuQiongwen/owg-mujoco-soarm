"""P2Y-5B: localize first-step loss of dynamic table clearance.

dY=0 only, no treatment comparison and no margin change. This decomposes the
corrected root, first corrected waypoint, controller goal update, and qvel / hold
ablations without collecting a close/lift outcome.
"""
from __future__ import annotations

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
    _finger_geom_ids, _geom_vertices_world, _table_top_z,
)
from scripts.piper_phase2y_driver import classify_contacts
from scripts.piper_phase2y_gate3_endpoint import (
    apply_corrected_root, build_corrected_segment,
)
from scripts.piper_phase2y_handoff import Capture, make_env
from scripts.piper_phase2y_variants import body_axes, load, make_variant, register_variant
from tango_robot.piper_robosuite import piper_pick_and_place as ppp

CONFIG = ROOT / "configs/piper/phase2y_clearance_corrected.yaml"
OUT = ROOT / "outputs/phase2y_5b_dynamic_clearance_audit.json"


def state_summary(env):
    model, data = env.sim.model._model, env.sim.data._data
    eef = model.site("robot0_eef_site").id
    clearances = [float(_geom_vertices_world(model, data, gid)[:, 2].min() - _table_top_z(env))
                  for gid in _finger_geom_ids(env)]
    contacts = classify_contacts(env, "pear")
    arm_dofs = [model.joint(name).dofadr[0] for name in ppp.JOINTS]
    grip_dofs = [model.joint(name).dofadr[0] for name in
                 ("gripper0_right_joint7", "gripper0_right_joint8")]
    return {
        "eef_pos_m": data.site_xpos[eef].tolist(),
        "eef_linear_velocity_m_s": env.sim.data.get_site_xvelp("robot0_eef_site").tolist(),
        "arm_qvel": data.qvel[arm_dofs].tolist(), "gripper_qvel": data.qvel[grip_dofs].tolist(),
        "minimum_finger_table_signed_distance_m": min(clearances),
        "finger_table_contact": contacts["finger_table_contact"],
        "finger_table_pairs": [p for p in contacts["pairs"] if p["finger_table"]],
        "ctrl": data.ctrl.tolist(),
    }


def controller_goals(env):
    out = {}
    for name, controller in env.robots[0].composite_controller.part_controllers.items():
        goal = getattr(controller, "goal_qpos", getattr(controller, "goal_qvel", None))
        out[name] = None if goal is None else np.asarray(goal).tolist()
    return out


def static_target_summary(env, bundle, corrected_root_qpos, target_qpos):
    apply_corrected_root(env, bundle, corrected_root_qpos)
    model, data = env.sim.model._model, env.sim.data._data
    for name, value in zip(ppp.JOINTS, target_qpos):
        data.qpos[model.joint(name).qposadr[0]] = value
    env.sim.forward()
    return state_summary(env)


def one_control_step(env, bundle, corrected_root_qpos, action, qvel_mode):
    apply_corrected_root(env, bundle, corrected_root_qpos)
    model, data = env.sim.model._model, env.sim.data._data
    if qvel_mode == "arm_zero":
        for name in ppp.JOINTS:
            data.qvel[model.joint(name).dofadr[0]] = 0.0
    elif qvel_mode == "all_zero":
        data.qvel[:] = 0.0
    elif qvel_mode != "restored":
        raise ValueError(qvel_mode)
    env.sim.forward()
    for controller in env.robots[0].composite_controller.part_controllers.values():
        controller.update(force=True)
    before = state_summary(env)
    goals_before = controller_goals(env)
    env.step(action)
    return {"before": before, "goals_before": goals_before,
            "after": state_summary(env), "goals_after": controller_goals(env)}


def goal_update_without_integration(env, bundle, corrected_root_qpos, action):
    apply_corrected_root(env, bundle, corrected_root_qpos)
    before = {"state": state_summary(env), "goals": controller_goals(env)}
    env.sim.forward()
    env._pre_action(np.asarray(action), policy_step=True)
    return {"before": before, "after_set_goal": {
        "state": state_summary(env), "goals": controller_goals(env)}}


def main():
    delta_z_m = yaml.safe_load(CONFIG.read_text())["correction"]["applied_vertical_mm"] / 1000.0
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

    axes = body_axes()
    path, tag = make_variant(0.0, axes)
    name = register_variant(path, tag)

    def fresh():
        return load(name)

    env = fresh()
    try:
        root_static = static_target_summary(env, capture.bundle, root_qpos, root_qpos)
        first_target_static = static_target_summary(
            env, capture.bundle, root_qpos, corrected_actions[0][:6])
        no_integrate = goal_update_without_integration(
            env, capture.bundle, root_qpos, corrected_actions[0])
    finally:
        env.close()

    probes = {}
    for label, action, qvel_mode in (
        ("first_action_restored_qvel", corrected_actions[0], "restored"),
        ("first_action_arm_qvel_zero", corrected_actions[0], "arm_zero"),
        ("first_action_all_qvel_zero", corrected_actions[0], "all_zero"),
        ("hold_root_restored_qvel", np.concatenate([root_qpos, corrected_actions[0][6:]]), "restored"),
        ("hold_root_all_qvel_zero", np.concatenate([root_qpos, corrected_actions[0][6:]]), "all_zero"),
    ):
        env = fresh()
        try:
            probes[label] = one_control_step(
                env, capture.bundle, root_qpos, action, qvel_mode)
        finally:
            env.close()

    report = {
        "audit": "P2Y-5B-dynamic-clearance-preservation",
        "dY_mm": 0.0, "delta_z_mm": delta_z_m * 1000,
        "root_static": root_static, "first_commanded_target_static": first_target_static,
        "goal_update_without_integration": no_integrate, "one_control_step_probes": probes,
        "first_action_ik": ik_diagnostics[0], "treatment_comparison_performed": False,
        "margin_changed": False,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "out": str(OUT.relative_to(ROOT)),
        "root_clearance_mm": root_static["minimum_finger_table_signed_distance_m"] * 1000,
        "first_target_static_clearance_mm": (
            first_target_static["minimum_finger_table_signed_distance_m"] * 1000),
        "root_eef_z_m": root_static["eef_pos_m"][2],
        "first_target_eef_z_m": first_target_static["eef_pos_m"][2],
        "root_eef_vz_m_s": root_static["eef_linear_velocity_m_s"][2],
        "probe_after_clearance_mm": {
            key: value["after"]["minimum_finger_table_signed_distance_m"] * 1000
            for key, value in probes.items()},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
