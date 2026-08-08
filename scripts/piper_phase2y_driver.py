"""Shared contracts for legal Phase 2Y diagnostic drivers."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def restore_reconstructed_root(env, bundle: dict[str, Any]) -> None:
    """Restore the validated P2Y cross-instance root reconstruction.

    Ordering is part of the contract. ``reset_goal`` is intentionally absent:
    it does not refresh reset-state controller caches.
    """
    model, data = env.sim.model._model, env.sim.data._data
    mujoco.mj_setState(model, data, bundle["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    env.robots[0].gripper["right"].current_action = np.asarray(
        bundle["gripper_current_action"], dtype=float
    ).copy()
    composite = env.robots[0].composite_controller
    composite.update_state()
    for controller in composite.part_controllers.values():
        controller.update(force=True)


def _matching_geom_ids(model, tokens) -> set[int]:
    return {i for i in range(model.ngeom)
            if all(token in (model.geom(i).name or "") for token in tokens)}


def collision_roles(env, target_object: str) -> dict[str, set[int]]:
    """Resolve semantic collision roles and verify MuJoCo mask compatibility."""
    from tango_robot.piper_robosuite import piper_pick_and_place as ppp

    model = env.sim.model._model
    important = env.robots[0].gripper["right"].important_geoms
    finger_names = {
        "finger7": important["left_finger"],
        "finger8": important["right_finger"],
    }
    fingers = {}
    for side, names in finger_names.items():
        ids = {model.geom(name).id for name in names}
        if not ids:
            raise RuntimeError(f"no registered collision geoms for {side}")
        fingers[side] = ids
    table = {model.geom("table_collision").id}
    finger_union = set().union(*fingers.values())
    target_candidates = set(ppp._object_contact_geoms(env, target_object))
    target = {other_id for other_id in target_candidates if any(
        (model.geom_contype[finger_id] & model.geom_conaffinity[other_id]) or
        (model.geom_contype[other_id] & model.geom_conaffinity[finger_id])
        for finger_id in finger_union
    )}
    if not target:
        raise RuntimeError("target object has no finger-compatible collision geoms")
    for finger_id in finger_union:
        for other_id in table | target:
            compatible = ((model.geom_contype[finger_id] & model.geom_conaffinity[other_id]) or
                          (model.geom_contype[other_id] & model.geom_conaffinity[finger_id]))
            if not compatible:
                raise RuntimeError("semantic collision roles are mask-incompatible")
    return {**fingers, "table": table, "target": target}


def classify_contacts(env, target_object: str) -> dict[str, Any]:
    """Classify contacts by explicit geom identities, never raw ``ncon``."""
    from tango_robot.piper_robosuite import piper_pick_and_place as ppp

    model, data = env.sim.model._model, env.sim.data._data
    roles = collision_roles(env, target_object)
    target, table = roles["target"], roles["table"]
    fingers = {side: roles[side] for side in ("finger7", "finger8")}

    result = {"target_finger_contact": False, "target_contact_sides": [],
              "finger_table_contact": False, "non_target_contact_count": 0,
              "pairs": []}
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        sides = [side for side, ids in fingers.items() if g1 in ids or g2 in ids]
        target_pair = bool(sides and ((g1 in target) ^ (g2 in target)))
        table_pair = bool(sides and ((g1 in table) ^ (g2 in table)))
        if target_pair:
            result["target_finger_contact"] = True
            result["target_contact_sides"].extend(sides)
        elif table_pair:
            result["finger_table_contact"] = True
        else:
            result["non_target_contact_count"] += 1
        result["pairs"].append({
            "index": index, "geom1": model.geom(g1).name, "geom2": model.geom(g2).name,
            "distance_m": float(contact.dist), "target_finger": target_pair,
            "finger_table": table_pair,
        })
    result["target_contact_sides"] = sorted(set(result["target_contact_sides"]))
    return result


def gate3_step_record(env, target_object: str, step: int) -> dict[str, Any]:
    """Formal per-step observables required by the future Gate 3 driver."""
    model, data = env.sim.model._model, env.sim.data._data
    contacts = classify_contacts(env, target_object)
    eef = model.site("robot0_eef_site").id
    body = env.object_body_ids[target_object]
    return {
        "step": int(step),
        "target_finger_contact": contacts["target_finger_contact"],
        "target_contact_side": contacts["target_contact_sides"],
        "finger_table_contact": contacts["finger_table_contact"],
        "non_target_contact_count": contacts["non_target_contact_count"],
        "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(), "qfrc_actuator": data.qfrc_actuator.tolist(),
        "qacc": data.qacc.tolist(), "eef_pos": data.site_xpos[eef].tolist(),
        "object_pos": data.xpos[body].tolist(), "object_quat": data.xquat[body].tolist(),
        "contact_pairs": contacts["pairs"],
    }
