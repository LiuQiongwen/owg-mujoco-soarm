"""P2Y-4D: dY=0 original-vs-fresh cross-instance forensic audit.

Diagnostics only. This script loads no treatment variant and mutates no model
parameter. It compares state at the first recorded close-segment action
boundary, then decomposes the first physics substep into controller output,
force calculation, and integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from scripts.piper_phase2y_handoff import Capture, make_env, restore
from scripts.piper_phase2y_driver import restore_reconstructed_root
from tango_robot.piper_robosuite import piper_pick_and_place as ppp


def _numeric_snapshot(obj: Any) -> dict[str, Any]:
    """Shallow snapshot of public numeric scalars / ndarrays."""
    out: dict[str, Any] = {}
    for name in sorted(set(dir(obj))):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (bool, int, float, np.number)):
            out[name] = np.asarray(value).copy()
        elif isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            out[name] = value.copy()
    return out


def _python_snapshot(env) -> dict[str, dict[str, Any]]:
    robot = env.robots[0]
    out = {
        "env": _numeric_snapshot(env),
        "robot": _numeric_snapshot(robot),
        "gripper.right": _numeric_snapshot(robot.gripper["right"]),
        "composite_controller": _numeric_snapshot(robot.composite_controller),
    }
    for name, controller in robot.composite_controller.part_controllers.items():
        out[f"controller.{name}"] = _numeric_snapshot(controller)
        for attr in ("interpolator", "interpolator_pos", "interpolator_ori"):
            interpolator = getattr(controller, attr, None)
            if interpolator is not None:
                out[f"controller.{name}.{attr}"] = _numeric_snapshot(interpolator)
    return out


def snapshot(env) -> dict[str, Any]:
    model = env.sim.model._model
    data = env.sim.data._data
    return {
        "model": _numeric_snapshot(model),
        "data": _numeric_snapshot(data),
        "python": _python_snapshot(env),
    }


def _digest(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def diff_numeric(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            rows.append({"field": name, "status": "missing", "left": name in left, "right": name in right})
            continue
        a, b = np.asarray(left[name]), np.asarray(right[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            rows.append({"field": name, "status": "schema", "left_shape": list(a.shape),
                         "right_shape": list(b.shape), "left_dtype": str(a.dtype),
                         "right_dtype": str(b.dtype)})
            continue
        equal = np.array_equal(a, b, equal_nan=True)
        if equal:
            continue
        finite = np.issubdtype(a.dtype, np.number) and a.dtype.kind not in "bc"
        max_abs = None
        first_index = None
        if finite and a.size:
            delta = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
            delta = np.abs(delta).astype(float)
            max_abs = float(np.nanmax(delta))
            mask = ~(np.equal(a, b) | (np.isnan(a) & np.isnan(b)))
            hit = np.argwhere(mask)
            first_index = hit[0].tolist() if len(hit) else None
        rows.append({"field": name, "status": "different", "shape": list(a.shape),
                     "max_abs": max_abs, "first_index": first_index,
                     "left_sha256": _digest(a), "right_sha256": _digest(b)})
    return {"different_count": len(rows), "differences": rows}


def diff_snapshot(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    python_keys = sorted(set(left["python"]) | set(right["python"]))
    return {
        "model": diff_numeric(left["model"], right["model"]),
        "data": diff_numeric(left["data"], right["data"]),
        "python": {
            key: diff_numeric(left["python"].get(key, {}), right["python"].get(key, {}))
            for key in python_keys
        },
    }


def finger_object_contact(env) -> dict[str, Any]:
    model, data = env.sim.model._model, env.sim.data._data
    object_geoms = set(ppp._object_contact_geoms(env, "pear"))
    fingers = {}
    for label in ("finger7_collision", "finger8_collision"):
        fingers[label] = {i for i in range(model.ngeom) if label in (model.geom(i).name or "")}
    pairs = {label: [] for label in fingers}
    for i in range(data.ncon):
        contact = data.contact[i]
        for label, geom_ids in fingers.items():
            if ((contact.geom1 in geom_ids and contact.geom2 in object_geoms) or
                    (contact.geom2 in geom_ids and contact.geom1 in object_geoms)):
                pairs[label].append({"contact_index": i, "geom1": int(contact.geom1),
                                     "geom2": int(contact.geom2), "dist": float(contact.dist)})
    return {"active": {key: bool(value) for key, value in pairs.items()}, "pairs": pairs}


FORCE_FIELDS = ("ctrl", "act", "qacc_warmstart", "qfrc_actuator", "qfrc_bias",
                "qfrc_constraint", "qfrc_applied", "xfrc_applied", "qacc", "qpos", "qvel")


def selected_data(env) -> dict[str, Any]:
    data = env.sim.data._data
    return {name: np.asarray(getattr(data, name)).copy() for name in FORCE_FIELDS if hasattr(data, name)}


def first_substep_probe(env, bundle, action) -> dict[str, Any]:
    restore(env, bundle, forward=True)
    env.robots[0].gripper["right"].current_action = bundle["gripper_current_action"].copy()
    before = selected_data(env)
    contact_before = finger_object_contact(env)
    # Mirror the non-lite robosuite step ordering for one physics substep.
    env.sim.forward()
    env._pre_action(np.asarray(action), policy_step=True)
    after_controller = selected_data(env)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    after_force = selected_data(env)
    mujoco.mj_step(env.sim.model._model, env.sim.data._data)
    after_step = selected_data(env)
    return {"before": before, "after_controller": after_controller,
            "after_force": after_force, "after_step": after_step,
            "finger_object_contact_before": contact_before}


def diff_probe(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {stage: diff_numeric(left[stage], right[stage])
            for stage in ("before", "after_controller", "after_force", "after_step")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2y_4d_cross_instance_audit.json"))
    args = parser.parse_args()

    original_boundary: dict[str, Any] = {}
    original = make_env(args.seed)
    capture = Capture(original, boundary_snapshotter=lambda env, bundle: original_boundary.update(snapshot(env)))
    try:
        ppp.run_pick_and_place(original, "pear", use_oriented_grasp=True, verbose=False,
                               candidate_selection=None, wrist_friendly_orientation=True,
                               step_hook=capture)
        capture._armed = False
        original.step = capture._orig_step
        actions = tuple(capture.actions)
        original_probe = first_substep_probe(original, capture.bundle, actions[0])

        fresh = make_env(args.seed)
        try:
            restore(fresh, capture.bundle, forward=True)
            fresh.robots[0].gripper["right"].current_action = \
                capture.bundle["gripper_current_action"].copy()
            fresh_boundary = snapshot(fresh)
            fresh_probe = first_substep_probe(fresh, capture.bundle, actions[0])
            restore_reconstructed_root(fresh, capture.bundle)
            fresh_synced_boundary = snapshot(fresh)
        finally:
            fresh.close()
    finally:
        original.close()

    report = {
        "audit": "P2Y-4D-cross-instance-state-equivalence",
        "seed": args.seed,
        "treatment": "dY=0-only",
        "n_actions": len(actions),
        "boundary_diff": diff_snapshot(original_boundary, fresh_boundary),
        "boundary_diff_after_forced_controller_sync": diff_snapshot(
            original_boundary, fresh_synced_boundary),
        "first_substep_diff": diff_probe(original_probe, fresh_probe),
        "finger_object_contact_before": {
            "original": original_probe["finger_object_contact_before"],
            "fresh": fresh_probe["finger_object_contact_before"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "model_differences": report["boundary_diff"]["model"]["different_count"],
        "data_differences": report["boundary_diff"]["data"]["different_count"],
        "python_sections_with_differences": sum(
            item["different_count"] > 0 for item in report["boundary_diff"]["python"].values()),
        "first_substep_differences": {
            stage: value["different_count"] for stage, value in report["first_substep_diff"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
