"""P2Y clearance-correction uniformity probe across five legal dY variants.

Geometry/contact diagnostic only: capture one dY=0 handoff root, reconstruct
that root into each separately compiled model, and measure the minimum vertical
translation needed to place both finger collision meshes above the table.
No action is stepped after reconstruction and no outcome is generated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from scripts.piper_phase2y_driver import classify_contacts, collision_roles, restore_reconstructed_root
from scripts.piper_phase2y_handoff import Capture, make_env
from scripts.piper_phase2y_variants import body_axes, make_variant, register_variant, load
from tango_robot.piper_robosuite import piper_pick_and_place as ppp

LEVELS_MM = (-15.0, -7.5, 0.0, 7.5, 15.0)


def _finger_geom_ids(env) -> list[int]:
    roles = collision_roles(env, "pear")
    return sorted(roles["finger7"] | roles["finger8"])


def _table_top_z(env) -> float:
    model, data = env.sim.model._model, env.sim.data._data
    table_id = next(iter(collision_roles(env, "pear")["table"]))
    if model.geom_type[table_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise RuntimeError("registered table collision geom is not a box")
    return float(data.geom_xpos[table_id, 2] + model.geom_size[table_id, 2])


def _geom_vertices_world(model, data, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise RuntimeError(f"finger geom {model.geom(geom_id).name} is not a mesh")
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start:start + count], dtype=float)
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    return vertices @ rotation.T + np.asarray(data.geom_xpos[geom_id])


def measure(env, dY_mm: float, margin_m: float) -> dict:
    model, data = env.sim.model._model, env.sim.data._data
    table_z = _table_top_z(env)
    fingers = []
    for geom_id in _finger_geom_ids(env):
        min_z = float(_geom_vertices_world(model, data, geom_id)[:, 2].min())
        fingers.append({
            "name": model.geom(geom_id).name,
            "minimum_mesh_z_m": min_z,
            "clearance_to_table_m": min_z - table_z,
            "required_vertical_correction_m": max(0.0, table_z + margin_m - min_z),
        })
    contacts = classify_contacts(env, "pear")
    table_pairs = [p for p in contacts["pairs"] if p["finger_table"]]
    return {
        "dY_mm": dY_mm, "table_top_z_m": table_z, "requested_margin_m": margin_m,
        "fingers": fingers,
        "required_vertical_correction_m": max(f["required_vertical_correction_m"] for f in fingers),
        "finger_table_contact_count": len(table_pairs),
        "maximum_contact_penetration_m": max([0.0] + [-p["distance_m"] for p in table_pairs]),
        "target_finger_contact": contacts["target_finger_contact"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--margin-mm", type=float, default=0.0)
    parser.add_argument("--uniform-tolerance-mm", type=float, default=None,
                        help="Optional preregistered tolerance; omitted means no binary verdict")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/phase2y_clearance_uniformity.json"))
    args = parser.parse_args()

    baseline = make_env(args.seed)
    capture = Capture(baseline)
    try:
        ppp.run_pick_and_place(baseline, "pear", use_oriented_grasp=True, verbose=False,
                               candidate_selection=None, wrist_friendly_orientation=True,
                               step_hook=capture)
    finally:
        baseline.close()

    axes = body_axes()
    rows = []
    variant_paths = {}
    for level in LEVELS_MM:
        path, tag = make_variant(level, axes)
        variant_paths[str(level)] = str(path.relative_to(ROOT))
        name = register_variant(path, tag)
        env = load(name)
        try:
            restore_reconstructed_root(env, capture.bundle)
            rows.append(measure(env, level, args.margin_mm / 1000.0))
        finally:
            env.close()

    corrections_mm = np.asarray([row["required_vertical_correction_m"] * 1000 for row in rows])
    spread_mm = float(corrections_mm.max() - corrections_mm.min())
    common_correction_mm = float(corrections_mm.max())
    relative_spread = spread_mm / float(corrections_mm.mean())
    for row, correction_mm in zip(rows, corrections_mm):
        row["common_correction_headroom_mm"] = common_correction_mm - float(correction_mm)
    report = {
        "audit": "P2Y-clearance-correction-uniformity",
        "seed": args.seed, "levels_mm": list(LEVELS_MM), "variant_paths": variant_paths,
        "margin_mm": args.margin_mm, "uniform_tolerance_mm": args.uniform_tolerance_mm,
        "correction_spread_mm": spread_mm,
        "relative_spread": relative_spread,
        "exactly_uniform": bool(spread_mm == 0.0),
        "uniform_within_tolerance": (None if args.uniform_tolerance_mm is None
                                     else bool(spread_mm <= args.uniform_tolerance_mm)),
        "minimum_common_correction_mm": common_correction_mm,
        "rows": rows,
        "interpretation": "geometry-only; no close/lift treatment outcome was executed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "corrections_mm": corrections_mm.tolist(),
        "spread_mm": spread_mm, "relative_spread": relative_spread,
        "minimum_common_correction_mm": common_correction_mm,
        "uniform_within_tolerance": report["uniform_within_tolerance"],
    }, indent=2))


if __name__ == "__main__":
    main()
