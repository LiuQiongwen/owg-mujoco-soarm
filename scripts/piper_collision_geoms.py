"""Shared collision-geom selection. R7 as an executable guard.

Ad-hoc `'table' in name` matching has now produced two wrong results in
this investigation:
  - robot0_g*_vis reported as non-colliding (they carry contype=1), nearly
    recorded as "the Piper arm has no collision geometry";
  - a table_visual geom pulled into a distance query, returning exactly
    0.000000 and invalidating a clearance margin measurement.

Use these helpers instead of matching names directly. Physics attributes
decide membership; names only filter within the colliding set.
"""
import numpy as np


def collision_geoms(model, *substrings):
    """Geom ids that can actually collide AND match every substring."""
    out = []
    for i in range(model.ngeom):
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        n = model.geom(i).name or ""
        if all(s in n for s in substrings):
            out.append(i)
    return out


def require_nonempty(model, name, ids):
    """A group that should exist but is empty is a matcher fault, not a
    pass. Callers must distinguish this from a genuine model property."""
    if not ids:
        raise ValueError(
            f"collision geom group '{name}' is EMPTY. This is a matcher "
            f"fault unless the model genuinely lacks that geometry -- "
            f"verify before treating as N/A (R7).")
    return ids


def min_distance(model, data, group_a, group_b, distmax=1.0):
    """Minimum signed distance between two COLLIDING geom groups."""
    import mujoco
    require_nonempty(model, "group_a", group_a)
    require_nonempty(model, "group_b", group_b)
    return min(float(mujoco.mj_geomDistance(model, data, a, b, distmax, None))
               for a in group_a for b in group_b)
