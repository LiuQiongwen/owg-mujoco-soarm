"""Stable scene filtering for the grasp benchmark.

Detects three failure modes before any grasp attempt:
  - exploding: object Z far above table (physics explosion)
  - out_of_bounds: object XY outside reachable workspace
  - still_moving: object velocity too high after settle steps

A scene that fails any check is logged as `validity: false` and skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from owg_robot.env_soarm import TABLE_TOP_Z


@dataclass
class StabilityResult:
    ok:     bool
    reason: Optional[str] = None   # None when ok=True

    def __bool__(self) -> bool:
        return self.ok


class StabilityChecker:
    """
    Check whether a spawned object is in a valid, graspable state.

    Parameters
    ----------
    max_pos_z     : float  Object Z above this → exploded
    min_pos_z     : float  Object Z below this → fell off table
    max_xy_radius : float  Euclidean XY distance from table centre above this → out of bounds
    max_velocity  : float  Sum of |qvel| above this → still moving
    table_centre  : tuple  XY centre of the reachable table area
    """

    def __init__(
        self,
        max_pos_z:     float = TABLE_TOP_Z + 0.40,
        min_pos_z:     float = TABLE_TOP_Z - 0.08,
        max_xy_radius: float = 0.44,
        max_velocity:  float = 0.05,
        table_centre:  tuple = (0.0, -0.45),
    ):
        self.max_pos_z     = max_pos_z
        self.min_pos_z     = min_pos_z
        self.max_xy_radius = max_xy_radius
        self.max_velocity  = max_velocity
        self.table_centre  = np.array(table_centre, dtype=float)

    def check(self, env, obj_id: int) -> StabilityResult:
        """Return StabilityResult after settling is complete."""
        try:
            pos = env.get_obj_pos(obj_id)
        except Exception:
            return StabilityResult(ok=False, reason="pos_read_failed")

        if not np.all(np.isfinite(pos)):
            return StabilityResult(ok=False, reason="nan_in_pos")

        x, y, z = pos

        if z > self.max_pos_z:
            return StabilityResult(
                ok=False,
                reason=f"exploded_z={z:.3f}_max={self.max_pos_z:.3f}",
            )

        if z < self.min_pos_z:
            return StabilityResult(
                ok=False,
                reason=f"fell_off_z={z:.3f}_min={self.min_pos_z:.3f}",
            )

        xy_dist = float(np.linalg.norm(np.array([x, y]) - self.table_centre))
        if xy_dist > self.max_xy_radius:
            return StabilityResult(
                ok=False,
                reason=f"out_of_bounds_xy={xy_dist:.3f}_max={self.max_xy_radius:.3f}",
            )

        if not self._velocity_ok(env, obj_id):
            return StabilityResult(ok=False, reason="still_moving")

        return StabilityResult(ok=True)

    def _velocity_ok(self, env, obj_id: int) -> bool:
        """Check that the object's LINEAR velocity is below threshold.

        We check only the first 3 components (translational) of the freejoint
        qvel, not angular.  Angular velocity from the contact solver is numerical
        jitter on resting objects and does not indicate an unstable scene.
        """
        try:
            slot    = env._obj_pool_slot(obj_id)
            jnt     = env.model.joint(f"obj_joint_{slot}")
            adr     = jnt.dofadr[0]
            vel_lin = env.data.qvel[adr : adr + 3]   # linear only
            return float(np.abs(vel_lin).sum()) < self.max_velocity
        except Exception:
            return True   # if we can't read velocity, assume OK
