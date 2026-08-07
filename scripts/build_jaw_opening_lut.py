"""Build a pure-geometry q <-> true fingertip opening LUT for the SO-101 jaw.

Step ① of the calibration order agreed 2026-08-07: kinematic mapping only, no
contact, no scene, no actuator. Both fingers are rigid, so `true_opening_m(q)`
(tango_robot/jaw_metrology.py) is a function of the hinge angle alone and does
not depend on -- and will not need to be rebuilt when -- any future contact
solver retuning. That is exactly why this step is safe to do before solver
work: it is a fact about the mesh, not about the physics engine's contact
model.

Does not touch env_soarm.py's control path. move_gripper()'s broken linear map
and GRIP_CLOSED/GRIP_OPEN are UNCHANGED by this script -- it only produces a
reference artifact against which scripts/validate_free_space_actuator.py (step
②) and, later, a real API fix (deferred) can be checked.

Output: calib/jaw_opening_lut.json
  forward: [[q_rad, true_opening_m], ...], dense and monotonic
  inverse: same data, meant to be interpolated the other way (opening -> q);
    stored once since the relation is monotonic and invertible by np.interp
    with swapped x/y, not duplicated as a second table.

Run:  conda run -n tango python scripts/build_jaw_opening_lut.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from tango_robot.jaw_metrology import JawMetrology, claimed_opening_m
from tango_robot.env_soarm import GRIP_CLOSED, GRIP_OPEN

SO101_XML = str(Path(__file__).resolve().parent.parent
                / "tango_robot" / "assets" / "so101" / "so101.xml")
N_POINTS = 201   # cheap (geometry-only, no physics stepping); dense LUT
OUT = Path(__file__).resolve().parent.parent / "calib" / "jaw_opening_lut.json"


def main():
    model = mujoco.MjModel.from_xml_path(SO101_XML)
    jm = JawMetrology(model)
    if not jm.available:
        raise SystemExit("JawMetrology unavailable on this model -- cannot build LUT")

    lo, hi = (float(v) for v in model.joint("gripper").range)
    qs = np.linspace(lo, hi, N_POINTS)
    openings = np.array([jm.true_opening_m(q) for q in qs])

    d = np.diff(openings)
    if not np.all(d > 0):
        raise SystemExit(f"true_opening_m is not strictly monotonic in q "
                         f"({int((d <= 0).sum())} non-increasing steps) -- "
                         f"LUT inversion would be ambiguous")

    lut = {
        "source": "tango_robot/jaw_metrology.py:JawMetrology.true_opening_m, "
                  "purely geometric (no contact, no scene)",
        "joint_range_rad": [lo, hi],
        "q_rad": qs.tolist(),
        "true_opening_m": openings.tolist(),
        # for context only -- NOT used to build the table, and not what the
        # table corrects: the legacy linear map, so anyone reading this file
        # can see the gap without recomputing it.
        "legacy_claimed_opening_m": [claimed_opening_m(q) for q in qs],
        "legacy_control_window_rad": [GRIP_CLOSED, GRIP_OPEN],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(lut, indent=1))

    lo_open, hi_open = openings[0], openings[-1]
    win_lo = jm.true_opening_m(GRIP_CLOSED)
    win_hi = jm.true_opening_m(GRIP_OPEN)
    print(f"wrote {OUT}  ({N_POINTS} points)")
    print(f"joint range {lo:.4f}..{hi:.4f} rad -> true opening "
          f"{lo_open*1000:.1f}..{hi_open*1000:.1f} mm")
    print(f"current legacy control window [GRIP_CLOSED={GRIP_CLOSED}, "
          f"GRIP_OPEN={GRIP_OPEN}] rad -> true opening "
          f"{win_lo*1000:.1f}..{win_hi*1000:.1f} mm "
          f"(legacy code claims {claimed_opening_m(GRIP_CLOSED)*1000:.1f}.."
          f"{claimed_opening_m(GRIP_OPEN)*1000:.1f} mm)")


if __name__ == "__main__":
    main()
