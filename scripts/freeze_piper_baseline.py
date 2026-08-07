"""P0: freeze PIPER_BASELINE_V1 -- a provenance manifest for the Piper
simulation platform, so every downstream experiment can state exactly what
it ran against.

Why this exists: four proposed failure mechanisms were withdrawn in quick
succession, and separately the `can` object went from a historical 0/25 to
6/6 today -- almost certainly because the historical runs predate the
2026-07-15 gripper double-scaling fix. Without a recorded platform
identity, there is no way to tell "this result disagrees with that one"
from "those two ran against different simulators".

STANDING RULE established with this freeze: results collected before
2026-07-15 (the gripper double-scaling fix in piper_controller_config.py)
do NOT enter current Piper experiments as baselines. They were produced
with an effective gripper travel of ~0.1mm and are not comparable.

Read-only. Writes calib/piper_baseline_v1.json.

Run:  conda run -n tango python scripts/freeze_piper_baseline.py
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

OUT = ROOT / "calib" / "piper_baseline_v1.json"

ASSETS = [
    "tango_robot/piper_assets/piper_gripper.xml",
    "tango_robot/piper_assets/robot_arm.xml",
    "tango_robot/piper_robosuite/piper_pick_and_place.py",
    "tango_robot/piper_robosuite/piper_gripper.py",
    "tango_robot/piper_robosuite/piper_robot.py",
    "tango_robot/piper_robosuite/piper_controller_config.py",
    "tango_robot/piper_robosuite/piper_multi_object_scene.py",
    "tango_robot/piper_robosuite/piper_ycb_objects.py",
]


def sha256(path):
    p = ROOT / path
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    from tango_robot.piper_robosuite import piper_pick_and_place as ppp
    from tango_robot.piper_robosuite.piper_controller_config import PIPER_JOINT_POSITION_CONFIG

    try:
        import robosuite
        rs_ver = robosuite.__version__
    except Exception:
        rs_ver = None
    import mujoco

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    git_dirty = bool(subprocess.run(["git", "status", "--porcelain",
                                     "tango_robot/piper_robosuite",
                                     "tango_robot/piper_assets"],
                                    cwd=ROOT, capture_output=True, text=True).stdout.strip())

    manifest = {
        "name": "PIPER_BASELINE_V1",
        "frozen_on": "2026-08-07",
        "git_sha": git_sha,
        "piper_tree_dirty_at_freeze": git_dirty,
        "versions": {
            "robosuite": rs_ver,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "asset_sha256_16": {a: sha256(a) for a in ASSETS},
        "control": {
            "controller_config": PIPER_JOINT_POSITION_CONFIG,
            "control_freq_hz": 20,
        },
        "geometry_constants": {
            "SAFE_TRANSIT_Z": ppp.SAFE_TRANSIT_Z,
            "GRASP_HEIGHT_OFFSET": ppp.GRASP_HEIGHT_OFFSET,
            "APPROACH_HEIGHT": ppp.APPROACH_HEIGHT,
            "TRAY_DROP_HEIGHT": ppp.TRAY_DROP_HEIGHT,
            "GRIPPER_OPEN": ppp.GRIPPER_OPEN,
            "GRIPPER_CLOSE": ppp.GRIPPER_CLOSE,
            "REAL_JOINT_LIMITS": [list(t) for t in ppp.REAL_JOINT_LIMITS],
            "OBJECT_TOP_OFFSET": ppp.OBJECT_TOP_OFFSET,
            "IK_TOL": ppp.IK_TOL, "ORI_TOL": ppp.ORI_TOL,
        },
        "measured": {
            "max_inner_face_opening_m": 0.100,
            "note": ("Measured on the live composed model at the actuator "
                     "ctrlrange floor (-0.05). Supersedes the stale '~7.6cm' "
                     "figure in OBJECT_NARROW_AXIS comments and is distinct "
                     "from REAL_GRIP_OPEN_M=0.12 (a real-hardware figure)."),
        },
        "success_definition": (
            "robosuite Lift._check_success inherited unmodified: object "
            "height > table + 0.04m. run_pick_and_place additionally reports "
            "dist_to_tray; 'success' in these experiments is "
            "run_pick_and_place's own returned flag."),
        "standing_rules": [
            "Data collected before 2026-07-15 (gripper double-scaling fix) "
            "is NOT a valid baseline for current Piper experiments.",
            "Variables derived from the outcome (e.g. dist_to_tray, which "
            "success is defined by) must never enter predictive or causal "
            "analysis.",
            "A candidate separator requires pooled signal + within-object "
            "signal + direction consistency across objects before any "
            "intervention is designed around it.",
        ],
        "withdrawn_claims": [
            "transit_high non-convergence as dominant failure mode "
            "(tautological label; AUC 0.50, zero outcome information)",
            "65.6mm eef_site-to-fingertip TCP offset (measurement artifact: "
            "the tip heuristic selected the finger mesh's proximal end)",
            "grasp-height-above-object separator for can/banana (falsified "
            "by intervention: can succeeds 6/6 at that height)",
            "bilateral-at-close as failure explanation (AUC 0.54)",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1, default=str))

    print(f"PIPER_BASELINE_V1 frozen at git {git_sha[:12]} (piper tree dirty: {git_dirty})")
    print(f"  robosuite={rs_ver}  mujoco={mujoco.__version__}  numpy={np.__version__}")
    for a, h in manifest["asset_sha256_16"].items():
        print(f"  {h}  {a}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
