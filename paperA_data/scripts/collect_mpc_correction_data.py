#!/usr/bin/env python3
"""
Phase 1 data collection (MPC-style real-time correction world model, see
/home/lina/.claude/plans/floating-crunching-yeti.md): builds a
(candidate target, correction delta) -> (resettled contact geometry) dataset
by calling EnvironmentSoArm._settle_at_pose() directly, without paying for a
full close/lift/contact-check cycle per sample.

For each (object, seed): spawn the object, sample one candidate grasp target
(same style as scripts/record_trajectory.py's _sample_action), measure its
baseline settled geometry (delta=0), then sample K_DELTAS small corrections
and re-measure. All delta measurements for one candidate reuse the SAME env
instance (cheap: only the settle sub-routine reruns, not full env setup).

Usage:
    conda run -n tango python paperA_data/scripts/collect_mpc_correction_data.py \
        --obj pear --seeds 1-40 --k-deltas 8 \
        --out paperA_data/worldmodel_trajs/mpc_correction_pear.jsonl
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from tango_robot.env_soarm import EnvironmentSoArm, TABLE_TOP_Z

OBJECT_REGISTRY = {
    "banana":   "YcbBanana",
    "pear":     "YcbPear",
    "mustard":  "YcbMustardBottle",
    "cracker":  "YcbCrackerBox",
    "drill":    "YcbPowerDrill",
    "can":      "YcbTomatoSoupCan",
    "cylinder": "YcbMediumClamp",
}

_CENTRE_Y     = -0.40
_SPREAD_XY    = 0.06
_DROP_Z       = TABLE_TOP_Z + 0.12
_SETTLE_STEPS = 300
_Z_OFFSET     = 0.025
_YAW_LO       = -math.pi / 2
_YAW_HI       = math.pi / 2
_OPEN_LO      = 0.04
_OPEN_HI      = 0.09

# Correction search range: small nudges around the candidate, on the same
# order of magnitude as the jaw_obj_xy_gap values observed in practice
# (0.02-0.05m), not the ~0.085m offset that characterised EBM v1's
# catastrophic failure -- this is meant to be a *local* correction, not a
# re-search of the whole candidate space.
_DELTA_XY_RANGE  = 0.03   # metres, uniform(-range, +range) per axis
_DELTA_YAW_RANGE = 0.20   # radians


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obj", required=True, choices=list(OBJECT_REGISTRY))
    p.add_argument("--seeds", required=True,
                   help="Seed range, e.g. '1-40' or comma list '1,2,3'")
    p.add_argument("--k-deltas", type=int, default=8,
                   help="Number of correction deltas sampled per candidate")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _parse_seeds(spec: str) -> list:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    ycb_name = OBJECT_REGISTRY[args.obj]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with open(out_path, "w") as fout:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            env = EnvironmentSoArm(vis=False, grasp_mode="physics_weld_after_bilateral")

            cx = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
            cy = _CENTRE_Y + float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
            obj_id = env.load_obj(ycb_name, name=args.obj, pos=[cx, cy, _DROP_Z])
            env._steps(_SETTLE_STEPS)
            obj_pos = env.get_obj_pos(obj_id)

            # Candidate target (same sampling style as record_trajectory.py)
            cand_x   = float(obj_pos[0] + rng.uniform(-0.04, 0.04))
            cand_y   = float(obj_pos[1] + rng.uniform(-0.04, 0.04))
            cand_z   = float(obj_pos[2] + _Z_OFFSET)
            cand_yaw = float(rng.uniform(_YAW_LO, _YAW_HI))
            opening  = float(rng.uniform(_OPEN_LO, _OPEN_HI)) * env.GRIP_REDUCTION

            env.reset_robot()
            base_metrics, _ = env._settle_at_pose(cand_x, cand_y, cand_z, cand_yaw, opening)
            # Directional offset vector (jaw midpoint - object centre), not just
            # its norm (jaw_obj_xy_gap) -- the scalar gap loses the information
            # needed to predict WHICH WAY a correction should go, only how far
            # off the current settle is. Recomputed from live sim state right
            # after each _settle_at_pose call (object may shift slightly between
            # settles, so re-read obj_pos each time rather than reuse the
            # pre-spawn obj_pos).
            jaw_mid_base = env._get_jaw_midpoint()[:2]
            obj_pos_base = env.get_obj_pos(obj_id)[:2]
            off_base = jaw_mid_base - obj_pos_base

            row_base = {
                "object": args.obj, "seed": seed,
                "cand_x": cand_x, "cand_y": cand_y, "cand_yaw": cand_yaw,
                "delta_x": 0.0, "delta_y": 0.0, "delta_yaw": 0.0,
                "base_jaw_gap": base_metrics.get("jaw_obj_xy_gap"),
                "base_off_x": float(off_base[0]), "base_off_y": float(off_base[1]),
                "jaw_obj_xy_gap": base_metrics.get("jaw_obj_xy_gap"),
                "ori_err_norm": base_metrics.get("ori_err_norm"),
                "symmetry_score": base_metrics.get("symmetry_score"),
                "bilateral_contacts": base_metrics.get("bilateral_contacts"),
            }
            fout.write(json.dumps(row_base) + "\n")
            n_rows += 1

            for _ in range(args.k_deltas):
                dx   = float(rng.uniform(-_DELTA_XY_RANGE, _DELTA_XY_RANGE))
                dy   = float(rng.uniform(-_DELTA_XY_RANGE, _DELTA_XY_RANGE))
                dyaw = float(rng.uniform(-_DELTA_YAW_RANGE, _DELTA_YAW_RANGE))

                env.reset_robot()
                metrics, _ = env._settle_at_pose(
                    cand_x + dx, cand_y + dy, cand_z, cand_yaw + dyaw, opening)

                row = {
                    "object": args.obj, "seed": seed,
                    "cand_x": cand_x, "cand_y": cand_y, "cand_yaw": cand_yaw,
                    "delta_x": dx, "delta_y": dy, "delta_yaw": dyaw,
                    "base_jaw_gap": base_metrics.get("jaw_obj_xy_gap"),
                    "base_off_x": float(off_base[0]), "base_off_y": float(off_base[1]),
                    "jaw_obj_xy_gap": metrics.get("jaw_obj_xy_gap"),
                    "ori_err_norm": metrics.get("ori_err_norm"),
                    "symmetry_score": metrics.get("symmetry_score"),
                    "bilateral_contacts": metrics.get("bilateral_contacts"),
                }
                fout.write(json.dumps(row) + "\n")
                n_rows += 1

            env.close()
            print(f"[collect] {args.obj} seed={seed}  base_gap={base_metrics.get('jaw_obj_xy_gap')}"
                  f"  ({n_rows} rows so far)")

    print(f"[collect] DONE: {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
