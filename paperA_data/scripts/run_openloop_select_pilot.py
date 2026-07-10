#!/usr/bin/env python3
"""
Phase 2, Part C follow-up (see /home/lina/.claude/plans/floating-crunching-yeti.md
and paperA_data/phase2_reactive_autopsy_predictive_blueprint.md): a real physical
pilot of the OPEN-LOOP candidate-selection protocol the retrospective analysis
predicted should work -- as opposed to the REACTIVE real-time correction protocol
already tested (net negative in all 3 rounds).

Protocol per (object, seed):
  1. Spawn the object exactly as tango_robot/ui.py's real evaluation path does
     (env.load_isolated_obj, pos=None default: r_x=U(-0.15,0.15), r_y=U(-0.35,-0.10)).
  2. Generate 5 candidates using the EXACT formula tango_robot/ui.py's
     _setup_grasps_mujoco uses for the Baseline condition (CoM +/- 0.06m in xy,
     yaw ~ U(-pi/2,pi/2), width ~ U(0.04,0.09), seeded with
     np.random.default_rng(seed) -- same as ui.py's own per-episode convention).
  3. Evaluate each of the 5 candidates with ONE CLEAN settle each (reset_robot()
     before every single evaluation, avoiding the non-idempotency Part B3
     documented for repeated in-episode settle calls) and score it with the
     trained bilateral correction model (mpc_correction_bilateral_v1.pt) --
     no candidate is corrected, only scored.
  4. Commit to and physically execute (close + lift + verify) ONLY the
     top-scored candidate, once. No verification, no revert, no reactive loop.

Compares against the locked pilot_baseline_{obj}.jsonl (same 3 objects, same
25 seeds, same candidate-generation seed convention -- but Baseline picks
candidate index 0 / first generated, not model-selected).

Usage:
    conda run -n tango python paperA_data/scripts/run_openloop_select_pilot.py
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from tango_robot.env_soarm import EnvironmentSoArm, GRASP_Z_TABLE_MARGIN

OBJECTS = {"Pear": "pear", "TomatoSoupCan": "can", "CrackerBox": "cracker"}
SEEDS = list(range(1, 26))
CKPT_PATH = "grasp_6dof/models/mpc_correction_bilateral_v1.pt"
N_CANDIDATES = 5
_LGGSN_SPREAD_XY = 0.06


class CorrectionNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = CorrectionNet(in_dim=ckpt["in_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, np.array(ckpt["x_mean"], dtype=np.float32), \
        np.array(ckpt["x_std"], dtype=np.float32), ckpt["objects"]


def featurize(off_x, off_y, cand_yaw, obj_key, objects):
    obj_onehot = [1.0 if obj_key == o else 0.0 for o in objects]
    # delta = 0 (pure candidate scoring, no correction applied)
    return [off_x, off_y, math.sin(cand_yaw), math.cos(cand_yaw),
            0.0, 0.0, math.sin(0.0), math.cos(0.0), *obj_onehot]


def run_one(env_obj_name: str, model_obj_key: str, seed: int, model, x_mean, x_std, objects) -> bool:
    env = EnvironmentSoArm(vis=False, grasp_mode="physics_weld_after_bilateral")
    obj_id = env.load_isolated_obj(env_obj_name, name=env_obj_name)
    env._steps(300)

    try:
        com = env.get_obj_com_pos(obj_id)
    except Exception:
        com = env.get_obj_pos(obj_id)
    gx, gy = float(com[0]), float(com[1])
    gz = float(com[2]) + GRASP_Z_TABLE_MARGIN

    rng = np.random.default_rng(seed)
    candidates = [
        (float(gx + rng.uniform(-_LGGSN_SPREAD_XY, _LGGSN_SPREAD_XY)),
         float(gy + rng.uniform(-_LGGSN_SPREAD_XY, _LGGSN_SPREAD_XY)),
         float(rng.uniform(-math.pi / 2, math.pi / 2)),
         float(rng.uniform(0.04, 0.09)))
        for _ in range(N_CANDIDATES)
    ]

    scores = []
    for (cx, cy, cyaw, cwidth) in candidates:
        env.reset_robot()   # clean state before EVERY evaluation (avoids B3 drift)
        opening = cwidth * env.GRIP_REDUCTION
        env._settle_at_pose(cx, cy, gz, cyaw, opening)
        jaw_mid = env._get_jaw_midpoint()[:2]
        obj_pos = env.get_obj_pos(obj_id)[:2]
        off_x, off_y = float(jaw_mid[0] - obj_pos[0]), float(jaw_mid[1] - obj_pos[1])
        feat = np.array(featurize(off_x, off_y, cyaw, model_obj_key, objects), dtype=np.float32)
        feat_n = (feat - x_mean) / x_std
        with torch.no_grad():
            score = model(torch.tensor(feat_n, dtype=torch.float32).unsqueeze(0)).item()
        scores.append(score)

    best_idx = int(np.argmax(scores))
    bx, by, byaw, bwidth = candidates[best_idx]

    env.reset_robot()   # clean state before the final, real commit
    success, _ = env._execute_grasp_physics_topdown(
        pos=(bx, by, gz), yaw=byaw, gripper_opening_length=bwidth, obj_height=0.05)

    env.close()
    return bool(success), best_idx, scores


def main():
    out_dir = Path("paperA_data/worldmodel_trajs")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, x_mean, x_std, objects = load_model()

    for env_obj_name, model_obj_key in OBJECTS.items():
        raw_path = out_dir / f"pilot_openloop_select_{env_obj_name}.jsonl"
        succ = 0
        with open(raw_path, "w") as fout:
            for seed in SEEDS:
                success, best_idx, scores = run_one(
                    env_obj_name, model_obj_key, seed, model, x_mean, x_std, objects)
                succ += int(success)
                fout.write(json.dumps({
                    "condition": "openloop_select", "object": env_obj_name, "seed": seed,
                    "success": "true" if success else "false",
                    "chosen_candidate_idx": best_idx, "scores": scores,
                }) + "\n")
                print(f"[openloop] {env_obj_name} seed={seed}  chosen={best_idx}  "
                      f"success={success}  ({succ} so far)")
        print(f"[openloop-{env_obj_name}] === DONE: {succ}/{len(SEEDS)} ===")


if __name__ == "__main__":
    main()
