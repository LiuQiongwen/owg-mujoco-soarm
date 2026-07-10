#!/usr/bin/env python3
"""
Phase 2, Part C (see /home/lina/.claude/plans/floating-crunching-yeti.md):
retrospective reanalysis of the already-collected correction dataset --
zero new simulation. Compares two ways of using the trained bilateral
correction model's predictions over the 120 candidate groups (1 base +
8 deltas each, real bilateral_contacts already recorded for every row):

  Open-loop / predictive : commit to the model's top-predicted row among
                            the 9, use its REAL outcome, no verification.
  Closed-loop / reactive (idealized): commit to the model's top-predicted
                            row, but only if its REAL outcome is >= the
                            group's own recorded base (delta=0) outcome;
                            otherwise fall back to the base row's REAL
                            outcome. This is the *idealized* best case for
                            the reactive protocol -- it assumes a perfectly
                            idempotent revert (reuses the row's own recorded
                            ground truth rather than re-simulating), which
                            Part B3 showed is NOT what the real system does.

If open-loop performs comparably to (or better than) even this idealized
closed-loop, that is evidence the correction mechanism's unreliability
itself (not just the revert's non-idempotency) limits the reactive
approach -- both problems point the same direction.

Usage:
    conda run -n tango python paperA_data/scripts/analyze_openloop_vs_closedloop.py
"""
import json

import numpy as np
import torch
import torch.nn as nn

OBJECTS = ["pear", "can", "cracker"]
DATA_GLOB = "paperA_data/worldmodel_trajs/mpc_correction_{obj}.jsonl"
CKPT_PATH = "grasp_6dof/models/mpc_correction_bilateral_v1.pt"


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


def load_rows():
    rows = []
    for obj in OBJECTS:
        rows.extend(json.loads(l) for l in open(DATA_GLOB.format(obj=obj)))
    return rows


def featurize(r, objects):
    obj_onehot = [1.0 if r["object"] == o else 0.0 for o in objects]
    return [
        r["base_off_x"], r["base_off_y"],
        np.sin(r["cand_yaw"]), np.cos(r["cand_yaw"]),
        r["delta_x"], r["delta_y"],
        np.sin(r["delta_yaw"]), np.cos(r["delta_yaw"]),
        *obj_onehot,
    ]


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = CorrectionNet(in_dim=ckpt["in_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x_mean = np.array(ckpt["x_mean"], dtype=np.float32)
    x_std = np.array(ckpt["x_std"], dtype=np.float32)
    objects = ckpt["objects"]

    rows = load_rows()
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['object']}_{r['seed']}", []).append(r)

    n_groups = 0
    n_openloop_success, n_closedloop_success, n_baseline_only_success = 0, 0, 0
    n_reverted = 0  # how often the idealized closed-loop protocol reverted

    for g, group_rows in groups.items():
        base_row = next((r for r in group_rows if r["delta_x"] == 0.0 and r["delta_y"] == 0.0
                          and r["delta_yaw"] == 0.0), None)
        if base_row is None:
            continue
        n_groups += 1

        feats = np.array([featurize(r, objects) for r in group_rows], dtype=np.float32)
        feats_n = (feats - x_mean) / x_std
        with torch.no_grad():
            preds = model(torch.tensor(feats_n, dtype=torch.float32)).numpy()
        best_idx = int(np.argmax(preds))
        best_row = group_rows[best_idx]

        # Open-loop: always commit to the model's top choice, no verification
        openloop_outcome = int(best_row["bilateral_contacts"])
        n_openloop_success += openloop_outcome

        # Closed-loop (idealized): keep the top choice only if its REAL
        # outcome >= base's REAL outcome, else revert to base's REAL outcome
        base_outcome = int(base_row["bilateral_contacts"])
        if best_row["bilateral_contacts"] >= base_row["bilateral_contacts"]:
            closedloop_outcome = openloop_outcome
        else:
            closedloop_outcome = base_outcome
            n_reverted += 1
        n_closedloop_success += closedloop_outcome

        n_baseline_only_success += base_outcome

    print(f"[analysis] {n_groups} candidate groups analysed (from {len(rows)} total rows)")
    print(f"[analysis] baseline-only (always delta=0, no model):     "
          f"{n_baseline_only_success}/{n_groups} = {100*n_baseline_only_success/n_groups:.1f}%")
    print(f"[analysis] open-loop / predictive (commit to top choice): "
          f"{n_openloop_success}/{n_groups} = {100*n_openloop_success/n_groups:.1f}%")
    print(f"[analysis] closed-loop / reactive (IDEALIZED, perfect revert): "
          f"{n_closedloop_success}/{n_groups} = {100*n_closedloop_success/n_groups:.1f}%  "
          f"(reverted in {n_reverted}/{n_groups} groups)")
    print(f"\n[analysis] Note: this idealized closed-loop is a BEST CASE -- it assumes a "
          f"perfectly idempotent revert (reuses the group's own recorded base outcome), "
          f"which Part B3's idempotency check showed does NOT hold in the real system "
          f"(mean|drift|=0.015-0.032 in jaw_obj_xy_gap even across repeated/reverted "
          f"_settle_at_pose calls). The real physical pilots (rounds 1-3) reflect the "
          f"actual non-idealized protocol and were all net negative vs baseline.")


if __name__ == "__main__":
    main()
