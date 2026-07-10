#!/usr/bin/env python3
"""
Phase 1 (MPC-style real-time correction world model, see
/home/lina/.claude/plans/floating-crunching-yeti.md): trains a lightweight
regressor that predicts the settled pre-close jaw_obj_xy_gap resulting from
applying a small correction (delta_x, delta_y, delta_yaw) to a candidate
grasp target, given the candidate's own baseline jaw_obj_xy_gap.

This is the offline-validation gate before wiring any correction search into
the physical evaluation harness: the reported held-out MAE is the number
that decides whether Phase 1 proceeds (per the plan's go/no-go at step 3),
not a full physical re-evaluation.

Data: paperA_data/worldmodel_trajs/mpc_correction_{pear,can,cracker}.jsonl
(paperA_data/scripts/collect_mpc_correction_data.py), 360 rows/object
(40 candidates x (1 base + 8 deltas)).

Usage:
    conda run -n tango python train_mpc_correction.py
    conda run -n tango python train_mpc_correction.py --epochs 500
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBJECTS = ["pear", "can", "cracker"]
DATA_GLOB = "paperA_data/worldmodel_trajs/mpc_correction_{obj}.jsonl"
CKPT_PATH = "grasp_6dof/models/mpc_correction_v1.pt"


class CorrectionNet(nn.Module):
    """MLP: (base_jaw_gap, sin/cos(cand_yaw), delta_x, delta_y, sin/cos(delta_yaw),
    obj_onehot(3)) -> predicted post-correction jaw_obj_xy_gap."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_dataset():
    rows = []
    for obj in OBJECTS:
        path = DATA_GLOB.format(obj=obj)
        rows.extend(json.loads(l) for l in open(path))
    print(f"[Data] {len(rows)} rows across {OBJECTS}")

    X, y, group_id, base_gap = [], [], [], []
    for r in rows:
        obj_onehot = [1.0 if r["object"] == o else 0.0 for o in OBJECTS]
        # base_off_x/y (the directional jaw-midpoint - object-centre offset
        # vector) replaces the scalar base_jaw_gap as model input: the scalar
        # magnitude alone can't tell the model WHICH WAY a correction should
        # go, only how far off the current settle is -- this was diagnosed via
        # a first training pass whose top-1 delta-selection accuracy (16.7%)
        # was barely above the ~11% chance level for 9 options/group.
        feat = [
            r["base_off_x"], r["base_off_y"],
            np.sin(r["cand_yaw"]), np.cos(r["cand_yaw"]),
            r["delta_x"], r["delta_y"],
            np.sin(r["delta_yaw"]), np.cos(r["delta_yaw"]),
            *obj_onehot,
        ]
        X.append(feat)
        y.append(r["jaw_obj_xy_gap"])
        group_id.append(f"{r['object']}_{r['seed']}")
        base_gap.append(r["base_jaw_gap"])
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(group_id), np.array(base_gap, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y, group_id, base_gap = load_dataset()
    rng = np.random.default_rng(args.seed)
    # Split by CANDIDATE group (object, seed), not by row -- each candidate
    # contributes 9 rows (1 base + 8 deltas) sharing the same base_jaw_gap,
    # so a row-level split would leak: a candidate's other-delta rows in the
    # training set would let the model partially memorize that candidate's
    # base_jaw_gap rather than genuinely learn the delta -> outcome mapping.
    # Grouped split matches the actual use case: predicting deltas for a
    # candidate never seen during training.
    groups = np.unique(group_id)
    rng.shuffle(groups)
    n_test_groups = max(1, int(len(groups) * args.test_frac))
    test_groups = set(groups[:n_test_groups])
    te_idx = np.array([i for i, g in enumerate(group_id) if g in test_groups])
    tr_idx = np.array([i for i, g in enumerate(group_id) if g not in test_groups])

    x_mean, x_std = X[tr_idx].mean(0), X[tr_idx].std(0).clip(min=1e-6)
    Xn = (X - x_mean) / x_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)

    model = CorrectionNet(in_dim=X.shape[1]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    Xtr, ytr = Xt[tr_idx], yt[tr_idx]
    Xte, yte = Xt[te_idx], yt[te_idx]
    print(f"[Train] n_train={len(tr_idx)}  n_test={len(te_idx)}  device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pred = model(Xtr)
        loss = F.mse_loss(pred, ytr)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_te = model(Xte)
                mae_te = (pred_te - yte).abs().mean().item()
                mae_tr = (model(Xtr) - ytr).abs().mean().item()
            print(f"  epoch {epoch:4d}/{args.epochs}  train_loss={loss.item():.6f}"
                  f"  train_MAE={mae_tr:.4f}  test_MAE={mae_te:.4f}")

    # Baseline comparison: "predict no change" (i.e. predicted gap = base_jaw_gap)
    naive_mae = np.abs(base_gap[te_idx] - y[te_idx]).mean()

    model.eval()
    with torch.no_grad():
        pred_te = model(Xte).cpu().numpy()
    model_mae = np.abs(pred_te - y[te_idx]).mean()

    print(f"\n[Eval] Held-out test MAE: model={model_mae:.4f}  "
          f"naive 'no-change' baseline={naive_mae:.4f}  "
          f"(model should beat naive to be useful for correction search)")

    # Ranking accuracy: the actual use case is "among the deltas tried for one
    # candidate, does the model pick the delta with the lowest REAL gap?" --
    # this is what the correction search cares about, not raw MAE.
    te_groups = np.unique(group_id[te_idx])
    n_correct_top1, n_groups_checked = 0, 0
    for g in te_groups:
        gi = te_idx[group_id[te_idx] == g]
        if len(gi) < 2:
            continue
        true_best = gi[np.argmin(y[gi])]
        local_pred = model(Xt[gi]).detach().cpu().numpy()
        pred_best = gi[np.argmin(local_pred)]
        n_groups_checked += 1
        if pred_best == true_best:
            n_correct_top1 += 1
    if n_groups_checked:
        print(f"[Eval] Top-1 delta-selection accuracy (model picks the actually-best "
              f"delta among candidates tried): {n_correct_top1}/{n_groups_checked} "
              f"({100*n_correct_top1/n_groups_checked:.1f}%) -- chance level for "
              f"~9 options/group is ~{100/9:.1f}%")

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "objects": OBJECTS, "in_dim": X.shape[1],
        "model_mae": float(model_mae), "naive_mae": float(naive_mae),
    }, CKPT_PATH)
    print(f"[Save] -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
