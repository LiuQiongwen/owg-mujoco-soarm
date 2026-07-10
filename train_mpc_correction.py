#!/usr/bin/env python3
"""
Phase 1 (MPC-style real-time correction world model, see
/home/lina/.claude/plans/floating-crunching-yeti.md): trains a lightweight
model that predicts the settled pre-close outcome resulting from applying a
small correction (delta_x, delta_y, delta_yaw) to a candidate grasp target.

Two targets, selectable via --target:
  gap       (original) regress jaw_obj_xy_gap, MSE loss.
  bilateral (default, added after the physical pilot) classify
            bilateral_contacts directly, BCE loss.

Why bilateral was added: the gap-regression model passed its own offline
gate (37.5% top-1 delta-selection accuracy vs ~11% chance) but a 3-object
n=25 physical pilot (paperA_data/scripts/run_mpc_correction_pilot.sh) showed
the correction mechanism net HURT success rate (-9.3pp pooled, driven by
Pear at -28pp) -- improving a continuous geometric proxy (jaw_obj_xy_gap)
does not necessarily improve what we actually care about (bilateral
contact / success). Training against bilateral_contacts directly removes
that proxy mismatch. No new data collection needed -- bilateral_contacts
was already recorded in every row by collect_mpc_correction_data.py.

This is the offline-validation gate before wiring any correction search into
the physical evaluation harness: the reported held-out metric is the number
that decides whether Phase 1 proceeds, not a full physical re-evaluation.

Data: paperA_data/worldmodel_trajs/mpc_correction_{pear,can,cracker}.jsonl
(paperA_data/scripts/collect_mpc_correction_data.py), 360 rows/object
(40 candidates x (1 base + 8 deltas)).

Usage:
    conda run -n tango python train_mpc_correction.py                    # bilateral (default)
    conda run -n tango python train_mpc_correction.py --target gap       # original regression
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
CKPT_PATH_GAP       = "grasp_6dof/models/mpc_correction_v1.pt"
CKPT_PATH_BILATERAL = "grasp_6dof/models/mpc_correction_bilateral_v1.pt"


class CorrectionNet(nn.Module):
    """MLP: (base_off_x/y, sin/cos(cand_yaw), delta_x, delta_y, sin/cos(delta_yaw),
    obj_onehot(3)) -> single logit. Regression (gap target) or classification
    (bilateral target, sigmoid applied at inference/loss time) share this
    architecture; only the loss and activation differ."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_dataset(target: str):
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
        y.append(r["jaw_obj_xy_gap"] if target == "gap" else float(r["bilateral_contacts"]))
        group_id.append(f"{r['object']}_{r['seed']}")
        base_gap.append(r["base_jaw_gap"])
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(group_id), np.array(base_gap, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["gap", "bilateral"], default="bilateral")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ckpt_path = CKPT_PATH_GAP if args.target == "gap" else CKPT_PATH_BILATERAL

    X, y, group_id, base_gap = load_dataset(args.target)
    rng = np.random.default_rng(args.seed)
    # Split by CANDIDATE group (object, seed), not by row -- each candidate
    # contributes 9 rows (1 base + 8 deltas) sharing the same base features,
    # so a row-level split would leak: a candidate's other-delta rows in the
    # training set would let the model partially memorize that candidate's
    # outcome rather than genuinely learn the delta -> outcome mapping.
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
    print(f"[Train] target={args.target}  n_train={len(tr_idx)}  n_test={len(te_idx)}  device={device}")

    pos_weight = None
    if args.target == "bilateral":
        pos_rate = ytr.mean().item()
        pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), device=device)
        print(f"[Train] bilateral positive rate (train)={pos_rate:.3f}  pos_weight={pos_weight.item():.2f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pred = model(Xtr)
        if args.target == "gap":
            loss = F.mse_loss(pred, ytr)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, ytr, pos_weight=pos_weight)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_te = model(Xte)
                if args.target == "gap":
                    metric_te = (pred_te - yte).abs().mean().item()
                    metric_tr = (model(Xtr) - ytr).abs().mean().item()
                    label = "MAE"
                else:
                    metric_te = ((torch.sigmoid(pred_te) > 0.5).float() == yte).float().mean().item()
                    metric_tr = ((torch.sigmoid(model(Xtr)) > 0.5).float() == ytr).float().mean().item()
                    label = "acc"
            print(f"  epoch {epoch:4d}/{args.epochs}  train_loss={loss.item():.6f}"
                  f"  train_{label}={metric_tr:.4f}  test_{label}={metric_te:.4f}")

    model.eval()
    with torch.no_grad():
        pred_te = model(Xte).cpu().numpy()

    if args.target == "gap":
        naive_mae = np.abs(base_gap[te_idx] - y[te_idx]).mean()
        model_mae = np.abs(pred_te - y[te_idx]).mean()
        print(f"\n[Eval] Held-out test MAE: model={model_mae:.4f}  "
              f"naive 'no-change' baseline={naive_mae:.4f}  "
              f"(model should beat naive to be useful for correction search)")
        naive_mae_v, model_mae_v = float(naive_mae), float(model_mae)
    else:
        prob_te = 1.0 / (1.0 + np.exp(-pred_te))
        pred_label = (prob_te > 0.5).astype(float)
        acc = (pred_label == y[te_idx]).mean()
        naive_acc = max(y[te_idx].mean(), 1 - y[te_idx].mean())  # always-majority-class baseline
        print(f"\n[Eval] Held-out test accuracy: model={acc:.4f}  "
              f"naive 'always-majority-class' baseline={naive_acc:.4f}")
        naive_mae_v, model_mae_v = float(naive_acc), float(acc)

    # Ranking accuracy: the actual use case is "among the deltas tried for one
    # candidate, does the model pick the delta with the best REAL outcome?" --
    # this is what the correction search cares about, not raw MAE/accuracy.
    # For gap: best = lowest jaw_obj_xy_gap. For bilateral: best = achieves
    # contact (label=1); only groups with variance (some 1s, some 0s) are
    # informative -- an all-0 or all-1 group has no ranking signal to test.
    te_groups = np.unique(group_id[te_idx])
    n_correct_top1, n_groups_checked = 0, 0
    for g in te_groups:
        gi = te_idx[group_id[te_idx] == g]
        if len(gi) < 2:
            continue
        if args.target == "bilateral" and len(set(y[gi].tolist())) < 2:
            continue  # no variance in this group, skip
        local_pred = model(Xt[gi]).detach().cpu().numpy()
        if args.target == "gap":
            true_best = gi[np.argmin(y[gi])]
            pred_best = gi[np.argmin(local_pred)]
        else:
            true_best = gi[np.argmax(y[gi])]
            pred_best = gi[np.argmax(local_pred)]
        n_groups_checked += 1
        if y[pred_best] == y[true_best]:  # matches best ACHIEVABLE label, not exact index
            n_correct_top1 += 1
    if n_groups_checked:
        print(f"[Eval] Top-1 delta-selection accuracy (model's chosen delta matches "
              f"the best achievable outcome among candidates tried): "
              f"{n_correct_top1}/{n_groups_checked} ({100*n_correct_top1/n_groups_checked:.1f}%) "
              f"-- chance level for ~9 options/group is ~{100/9:.1f}%")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "objects": OBJECTS, "in_dim": X.shape[1], "target": args.target,
        "model_metric": model_mae_v, "naive_metric": naive_mae_v,
    }, ckpt_path)
    print(f"[Save] -> {ckpt_path}")


if __name__ == "__main__":
    main()
