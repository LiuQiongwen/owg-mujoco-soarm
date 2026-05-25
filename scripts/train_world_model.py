#!/usr/bin/env python3
"""Train the WorldModelMLP from diverse benchmark trial data.

Features (9-dim per trial):
  [0:6]  grasp candidate  (x, y, z, yaw, opening, obj_height)
  [6:9]  settled obj_pos  (x, y, z) — from scene_file

Labels: binary success (1 = lifted, 0 = failed).

Usage
-----
    conda run -n owg-mujoco python scripts/train_world_model.py
    conda run -n owg-mujoco python scripts/train_world_model.py --epochs 100 --hidden 128
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── feature extraction ────────────────────────────────────────────────────────

def _load_dataset(results_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from all trials.jsonl files.

    Reconstructs the exact candidate that was executed using the same RNG
    seed as the benchmark runner (seed + 9999).
    """
    from benchmark.runner import SamplingConfig, _sample_candidates
    from owg_robot.env_soarm import TABLE_TOP_Z

    # default sampling config — matches diverse_*.yaml
    cfg_samp = SamplingConfig(
        spread_xy=0.04, z_offset=0.025,
        yaw_lo=-1.5708, yaw_hi=1.5708,
        opening_lo=0.04, opening_hi=0.09,
    )

    class _FakeCfg:
        n_grasp_candidates = 10
        sampling = cfg_samp

    fake_cfg = _FakeCfg()

    Xs, ys = [], []
    skipped = 0

    for rd in results_dirs:
        p = rd / "trials.jsonl"
        if not p.exists():
            continue
        # dedupe by (object, seed, method) — keep last
        seen = {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            seen[(r["object"], r["seed"], r["method"])] = r

        for r in seen.values():
            if r.get("stability_valid") is False:
                continue
            if r.get("success") is None:
                continue

            # load settled obj_pos from scene file
            sf = r.get("scene_file")
            if not sf or not Path(sf).exists():
                skipped += 1
                continue
            scene = json.loads(Path(sf).read_text())
            obj_pos = np.array(scene["obj_pos"], dtype=np.float32)

            # reconstruct candidates (deterministic: seed + 9999)
            seed = r["seed"]
            rng = np.random.default_rng(seed + 9999)
            candidates = _sample_candidates(obj_pos, rng, fake_cfg)

            gi = r.get("grasp_index")
            if gi is None or gi >= len(candidates):
                skipped += 1
                continue

            g = candidates[int(gi)]           # (6,): x,y,z,yaw,opening,H
            feat = np.concatenate([g, obj_pos])  # (9,)
            Xs.append(feat)
            ys.append(float(bool(r["success"])))

    if skipped:
        print(f"  [data] skipped {skipped} trials (missing scene_file or grasp_index)")

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── training ──────────────────────────────────────────────────────────────────

def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from benchmark._wm_mlp import WorldModelMLP

    results_dirs = sorted(ROOT.glob("results/diverse_*"))
    if not results_dirs:
        print("No results/diverse_* directories found. Run the benchmark first.")
        sys.exit(1)

    print(f"Loading data from {[r.name for r in results_dirs]} ...")
    X, y = _load_dataset(results_dirs)
    print(f"  {len(X)} samples  |  pos={y.sum():.0f}  neg={(1-y).sum():.0f}  "
          f"balance={y.mean():.1%}")

    # train/val split (80/20, stratified by label)
    rng = np.random.default_rng(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)

    def _split(idx):
        n = len(idx)
        return idx[:int(n * 0.8)], idx[int(n * 0.8):]

    tr_pos, va_pos = _split(pos_idx)
    tr_neg, va_neg = _split(neg_idx)
    tr_idx = np.concatenate([tr_pos, tr_neg])
    va_idx = np.concatenate([va_pos, va_neg])

    X_tr, y_tr = torch.tensor(X[tr_idx]), torch.tensor(y[tr_idx])
    X_va, y_va = torch.tensor(X[va_idx]), torch.tensor(y[va_idx])

    # normalise features
    mu = X_tr.mean(0, keepdim=True)
    sd = X_tr.std(0, keepdim=True).clamp(min=1e-6)
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd

    train_ds = TensorDataset(X_tr, y_tr.unsqueeze(1))
    val_ds   = TensorDataset(X_va, y_va.unsqueeze(1))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=256)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = WorldModelMLP(input_dim=9, hidden=args.hidden).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # class-weighted BCE to handle imbalance
    pos_weight = torch.tensor([(1 - y.mean()) / y.mean()]).to(device)
    criterion  = nn.BCELoss(reduction="none")

    print(f"\nTraining {model.__class__.__name__}  input=9  hidden={args.hidden}  "
          f"device={device}  epochs={args.epochs}")
    print(f"{'Epoch':>6}  {'loss':>8}  {'val_acc':>8}  {'val_auc':>8}")

    best_val_acc = 0.0
    best_state   = None

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            w    = torch.where(yb == 1, pos_weight, torch.ones_like(yb))
            loss = (criterion(pred, yb) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(xb)
        sched.step()

        if ep % 10 == 0 or ep == args.epochs:
            model.eval()
            all_pred, all_true = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    all_pred.append(model(xb.to(device)).cpu())
                    all_true.append(yb)
            preds = torch.cat(all_pred).squeeze()
            trues = torch.cat(all_true).squeeze()
            val_acc = ((preds > 0.5).float() == trues).float().mean().item()

            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(trues.numpy(), preds.numpy())
            except Exception:
                auc = float("nan")

            avg_loss = total_loss / len(train_ds)
            print(f"{ep:>6}  {avg_loss:>8.4f}  {val_acc:>8.3f}  {auc:>8.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # save best checkpoint
    out = ROOT / "data" / "world_model_ckpt.pt"
    torch.save({
        "model_state": best_state,
        "input_dim":   9,
        "hidden":      args.hidden,
        "norm_mean":   mu.squeeze().tolist(),
        "norm_std":    sd.squeeze().tolist(),
        "feature_names": ["x","y","z","yaw","opening","obj_height",
                          "obj_x","obj_y","obj_z"],
        "val_acc":     best_val_acc,
    }, out)
    print(f"\nSaved → {out}  (val_acc={best_val_acc:.3f})")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--batch",  type=int, default=64)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
