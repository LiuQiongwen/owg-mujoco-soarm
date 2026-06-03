#!/usr/bin/env python3
"""
LGGSN v6 — 18-dim (v5d 17-dim geom + pe_ik) + 4-dim query embedding.

Changes from v5d:
  - 18th feature: pe_ik (IK position error for CoM target, episode-level)
  - Trained on lggsn_candidates_v6.jsonl (5 objects × 100 seeds)
  - geom_dim=18, mlp_in=22

Baseline: v3 val pair_acc = 0.660  (17-dim, no query embedding)
          v5d val pair_acc = 0.750  (17-dim + query emb, dropout=0.2)
"""

import collections
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lggsn_model import LGGSN

# ── config ────────────────────────────────────────────────────────────────────
JSONL_PATH = os.environ.get(
    "LGGSN_JSONL", "grasp_6dof/dataset/lggsn_candidates_v6.jsonl"
)
CKPT_PATH  = os.environ.get(
    "LGGSN_CKPT",  "grasp_6dof/models/lggsn_pairwise_live_v6.pt"
)
N_EPOCHS   = int(os.environ.get("LGGSN_EPOCHS", 30))
LR         = float(os.environ.get("LGGSN_LR", 1e-3))
BATCH      = int(os.environ.get("LGGSN_BATCH", 32))
DROPOUT    = float(os.environ.get("LGGSN_DROPOUT", "0.2"))
SEED       = 42

OBJECT_CLASSES = {
    "banana":   0,
    "can":      1,
    "cracker":  2,
    "cylinder": 3,
    "drill":    4,
    "mustard":  5,
    "pear":     6,
}
TRAIN_OBJECTS = {"banana", "can", "drill", "mustard", "pear"}

FEATURE_COLS = [
    "x", "y", "z",
    "roll", "pitch", "yaw",
    "width", "score",
    "dz", "dz_lift", "need_dz", "H",
    "dist_to_centroid", "z_rel",
    "local_point_density", "normal_consistency", "contact_width_ratio",
    "pe_ik",
]
assert len(FEATURE_COLS) == 18
# ─────────────────────────────────────────────────────────────────────────────


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q_id, pos, neg = self.pairs[idx]
        return (
            torch.tensor(pos, dtype=torch.float32),
            torch.tensor(neg, dtype=torch.float32),
            torch.tensor(q_id, dtype=torch.long),
        )


def load_episodes(path):
    rows = [json.loads(l) for l in open(path)]

    ep_rows = collections.defaultdict(list)
    for r in rows:
        if r["query"] not in TRAIN_OBJECTS:
            continue
        ep_rows[(r["query"], r["scene_id"])].append(r)

    ep_by_query = collections.defaultdict(lambda: {"pos": [], "neg": [], "class_id": 0})
    for (query, _sid), cands in ep_rows.items():
        labels = [c["label"] for c in cands]
        n_pos  = sum(labels)
        n_neg  = len(labels) - n_pos
        if n_pos == n_neg:
            continue
        ep_label = 1 if n_pos > n_neg else 0

        feats = [[c[f] for f in FEATURE_COLS] for c in cands]
        side  = "pos" if ep_label == 1 else "neg"
        ep_by_query[query][side].append(feats)
        ep_by_query[query]["class_id"] = OBJECT_CLASSES[query]

    return ep_by_query


def build_pairs(ep_by_query, val_frac=0.2, seed=SEED):
    rng = random.Random(seed)
    train_pairs, val_pairs = [], []

    for query, sides in ep_by_query.items():
        pos_eps  = sides["pos"][:]
        neg_eps  = sides["neg"][:]
        class_id = sides["class_id"]
        if not pos_eps or not neg_eps:
            continue

        rng.shuffle(pos_eps)
        rng.shuffle(neg_eps)

        n_pos_val = max(1, round(len(pos_eps) * val_frac))
        n_neg_val = max(1, round(len(neg_eps) * val_frac))
        pos_val, pos_train = pos_eps[:n_pos_val], pos_eps[n_pos_val:]
        neg_val, neg_train = neg_eps[:n_neg_val], neg_eps[n_neg_val:]

        def cartesian(pos_list, neg_list):
            out = []
            for p_ep in pos_list:
                for n_ep in neg_list:
                    for p_feat in p_ep:
                        for n_feat in n_ep:
                            out.append((class_id, p_feat, n_feat))
            return out

        train_pairs.extend(cartesian(pos_train, neg_train))
        val_pairs.extend(cartesian(pos_val, neg_val))

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    return train_pairs, val_pairs


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_correct, n_total = 0.0, 0, 0
    for pos_geom, neg_geom, q_id in loader:
        pos_geom = pos_geom.to(device)
        neg_geom = neg_geom.to(device)
        q_id     = q_id.to(device)

        with torch.set_grad_enabled(is_train):
            logit_pos = model(pos_geom, q_id).view(-1)
            logit_neg = model(neg_geom, q_id).view(-1)
            loss = -F.logsigmoid(logit_pos - logit_neg).mean()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(q_id)
        n_correct  += (logit_pos > logit_neg).sum().item()
        n_total    += len(q_id)

    return total_loss / max(n_total, 1), n_correct / max(n_total, 1)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading {JSONL_PATH} ...")
    ep_by_query = load_episodes(JSONL_PATH)

    print(f"\nTraining objects ({len(ep_by_query)}): {sorted(ep_by_query.keys())}")
    print("Per-object episode counts:")
    total_pos, total_neg = 0, 0
    for q in sorted(ep_by_query):
        s = ep_by_query[q]
        p, n = len(s["pos"]), len(s["neg"])
        total_pos += p
        total_neg += n
        print(f"  {q:<12}  class_id={s['class_id']}  pos_ep={p:3d}  neg_ep={n:3d}")
    print(f"  {'TOTAL':<12}  pos_ep={total_pos:3d}  neg_ep={total_neg:3d}")

    train_pairs, val_pairs = build_pairs(ep_by_query, val_frac=0.2)
    print(f"\nTrain pairs : {len(train_pairs)}")
    print(f"Val   pairs : {len(val_pairs)}")
    print(f"Majority baseline : 0.500")
    print(f"v3 val pair_acc   : 0.660  (17-dim, no query emb)")
    print(f"v5d val pair_acc  : 0.750  (17-dim + query emb, dropout=0.2)")

    train_loader = DataLoader(
        PairDataset(train_pairs), batch_size=BATCH, shuffle=True
    )
    val_loader = DataLoader(
        PairDataset(val_pairs), batch_size=BATCH, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LGGSN(
        n_queries=len(OBJECT_CLASSES),   # 7
        geom_dim=len(FEATURE_COLS),      # 18
        query_dim=4,
        hidden_dim=40,
        dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nDevice     : {device}")
    print(f"Model      : LGGSN  n_queries=7  geom_dim=18  query_dim=4  "
          f"mlp_in=22  hidden=40  dropout={DROPOUT}  params={n_params}")
    print(f"Optimizer  : Adam  lr={LR}  batch={BATCH}  epochs={N_EPOCHS}")
    print(f"Output     : {CKPT_PATH}\n")

    best_val_acc = 0.0
    best_state   = None
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    print(f"{'Epoch':>5}  {'tr_loss':>8}  {'tr_acc':>7}  {'va_loss':>8}  {'va_acc':>7}")
    print("-" * 48)
    for epoch in range(N_EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   device)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        marker = " ←" if va_acc == best_val_acc else ""
        print(
            f"{epoch:5d}  {tr_loss:8.4f}  {tr_acc:7.3f}  "
            f"{va_loss:8.4f}  {va_acc:7.3f}{marker}"
        )

    torch.save(best_state, CKPT_PATH)
    print(f"\nBest val pair_acc : {best_val_acc:.3f}")
    print(f"  v3  baseline    : 0.660")
    print(f"  v5d baseline    : 0.750")
    print(f"Saved v6 checkpoint → {CKPT_PATH}")


if __name__ == "__main__":
    main()
