#!/usr/bin/env python3
"""
Pairwise BPR training for LGGSN on live YCB candidate data.

Pair construction: cross-episode, same-query.
  pos row = any candidate from a success episode  (label=1)
  neg row = any candidate from a failure episode  (label=0)
  supervision: pos should score strictly higher than neg

Loss: BPR  =  -log σ(logit_pos - logit_neg)
Val metric: pairwise accuracy  (majority baseline = 0.50)

Env vars
--------
  LGGSN_JSONL   path to lggsn_live_candidates.jsonl
                (default: logs/lggsn_live_candidates.jsonl)
  LGGSN_CKPT    output checkpoint path
                (default: grasp_6dof/models/lggsn_pairwise_live.pt)
  LGGSN_EPOCHS  number of training epochs  (default: 30)
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
JSONL_PATH  = os.environ.get("LGGSN_JSONL",  "logs/lggsn_live_candidates.jsonl")
CKPT_PATH   = os.environ.get("LGGSN_CKPT",   "grasp_6dof/models/lggsn_pairwise_live_v2.pt")
N_EPOCHS    = int(os.environ.get("LGGSN_EPOCHS", 30))
SEED        = 42
_USE_DIST   = os.environ.get("FEAT_DIST", "1") == "1"   # ablation toggle
_USE_ZREL   = os.environ.get("FEAT_ZREL", "1") == "1"   # ablation toggle

FEATURE_COLS = [
    "x", "y", "z",
    "roll", "pitch", "yaw",
    "width", "score",
    "dz", "dz_lift", "need_dz", "H",
] + (["dist_to_centroid"] if _USE_DIST else []) \
  + (["z_rel"]            if _USE_ZREL else [])
# ─────────────────────────────────────────────────────────────────────────────


class PairDataset(Dataset):
    """Each item is (pos_features, neg_features, dummy_query_id=0)."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos, neg = self.pairs[idx]
        q = torch.tensor(0, dtype=torch.long)
        return (
            torch.tensor(pos, dtype=torch.float32),
            torch.tensor(neg, dtype=torch.float32),
            q,
        )


def load_episodes(path):
    """
    Read JSONL and group rows by (query, scene_id).

    Mixed-label episodes (artefact of double-collection) are resolved by
    majority vote; a tied episode is dropped.

    Returns
    -------
    dict[query] -> {'pos': [[row_feats, ...], ...],   # one inner list per episode
                    'neg': [[row_feats, ...], ...]}
    """
    rows = [json.loads(l) for l in open(path)]

    # group rows by episode key
    ep_rows = collections.defaultdict(list)
    for r in rows:
        ep_rows[(r["query"], r["scene_id"])].append(r)

    ep_by_query = collections.defaultdict(lambda: {"pos": [], "neg": []})
    for (query, _sid), cands in ep_rows.items():
        labels = [c["label"] for c in cands]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos == n_neg:
            continue  # tied / ambiguous — skip
        ep_label = 1 if n_pos > n_neg else 0

        # compute context features only when toggled on
        if _USE_DIST or _USE_ZREL:
            import math
            xs = [c["x"] for c in cands]
            ys = [c["y"] for c in cands]
            zs = [c["z"] for c in cands]
            cx    = sum(xs) / len(xs)
            cy    = sum(ys) / len(ys)
            z_min = min(zs)
            z_max = max(zs)
            for i, c in enumerate(cands):
                if _USE_DIST:
                    c["dist_to_centroid"] = math.sqrt((xs[i]-cx)**2 + (ys[i]-cy)**2)
                if _USE_ZREL:
                    c["z_rel"] = (zs[i] - z_min) / (z_max - z_min + 1e-8)

        feats = [[c[f] for f in FEATURE_COLS] for c in cands]
        side = "pos" if ep_label == 1 else "neg"
        ep_by_query[query][side].append(feats)

    return ep_by_query


def build_pairs(ep_by_query, val_frac=0.2, seed=SEED):
    """
    Split episodes 80/20 at the episode level (no leakage across splits).
    Form all cross-episode (pos_cand, neg_cand) pairs within each split.

    Returns
    -------
    train_pairs, val_pairs : list of (pos_feat_list, neg_feat_list)
    """
    rng = random.Random(seed)
    train_pairs, val_pairs = [], []

    for query, sides in ep_by_query.items():
        pos_eps = sides["pos"][:]
        neg_eps = sides["neg"][:]
        if not pos_eps or not neg_eps:
            continue

        rng.shuffle(pos_eps)
        rng.shuffle(neg_eps)

        n_pos_val = max(1, round(len(pos_eps) * val_frac))
        n_neg_val = max(1, round(len(neg_eps) * val_frac))

        pos_val,   pos_train = pos_eps[:n_pos_val],  pos_eps[n_pos_val:]
        neg_val,   neg_train = neg_eps[:n_neg_val],  neg_eps[n_neg_val:]

        def cartesian(pos_list, neg_list):
            out = []
            for p_ep in pos_list:
                for n_ep in neg_list:
                    for p_feat in p_ep:
                        for n_feat in n_ep:
                            out.append((p_feat, n_feat))
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
            # BPR loss: maximise P(score_pos > score_neg)
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
    # 1) load & pair
    ep_by_query = load_episodes(JSONL_PATH)
    train_pairs, val_pairs = build_pairs(ep_by_query, val_frac=0.2)

    print(f"Train pairs : {len(train_pairs)}")
    print(f"Val   pairs : {len(val_pairs)}")
    print(f"Pairwise majority baseline : 0.500")

    # pair-level query breakdown
    print("\nEpisode split per query:")
    for q, s in sorted(ep_by_query.items()):
        print(f"  {q:<20s}  pos_ep={len(s['pos'])}  neg_ep={len(s['neg'])}")

    train_loader = DataLoader(PairDataset(train_pairs), batch_size=32, shuffle=True)
    val_loader   = DataLoader(PairDataset(val_pairs),   batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 2) model — same architecture as BCE version
    model = LGGSN(
        n_queries=1,
        geom_dim=len(FEATURE_COLS),
        query_dim=0,
        hidden_dim=40,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3) train
    for epoch in range(N_EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   device)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss={tr_loss:.4f} pair_acc={tr_acc:.3f} | "
            f"val  loss={va_loss:.4f} pair_acc={va_acc:.3f}"
        )

    # 4) save
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nSaved pairwise LGGSN to {CKPT_PATH}")


if __name__ == "__main__":
    main()
