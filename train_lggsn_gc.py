#!/usr/bin/env python3
"""
GC-LGGSN training script — Geometry-Conditioned feature gating.

Each training pair carries the episode context z = [flat_frac, sigma_H, sigma_yaw]
for both the positive and negative candidate.  The GatingNetwork learns to
soft-mask features based on that context before the LGGSN scorer runs.

Architecture change vs v2:  +302 parameters (GatingNetwork 3→16→14).
Loss, optimiser, epochs:     unchanged (BPR, Adam lr=1e-3, 30 epochs).

Env vars
--------
  LGGSN_JSONL    input candidate log  (default: logs/lggsn_live_candidates.jsonl)
  LGGSN_CKPT     output checkpoint    (default: grasp_6dof/models/lggsn_gc.pt)
  LGGSN_EPOCHS   number of epochs     (default: 30)
"""
import collections
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lggsn_model import GC_LGGSN

# ── config ─────────────────────────────────────────────────────────────────────
JSONL_PATH = os.environ.get("LGGSN_JSONL",  "logs/lggsn_live_candidates.jsonl")
CKPT_PATH  = os.environ.get("LGGSN_CKPT",   "grasp_6dof/models/lggsn_gc.pt")
N_EPOCHS   = int(os.environ.get("LGGSN_EPOCHS", 30))
SEED       = 42

FEATURE_COLS = [
    "x", "y", "z",
    "roll", "pitch", "yaw",
    "width", "score",
    "dz", "dz_lift", "need_dz", "H",
    "dist_to_centroid", "z_rel",
]
GEOM_DIM    = len(FEATURE_COLS)   # 14
CONTEXT_DIM = 3                   # [flat_frac, sigma_H, sigma_yaw]
# ───────────────────────────────────────────────────────────────────────────────


def _episode_context(cands):
    """
    Compute z = [flat_frac, sigma_H, sigma_yaw] for one episode.

    flat_frac  — fraction of candidates with H < 0.001 (depth degeneracy rate)
    sigma_H    — std of H across candidates
    sigma_yaw  — std of yaw across candidates
    """
    Hs   = [c["H"]   for c in cands]
    yaws = [c["yaw"] for c in cands]
    flat_frac = sum(1 for h in Hs if h < 0.001) / max(len(Hs), 1)
    sigma_H   = float(np.std(Hs))   if len(Hs)   > 1 else 0.0
    sigma_yaw = float(np.std(yaws)) if len(yaws) > 1 else 0.0
    return [flat_frac, sigma_H, sigma_yaw]


def load_episodes(path):
    """
    Read JSONL, group by (query, scene_id), resolve mixed-label episodes.

    Returns
    -------
    dict[query] -> {
        'pos': [(feats_list, ctx), ...],   # one tuple per episode
        'neg': [(feats_list, ctx), ...],
    }
    where ctx = [flat_frac, sigma_H, sigma_yaw] for that episode.
    """
    rows = [json.loads(l) for l in open(path)]

    ep_rows = collections.defaultdict(list)
    for r in rows:
        ep_rows[(r["query"], r["scene_id"])].append(r)

    ep_by_query = collections.defaultdict(lambda: {"pos": [], "neg": []})
    for (query, _sid), cands in ep_rows.items():
        labels = [c["label"] for c in cands]
        n_pos  = sum(labels)
        n_neg  = len(labels) - n_pos
        if n_pos == n_neg:
            continue  # tied / ambiguous — skip
        ep_label = 1 if n_pos > n_neg else 0

        # context-aware (dist_to_centroid, z_rel) features
        xs = [c["x"] for c in cands]; ys = [c["y"] for c in cands]
        zs = [c["z"] for c in cands]
        cx = sum(xs) / len(xs);       cy = sum(ys) / len(ys)
        z_min, z_max = min(zs), max(zs)
        for i, c in enumerate(cands):
            c["dist_to_centroid"] = math.sqrt((xs[i]-cx)**2 + (ys[i]-cy)**2)
            c["z_rel"] = (zs[i] - z_min) / (z_max - z_min + 1e-8)

        feats = [[c[f] for f in FEATURE_COLS] for c in cands]
        ctx   = _episode_context(cands)
        side  = "pos" if ep_label == 1 else "neg"
        ep_by_query[query][side].append((feats, ctx))

    return ep_by_query


def build_pairs(ep_by_query, val_frac=0.2, seed=SEED):
    """
    Split at the episode level (80/20), form cartesian pos×neg pairs.

    Each pair: (pos_feat, pos_ctx, neg_feat, neg_ctx, query)
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
            for p_feats, p_ctx in pos_list:
                for n_feats, n_ctx in neg_list:
                    for p_feat in p_feats:
                        for n_feat in n_feats:
                            out.append((p_feat, p_ctx, n_feat, n_ctx, query))
            return out

        train_pairs.extend(cartesian(pos_train, neg_train))
        val_pairs.extend(cartesian(pos_val, neg_val))

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    return train_pairs, val_pairs


class GCPairDataset(Dataset):
    """Each item: (pos_feats, pos_ctx, neg_feats, neg_ctx, query_str)."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p_feat, p_ctx, n_feat, n_ctx, query = self.pairs[idx]
        return (
            torch.tensor(p_feat, dtype=torch.float32),
            torch.tensor(p_ctx,  dtype=torch.float32),
            torch.tensor(n_feat, dtype=torch.float32),
            torch.tensor(n_ctx,  dtype=torch.float32),
            query,
        )


def run_epoch(model, loader, device, optimizer=None):
    """Returns (loss, overall_pair_acc, dict[query->pair_acc])."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct_by_q  = collections.defaultdict(int)
    total_by_q    = collections.defaultdict(int)

    for p_geom, p_ctx, n_geom, n_ctx, queries in loader:
        p_geom = p_geom.to(device); p_ctx = p_ctx.to(device)
        n_geom = n_geom.to(device); n_ctx = n_ctx.to(device)
        q_id   = torch.zeros(len(queries), dtype=torch.long, device=device)

        with torch.set_grad_enabled(is_train):
            logit_pos = model(p_geom, q_id, p_ctx).view(-1)
            logit_neg = model(n_geom, q_id, n_ctx).view(-1)
            loss = -F.logsigmoid(logit_pos - logit_neg).mean()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(queries)
        correct = (logit_pos > logit_neg).cpu()
        for i, q in enumerate(queries):
            correct_by_q[q]  += int(correct[i])
            total_by_q[q]    += 1

    n_total   = sum(total_by_q.values())
    n_correct = sum(correct_by_q.values())
    per_q_acc = {q: correct_by_q[q] / total_by_q[q] for q in total_by_q}
    return total_loss / max(n_total, 1), n_correct / max(n_total, 1), per_q_acc


def main():
    random.seed(SEED); torch.manual_seed(SEED)

    ep_by_query = load_episodes(JSONL_PATH)
    train_pairs, val_pairs = build_pairs(ep_by_query)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val   pairs: {len(val_pairs)}")
    print(f"GEOM_DIM={GEOM_DIM}  CONTEXT_DIM={CONTEXT_DIM}")
    print("\nEpisode split per query:")
    for q, s in sorted(ep_by_query.items()):
        print(f"  {q:<20s}  pos_ep={len(s['pos'])}  neg_ep={len(s['neg'])}")

    train_loader = DataLoader(GCPairDataset(train_pairs), batch_size=32, shuffle=True)
    val_loader   = DataLoader(GCPairDataset(val_pairs),   batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = GC_LGGSN(
        n_queries=1,
        geom_dim=GEOM_DIM,
        query_dim=0,
        hidden_dim=40,
        context_dim=CONTEXT_DIM,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GC-LGGSN total parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    REPORT_OBJECTS = ["Scissors", "CrackerBox", "MustardBottle", "PowerDrill"]

    for epoch in range(N_EPOCHS):
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc, va_by_q = run_epoch(model, val_loader, device)

        obj_str = "  ".join(
            f"{o[:6]}={va_by_q.get(o, float('nan')):.3f}"
            for o in REPORT_OBJECTS
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train loss={tr_loss:.4f} pair_acc={tr_acc:.3f} | "
            f"val loss={va_loss:.4f} pair_acc={va_acc:.3f} | "
            f"{obj_str}"
        )

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nSaved GC-LGGSN to {CKPT_PATH}")

    # ── Phase 1 summary ──────────────────────────────────────────────────────
    _, va_acc_final, va_by_q_final = run_epoch(model, val_loader, device)
    print("\n=== Phase 1 Validation Summary ===")
    print(f"Overall val pair_acc: {va_acc_final:.4f}")
    for obj in REPORT_OBJECTS:
        acc = va_by_q_final.get(obj, float("nan"))
        print(f"  {obj:<20s}: {acc:.4f}")


if __name__ == "__main__":
    main()
