#!/usr/bin/env python3
"""
LGGSN v9 — geometry + SAM visual feature (concatenation).

Differences from v7:
  - Each row in the JSONL carries a "visual_feat" list[256] from
    collect_lggsn_data.py (SAM ViT-H ROI mean-pool, episode-level constant).
  - Feature vector fed to the model: 18 geom + 256 vis = 274 dims.
  - Model: LGGSNVision(geom_dim=18, vis_dim=256, query_dim=4, hidden_dim=64).
  - PairDataset returns 5-tuple: (pos_geom, pos_vis, neg_geom, neg_vis, q_id).
  - BPR loss and within/cross episode pair construction are unchanged from v7.

Baseline: v5d val pair_acc = 0.750, v7 val pair_acc = TBD
"""

import collections
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lggsn_model import LGGSNVision

# ── config ────────────────────────────────────────────────────────────────────
JSONL_PATH = os.environ.get(
    "LGGSN_JSONL", "grasp_6dof/dataset/lggsn_candidates_v9.jsonl"
)
CKPT_PATH  = os.environ.get(
    "LGGSN_CKPT",  "grasp_6dof/models/lggsn_pairwise_live_v9.pt"
)
N_EPOCHS   = int(os.environ.get("LGGSN_EPOCHS", 30))
LR         = float(os.environ.get("LGGSN_LR", 1e-3))
BATCH      = int(os.environ.get("LGGSN_BATCH", 32))
SEED       = 42

VIS_DIM = 256   # SAM ViT-H ROI embedding dimension

OBJECT_CLASSES: dict[str, int] = {
    "banana":   0,
    "can":      1,
    "cracker":  2,
    "cylinder": 3,
    "drill":    4,
    "mustard":  5,
    "pear":     6,
}

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
        # pairs: list of (q_id, pos_geom, pos_vis, neg_geom, neg_vis)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q_id, pos_geom, pos_vis, neg_geom, neg_vis = self.pairs[idx]
        return (
            torch.tensor(pos_geom, dtype=torch.float32),
            torch.tensor(pos_vis,  dtype=torch.float32),
            torch.tensor(neg_geom, dtype=torch.float32),
            torch.tensor(neg_vis,  dtype=torch.float32),
            torch.tensor(q_id,     dtype=torch.long),
        )


def load_episodes(path: str) -> dict:
    """
    Read v9 JSONL. Group by (query, scene_id).

    Returns dict[(query, scene_id)] ->
        {
          "pos_geom":  [[18 floats], ...],   # candidates with label=1
          "pos_vis":   [[256 floats], ...],
          "neg_geom":  [[18 floats], ...],   # candidates with label=0
          "neg_vis":   [[256 floats], ...],
          "class_id":  int,
          "query":     str,
        }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/collect_lggsn_data.py first."
        )
    rows = [json.loads(l) for l in open(path)]

    ep_raw: dict = collections.defaultdict(list)
    for r in rows:
        ep_raw[(r["query"], r["scene_id"])].append(r)

    episodes = {}
    for (query, scene_id), cands in ep_raw.items():
        pos_cands = [c for c in cands if c["label"] == 1]
        neg_cands = [c for c in cands if c["label"] == 0]
        episodes[(query, scene_id)] = {
            "pos_geom":  [[c[f] for f in FEATURE_COLS] for c in pos_cands],
            "pos_vis":   [c.get("visual_feat", [0.0] * VIS_DIM) for c in pos_cands],
            "neg_geom":  [[c[f] for f in FEATURE_COLS] for c in neg_cands],
            "neg_vis":   [c.get("visual_feat", [0.0] * VIS_DIM) for c in neg_cands],
            "class_id":  OBJECT_CLASSES.get(query, 0),
            "query":     query,
        }
    return episodes


def build_pairs(
    episodes: dict,
    val_frac: float = 0.2,
    seed: int = SEED,
) -> tuple[list, list, int, int]:
    """
    80/20 stratified episode split → within-episode + cross-episode BPR pairs.

    Pair format: (class_id, pos_geom, pos_vis, neg_geom, neg_vis).
    """
    rng = random.Random(seed)

    by_query: dict = collections.defaultdict(list)
    for key in sorted(episodes.keys()):
        by_query[key[0]].append(key)

    train_eps: set = set()
    val_eps:   set = set()
    for query, keys in by_query.items():
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_frac))
        for k in keys[:n_val]:
            val_eps.add(k)
        for k in keys[n_val:]:
            train_eps.add(k)

    def _make_pairs(ep_keys):
        within = []
        x_pos: dict = collections.defaultdict(list)
        x_neg: dict = collections.defaultdict(list)

        for key in ep_keys:
            ep       = episodes[key]
            query    = ep["query"]
            class_id = ep["class_id"]
            pg, pv   = ep["pos_geom"], ep["pos_vis"]
            ng, nv   = ep["neg_geom"], ep["neg_vis"]

            # 1. within-episode pairs
            within.extend(
                (class_id, p_g, p_v, n_g, n_v)
                for p_g, p_v in zip(pg, pv)
                for n_g, n_v in zip(ng, nv)
            )

            # 2. cross-episode bookkeeping
            all_g = pg + ng
            all_v = pv + nv
            if not all_g:
                continue
            if pg and len(pg) >= len(ng):
                x_pos[query].append((list(zip(all_g, all_v)), class_id))
            elif ng:
                x_neg[query].append((list(zip(all_g, all_v)), class_id))

        cross = []
        for query in set(x_pos) & set(x_neg):
            for p_pairs, cid in x_pos[query]:
                for n_pairs, _ in x_neg[query]:
                    cross.extend(
                        (cid, p_g, p_v, n_g, n_v)
                        for p_g, p_v in p_pairs
                        for n_g, n_v in n_pairs
                    )
        return within, cross

    train_within, train_cross = _make_pairs(sorted(train_eps))
    val_within,   val_cross   = _make_pairs(sorted(val_eps))

    train_pairs = train_within + train_cross
    val_pairs   = val_within   + val_cross
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    n_within = len(train_within) + len(val_within)
    n_cross  = len(train_cross)  + len(val_cross)
    return train_pairs, val_pairs, n_within, n_cross


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_correct, n_total = 0.0, 0, 0
    for pos_geom, pos_vis, neg_geom, neg_vis, q_id in loader:
        pos_geom = pos_geom.to(device)
        pos_vis  = pos_vis.to(device)
        neg_geom = neg_geom.to(device)
        neg_vis  = neg_vis.to(device)
        q_id     = q_id.to(device)

        with torch.set_grad_enabled(is_train):
            logit_pos = model(pos_geom, pos_vis, q_id).view(-1)
            logit_neg = model(neg_geom, neg_vis, q_id).view(-1)
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
    episodes = load_episodes(JSONL_PATH)

    # ── per-object stats ──────────────────────────────────────────────────────
    stat: dict = collections.defaultdict(
        lambda: {"n_ep": 0, "pos": 0, "neg": 0, "within": 0}
    )
    for ep in episodes.values():
        s = stat[ep["query"]]
        s["n_ep"]  += 1
        s["pos"]   += len(ep["pos_geom"])
        s["neg"]   += len(ep["neg_geom"])
        s["within"] += len(ep["pos_geom"]) * len(ep["neg_geom"])

    print(f"\nPer-object episode stats  ({len(episodes)} total episodes):")
    print(f"  {'object':<12}  {'cls':>3}  {'ep':>5}  "
          f"{'pos_cand':>9}  {'neg_cand':>9}  {'SR':>6}  {'within_pairs':>13}")
    t_pos = t_neg = t_within = 0
    for q in sorted(stat):
        s = stat[q]
        t_pos += s["pos"]; t_neg += s["neg"]; t_within += s["within"]
        sr = s["pos"] / max(s["pos"] + s["neg"], 1)
        print(f"  {q:<12}  {OBJECT_CLASSES[q]:>3}  {s['n_ep']:>5}  "
              f"{s['pos']:>9}  {s['neg']:>9}  {sr:>6.1%}  {s['within']:>13,}")
    sr_tot = t_pos / max(t_pos + t_neg, 1)
    print(f"  {'TOTAL':<12}  {'':>3}  {'':>5}  "
          f"{t_pos:>9}  {t_neg:>9}  {sr_tot:>6.1%}  {t_within:>13,}")

    mixed = sum(1 for ep in episodes.values() if ep["pos_geom"] and ep["neg_geom"])
    print(f"\nEpisodes with mixed labels (have within-ep pairs): "
          f"{mixed}/{len(episodes)}")

    # ── build pairs ───────────────────────────────────────────────────────────
    train_pairs, val_pairs, n_within, n_cross = build_pairs(episodes)
    print(f"\nTrain pairs   : {len(train_pairs):,}")
    print(f"Val   pairs   : {len(val_pairs):,}")
    print(f"  Within-ep   : {n_within:,}")
    print(f"  Cross-ep    : {n_cross:,}")
    print(f"v5d baseline val pair_acc : 0.750")

    train_loader = DataLoader(
        PairDataset(train_pairs), batch_size=BATCH, shuffle=True
    )
    val_loader = DataLoader(
        PairDataset(val_pairs), batch_size=BATCH, shuffle=False
    )

    # ── model ─────────────────────────────────────────────────────────────────
    DROPOUT = float(os.environ.get("LGGSN_DROPOUT", "0.0"))
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = LGGSNVision(
        n_queries=len(OBJECT_CLASSES),  # 7
        geom_dim=len(FEATURE_COLS),     # 18
        vis_dim=VIS_DIM,                # 256
        query_dim=4,
        hidden_dim=64,
        dropout=DROPOUT,
    ).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nDevice    : {device}")
    print(f"Model     : LGGSNVision  geom=18  vis=256  query=4  "
          f"in=278  hidden=64  dropout={DROPOUT}  params={n_params}")
    print(f"Optimizer : Adam  lr={LR}  batch={BATCH}  epochs={N_EPOCHS}")
    print(f"Output    : {CKPT_PATH}\n")

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_acc = -1.0   # -1 ensures epoch 0 always saves when val is empty
    best_state   = None
    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)
    print(f"{'Epoch':>5}  {'tr_loss':>8}  {'tr_acc':>7}  "
          f"{'va_loss':>8}  {'va_acc':>7}")
    print("-" * 48)
    for epoch in range(N_EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   device)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
        marker = " ←" if va_acc == best_val_acc else ""
        print(f"{epoch:5d}  {tr_loss:8.4f}  {tr_acc:7.3f}  "
              f"{va_loss:8.4f}  {va_acc:7.3f}{marker}")

    if best_state is None:   # guard: val was empty every epoch, use last state
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, CKPT_PATH)
    print(f"\nBest val pair_acc : {best_val_acc:.3f}")
    print(f"Saved → {CKPT_PATH}")


if __name__ == "__main__":
    main()
