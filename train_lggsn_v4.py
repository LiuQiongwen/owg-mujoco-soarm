#!/usr/bin/env python3
"""
Pairwise BPR training for LGGSN v4 — 17-dim features, per-candidate labels.

Key difference from v3: uses WITHIN-EPISODE pairs instead of cross-episode.
  pos = evaluated candidate with label=1 from scene S
  neg = evaluated candidate with label=0 from scene S (same scene)

Within-scene pairs are much stronger supervision: identical object pose,
identical point-cloud context — only the grasp pose differs.

Falls back to cross-episode pairs (same object class) when a scene has only
successes or only failures among evaluated candidates.

Data source: grasp_6dof/dataset/lggsn_candidates_v4.jsonl
  Each row has: scene_id, query, label (per-candidate), evaluated, 17 features
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
    "LGGSN_JSONL", "grasp_6dof/dataset/lggsn_candidates_v4.jsonl"
)
CKPT_PATH  = os.environ.get(
    "LGGSN_CKPT",  "grasp_6dof/models/lggsn_pairwise_live_v4.pt"
)
N_EPOCHS     = int(os.environ.get("LGGSN_EPOCHS", 40))
LR           = float(os.environ.get("LGGSN_LR", 5e-4))
WEIGHT_DECAY = float(os.environ.get("LGGSN_WD", 1e-2))
BATCH        = int(os.environ.get("LGGSN_BATCH", 32))
SEED         = 42

FEATURE_COLS = [
    "x", "y", "z",
    "roll", "pitch", "yaw",
    "width", "score",
    "dz", "dz_lift", "need_dz", "H",
    "dist_to_centroid", "z_rel",
    "local_point_density", "normal_consistency", "contact_width_ratio",
]
assert len(FEATURE_COLS) == 17
# ─────────────────────────────────────────────────────────────────────────────


class PairDataset(Dataset):
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


def load_scenes(path):
    """
    Read v4 JSONL and group evaluated candidates by (query, scene_id).

    Returns
    -------
    dict[query] -> dict[scene_id] -> {'pos': [feat_list], 'neg': [feat_list]}
    """
    scenes = collections.defaultdict(
        lambda: collections.defaultdict(lambda: {"pos": [], "neg": []})
    )
    n_rows = n_eval = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_rows += 1
            if not r.get("evaluated", 1):
                continue  # skip unevaluated (conservative label=0)
            n_eval += 1
            feats = [r[c] for c in FEATURE_COLS]
            side  = "pos" if r["label"] == 1 else "neg"
            scenes[r["query"]][r["scene_id"]][side].append(feats)
    print(f"  Rows total: {n_rows}  |  evaluated: {n_eval}")
    return scenes


def build_pairs(scenes, val_frac=0.2, seed=SEED):
    """
    Build within-scene and cross-scene pairs; 80/20 scene-level train/val split.

    Within-scene pair: (pos_cand, neg_cand) from the SAME scene_id.
    Cross-scene pair : (pos_cand, neg_cand) from DIFFERENT scene_ids, same query.

    Returns (train_pairs, val_pairs, stats_dict)
    """
    rng = random.Random(seed)

    train_within, val_within = [], []
    train_cross,  val_cross  = [], []

    for query, scene_dict in scenes.items():
        scene_ids = sorted(scene_dict.keys())
        rng.shuffle(scene_ids)

        n_val = max(1, round(len(scene_ids) * val_frac))
        val_ids   = set(scene_ids[:n_val])
        train_ids = set(scene_ids[n_val:])

        def within_pairs(sids):
            pairs = []
            for sid in sids:
                s = scene_dict[sid]
                if s["pos"] and s["neg"]:
                    for p in s["pos"]:
                        for n in s["neg"]:
                            pairs.append((p, n))
            return pairs

        def cross_pairs(sids):
            """All (pos_cand from scene A, neg_cand from scene B) where A≠B."""
            all_pos = [f for sid in sids for f in scene_dict[sid]["pos"]]
            all_neg = [f for sid in sids for f in scene_dict[sid]["neg"]]
            if not all_pos or not all_neg:
                return []
            pairs = [(p, n) for p in all_pos for n in all_neg]
            return pairs

        train_within.extend(within_pairs(train_ids))
        val_within.extend(within_pairs(val_ids))
        train_cross.extend(cross_pairs(train_ids))
        val_cross.extend(cross_pairs(val_ids))

    # Cap cross-scene pairs to 2× within-scene count to keep supervision balanced
    n_tr_w = len(train_within)
    n_va_w = len(val_within)
    cap_tr = max(n_tr_w * 2, 1000)
    cap_va = max(n_va_w * 2, 200)
    rng.shuffle(train_cross)
    rng.shuffle(val_cross)
    train_cross = train_cross[:cap_tr]
    val_cross   = val_cross[:cap_va]

    train_pairs = train_within + train_cross
    val_pairs   = val_within   + val_cross

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    stats = {
        "train_within": n_tr_w,
        "train_cross":  len(train_cross),
        "val_within":   n_va_w,
        "val_cross":    len(val_cross),
    }
    return train_pairs, val_pairs, stats


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

    # ── dataset ───────────────────────────────────────────────────────────────
    print(f"Loading {JSONL_PATH} ...")
    scenes = load_scenes(JSONL_PATH)

    print("\nPer-object scene counts (with evaluated candidates):")
    for q in sorted(scenes.keys()):
        n_scenes  = len(scenes[q])
        n_pos_sc  = sum(1 for s in scenes[q].values() if s["pos"])
        n_neg_sc  = sum(1 for s in scenes[q].values() if s["neg"])
        n_both    = sum(1 for s in scenes[q].values() if s["pos"] and s["neg"])
        print(f"  {q:<12}  scenes={n_scenes:3d}  "
              f"has_pos={n_pos_sc:3d}  has_neg={n_neg_sc:3d}  "
              f"both(within-pair)={n_both:3d}")

    train_pairs, val_pairs, stats = build_pairs(scenes, val_frac=0.2)

    print(f"\nPair breakdown:")
    print(f"  Train within-scene : {stats['train_within']:5d}")
    print(f"  Train cross-scene  : {stats['train_cross']:5d}")
    print(f"  Val   within-scene : {stats['val_within']:5d}")
    print(f"  Val   cross-scene  : {stats['val_cross']:5d}")
    print(f"  Train total        : {len(train_pairs):5d}")
    print(f"  Val   total        : {len(val_pairs):5d}")
    print(f"  Majority baseline  : 0.500")
    print(f"  v3 val pair_acc    : ~0.660  (cross-episode only)")

    train_loader = DataLoader(
        PairDataset(train_pairs), batch_size=BATCH, shuffle=True
    )
    val_loader = DataLoader(
        PairDataset(val_pairs), batch_size=BATCH, shuffle=False
    )

    # ── model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LGGSN(
        n_queries=1,
        geom_dim=len(FEATURE_COLS),   # 17
        query_dim=0,
        hidden_dim=40,
    ).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)

    print(f"\nDevice     : {device}")
    print(f"Model      : LGGSN  geom_dim=17  hidden=40  params={n_params}")
    print(f"Optimizer  : Adam  lr={LR}  wd={WEIGHT_DECAY}  batch={BATCH}  epochs={N_EPOCHS}")
    print(f"Output     : {CKPT_PATH}\n")

    # ── training loop ─────────────────────────────────────────────────────────
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
    print(f"\nBest val pair_acc : {best_val_acc:.3f}  (v3 baseline: ~0.660)")
    print(f"Saved v4 checkpoint → {CKPT_PATH}")


if __name__ == "__main__":
    main()
