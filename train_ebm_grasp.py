#!/usr/bin/env python3
"""
Energy-based (implicit) grasp candidate scoring model, v2 -- InfoNCE with
adversarial (self-mined) negatives.

Motivation (2026-07-10): the explicit generative approach (OT-CFM, minibatch
optimal-transport-coupled conditional flow matching) was found to
significantly underperform random-CoM sampling on physically-executed grasp
trials (n=50/object, pooled -10.0pp, p=0.0025), and a condition-aware OT
coupling fix (C2OT, ICCV 2025, discrete-class variant) only partially and
inconsistently mitigated this. Plain CFM (no OT) and DDPM (same architecture,
same data) tracked or beat baseline on 2/3 objects tested, showing the
learned-distribution approach is viable -- the problem was specifically the
OT-coupled training trajectory needing more per-condition data than we have
(~400 examples/object) to learn a stable, uncrossed flow.

v1 of this file trained the EnergyNet with plain binary cross-entropy against
the dataset's static label=1/label=0 rows, then sampled via cross-entropy-
method (CEM) search at inference. That failed catastrophically (0-16% success
across 3 objects, worse than every other method tried this session). Root
cause, confirmed by direct inspection: CEM search converged to a normalised
pose ~1.7 std from the training mean where the v1 model assigned an
artificially high score (logit=4.55, i.e. ~99% confidence) despite this
region being essentially unconstrained by training data -- each marginal
dimension looked plausible in isolation, but the specific combination was a
region the static-negative BCE loss never taught the model to avoid. This is
exactly the pitfall Implicit Behavioral Cloning (Florence et al., CoRL 2021)
warns about and is why IBC does NOT train with plain BCE against a fixed
negative set: it trains with an InfoNCE-style loss against a NEGATIVE POOL
THAT INCLUDES SAMPLES MINED FROM THE MODEL'S OWN CURRENT BELIEFS DURING
TRAINING (a "derivative-free optimizer" / counter-example procedure), so any
region the model starts to over-score gets explicitly pushed back down before
training ends, instead of only ever seeing the dataset's original failures.

v2 (this file) trains that way: for every positive example, a negative pool
is assembled from three sources per training step --
  1. STATIC negatives: real logged label=0 (failed) rows for the same object
     (free, already exist in the dataset, ~1.5x more failures than
     successes on average).
  2. UNIFORM negatives: candidates drawn uniformly across the training data's
     observed pose range, so the model always has some signal about the
     broad feasible region, not just points near real trials.
  3. HARD (self-mined) negatives: a short CEM search using the model's OWN
     CURRENT weights (this training step, detached/no-grad) per object
     present in the batch -- whatever pose the model currently over-scores
     for that object becomes a negative it is immediately trained against.
     This is the piece that was missing in v1 and is the actual mechanism
     that prevents the CEM-at-inference-time exploit found there.
Loss: softmax cross-entropy over {positive, all negatives} with the positive
as the target class (InfoNCE / Implicit BC loss), not plain BCE.

Cond   : L2-normalised SAM 256-dim visual feature (same as CFM/DDPM, for a
         fair, apples-to-apples comparison in the evaluation harness).
Input  : (x, y, sin(yaw), cos(yaw)) -- the only dimensions that vary in this
         dataset (z/roll/pitch are constant top-down grasps, same as CFM).

Inference: same CEM search as v1 (ui.py:_ebm_sample_candidates, unchanged) --
only the training procedure changed. This is the meaningful test: if the
same inference-time search now succeeds where it catastrophically failed
before, the cause and fix are both confirmed.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

JSONL_PATH   = os.environ.get("EBM_JSONL",  "grasp_6dof/dataset/lggsn_candidates_v9.jsonl")
CKPT_PATH    = os.environ.get("EBM_CKPT",   "grasp_6dof/models/ebm_allobj.pt")
OBJECTS      = ["banana","can","cracker","cylinder","drill","mustard","pear"]
N_EPOCHS     = int(os.environ.get("EBM_EPOCHS",    "1000"))
BATCH_SIZE   = int(os.environ.get("EBM_BATCH",     "128"))
LR           = float(os.environ.get("EBM_LR",      "3e-4"))
WEIGHT_DECAY = float(os.environ.get("EBM_WD",      "1e-4"))
K_STATIC     = int(os.environ.get("EBM_K_STATIC",  "4"))   # real logged failures
K_UNIFORM    = int(os.environ.get("EBM_K_UNIFORM", "4"))   # broad-coverage negatives
K_HARD       = int(os.environ.get("EBM_K_HARD",    "4"))   # self-mined adversarial negatives
HARD_POP     = int(os.environ.get("EBM_HARD_POP",  "32"))  # CEM population for mining
HARD_ITERS   = int(os.environ.get("EBM_HARD_ITERS", "3"))  # CEM iterations for mining
VIS_DIM      = 256
POSE_DIM     = 4  # x, y, sin(yaw), cos(yaw)
IN_DIM       = POSE_DIM + VIS_DIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. Dataset ────────────────────────────────────────────────────────────────

class GraspScoreDataset(Dataset):
    """All-object (pose, vis_feat, label) triples -- both label=1 and label=0."""

    def __init__(self, jsonl_path: str):
        rows = [json.loads(l) for l in open(jsonl_path)]

        xy      = np.array([[r["x"], r["y"]] for r in rows], dtype=np.float32)
        yaw     = np.array([r["yaw"] for r in rows], dtype=np.float32)
        vis_raw = np.array([r["visual_feat"] for r in rows], dtype=np.float32)
        labels  = np.array([r["label"] for r in rows], dtype=np.float32)
        queries = [r["query"] for r in rows]

        self.xy_mean = xy.mean(0)
        self.xy_std  = xy.std(0).clip(min=1e-6)
        xy_norm      = (xy - self.xy_mean) / self.xy_std
        # Empirical range of the training data in normalised space -- used to
        # bound the UNIFORM negative sampler and the hard-negative miner's
        # initial search population, so mining stays within the region the
        # model is actually meant to reason about.
        self.xy_norm_lo = xy_norm.min(0) - 0.2
        self.xy_norm_hi = xy_norm.max(0) + 0.2

        pose_feat = np.concatenate([xy_norm, np.sin(yaw)[:, None], np.cos(yaw)[:, None]], axis=1)

        norms    = np.linalg.norm(vis_raw, axis=1, keepdims=True).clip(min=1e-8)
        vis_norm = vis_raw / norms

        self.pose  = torch.from_numpy(pose_feat.astype(np.float32))  # (N, 4)
        self.vis   = torch.from_numpy(vis_norm.astype(np.float32))   # (N, 256)
        self.label = labels                                          # (N,) numpy, label=1/0
        self.queries = queries
        self.N     = len(rows)

        by_obj, pos_by_obj, neg_by_obj = {}, {}, {}
        for i, q in enumerate(queries):
            by_obj.setdefault(q, []).append(i)
            (pos_by_obj if labels[i] == 1 else neg_by_obj).setdefault(q, []).append(i)
        print(f"\n[Dataset] {self.N} total rows (label=1 + label=0)")
        print(f"  {'object':<12} {'n_pos':>6} {'n_neg':>6}")
        for obj in OBJECTS:
            print(f"  {obj:<12} {len(pos_by_obj.get(obj, [])):>6} {len(neg_by_obj.get(obj, [])):>6}")
        self._by_obj = by_obj
        self.pos_by_obj = pos_by_obj
        self.neg_by_obj = neg_by_obj
        self.pos_indices = [i for i in range(self.N) if labels[i] == 1]

    def mean_vis(self, obj: str) -> torch.Tensor:
        idx = self._by_obj[obj]
        return self.vis[idx].mean(0)


# ── 2. Energy model ────────────────────────────────────────────────────────────

class EnergyNet(nn.Module):
    """4-layer MLP: (pose_feat, cond) -> scalar logit (higher = more likely to succeed)."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, hidden),  nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, pose_feat, cond):
        inp = torch.cat([pose_feat, cond], dim=-1)
        return self.net(inp).squeeze(-1)


# ── 3. Hard-negative mining (short CEM using the model's current weights) ────

def _mine_hard_negatives(model: EnergyNet, cond_vec: torch.Tensor,
                          xy_lo: np.ndarray, xy_hi: np.ndarray,
                          rng: np.random.Generator, k: int) -> np.ndarray:
    """Short CEM search over (x, y, yaw) in normalised space, using the
    model's CURRENT weights (no grad -- this is data generation, not part of
    the backward pass). Returns the top-k poses the model currently over-
    scores for this condition, as (k, 4) pose features -- these become
    negatives the very same training step trains the model away from.
    Bounded to the training data's own observed range (+margin) so mining
    explores the region the model is actually meant to reason about, not
    arbitrary far-away points."""
    pop_xy  = rng.uniform(xy_lo, xy_hi, size=(HARD_POP, 2)).astype(np.float32)
    pop_yaw = rng.uniform(-np.pi / 2, np.pi / 2, size=HARD_POP).astype(np.float32)
    dev = cond_vec.device

    for _ in range(HARD_ITERS):
        feat = np.concatenate([pop_xy, np.sin(pop_yaw)[:, None], np.cos(pop_yaw)[:, None]], axis=1)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=dev)
        cond_b = cond_vec.unsqueeze(0).expand(HARD_POP, -1)
        with torch.no_grad():
            logits = model(feat_t, cond_b).cpu().numpy()
        elite_n = max(2, HARD_POP // 4)
        elite_idx = np.argsort(-logits)[:elite_n]
        mean_xy = pop_xy[elite_idx].mean(0)
        std_xy  = pop_xy[elite_idx].std(0).clip(min=0.1)
        elite_yaw = pop_yaw[elite_idx]
        yaw_mean = float(np.arctan2(np.sin(elite_yaw).mean(), np.cos(elite_yaw).mean()))
        yaw_std  = max(0.15, float(elite_yaw.std()))
        pop_xy = np.clip(rng.normal(mean_xy, std_xy, size=(HARD_POP, 2)), xy_lo, xy_hi).astype(np.float32)
        pop_yaw = np.clip(rng.normal(yaw_mean, yaw_std, size=HARD_POP), -np.pi / 2, np.pi / 2).astype(np.float32)

    feat = np.concatenate([pop_xy, np.sin(pop_yaw)[:, None], np.cos(pop_yaw)[:, None]], axis=1)
    feat_t = torch.tensor(feat, dtype=torch.float32, device=dev)
    cond_b = cond_vec.unsqueeze(0).expand(HARD_POP, -1)
    with torch.no_grad():
        logits = model(feat_t, cond_b).cpu().numpy()
    top_idx = np.argsort(-logits)[:k]
    return feat[top_idx]  # (k, 4)


# ── 4. Training (InfoNCE with static + uniform + self-mined hard negatives) ──

def train(dataset: GraspScoreDataset) -> EnergyNet:
    model = EnergyNet(hidden=256).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS, eta_min=1e-5)
    rng   = np.random.default_rng(0)

    n_params = sum(p.numel() for p in model.parameters())
    n_batches = max(1, len(dataset.pos_indices) // BATCH_SIZE)
    print(f"[Train] EnergyNet params: {n_params:,}  device: {device}")
    print(f"[Train] {N_EPOCHS} epochs  batch={BATCH_SIZE}  lr={LR}  wd={WEIGHT_DECAY}  "
          f"pos_anchors={len(dataset.pos_indices)}  batches/epoch={n_batches}")
    print(f"[Train] negatives/positive: {K_STATIC} static + {K_UNIFORM} uniform + {K_HARD} hard "
          f"(mined via {HARD_ITERS}-iter CEM, pop={HARD_POP})")

    xy_lo, xy_hi = dataset.xy_norm_lo, dataset.xy_norm_hi
    obj_mean_vis = {obj: dataset.mean_vis(obj).to(device) for obj in OBJECTS if obj in dataset._by_obj}

    best_loss, best_state = float("inf"), None
    pos_indices = np.array(dataset.pos_indices)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        rng.shuffle(pos_indices)
        total_loss, total_correct, total_n = 0.0, 0, 0

        for b in range(n_batches):
            batch_idx = pos_indices[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            if len(batch_idx) == 0:
                continue
            objs_b    = [dataset.queries[i] for i in batch_idx]
            cond_b    = dataset.vis[batch_idx].to(device)
            pos_feat  = dataset.pose[batch_idx].to(device)
            B = len(batch_idx)

            # hard negatives: one short CEM mine per unique object in this batch,
            # shared across that object's positives in the batch (cheap: at most
            # 7 mining calls per batch, not B).
            model.eval()
            hard_by_obj = {}
            for obj in set(objs_b):
                cv = obj_mean_vis.get(obj, cond_b[0])
                hard_by_obj[obj] = _mine_hard_negatives(model, cv, xy_lo, xy_hi, rng, K_HARD)
            model.train()

            neg_feats = np.zeros((B, K_STATIC + K_UNIFORM + K_HARD, POSE_DIM), dtype=np.float32)
            for i, obj in enumerate(objs_b):
                # static (real logged failures for this object; fall back to any object if none)
                neg_pool = dataset.neg_by_obj.get(obj) or list(range(dataset.N))
                static_idx = rng.choice(neg_pool, size=K_STATIC, replace=len(neg_pool) < K_STATIC)
                neg_feats[i, :K_STATIC] = dataset.pose[static_idx].numpy()
                # uniform
                u_xy  = rng.uniform(xy_lo, xy_hi, size=(K_UNIFORM, 2))
                u_yaw = rng.uniform(-np.pi / 2, np.pi / 2, size=K_UNIFORM)
                neg_feats[i, K_STATIC:K_STATIC + K_UNIFORM] = np.concatenate(
                    [u_xy, np.sin(u_yaw)[:, None], np.cos(u_yaw)[:, None]], axis=1)
                # hard (self-mined, shared within this object group)
                neg_feats[i, K_STATIC + K_UNIFORM:] = hard_by_obj[obj]

            K_total = K_STATIC + K_UNIFORM + K_HARD
            neg_feats_t = torch.tensor(neg_feats, dtype=torch.float32, device=device)  # (B, K, 4)
            cond_exp = cond_b.unsqueeze(1).expand(B, K_total, VIS_DIM).reshape(B * K_total, VIS_DIM)
            neg_flat = neg_feats_t.reshape(B * K_total, POSE_DIM)
            neg_logit = model(neg_flat, cond_exp).reshape(B, K_total)
            pos_logit = model(pos_feat, cond_b)

            logits_all = torch.cat([pos_logit.unsqueeze(1), neg_logit], dim=1)  # (B, 1+K), positive at col 0
            target = torch.zeros(B, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits_all, target)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item() * B
            total_correct += (logits_all.argmax(dim=1) == 0).sum().item()
            total_n += B

        sched.step()
        avg_loss = total_loss / max(1, total_n)
        acc = total_correct / max(1, total_n)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 200 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{N_EPOCHS}  loss={avg_loss:.5f}  "
                  f"positive-ranked-top1_acc={acc:.3f}  best_loss={best_loss:.5f}")

    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)
    torch.save(best_state, CKPT_PATH)
    print(f"[Train] Best loss: {best_loss:.5f}  -> {CKPT_PATH}")
    model.load_state_dict(best_state)
    return model


def save_inference_stats(dataset: GraspScoreDataset, ckpt_path: str):
    stats = {
        "xy_mean": dataset.xy_mean.tolist(),
        "xy_std":  dataset.xy_std.tolist(),
        "mean_vis_per_obj": {
            obj: dataset.vis[idx].mean(0).numpy().tolist()
            for obj, idx in dataset._by_obj.items()
        },
    }
    stats_path = ckpt_path.replace(".pt", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    print(f"[Stats] Saved -> {stats_path}")


def main():
    dataset = GraspScoreDataset(JSONL_PATH)
    model = train(dataset)
    save_inference_stats(dataset, CKPT_PATH)


if __name__ == "__main__":
    main()
