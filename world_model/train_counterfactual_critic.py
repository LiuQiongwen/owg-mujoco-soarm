#!/usr/bin/env python3
"""Train scene-grouped grasp critics from strict paired MuJoCo outcomes.

Variants:
  global_bce          absolute pose + scene context, BCE only
  object_bce          object-relative pose + scene context, BCE only
  object_counterfactual  object-relative features + BCE + within-scene BPR
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBJECTS = ["cracker", "mustard", "drill"]


class Critic(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def feature(rec: dict, cand: dict, relative: bool) -> list[float]:
    pose = np.asarray(cand["candidate_pose"], dtype=np.float32)
    obj = np.asarray(rec["obj_pos_before"], dtype=np.float32)
    xyz = pose[:3] - obj if relative else pose[:3]
    yaw = float(pose[3])
    base = [*xyz.tolist(), math.sin(yaw), math.cos(yaw),
            float(pose[4]), float(pose[5])]
    pc = [float(v) for v in rec["pc_stats_before"]]
    onehot = [float(rec["object"] == name) for name in OBJECTS]
    return base + pc + onehot


def load_scenes(path: Path, relative: bool):
    scenes = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        xs = [feature(rec, c, relative) for c in rec["oracle_per_candidate"]]
        ys = [float(c["success"]) for c in rec["oracle_per_candidate"]]
        scenes.append({"key": (rec["object"], rec["seed"]),
                       "object": rec["object"], "x": xs, "y": ys})
    return scenes


def load_ensemble(model_dir: Path, variant: str):
    bundles = []
    for path in sorted(model_dir.glob(f"{variant}_seed*.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = Critic(int(ckpt["dim"]))
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        bundles.append((model, ckpt["mean"], ckpt["std"]))
    if not bundles:
        raise FileNotFoundError(f"no {variant}_seed*.pt in {model_dir}")
    return bundles


def score_candidates(rec: dict, candidates: list[dict], bundles, relative: bool):
    x = torch.tensor([feature(rec, c, relative) for c in candidates],
                     dtype=torch.float32)
    with torch.no_grad():
        scores = [model((x - mean) / std).sigmoid().numpy()
                  for model, mean, std in bundles]
    arr = np.stack(scores)
    return arr.mean(0), arr.std(0)


def auc(y, score):
    y, score = np.asarray(y), np.asarray(score)
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
                 / (len(pos) * len(neg)))


def evaluate(model, scenes, mean, std):
    all_y, all_s = [], []
    top_ok, oracle_ok, mixed_top, mixed_n = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for scene in scenes:
            x = (torch.tensor(scene["x"], dtype=torch.float32) - mean) / std
            score = model(x).cpu().numpy()
            y = np.asarray(scene["y"], dtype=int)
            all_y.extend(y.tolist()); all_s.extend(score.tolist())
            top_ok += int(y[int(np.argmax(score))])
            oracle_ok += int(y.any())
            if y.any() and not y.all():
                mixed_n += 1
                mixed_top += int(y[int(np.argmax(score))])
    n = max(len(scenes), 1)
    return {"auc": auc(all_y, all_s), "top1": top_ok / n,
            "oracle": oracle_ok / n,
            "mixed_top1": mixed_top / max(mixed_n, 1), "mixed_n": mixed_n}


def train_one(scenes, variant: str, seed: int, epochs: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Scene-grouped 80/20 split, stratified by object and reproducible per seed.
    train, val = [], []
    for obj in OBJECTS:
        group = [s for s in scenes if s["object"] == obj]
        order = rng.permutation(len(group))
        n_val = max(1, int(round(0.2 * len(group))))
        val.extend(group[i] for i in order[:n_val])
        train.extend(group[i] for i in order[n_val:])

    x_train = torch.tensor([x for s in train for x in s["x"]], dtype=torch.float32)
    mean = x_train.mean(0)
    std = x_train.std(0).clamp_min(1e-6)
    model = Critic(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    use_pair = variant == "object_counterfactual"

    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        bce_terms, pair_terms = [], []
        for scene in train:
            x = (torch.tensor(scene["x"], dtype=torch.float32) - mean) / std
            y = torch.tensor(scene["y"], dtype=torch.float32)
            logits = model(x)
            bce_terms.append(F.binary_cross_entropy_with_logits(logits, y))
            if use_pair:
                pos, neg = logits[y > 0.5], logits[y < 0.5]
                if len(pos) and len(neg):
                    pair_terms.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())
        loss = torch.stack(bce_terms).mean()
        if pair_terms:
            loss = 0.5 * loss + torch.stack(pair_terms).mean()
        loss.backward(); opt.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            metric = evaluate(model, val, mean, std)
            score = metric["mixed_top1"]
            if score > best:
                best = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, mean, std, train, val, evaluate(model, val, mean, std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for variant in ("global_bce", "object_bce", "object_counterfactual"):
        relative = variant != "global_bce"
        scenes = load_scenes(Path(args.data), relative)
        for seed in range(args.seeds):
            model, mean, std, train, val, metrics = train_one(
                scenes, variant, seed, args.epochs)
            ckpt = {"variant": variant, "seed": seed,
                    "state_dict": model.state_dict(), "dim": len(mean),
                    "mean": mean, "std": std, "objects": OBJECTS,
                    "train_keys": [s["key"] for s in train],
                    "val_keys": [s["key"] for s in val]}
            path = out / f"{variant}_seed{seed}.pt"
            torch.save(ckpt, path)
            row = {"variant": variant, "seed": seed, **metrics,
                   "checkpoint": str(path)}
            summary.append(row)
            print(json.dumps(row))
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
