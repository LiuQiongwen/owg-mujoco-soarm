#!/usr/bin/env python3
"""
Train the multi-head contact/lift/success critic (C.3).

Scoping (per user decision, 2026-07-30 -- see results/risk_gated_vla/phase1/
multitask_outcome_critic/DATA_AUDIT.md for the empirical basis):
  - Heads: bilateral_contact, lifted, success (all binary), failure_type
    (3-class: success/no_contact/weld_no_lift -- the 6-class future schema is
    NOT trained; see world_model/multihead_labels.py).
  - retained_grasp_proxy is NOT a head. It is a diagnostic alias
    (== weld_triggered == bilateral_contact in all current data) reported
    separately, never in the loss or main results.
  - This experiment answers "is a multi-head contact/lift/success
    representation more physically interpretable than the single-head
    success critic?" -- it does NOT claim a general retention or failure
    model.

Reuses (does not modify) world_model/train_counterfactual_critic.py's
`feature()` and `OBJECTS`. Duplicates its scene-grouped split logic verbatim
(Option B, per explicit user decision) -- see
tests/test_multihead_critic.py::test_split_matches_existing_train_one for the
equivalence proof against the original, unmodified train_one().

Baseline for comparison is the EXISTING, already-trained object_bce /
object_counterfactual checkpoints in
results/risk_gated_vla/counterfactual_models_20260730/ -- not retrained here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_model.train_counterfactual_critic import feature, OBJECTS
from world_model.multihead_labels import (
    derive_success, failure_type_3class, FAILURE_TYPES_REALIZED,
)

LOSS_WEIGHTINGS = ("equal", "success_weighted")


# ── Scene-grouped split (duplicated verbatim from train_counterfactual_critic.py's
#    train_one(), Option B -- do not modify the original) ──────────────────

def scene_grouped_split(scenes, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for obj in OBJECTS:
        group = [s for s in scenes if s["object"] == obj]
        order = rng.permutation(len(group))
        n_val = max(1, int(round(0.2 * len(group))))
        val.extend(group[i] for i in order[:n_val])
        train.extend(group[i] for i in order[n_val:])
    return train, val


# ── Data loading ─────────────────────────────────────────────────────────────

def load_scenes_multihead(path: Path, relative: bool = True) -> list:
    scenes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cands = rec["oracle_per_candidate"]
        xs = [feature(rec, c, relative) for c in cands]
        ys = []
        for c in cands:
            ft = failure_type_3class(c)  # raises loudly if data has drifted -- do not catch
            ys.append({
                "bilateral_contact": float(c["bilateral_contact"]),
                "lifted": float(c["lifted"]),
                "success": float(derive_success(c)),
                "failure_type_idx": float(FAILURE_TYPES_REALIZED.index(ft)),
            })
        scenes.append({
            "key": (rec["object"], rec["seed"]), "object": rec["object"],
            "x": xs, "y": ys,
        })
    return scenes


def print_class_distribution(scenes: list, label: str) -> None:
    """Required before any training run -- overall and per-object."""
    from collections import Counter
    overall = Counter()
    by_obj = {}
    for s in scenes:
        for y in s["y"]:
            ft = FAILURE_TYPES_REALIZED[int(y["failure_type_idx"])]
            overall[ft] += 1
            by_obj.setdefault(s["object"], Counter())[ft] += 1
    print(f"\n[{label}] failure_type distribution (n_scenes={len(scenes)}, "
          f"n_candidates={sum(len(s['y']) for s in scenes)}):")
    print(f"  overall: {dict(overall)}")
    for obj in OBJECTS:
        print(f"  {obj}: {dict(by_obj.get(obj, {}))}")
    if "weld_no_lift" in overall:
        drill_weld_no_lift = by_obj.get("drill", Counter()).get("weld_no_lift", 0)
        total_weld_no_lift = overall["weld_no_lift"]
        non_drill_weld_no_lift = total_weld_no_lift - drill_weld_no_lift
        pct = 100 * drill_weld_no_lift / total_weld_no_lift if total_weld_no_lift else float("nan")
        print(f"  weld_no_lift drill-share: {drill_weld_no_lift}/{total_weld_no_lift} "
              f"({pct:.1f}%)  non-drill support: {non_drill_weld_no_lift}")


# ── Model ────────────────────────────────────────────────────────────────────

class MultiHeadCritic(nn.Module):
    """Shared trunk + bilateral_contact/lifted/success binary heads +
    failure_type (3-class) head. The heads are deliberately redundant with
    each other (failure_type is a deterministic function of the binary
    signals) -- the experiment's question is whether this decomposed
    representation is more physically interpretable, not whether the heads
    are statistically independent."""

    def __init__(self, dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, 64), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.head_bilateral_contact = nn.Linear(64, 1)
        self.head_lifted = nn.Linear(64, 1)
        self.head_success = nn.Linear(64, 1)
        self.head_failure_type = nn.Linear(64, len(FAILURE_TYPES_REALIZED))

    def forward(self, x):
        h = self.trunk(x)
        return {
            "bilateral_contact": self.head_bilateral_contact(h).squeeze(-1),
            "lifted": self.head_lifted(h).squeeze(-1),
            "success": self.head_success(h).squeeze(-1),
            "failure_type": self.head_failure_type(h),
        }


def compute_loss(preds: dict, targets: dict, weighting: str) -> torch.Tensor:
    l_bc = F.binary_cross_entropy_with_logits(preds["bilateral_contact"], targets["bilateral_contact"])
    l_lift = F.binary_cross_entropy_with_logits(preds["lifted"], targets["lifted"])
    l_succ = F.binary_cross_entropy_with_logits(preds["success"], targets["success"])
    l_ft = F.cross_entropy(preds["failure_type"], targets["failure_type_idx"].long())
    if weighting == "equal":
        return l_bc + l_lift + l_succ + l_ft
    if weighting == "success_weighted":
        return 2.0 * l_succ + l_bc + l_lift + l_ft
    raise ValueError(f"unknown weighting {weighting!r}")


def _stack_targets(scene, mean=None, std=None):
    x = torch.tensor(scene["x"], dtype=torch.float32)
    if mean is not None:
        x = (x - mean) / std
    y = scene["y"]
    targets = {
        "bilateral_contact": torch.tensor([r["bilateral_contact"] for r in y], dtype=torch.float32),
        "lifted": torch.tensor([r["lifted"] for r in y], dtype=torch.float32),
        "success": torch.tensor([r["success"] for r in y], dtype=torch.float32),
        "failure_type_idx": torch.tensor([r["failure_type_idx"] for r in y], dtype=torch.float32),
    }
    return x, targets


def evaluate_mixed_top1(model, scenes, mean, std) -> dict:
    """Mixed-scene top-1 accuracy selecting by the success head (matches the
    existing single-head critic's own selection convention, for direct
    comparability)."""
    model.eval()
    top_ok, oracle_ok, mixed_top, mixed_n = 0, 0, 0, 0
    with torch.no_grad():
        for scene in scenes:
            x, targets = _stack_targets(scene, mean, std)
            score = model(x)["success"].cpu().numpy()
            y = targets["success"].numpy().astype(int)
            top_ok += int(y[int(np.argmax(score))])
            oracle_ok += int(y.any())
            if y.any() and not y.all():
                mixed_n += 1
                mixed_top += int(y[int(np.argmax(score))])
    n = max(len(scenes), 1)
    return {"top1": top_ok / n, "oracle": oracle_ok / n,
            "mixed_top1": mixed_top / max(mixed_n, 1), "mixed_n": mixed_n}


def train_one_multihead(scenes: list, weighting: str, seed: int, epochs: int) -> dict:
    torch.manual_seed(seed)
    train, val = scene_grouped_split(scenes, seed)

    x_train = torch.tensor([x for s in train for x in s["x"]], dtype=torch.float32)
    mean = x_train.mean(0)
    std = x_train.std(0).clamp_min(1e-6)

    model = MultiHeadCritic(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        losses = []
        for scene in train:
            x, targets = _stack_targets(scene, mean, std)
            preds = model(x)
            losses.append(compute_loss(preds, targets, weighting))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            metric = evaluate_mixed_top1(model, val, mean, std)
            if metric["mixed_top1"] > best:
                best = metric["mixed_top1"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return {
        "model": model, "mean": mean, "std": std,
        "train_keys": [s["key"] for s in train], "val_keys": [s["key"] for s in val],
        "dev_metric": evaluate_mixed_top1(model, val, mean, std),
        "dim": int(x_train.shape[1]), "weighting": weighting, "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    scenes = load_scenes_multihead(Path(args.train_data))
    print_class_distribution(scenes, "train")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for weighting in LOSS_WEIGHTINGS:
        for seed in range(args.seeds):
            result = train_one_multihead(scenes, weighting, seed, args.epochs)
            ckpt = {
                "weighting": weighting, "seed": seed, "dim": result["dim"],
                "state_dict": result["model"].state_dict(),
                "mean": result["mean"], "std": result["std"],
                "objects": OBJECTS,
                "failure_types_realized": FAILURE_TYPES_REALIZED,
                "train_keys": result["train_keys"], "val_keys": result["val_keys"],
            }
            path = out / f"multihead_{weighting}_seed{seed}.pt"
            torch.save(ckpt, path)
            row = {"weighting": weighting, "seed": seed, **result["dev_metric"],
                   "checkpoint": str(path)}
            summary.append(row)
            print(json.dumps(row))
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[train] wrote {out / 'training_summary.json'}")


if __name__ == "__main__":
    main()
