#!/usr/bin/env python3
"""Object one-hot / point-cloud feature ablation for the object-relative
counterfactual critic, on the paper's own base-100 (train) / base-200
(dev-test) scene population -- NOT the earlier tomato-can (base-4100/4200)
population, which used a different, non-drop-in-citable data population.

Answers: how much of the critic's benefit comes from candidate-local
geometry (pc_stats_local) vs. from an object-identity prior (the one-hot),
independent of relative pose? Four feature variants, object_counterfactual
architecture (BPR pairwise + BCE, matching the paper's actual deployed
variant), 5 seeds x 400 epochs each via the real, unmodified
train_one()/evaluate() from world_model/train_counterfactual_critic.py:

  pose_only        object-relative pose only
  pose_pc          + candidate-local point cloud, no object one-hot
  pose_onehot      + object one-hot, no point cloud
  full             + both (the paper's actual "corrected_local" variant)

Trained on results/risk_gated_vla/pc_fix_train_base100/scenes.jsonl (120
scenes, base-seed=100), evaluated on
results/risk_gated_vla/pc_fix_devtest_base200/scenes.jsonl (90 scenes,
base-seed=200) -- the same real MuJoCo-collected data already backing
paper_risk_gated_vla.tex's sec:pcfix, reused here for a different question.
CONFIRMATORY BATCH (base-seed 300) IS NEVER TOUCHED.

Usage:
    conda run -n tango python scripts/pc_fix_onehot_ablation.py \\
        --train results/risk_gated_vla/pc_fix_train_base100/scenes.jsonl \\
        --devtest results/risk_gated_vla/pc_fix_devtest_base200/scenes.jsonl \\
        --out-dir results/risk_gated_vla/onehot_ablation \\
        --epochs 400 --seeds 5
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from world_model.train_counterfactual_critic import Critic, evaluate, train_one, OBJECTS
from scripts.risk_gated_vla_phase1_stats import mcnemar_exact

VARIANTS = ["pose_only", "pose_pc", "pose_onehot", "full"]


def feature(rec: dict, cand: dict, variant: str) -> list[float]:
    pose = np.asarray(cand["candidate_pose"], dtype=np.float32)
    obj = np.asarray(rec["obj_pos_before"], dtype=np.float32)
    xyz = pose[:3] - obj
    yaw = float(pose[3])
    base = [*xyz.tolist(), math.sin(yaw), math.cos(yaw),
            float(pose[4]), float(pose[5])]
    pc = [float(v) for v in cand.get("pc_stats_local", rec["pc_stats_before"])]
    onehot = [float(rec["object"] == name) for name in OBJECTS]
    if variant == "pose_only":
        return base
    elif variant == "pose_pc":
        return base + pc
    elif variant == "pose_onehot":
        return base + onehot
    elif variant == "full":
        return base + pc + onehot
    raise ValueError(variant)


def load_scenes(path: Path, variant: str):
    scenes = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        xs = [feature(rec, c, variant) for c in rec["oracle_per_candidate"]]
        ys = [float(c["success"]) for c in rec["oracle_per_candidate"]]
        scenes.append({"key": (rec["object"], rec["seed"]),
                        "object": rec["object"], "x": xs, "y": ys})
    return scenes


def load_devtest_with_geometry(path: Path, variant: str):
    """Like load_scenes, but also carries the live-executed geometry outcome
    and candidate index, for the geometry-baseline comparison row."""
    scenes = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        xs = [feature(rec, c, variant) for c in rec["oracle_per_candidate"]]
        ys = [float(c["success"]) for c in rec["oracle_per_candidate"]]
        scenes.append({
            "key": (rec["object"], rec["seed"]), "object": rec["object"],
            "x": xs, "y": ys,
            "geometry_success": bool(rec["outcomes"]["geometry"]["success"]),
        })
    return scenes


def score_ensemble_pick(scene, ensemble):
    scores = []
    for model, mean, std in ensemble:
        x = (torch.tensor(scene["x"], dtype=torch.float32) - mean) / std
        with torch.no_grad():
            scores.append(model(x).sigmoid().numpy())
    mean_score = np.mean(scores, axis=0)
    idx = int(np.argmax(mean_score))
    return idx, bool(scene["y"][idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--devtest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_devtest_picks = {}  # variant -> list of (success bool) per scene, seed-0 model only, for McNemar
    all_ensemble_devtest_top1 = {}  # variant -> pooled top1 success rate (ensemble mean score)
    summary = []

    for variant in VARIANTS:
        train_scenes = load_scenes(Path(args.train), variant)
        devtest_scenes = load_devtest_with_geometry(Path(args.devtest), variant)

        ensemble = []
        val_metrics_per_seed = []
        for seed in range(args.seeds):
            model, mean, std, train, val, val_metrics = train_one(
                train_scenes, "object_counterfactual", seed, args.epochs)
            ensemble.append((model, mean, std))
            val_metrics_per_seed.append(val_metrics)
            ckpt = {"variant": variant, "seed": seed, "state_dict": model.state_dict(),
                    "dim": len(mean), "mean": mean, "std": std}
            torch.save(ckpt, out / f"{variant}_seed{seed}.pt")
            row = {"variant": variant, "seed": seed, **val_metrics}
            summary.append(row)
            print(json.dumps(row))

        # Ensemble-mean pick on dev-test (matches pc_fix_compare_checkpoints.py's convention)
        picks = [score_ensemble_pick(s, ensemble) for s in devtest_scenes]
        n = len(devtest_scenes)
        n_success = sum(p[1] for p in picks)
        all_devtest_picks[variant] = [p[1] for p in picks]
        all_ensemble_devtest_top1[variant] = n_success / n
        print(f"\n[{variant}] dev-test ensemble top1: {n_success}/{n} ({100*n_success/n:.1f}%)\n")

    n_geo = sum(s["geometry_success"] for s in load_devtest_with_geometry(Path(args.devtest), "full"))
    n = len(all_devtest_picks["full"])
    print(f"Geometry (live-executed): {n_geo}/{n} ({100*n_geo/n:.1f}%)")
    for v, top1 in all_ensemble_devtest_top1.items():
        print(f"{v:>12}: {top1*n:.0f}/{n} ({100*top1:.1f}%)")

    print("\nPairwise McNemar (dev-test, ensemble-mean picks):")
    pairwise = {}
    for i, v1 in enumerate(VARIANTS):
        for v2 in VARIANTS[i + 1:]:
            m = mcnemar_exact(all_devtest_picks[v1], all_devtest_picks[v2])
            pairwise[f"{v1}_vs_{v2}"] = m
            print(f"  {v1} vs {v2}: {m}")

    (out / "onehot_ablation_summary.json").write_text(json.dumps({
        "n_devtest_scenes": n, "n_geometry_live": n_geo,
        "devtest_top1_by_variant": all_ensemble_devtest_top1,
        "devtest_picks_by_variant": all_devtest_picks,
        "pairwise_mcnemar": pairwise,
        "per_seed_val_metrics": summary,
    }, indent=2))
    print(f"\nWrote {out / 'onehot_ablation_summary.json'}")


if __name__ == "__main__":
    main()
