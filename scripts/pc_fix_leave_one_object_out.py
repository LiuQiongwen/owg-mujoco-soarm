#!/usr/bin/env python3
"""Leave-one-object-out generalization test: does candidate-local point-cloud
geometry (pc_stats_local) carry real, generalizable per-candidate signal, or
is its apparent benefit (results/risk_gated_vla/onehot_ablation/) just
correlated with object identity in a closed 3-object world?

Breaks the point-cloud/object-identity collinearity by construction: for
each object, train on the OTHER TWO objects only (object one-hot is
therefore meaningless for the held-out class -- omitted entirely, not just
zeroed, so there is no identity crutch available), then evaluate on the
held-out object's own dev-test scenes, which the model never saw during
training in any form. If pose+point-cloud beats pose-only on truly unseen
objects, that is decisive evidence of real geometric generalization, not an
identity shortcut. Pooled across all three leave-one-out folds for a
total n matching the original dev-test population (90 scenes).

Uses the SAME real data already backing sec:pcfix and the onehot ablation
(results/risk_gated_vla/pc_fix_train_base100, pc_fix_devtest_base200) --
no new MuJoCo collection. CONFIRMATORY BATCH (base-seed 300) never touched.

Usage:
    conda run -n tango python scripts/pc_fix_leave_one_object_out.py \\
        --train results/risk_gated_vla/pc_fix_train_base100/scenes.jsonl \\
        --devtest results/risk_gated_vla/pc_fix_devtest_base200/scenes.jsonl \\
        --out-dir results/risk_gated_vla/leave_one_object_out \\
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
from world_model.train_counterfactual_critic import train_one, OBJECTS
from scripts.risk_gated_vla_phase1_stats import mcnemar_exact

VARIANTS = ["pose_only", "pose_pc"]


def feature(rec: dict, cand: dict, variant: str) -> list[float]:
    pose = np.asarray(cand["candidate_pose"], dtype=np.float32)
    obj = np.asarray(rec["obj_pos_before"], dtype=np.float32)
    xyz = pose[:3] - obj
    yaw = float(pose[3])
    base = [*xyz.tolist(), math.sin(yaw), math.cos(yaw),
            float(pose[4]), float(pose[5])]
    pc = [float(v) for v in cand.get("pc_stats_local", rec["pc_stats_before"])]
    if variant == "pose_only":
        return base
    elif variant == "pose_pc":
        return base + pc
    raise ValueError(variant)


def build_scenes(records, variant):
    scenes = []
    for rec in records:
        xs = [feature(rec, c, variant) for c in rec["oracle_per_candidate"]]
        ys = [float(c["success"]) for c in rec["oracle_per_candidate"]]
        scenes.append({"key": (rec["object"], rec["seed"]),
                        "object": rec["object"], "x": xs, "y": ys})
    return scenes


def build_eval_scenes(records, variant):
    scenes = build_scenes(records, variant)
    for s, rec in zip(scenes, records):
        s["geometry_success"] = bool(rec["outcomes"]["geometry"]["success"])
    return scenes


def score_ensemble_pick(scene, ensemble):
    scores = []
    for model, mean, std in ensemble:
        x = (torch.tensor(scene["x"], dtype=torch.float32) - mean) / std
        with torch.no_grad():
            scores.append(model(x).sigmoid().numpy())
    mean_score = np.mean(scores, axis=0)
    idx = int(np.argmax(mean_score))
    return bool(scene["y"][idx])


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

    train_records = [json.loads(l) for l in Path(args.train).read_text().splitlines()]
    devtest_records = [json.loads(l) for l in Path(args.devtest).read_text().splitlines()]

    pooled_picks = {v: [] for v in VARIANTS}
    pooled_geo = []
    per_object_results = {}

    for held_out in OBJECTS:
        train_subset = [r for r in train_records if r["object"] != held_out]
        eval_subset = [r for r in devtest_records if r["object"] == held_out]
        print(f"\n=== held out: {held_out} (train on {sorted(set(r['object'] for r in train_subset))}, "
              f"{len(train_subset)} scenes; eval on {len(eval_subset)} held-out scenes) ===")

        fold_results = {}
        for variant in VARIANTS:
            train_scenes = build_scenes(train_subset, variant)
            eval_scenes = build_eval_scenes(eval_subset, variant)

            ensemble = []
            for seed in range(args.seeds):
                model, mean, std, _, _, val_metrics = train_one(
                    train_scenes, "object_counterfactual", seed, args.epochs)
                ensemble.append((model, mean, std))
                ckpt = {"held_out": held_out, "variant": variant, "seed": seed,
                        "state_dict": model.state_dict(), "dim": len(mean),
                        "mean": mean, "std": std}
                torch.save(ckpt, out / f"loo_{held_out}_{variant}_seed{seed}.pt")
                print(f"  [{held_out}/{variant}] seed={seed} train-internal-val: {json.dumps(val_metrics)}")

            picks = [score_ensemble_pick(s, ensemble) for s in eval_scenes]
            n_success = sum(picks)
            fold_results[variant] = picks
            pooled_picks[variant].extend(picks)
            print(f"  [{held_out}/{variant}] held-out-object top1: {n_success}/{len(picks)} "
                  f"({100*n_success/len(picks):.1f}%)")

        geo = [s["geometry_success"] for s in build_eval_scenes(eval_subset, "pose_only")]
        pooled_geo.extend(geo)
        n_geo = sum(geo)
        print(f"  [{held_out}/geometry] {n_geo}/{len(geo)} ({100*n_geo/len(geo):.1f}%)")

        per_object_results[held_out] = {
            "n": len(eval_subset),
            "geometry": n_geo,
            **{v: sum(fold_results[v]) for v in VARIANTS},
        }

    n = len(pooled_geo)
    print(f"\n=== Pooled leave-one-object-out results (n={n}) ===")
    print(f"geometry:  {sum(pooled_geo)}/{n} ({100*sum(pooled_geo)/n:.1f}%)")
    for v in VARIANTS:
        k = sum(pooled_picks[v])
        print(f"{v:>10}: {k}/{n} ({100*k/n:.1f}%)")

    print("\nPairwise McNemar (pooled, held-out-object scenes only):")
    comparisons = {}
    comparisons["pose_only_vs_pose_pc"] = mcnemar_exact(pooled_picks["pose_only"], pooled_picks["pose_pc"])
    comparisons["geometry_vs_pose_only"] = mcnemar_exact(pooled_geo, pooled_picks["pose_only"])
    comparisons["geometry_vs_pose_pc"] = mcnemar_exact(pooled_geo, pooled_picks["pose_pc"])
    for k, v in comparisons.items():
        print(f"  {k}: {v}")

    (out / "leave_one_object_out_summary.json").write_text(json.dumps({
        "n_pooled": n,
        "pooled_geometry": sum(pooled_geo),
        "pooled_top1_by_variant": {v: sum(pooled_picks[v]) for v in VARIANTS},
        "per_object_results": per_object_results,
        "pairwise_mcnemar": comparisons,
        "pooled_picks_by_variant": pooled_picks,
        "pooled_geometry_picks": pooled_geo,
    }, indent=2))
    print(f"\nWrote {out / 'leave_one_object_out_summary.json'}")


if __name__ == "__main__":
    main()
