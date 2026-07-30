#!/usr/bin/env python3
"""Offline top-1 evaluation on fully executed, held-out candidate pools."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_model.train_counterfactual_critic import load_ensemble, score_candidates


def exact_mcnemar(a, b):
    a_win = sum(x and not y for x, y in zip(a, b))
    b_win = sum(y and not x for x, y in zip(a, b))
    n = a_win + b_win
    if not n:
        return a_win, b_win, 1.0
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a_win, b_win) + 1)) / 2**n)
    return a_win, b_win, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.data).read_text().splitlines()]
    results = {}

    geo = []
    for rec in rows:
        cs = rec["oracle_per_candidate"]
        geo.append(bool(cs[int(np.argmax([c["geo_score"] for c in cs]))]["success"]))
    results["geometry"] = {"outcomes": geo}

    for variant in ("global_bce", "object_bce", "object_counterfactual"):
        bundles = load_ensemble(Path(args.model_dir), variant)
        relative = variant != "global_bce"
        outcomes, uncertainties = [], []
        for rec in rows:
            cs = rec["oracle_per_candidate"]
            score, unc = score_candidates(rec, cs, bundles, relative)
            idx = int(np.argmax(score))
            outcomes.append(bool(cs[idx]["success"]))
            uncertainties.append(float(unc[idx]))
        results[variant] = {"outcomes": outcomes, "uncertainty": uncertainties}

    summary = {}
    objects = sorted(set(r["object"] for r in rows))
    for name, vals in results.items():
        outcomes = vals["outcomes"]
        item = {"n": len(rows), "successes": sum(outcomes),
                "success_rate": sum(outcomes) / len(outcomes), "per_object": {}}
        for obj in objects:
            ids = [i for i, r in enumerate(rows) if r["object"] == obj]
            yy = [outcomes[i] for i in ids]
            item["per_object"][obj] = {"n": len(yy), "successes": sum(yy),
                                        "success_rate": sum(yy) / len(yy)}
        if name != "geometry":
            wins, losses, p = exact_mcnemar(outcomes, geo)
            item.update({"wins_vs_geometry": wins, "losses_vs_geometry": losses,
                         "mcnemar_p": p,
                         "delta_pp": 100 * (item["success_rate"] - np.mean(geo))})
        summary[name] = item

    comparisons = {}
    names = list(results)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            wins, losses, p = exact_mcnemar(
                results[left]["outcomes"], results[right]["outcomes"])
            comparisons[f"{left}_vs_{right}"] = {
                "left_wins": wins, "right_wins": losses, "mcnemar_p": p
            }
    summary["paired_comparisons"] = comparisons

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
