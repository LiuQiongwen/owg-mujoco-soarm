#!/usr/bin/env python3
"""
Risk-Gated VLA -- Phase 1 statistics.

Reads scenes.jsonl produced by scripts/risk_gated_vla_phase1_eval.py and computes,
per the pre-registered plan (results/risk_gated_vla/preregistration.yaml):

  - paired success table (per object + pooled) for random / geometry / world_critic / oracle
  - McNemar's exact test (scipy.stats.binomtest on discordant pairs -- this project's
    established convention, see RULED_OUT_METHODS.md; no statsmodels dependency)
  - bootstrap 95% CI on the success-rate delta (world_critic vs geometry)
  - critic AUROC / AUPRC against real per-candidate ground truth (oracle_per_candidate)
  - calibration (ECE, 10 bins) of the critic's success_prob
  - risk-coverage curve using |success_prob - 0.5| as the confidence proxy
  - oracle headroom (oracle SR - best single method SR)

Usage:
  conda run -n tango python scripts/risk_gated_vla_phase1_stats.py \\
      --scenes results/risk_gated_vla/phase1/scenes.jsonl \\
      --out results/risk_gated_vla/tables/phase1_stats.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

METHODS = ["random", "geometry", "world_critic"]


def load_scenes(path: Path) -> list:
    return [json.loads(line) for line in open(path) if line.strip()]


# ── Paired success table + McNemar ──────────────────────────────────────────

def mcnemar_exact(a_succ: list, b_succ: list) -> dict:
    """a_succ, b_succ: parallel boolean lists (same trials). Exact McNemar via
    scipy.stats.binomtest on discordant pairs (project convention)."""
    a = np.asarray(a_succ, dtype=bool)
    b = np.asarray(b_succ, dtype=bool)
    n01 = int(np.sum(~a & b))   # a fails, b succeeds
    n10 = int(np.sum(a & ~b))   # a succeeds, b fails
    n_disc = n01 + n10
    if n_disc == 0:
        p = 1.0
    else:
        p = stats.binomtest(min(n01, n10), n_disc, 0.5, alternative="two-sided").pvalue
    return {"n01": n01, "n10": n10, "n_discordant": n_disc, "p_value": p}


def bootstrap_ci_delta(a_succ: list, b_succ: list, n_boot: int = 10000, seed: int = 0) -> dict:
    """Bootstrap 95% CI on mean(b) - mean(a), resampling paired trials jointly."""
    a = np.asarray(a_succ, dtype=float)
    b = np.asarray(b_succ, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = b[idx].mean() - a[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_mean": float(b.mean() - a.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2n = z * z / n
    denom = 1.0 + z2n
    center = (p + z2n / 2.0) / denom
    half = z * ((p * (1 - p) / n + z2n / (4 * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ── Critic quality: AUROC/AUPRC/ECE/risk-coverage (per-candidate ground truth) ─

def critic_quality(scenes: list) -> dict:
    y, score = [], []
    for rec in scenes:
        for c in rec["oracle_per_candidate"]:
            y.append(int(c["success"]))
            score.append(float(c["success_prob"]))
    y = np.asarray(y)
    score = np.asarray(score)

    auroc = float(roc_auc_score(y, score)) if len(set(y.tolist())) > 1 else float("nan")
    auprc = float(average_precision_score(y, score)) if len(set(y.tolist())) > 1 else float("nan")

    # ECE: 10 equal-width bins on predicted score
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if mask.sum() == 0:
            continue
        conf = float(score[mask].mean())
        acc = float(y[mask].mean())
        w = mask.sum() / len(y)
        ece += w * abs(conf - acc)
        bin_table.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                          "mean_pred": conf, "mean_actual": acc})

    # Risk-coverage: confidence = |score - 0.5|, sorted descending, cumulative
    # selective accuracy at each coverage level
    conf = np.abs(score - 0.5)
    order = np.argsort(-conf)
    y_sorted = y[order]
    coverages = np.arange(1, len(y) + 1) / len(y)
    cum_acc = np.cumsum(y_sorted) / np.arange(1, len(y) + 1)
    rc_curve = [{"coverage": float(c), "selective_accuracy": float(a)}
                for c, a in zip(coverages[::max(1, len(y)//50)], cum_acc[::max(1, len(y)//50)])]

    return {"n_candidates": int(len(y)), "auroc": auroc, "auprc": auprc,
            "ece": float(ece), "calibration_bins": bin_table,
            "risk_coverage_curve": rc_curve}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scenes = load_scenes(Path(args.scenes))
    objects = sorted(set(r["object"] for r in scenes))

    report = {"n_scenes": len(scenes), "objects": objects, "per_object": {}, "pooled": {}}

    by_obj = {o: [r for r in scenes if r["object"] == o] for o in objects}

    def method_succ(rows, method):
        return [bool(r["outcomes"][method]["success"]) for r in rows]

    def oracle_succ(rows):
        return [bool(r["oracle_success"]) for r in rows]

    for obj, rows in by_obj.items():
        obj_report = {"n": len(rows)}
        succ = {m: method_succ(rows, m) for m in METHODS}
        succ["oracle"] = oracle_succ(rows)
        for m in METHODS + ["oracle"]:
            k = sum(succ[m])
            n = len(succ[m])
            lo, hi = wilson_ci(k, n)
            obj_report[m] = {"sr": k / n if n else float("nan"), "k": k, "n": n,
                             "ci_lo": lo, "ci_hi": hi}
        obj_report["mcnemar_world_critic_vs_geometry"] = mcnemar_exact(succ["geometry"], succ["world_critic"])
        obj_report["mcnemar_world_critic_vs_random"] = mcnemar_exact(succ["random"], succ["world_critic"])
        obj_report["bootstrap_delta_world_critic_vs_geometry"] = bootstrap_ci_delta(
            succ["geometry"], succ["world_critic"])
        obj_report["oracle_headroom_vs_best_method"] = (
            obj_report["oracle"]["sr"] - max(obj_report[m]["sr"] for m in METHODS)
        )
        report["per_object"][obj] = obj_report

    # pooled (concatenate all objects' paired trials -- valid because pairing is
    # within-scene, not within-object; pooling preserves the pairing structure)
    all_rows = scenes
    succ = {m: method_succ(all_rows, m) for m in METHODS}
    succ["oracle"] = oracle_succ(all_rows)
    pooled = {"n": len(all_rows)}
    for m in METHODS + ["oracle"]:
        k = sum(succ[m]); n = len(succ[m])
        lo, hi = wilson_ci(k, n)
        pooled[m] = {"sr": k / n if n else float("nan"), "k": k, "n": n, "ci_lo": lo, "ci_hi": hi}
    pooled["mcnemar_world_critic_vs_geometry"] = mcnemar_exact(succ["geometry"], succ["world_critic"])
    pooled["mcnemar_world_critic_vs_random"] = mcnemar_exact(succ["random"], succ["world_critic"])
    pooled["bootstrap_delta_world_critic_vs_geometry"] = bootstrap_ci_delta(succ["geometry"], succ["world_critic"])
    pooled["oracle_headroom_vs_best_method"] = pooled["oracle"]["sr"] - max(pooled[m]["sr"] for m in METHODS)
    report["pooled"] = pooled

    report["critic_quality"] = critic_quality(scenes)

    # ── Pre-registered gate check (world_critic vs geometry, the direction the
    # world-model literature/wm_reranking_results.md claimed) ──
    pooled_delta = pooled["world_critic"]["sr"] - pooled["geometry"]["sr"]
    per_obj_deltas = [report["per_object"][o]["world_critic"]["sr"] - report["per_object"][o]["geometry"]["sr"]
                       for o in objects]
    n_same_direction = sum(1 for d in per_obj_deltas if (d > 0) == (pooled_delta > 0) and d != 0)
    gate = {
        "pooled_delta_pp": pooled_delta * 100,
        "threshold_pp": 8.0,
        "meets_effect_size": pooled_delta * 100 >= 8.0,
        "mcnemar_p": pooled["mcnemar_world_critic_vs_geometry"]["p_value"],
        "meets_significance": pooled["mcnemar_world_critic_vs_geometry"]["p_value"] < 0.05,
        "n_objects_same_direction": n_same_direction,
        "n_objects_total": len(objects),
        "meets_direction_consistency": n_same_direction >= 2,
    }
    gate["PASS"] = (gate["meets_effect_size"] and gate["meets_significance"]
                     and gate["meets_direction_consistency"])
    report["preregistered_gate_world_critic_vs_geometry"] = gate

    def _json_default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=_json_default))
    print(json.dumps(report, indent=2, default=_json_default))
    print(f"\n[stats] wrote {out_path}")
    print(f"\n[gate] world_critic vs geometry: {'PASS' if gate['PASS'] else 'FAIL'}  "
          f"(pooled_delta={gate['pooled_delta_pp']:+.1f}pp, p={gate['mcnemar_p']:.4f}, "
          f"{gate['n_objects_same_direction']}/{gate['n_objects_total']} objects same direction)")


if __name__ == "__main__":
    main()
