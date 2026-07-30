#!/usr/bin/env python3
"""
Evaluate the multi-head contact/lift/success critic (C.3).

Two-phase protocol, matching the user's explicit requirement (dev selects
everything; confirmatory is read exactly once, at the end):

  --mode select   Score every trained checkpoint (all weightings x seeds) on
                   the dev-200 batch, pick the best (weighting, seed) by
                   mixed-scene top-1 accuracy (success head), sweep and
                   freeze per-head classification thresholds on the SAME
                   dev-200 data. Writes model_selection.json. Iterate here
                   freely -- this is not the one-time confirmatory look.

  --mode confirm  Load ONLY the frozen selection from model_selection.json,
                   evaluate once on the confirmatory-300 batch, write the
                   full report. Appends a timestamped entry to
                   confirmatory_run_log.jsonl every time this mode is
                   invoked, so any repeat invocation is visible in the
                   record rather than silently overwriting a "single" look.

ALL numbers reported here are OFFLINE RE-SCORING (this checkpoint was never
the live-selected "world_critic" during any scenes.jsonl collection run) --
must never be presented as, or mixed with, this project's live-executed
primary results (final_report.md's dev-test/confirmatory critic-vs-geometry
numbers).
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_model.train_counterfactual_critic import feature, OBJECTS
from world_model.train_multihead_critic import MultiHeadCritic
from world_model.multihead_labels import (
    derive_success, derive_retained_grasp_proxy, failure_type_3class,
    FAILURE_TYPES_REALIZED,
)

HEADS_BINARY = ("bilateral_contact", "lifted", "success")


def load_scenes_raw(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MultiHeadCritic(int(ckpt["dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {**ckpt, "model": model}


def score_scene(rec: dict, ckpt: dict) -> dict:
    """Returns per-candidate arrays: sigmoid probs for the 3 binary heads,
    softmax probs for failure_type (3-class), plus the raw candidate labels."""
    cands = rec["oracle_per_candidate"]
    x = torch.tensor([feature(rec, c, relative=True) for c in cands], dtype=torch.float32)
    x = (x - ckpt["mean"]) / ckpt["std"]
    with torch.no_grad():
        preds = ckpt["model"](x)
    out = {
        "bilateral_contact": torch.sigmoid(preds["bilateral_contact"]).numpy(),
        "lifted": torch.sigmoid(preds["lifted"]).numpy(),
        "success": torch.sigmoid(preds["success"]).numpy(),
        "failure_type_prob": torch.softmax(preds["failure_type"], dim=-1).numpy(),
    }
    labels = {
        "bilateral_contact": np.array([float(c["bilateral_contact"]) for c in cands]),
        "lifted": np.array([float(c["lifted"]) for c in cands]),
        "success": np.array([float(derive_success(c)) for c in cands]),
        "failure_type_idx": np.array(
            [FAILURE_TYPES_REALIZED.index(failure_type_3class(c)) for c in cands]
        ),
        "retained_grasp_proxy_diagnostic": np.array(
            [float(derive_retained_grasp_proxy(c)) for c in cands]
        ),
        "geo_score": np.array([float(c["geo_score"]) for c in cands]),
    }
    return out, labels


def ece(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if mask.sum() == 0:
            continue
        conf = float(score[mask].mean())
        acc = float(y[mask].mean())
        total += (mask.sum() / len(y)) * abs(conf - acc)
    return float(total)


def safe_auroc(y, score):
    return float(roc_auc_score(y, score)) if len(set(y.tolist())) > 1 else float("nan")


def safe_auprc(y, score):
    return float(average_precision_score(y, score)) if len(set(y.tolist())) > 1 else float("nan")


def mcnemar_exact(a_succ, b_succ) -> dict:
    a, b = np.asarray(a_succ, dtype=bool), np.asarray(b_succ, dtype=bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    p = sps.binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue if n else 1.0
    return {"n01": n01, "n10": n10, "n_discordant": n, "p_value": p}


def collect_all(rows: list, ckpt: dict, threshold_success: float = 0.5) -> dict:
    """Pools per-candidate predictions/labels across all scenes, plus per-scene
    top-1 (mixed-scene accuracy) and per-scene geometry-vs-critic outcome
    (offline re-scored, for McNemar)."""
    pooled_pred = {h: [] for h in HEADS_BINARY}
    pooled_label = {h: [] for h in HEADS_BINARY}
    pooled_ft_true, pooled_ft_pred = [], []
    pooled_object = []
    pooled_proxy = []

    top_ok, oracle_ok, mixed_top, mixed_n = 0, 0, 0, 0
    critic_scene_success, geo_scene_success = [], []
    scene_objects = []

    for rec in rows:
        preds, labels = score_scene(rec, ckpt)
        for h in HEADS_BINARY:
            pooled_pred[h].extend(preds[h].tolist())
            pooled_label[h].extend(labels[h].tolist())
        pooled_ft_true.extend(labels["failure_type_idx"].tolist())
        pooled_ft_pred.extend(np.argmax(preds["failure_type_prob"], axis=-1).tolist())
        pooled_object.extend([rec["object"]] * len(labels["success"]))
        pooled_proxy.extend(labels["retained_grasp_proxy_diagnostic"].tolist())

        y_success = labels["success"].astype(int)
        idx_critic = int(np.argmax(preds["success"]))
        idx_geo = int(np.argmax(labels["geo_score"]))
        top_ok += int(y_success[idx_critic])
        oracle_ok += int(y_success.any())
        if y_success.any() and not y_success.all():
            mixed_n += 1
            mixed_top += int(y_success[idx_critic])
        critic_scene_success.append(bool(y_success[idx_critic]))
        geo_scene_success.append(bool(y_success[idx_geo]))
        scene_objects.append(rec["object"])

    n_scenes = len(rows)
    return {
        "pooled_pred": pooled_pred, "pooled_label": pooled_label,
        "pooled_ft_true": pooled_ft_true, "pooled_ft_pred": pooled_ft_pred,
        "pooled_object": pooled_object, "pooled_proxy": pooled_proxy,
        "mixed_top1": {"top1": top_ok / max(n_scenes, 1), "oracle": oracle_ok / max(n_scenes, 1),
                       "mixed_top1": mixed_top / max(mixed_n, 1), "mixed_n": mixed_n},
        "critic_scene_success": critic_scene_success, "geo_scene_success": geo_scene_success,
        "scene_objects": scene_objects,
    }


def build_report(rows: list, ckpt: dict, split_name: str) -> dict:
    agg = collect_all(rows, ckpt)
    report = {"split": split_name, "scoring": "OFFLINE_RESCORING_NOT_LIVE_EXECUTED",
              "n_scenes": len(rows), "weighting": ckpt.get("weighting"), "seed": ckpt.get("seed")}

    # per-head AUROC/AUPRC/ECE, overall and per-object
    report["heads"] = {}
    for h in HEADS_BINARY:
        y = np.array(agg["pooled_label"][h])
        s = np.array(agg["pooled_pred"][h])
        report["heads"][h] = {
            "overall": {"auroc": safe_auroc(y, s), "auprc": safe_auprc(y, s), "ece": ece(y, s), "n": len(y)},
            "per_object": {},
        }
        for obj in OBJECTS:
            mask = np.array([o == obj for o in agg["pooled_object"]])
            if mask.sum() == 0:
                continue
            report["heads"][h]["per_object"][obj] = {
                "auroc": safe_auroc(y[mask], s[mask]), "auprc": safe_auprc(y[mask], s[mask]),
                "ece": ece(y[mask], s[mask]), "n": int(mask.sum()),
            }

    # failure_type confusion matrix, overall and per-object
    yt = np.array(agg["pooled_ft_true"], dtype=int)
    yp = np.array(agg["pooled_ft_pred"], dtype=int)
    cm = confusion_matrix(yt, yp, labels=list(range(len(FAILURE_TYPES_REALIZED))))
    report["failure_type_confusion_matrix"] = {
        "labels": list(FAILURE_TYPES_REALIZED), "matrix": cm.tolist(),
    }
    per_obj_cm = {}
    for obj in OBJECTS:
        mask = np.array([o == obj for o in agg["pooled_object"]])
        if mask.sum() == 0:
            continue
        per_obj_cm[obj] = confusion_matrix(
            yt[mask], yp[mask], labels=list(range(len(FAILURE_TYPES_REALIZED)))
        ).tolist()
    report["failure_type_confusion_matrix_per_object"] = per_obj_cm

    # weld_no_lift drill-share + non-drill support (req #2/#3) -- never
    # phrase this as a cross-object generalization claim (req #4)
    ft_counts = Counter(FAILURE_TYPES_REALIZED[i] for i in yt)
    drill_mask = np.array([o == "drill" for o in agg["pooled_object"]])
    drill_weld_no_lift = int(np.sum((yt == FAILURE_TYPES_REALIZED.index("weld_no_lift")) & drill_mask))
    total_weld_no_lift = int(ft_counts.get("weld_no_lift", 0))
    report["weld_no_lift_object_attribution"] = {
        "total": total_weld_no_lift, "drill": drill_weld_no_lift,
        "non_drill": total_weld_no_lift - drill_weld_no_lift,
        "drill_share_pct": (100 * drill_weld_no_lift / total_weld_no_lift) if total_weld_no_lift else float("nan"),
        "note": ("weld_no_lift support outside drill is near-zero in this dataset -- any "
                 "weld_no_lift metric (AUROC/AUPRC/confusion-matrix cell) is effectively a "
                 "within-drill measurement, not evidence of a cross-object generalization "
                 "pattern. Do not describe it as one."),
    }

    # retained_grasp_proxy diagnostic-only cross-check (never a head, never in loss)
    proxy = np.array(agg["pooled_proxy"])
    bc = np.array(agg["pooled_label"]["bilateral_contact"])
    report["retained_grasp_proxy_diagnostic_only"] = {
        "note": "DIAGNOSTIC ALIAS ONLY -- weld_triggered, not independent, not a trained head, "
                "not equivalent to true post-settle retention.",
        "matches_bilateral_contact_label": bool(np.array_equal(proxy, bc)),
    }

    # mixed-scene top-1 (success head) + geometry comparison, offline re-scored
    report["mixed_scene_top1"] = agg["mixed_top1"]
    geo_succ = agg["geo_scene_success"]
    critic_succ = agg["critic_scene_success"]
    report["geometry_baseline_comparison_offline"] = {
        "geometry_sr": float(np.mean(geo_succ)), "critic_sr": float(np.mean(critic_succ)),
        "delta_pp": float(np.mean(critic_succ) - np.mean(geo_succ)) * 100,
        "mcnemar_exact": mcnemar_exact(geo_succ, critic_succ),
    }
    per_obj_geo = {}
    objs_arr = np.array(agg["scene_objects"])
    for obj in OBJECTS:
        mask = objs_arr == obj
        if mask.sum() == 0:
            continue
        g = np.array(geo_succ)[mask]
        c = np.array(critic_succ)[mask]
        per_obj_geo[obj] = {
            "n": int(mask.sum()), "geometry_sr": float(g.mean()), "critic_sr": float(c.mean()),
            "delta_pp": float(c.mean() - g.mean()) * 100,
            "mcnemar_exact": mcnemar_exact(g.tolist(), c.tolist()),
        }
    report["geometry_baseline_comparison_offline"]["per_object"] = per_obj_geo

    return report


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(f"not serializable: {type(o)}")


def mode_select(args):
    dev_rows = load_scenes_raw(Path(args.dev_data))
    model_dir = Path(args.model_dir)
    candidates = []
    for weighting in ("equal", "success_weighted"):
        for path in sorted(model_dir.glob(f"multihead_{weighting}_seed*.pt")):
            ckpt = load_checkpoint(path)
            rep = build_report(dev_rows, ckpt, split_name="dev-200(selection)")
            candidates.append({
                "weighting": weighting, "seed": ckpt["seed"], "path": str(path),
                "dev_mixed_top1": rep["mixed_scene_top1"]["mixed_top1"],
                "dev_success_auroc": rep["heads"]["success"]["overall"]["auroc"],
            })
            print(f"[select] {weighting} seed={ckpt['seed']}: "
                  f"dev_mixed_top1={rep['mixed_scene_top1']['mixed_top1']:.3f}  "
                  f"dev_success_auroc={rep['heads']['success']['overall']['auroc']:.3f}")

    best = max(candidates, key=lambda r: r["dev_mixed_top1"])
    print(f"\n[select] chosen: weighting={best['weighting']} seed={best['seed']} "
          f"(dev_mixed_top1={best['dev_mixed_top1']:.3f})")

    selection = {"chosen": best, "all_candidates": candidates, "selected_at": time.time()}
    out_path = Path(args.out_dir) / "model_selection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selection, indent=2, default=_json_default))
    print(f"[select] wrote {out_path} -- this selection is now FROZEN for --mode confirm")


def mode_confirm(args):
    sel_path = Path(args.out_dir) / "model_selection.json"
    if not sel_path.exists():
        raise SystemExit(f"no model_selection.json at {sel_path} -- run --mode select first")
    selection = json.loads(sel_path.read_text())
    chosen = selection["chosen"]

    log_path = Path(args.out_dir) / "confirmatory_run_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": time.time(), "chosen_weighting": chosen["weighting"],
            "chosen_seed": chosen["seed"], "confirmatory_data": args.confirmatory_data,
        }) + "\n")
    n_prior_runs = sum(1 for _ in open(log_path))
    if n_prior_runs > 1:
        print(f"\n*** WARNING: this is confirmatory run #{n_prior_runs} against "
              f"{log_path} -- the 'evaluate exactly once' discipline has been violated if "
              f"any earlier run's result was already reported. Check {log_path} before "
              f"trusting this output as the paper's number. ***\n")

    ckpt = load_checkpoint(Path(chosen["path"]))
    rows = load_scenes_raw(Path(args.confirmatory_data))
    report = build_report(rows, ckpt, split_name="confirmatory-300(FINAL)")
    report["frozen_selection"] = chosen
    report["confirmatory_run_count"] = n_prior_runs

    out_path = Path(args.out_dir) / "confirmatory_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=_json_default))
    print(json.dumps(report, indent=2, default=_json_default))
    print(f"\n[confirm] wrote {out_path}  (run #{n_prior_runs}, log: {log_path})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["select", "confirm"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dev-data", default="results/risk_gated_vla/counterfactual_test_n30_20260730/scenes.jsonl")
    ap.add_argument("--confirmatory-data", default="results/risk_gated_vla/confirmatory_n50_seed300_20260730/scenes.jsonl")
    args = ap.parse_args()
    if args.mode == "select":
        mode_select(args)
    else:
        mode_confirm(args)


if __name__ == "__main__":
    main()
