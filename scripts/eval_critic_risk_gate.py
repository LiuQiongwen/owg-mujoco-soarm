#!/usr/bin/env python3
"""Calibrate on a development set, then evaluate a frozen critic risk gate.

All outcomes come from fully executed, shared candidate pools.  Calibration
selects only an ensemble-uncertainty threshold; the confirmatory file is never
read until that threshold has been frozen.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_model.risk_gate import CriticRiskGate
from world_model.train_counterfactual_critic import load_ensemble


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def scene_arrays(rec):
    cs = rec["oracle_per_candidate"]
    return ([c["candidate_pose"] for c in cs],
            [float(c["geo_score"]) for c in cs],
            np.asarray([bool(c["success"]) for c in cs]))


def decisions(records, ensemble, threshold):
    gate = CriticRiskGate(ensemble, threshold, min_advantage=0.0, relative=True)
    rows = []
    for rec in records:
        poses, geo, y = scene_arrays(rec)
        d = gate.decide(rec, poses, geo)
        rows.append({"object": rec["object"], "seed": rec["seed"],
                     "gate_success": bool(y[d.candidate_idx]),
                     "critic_success": bool(y[d.policy_idx]),
                     "geometry_success": bool(y[d.fallback_idx]),
                     "accepted": d.accepted, "source": d.source,
                     "candidate_idx": d.candidate_idx,
                     "policy_idx": d.policy_idx,
                     "fallback_idx": d.fallback_idx,
                     "uncertainty": d.uncertainty,
                     "critic_score": d.critic_score,
                     "fallback_score": d.fallback_score})
    return rows


def summarize(rows):
    out = {}
    for obj in sorted({r["object"] for r in rows}) + ["pooled"]:
        rs = rows if obj == "pooled" else [r for r in rows if r["object"] == obj]
        out[obj] = {"n": len(rs),
                    "gate_success": sum(r["gate_success"] for r in rs) / len(rs),
                    "critic_success": sum(r["critic_success"] for r in rs) / len(rs),
                    "geometry_success": sum(r["geometry_success"] for r in rs) / len(rs),
                    "critic_coverage": sum(r["accepted"] for r in rs) / len(rs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-data", required=True)
    ap.add_argument("--confirmatory-data", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--variant", default="object_counterfactual")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ensemble = load_ensemble(Path(args.model_dir), args.variant)
    calibration = load(args.calibration_data)

    # Candidate thresholds are frozen using calibration outcomes only. Include
    # accept-none/all endpoints and empirical uncertainty quantiles.
    probe = decisions(calibration, ensemble, float("inf"))
    u = np.asarray([r["uncertainty"] for r in probe])
    thresholds = sorted(set([-1.0, *np.quantile(u, np.linspace(0, 1, 21)).tolist(), float("inf")]))
    candidates = []
    for threshold in thresholds:
        rows = decisions(calibration, ensemble, threshold)
        summary = summarize(rows)["pooled"]
        candidates.append((summary["gate_success"], -summary["critic_coverage"], threshold, summary))
    # Accuracy first; conservative (lower critic coverage) tie break.
    _, _, frozen_threshold, calibration_summary = max(candidates, key=lambda x: (x[0], x[1]))

    confirmatory = load(args.confirmatory_data)  # first access after freezing
    rows = decisions(confirmatory, ensemble, frozen_threshold)
    result = {"variant": args.variant,
              "calibration_data": args.calibration_data,
              "confirmatory_data": args.confirmatory_data,
              "frozen_uncertainty_threshold": frozen_threshold,
              "calibration_summary": calibration_summary,
              "confirmatory_summary": summarize(rows), "decisions": rows}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, indent=2))


if __name__ == "__main__":
    main()
