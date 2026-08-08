"""P1.1: separation analysis over the PIPER_BASELINE_V1 outcome dataset.

Applies every guard that earlier passes learned the hard way:
  - outcome-derived variables are refused (dist_to_tray etc.);
  - pooled AUC is never reported alone -- within-object AUC and
    leave-one-object-out AUC accompany it;
  - direction consistency across objects is required;
  - bootstrap CI on pooled AUC, so "0.71" with a CI spanning 0.5 is not
    mistaken for evidence.

A variable is promoted to CANDIDATE SEPARATOR only if it passes all of:
  pooled CI excludes 0.5, every object with enough data agrees in
  direction, and the weakest within-object AUC still deviates from 0.5.

Run:  conda run -n tango python scripts/analyze_piper_outcome_dataset.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.piper_outcome_dataset import OUTCOME_DERIVED  # noqa: E402

DATA = ROOT / "outputs" / "piper_outcome_dataset.jsonl"
MIN_PER_CLASS = 4


def auc(pos, neg):
    if len(pos) < 1 or len(neg) < 1:
        return None
    w = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def auc_ci(pos, neg, n_boot=2000, seed=0):
    if len(pos) < 3 or len(neg) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)
    vals = []
    for _ in range(n_boot):
        a = rng.choice(pos, len(pos), replace=True)
        b = rng.choice(neg, len(neg), replace=True)
        vals.append(auc(list(a), list(b)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    rows = [json.loads(l) for l in DATA.open()]
    objects = sorted({r["object"] for r in rows})
    print(f"{len(rows)} rollouts, {sum(r['success'] for r in rows)} successes")
    for o in objects:
        sub = [r for r in rows if r["object"] == o]
        print(f"  {o:9s} {sum(r['success'] for r in sub):3d}/{len(sub):3d}")

    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float, bool)) and k not in ("seed",):
                keys.add(k)
    keys -= OUTCOME_DERIVED
    keys.discard("success")
    leaked = keys & OUTCOME_DERIVED
    if leaked:
        raise ValueError(f"outcome-derived variable reached analysis: {leaked}")

    results = []
    for k in sorted(keys):
        P = [float(r[k]) for r in rows if r["success"] and r.get(k) is not None]
        N = [float(r[k]) for r in rows if not r["success"] and r.get(k) is not None]
        if len(P) < MIN_PER_CLASS or len(N) < MIN_PER_CLASS:
            continue
        pooled = auc(P, N)
        lo, hi = auc_ci(P, N)

        per_obj = {}
        for o in objects:
            sub = [r for r in rows if r["object"] == o]
            p = [float(r[k]) for r in sub if r["success"] and r.get(k) is not None]
            n = [float(r[k]) for r in sub if not r["success"] and r.get(k) is not None]
            per_obj[o] = auc(p, n) if (len(p) >= 3 and len(n) >= 3) else None

        loo = {}
        for o in objects:
            sub = [r for r in rows if r["object"] != o]
            p = [float(r[k]) for r in sub if r["success"] and r.get(k) is not None]
            n = [float(r[k]) for r in sub if not r["success"] and r.get(k) is not None]
            loo[o] = auc(p, n) if (len(p) >= 3 and len(n) >= 3) else None

        usable = [v for v in per_obj.values() if v is not None]
        ci_excludes_half = (lo is not None and (lo > 0.5 or hi < 0.5))
        consistent = len(usable) >= 2 and (
            all(v > 0.5 for v in usable) or all(v < 0.5 for v in usable))
        weakest = min((abs(v - 0.5) for v in usable), default=0.0)
        promoted = bool(ci_excludes_half and consistent and weakest >= 0.10)

        results.append(dict(key=k, pooled=pooled, lo=lo, hi=hi, per_obj=per_obj,
                            loo=loo, promoted=promoted, weakest=weakest,
                            consistent=consistent, ci_ok=ci_excludes_half))

    results.sort(key=lambda d: (d["promoted"], abs(d["pooled"] - 0.5)), reverse=True)

    print("\n" + "=" * 104)
    print("pooled AUC [95% bootstrap CI] | per-object AUC | promoted?")
    print("=" * 104)
    hdr = f"{'variable':38s} {'AUC':>5s} {'95% CI':>14s}  " + " ".join(f"{o[:7]:>7s}" for o in objects)
    print(hdr)
    print("-" * 104)
    for d in results[:20]:
        ci = f"[{d['lo']:.2f},{d['hi']:.2f}]" if d["lo"] is not None else "     -"
        cells = " ".join((f"{d['per_obj'][o]:7.2f}" if d["per_obj"][o] is not None else "      -")
                         for o in objects)
        mark = "  PROMOTED" if d["promoted"] else ""
        print(f"{d['key']:38s} {d['pooled']:5.2f} {ci:>14s}  {cells}{mark}")

    print("\npromotion requires: CI excludes 0.50, all objects agree in direction,")
    print("and the weakest per-object |AUC-0.50| >= 0.10.")

    promoted = [d for d in results if d["promoted"]]
    print(f"\n{len(promoted)} variable(s) promoted to candidate separator:")
    for d in promoted:
        print(f"  {d['key']}  pooled={d['pooled']:.2f} [{d['lo']:.2f},{d['hi']:.2f}]  "
              f"LOO={{{', '.join(f'{o}:{v:.2f}' for o, v in d['loo'].items() if v is not None)}}}")
    if not promoted:
        print("  (none -- no variable in this dataset separates outcomes robustly)")

    (ROOT / "outputs" / "piper_separation_report.json").write_text(
        json.dumps(results, indent=1, default=str))
    print(f"\nwrote {ROOT / 'outputs' / 'piper_separation_report.json'}")


if __name__ == "__main__":
    main()
