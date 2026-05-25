#!/usr/bin/env python3
"""Generate paper_results_summary.md from diverse benchmark outputs.

Reads:
  results/diverse_easy/trials.jsonl
  results/diverse_medium/trials.jsonl
  results/diverse_hard/trials.jsonl
  (or a single --run-dir)

Writes:
  results/paper_results_summary.md

Usage
-----
    python scripts/generate_paper_summary.py
    python scripts/generate_paper_summary.py --run-dir results/diverse_medium
    python scripts/generate_paper_summary.py --all
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.logger import wilson_ci

# reuse stat functions from analyze script without import cycle
def _cohen_h(p1: float, p2: float) -> float:
    if p1 < 0 or p1 > 1 or p2 < 0 or p2 > 1:
        return float("nan")
    return (2 * math.asin(math.sqrt(max(0, min(1, p1)))) -
            2 * math.asin(math.sqrt(max(0, min(1, p2)))))


def _hg_pmf(k: int, n: int, K: int, N: int) -> float:
    from math import comb
    if k < max(0, n + K - N) or k > min(n, K):
        return 0.0
    return comb(K, k) * comb(N - K, n - k) / comb(N, n)


def _fisher_p(k1, n1, k2, n2) -> float:
    N, K, n = n1 + n2, k1 + k2, n1
    if n1 == 0 or n2 == 0 or N == 0:
        return float("nan")
    p_obs = _hg_pmf(k1, n, K, N)
    return float(min(1.0, sum(
        _hg_pmf(k, n, K, N)
        for k in range(max(0, n + K - N), min(n, K) + 1)
        if _hg_pmf(k, n, K, N) <= p_obs + 1e-10
    )))


def _fmt_pct(v: float) -> str:
    return f"{v*100:.1f}%" if not math.isnan(v) else "—"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo*100:.1f}%, {hi*100:.1f}%]"


def load_trials(run_dir: Path) -> List[dict]:
    path = run_dir / "trials.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def agg_by(records: List[dict], *keys) -> Dict[tuple, dict]:
    rows = defaultdict(lambda: {"k": 0, "n": 0})
    for r in records:
        if not r.get("stability_valid"):
            continue
        key = tuple(r.get(k) for k in keys)
        rows[key]["n"] += 1
        if r.get("success"):
            rows[key]["k"] += 1
    return dict(rows)


def generate_md(all_records: List[dict], run_label: str) -> str:
    valid = [r for r in all_records if r.get("stability_valid")]
    if not valid:
        return "No valid trials found.\n"

    methods   = sorted({r["method"]     for r in valid})
    objects   = sorted({r["object"]     for r in valid})
    diffs     = sorted({r.get("difficulty") or "?" for r in valid},
                       key=lambda x: {"easy": 0, "medium": 1, "hard": 2}.get(x, 9))
    n_total   = len(valid)

    lines = []
    lines.append(f"# OWG SO-ARM101 Grasp Benchmark — Results Summary")
    lines.append(f"")
    lines.append(f"**Execution mode**: `physics_weld_after_bilateral`  ")
    lines.append(f"**Methods**: {', '.join(f'`{m}`' for m in methods)}  ")
    lines.append(f"**Difficulties**: {', '.join(diffs)}  ")
    lines.append(f"**Objects**: {', '.join(objects)}  ")
    lines.append(f"**Total valid trials**: {n_total}  ")
    lines.append(f"**Run**: {run_label}  ")
    lines.append(f"")

    # ── 1. Overall success rate ───────────────────────────────────────────────
    lines.append("## 1. Overall Success Rate")
    lines.append("")
    lines.append("| Method | N | Success | 95% CI |")
    lines.append("|--------|---|---------|--------|")
    m_agg = agg_by(valid, "method")
    for m in methods:
        d = m_agg.get((m,), {"k": 0, "n": 0})
        rate, lo, hi = wilson_ci(d["k"], d["n"])
        lines.append(f"| `{m}` | {d['n']} | {_fmt_pct(rate)} | {_fmt_ci(lo, hi)} |")
    lines.append("")

    if len(methods) == 2:
        d1 = m_agg.get((methods[0],), {"k": 0, "n": 0})
        d2 = m_agg.get((methods[1],), {"k": 0, "n": 0})
        r1, lo1, hi1 = wilson_ci(d1["k"], d1["n"])
        r2, lo2, hi2 = wilson_ci(d2["k"], d2["n"])
        h   = _cohen_h(r1, r2)
        p   = _fisher_p(d1["k"], d1["n"], d2["k"], d2["n"])
        sig = "p < 0.05 ✓" if p < 0.05 else f"p = {p:.3f}"
        h_label = ("large" if abs(h) > 0.5 else "medium" if abs(h) > 0.2 else "small")
        lines.append(f"**`{methods[0]}` vs `{methods[1]}`**: "
                     f"Cohen's h = {h:.3f} ({h_label}), Fisher's exact {sig}  ")
        lines.append("")

    # ── 2. By difficulty ─────────────────────────────────────────────────────
    lines.append("## 2. Success Rate by Difficulty")
    lines.append("")
    header = "| Difficulty | " + " | ".join(f"`{m}`" for m in methods) + " |"
    sep    = "| --- | " + " | ".join(["---"] * len(methods)) + " |"
    lines.extend([header, sep])
    d_agg = agg_by(valid, "difficulty", "method")
    for diff in diffs:
        cells = []
        for m in methods:
            d = d_agg.get((diff, m), {"k": 0, "n": 0})
            rate, lo, hi = wilson_ci(d["k"], d["n"])
            cells.append(f"{_fmt_pct(rate)} {_fmt_ci(lo, hi)} (n={d['n']})")
        lines.append(f"| {diff} | " + " | ".join(cells) + " |")
    lines.append("")

    # difficulty effect (for geometry)
    if diffs and "geometry" in methods:
        geo_rates = []
        for diff in diffs:
            d = d_agg.get((diff, "geometry"), {"k": 0, "n": 0})
            if d["n"] > 0:
                rate, _, _ = wilson_ci(d["k"], d["n"])
                geo_rates.append((diff, rate, d["k"], d["n"]))
        if len(geo_rates) >= 2:
            lines.append("**Difficulty effect on `geometry`**:  ")
            for diff, rate, k, n in geo_rates:
                lines.append(f"- {diff}: {_fmt_pct(rate)} (n={n})")
            # easy vs hard
            if geo_rates[0][0] == "easy" and geo_rates[-1][0] == "hard":
                d_easy = (geo_rates[0][2], geo_rates[0][3])
                d_hard = (geo_rates[-1][2], geo_rates[-1][3])
                p_eh = _fisher_p(*d_easy, *d_hard)
                h_eh = _cohen_h(geo_rates[0][1], geo_rates[-1][1])
                lines.append(f"\nEasy vs Hard: Cohen's h = {h_eh:.3f}, "
                              f"Fisher's p = {p_eh:.4f}  ")
        lines.append("")

    # ── 3. Per-object breakdown ───────────────────────────────────────────────
    lines.append("## 3. Per-Object Success Rate")
    lines.append("")
    header = "| Object | " + " | ".join(f"`{m}`" for m in methods) + " |"
    sep    = "| --- | " + " | ".join(["---"] * len(methods)) + " |"
    lines.extend([header, sep])
    o_agg = agg_by(valid, "object", "method")
    for obj in objects:
        cells = []
        for m in methods:
            d = o_agg.get((obj, m), {"k": 0, "n": 0})
            rate, lo, hi = wilson_ci(d["k"], d["n"])
            cells.append(f"{_fmt_pct(rate)} {_fmt_ci(lo, hi)}")
        lines.append(f"| {obj} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── 4. Contact metrics ────────────────────────────────────────────────────
    lines.append("## 4. Bilateral Contact and Weld Rate")
    lines.append("")
    lines.append("| Method | Bilateral Rate | Weld Rate | Table Contact Rate |")
    lines.append("|--------|---------------|-----------|-------------------|")
    for m in methods:
        recs_m = [r for r in valid if r.get("method") == m]
        bi_k = sum(1 for r in recs_m if r.get("bilateral_contact"))
        bi_n = sum(1 for r in recs_m if r.get("bilateral_contact") is not None)
        weld_k = sum(1 for r in recs_m if r.get("weld_triggered"))
        tc_k   = sum(1 for r in recs_m if r.get("table_contact"))
        bi_rate, _, _  = wilson_ci(bi_k, bi_n) if bi_n else (float("nan"), 0, 0)
        w_rate, _, _   = wilson_ci(weld_k, bi_n) if bi_n else (float("nan"), 0, 0)
        tc_rate, _, _  = wilson_ci(tc_k,   bi_n) if bi_n else (float("nan"), 0, 0)
        lines.append(f"| `{m}` | {_fmt_pct(bi_rate)} | {_fmt_pct(w_rate)} | {_fmt_pct(tc_rate)} |")
    lines.append("")

    # ── 5. Failure analysis ───────────────────────────────────────────────────
    lines.append("## 5. Failure Analysis")
    lines.append("")
    fail_recs = [r for r in valid if not r.get("success")]
    if fail_recs:
        reason_counts: dict = defaultdict(lambda: defaultdict(int))
        for r in fail_recs:
            method = r.get("method", "?")
            reason = r.get("failure_reason") or "unknown"
            reason_counts[method][reason] += 1

        lines.append("| Method | Failure Reason | Count | % of Failures |")
        lines.append("|--------|---------------|-------|--------------|")
        for m in methods:
            m_fail_total = sum(1 for r in fail_recs if r.get("method") == m)
            for reason, count in sorted(reason_counts[m].items(), key=lambda x: -x[1]):
                pct = count / m_fail_total * 100 if m_fail_total else 0
                lines.append(f"| `{m}` | {reason} | {count} | {pct:.1f}% |")
    lines.append("")

    # ── 6. Significance summary ───────────────────────────────────────────────
    if len(methods) == 2:
        lines.append("## 6. Statistical Significance Summary")
        lines.append("")
        lines.append("Fisher's exact test, geometry vs random, per condition:")
        lines.append("")
        lines.append("| Condition | geometry | random | OR | p | sig | Cohen's h |")
        lines.append("|-----------|---------|--------|----|----|-----|----------|")

        # by difficulty
        for diff in diffs:
            d1 = d_agg.get((diff, methods[0]), {"k": 0, "n": 0})
            d2 = d_agg.get((diff, methods[1]), {"k": 0, "n": 0})
            if not d1["n"] or not d2["n"]:
                continue
            r1, _, _ = wilson_ci(d1["k"], d1["n"])
            r2, _, _ = wilson_ci(d2["k"], d2["n"])
            p    = _fisher_p(d1["k"], d1["n"], d2["k"], d2["n"])
            a, b = d1["k"], d1["n"] - d1["k"]
            c, d_v = d2["k"], d2["n"] - d2["k"]
            or_v = (a * d_v / (b * c)) if (b > 0 and c > 0) else float("nan")
            h = _cohen_h(r1, r2)
            sig = "✓" if (not math.isnan(p) and p < 0.05) else "✗"
            or_s = f"{or_v:.2f}" if not math.isnan(or_v) else "—"
            p_s  = f"{p:.4f}"   if not math.isnan(p)    else "—"
            h_s  = f"{h:.3f}"   if not math.isnan(h)    else "—"
            lines.append(f"| {diff} | {_fmt_pct(r1)} | {_fmt_pct(r2)} | "
                          f"{or_s} | {p_s} | {sig} | {h_s} |")

        # overall row
        m0_d = m_agg.get((methods[0],), {"k": 0, "n": 0})
        m1_d = m_agg.get((methods[1],), {"k": 0, "n": 0})
        r0, _, _ = wilson_ci(m0_d["k"], m0_d["n"])
        r1, _, _ = wilson_ci(m1_d["k"], m1_d["n"])
        p    = _fisher_p(m0_d["k"], m0_d["n"], m1_d["k"], m1_d["n"])
        a0, b0 = m0_d["k"], m0_d["n"] - m0_d["k"]
        c0, d0 = m1_d["k"], m1_d["n"] - m1_d["k"]
        or_v = (a0 * d0 / (b0 * c0)) if (b0 > 0 and c0 > 0) else float("nan")
        h = _cohen_h(r0, r1)
        sig = "✓" if (not math.isnan(p) and p < 0.05) else "✗"
        or_s2 = f"{or_v:.2f}" if not math.isnan(or_v) else "—"
        lines.append(f"| **Overall** | **{_fmt_pct(r0)}** | **{_fmt_pct(r1)}** | "
                      f"{or_s2} | {p:.4f} | {sig} | {h:.3f} |")
        lines.append("")

    # ── 7. Notes ──────────────────────────────────────────────────────────────
    lines.append("## 7. Notes")
    lines.append("")
    lines.append("- **Execution**: `physics_weld_after_bilateral` — kinematic weld "
                 "conditioned on bilateral jaw-object contact.  No fake successes.")
    lines.append("- **world_model** checkpoint not available; comparison is "
                 "`geometry` (centroid-based ranking) vs `random` (uniform shuffle).")
    lines.append("- CI: Wilson 95% confidence interval.  Effect sizes: Cohen's h.")
    lines.append("- Scene states saved to `results/<run>/scenes/` for failure replay.")
    lines.append("- Stability-invalid trials (explosion, fell off table, "
                 "still moving) are excluded from all rates.")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",  default=None)
    ap.add_argument("--all",      action="store_true")
    ap.add_argument("--out",      default=None, help="Output path for .md")
    args = ap.parse_args()

    results_root = ROOT / "results"

    if args.all:
        run_dirs = sorted(results_root.glob("diverse_*"))
        all_records = []
        labels = []
        for rd in run_dirs:
            recs = load_trials(rd)
            if recs:
                all_records += recs
                labels.append(rd.name)
        if not all_records:
            print("No trials found in results/diverse_*/")
            return
        md = generate_md(all_records, " + ".join(labels))
        out_path = Path(args.out) if args.out else results_root / "paper_results_summary.md"
    elif args.run_dir:
        rd = Path(args.run_dir)
        records = load_trials(rd)
        md = generate_md(records, rd.name)
        out_path = Path(args.out) if args.out else rd / "paper_results_summary.md"
    else:
        # default: all diverse_* runs combined
        run_dirs = sorted(results_root.glob("diverse_*"))
        all_records = []
        for rd in run_dirs:
            all_records += load_trials(rd)
        if not all_records:
            print("No results found. Run: python scripts/run_diverse_benchmark.py --all")
            return
        md = generate_md(all_records, "diverse_easy+medium+hard")
        out_path = Path(args.out) if args.out else results_root / "paper_results_summary.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"[paper_summary] written → {out_path}")
    # preview first table
    for line in md.split("\n")[:40]:
        print(line)


if __name__ == "__main__":
    main()
