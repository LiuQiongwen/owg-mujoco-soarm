"""
Score-spread uncertainty gating ablation.

Runs a paired evaluation: for each (seed, object) pair, Stage 3 and three
Stage 4 gate-delta conditions execute on the same MuJoCo scene.

Conditions:
  s3          — Stage 3 baseline (no LGGSN)
  gate_0.0    — Stage 4, no gate (delta=0.0)
  gate_0.1    — Stage 4, skip reranking if spread < 0.1
  gate_0.2    — Stage 4, skip reranking if spread < 0.2

Usage:
  conda run -n owg-mujoco python scripts/run_gate_ablation.py \
      [--seeds 1-25] [--timeout 90] [--out results/gate_uncertainty_ablation.txt]
"""

import argparse
import subprocess
import sys
import os
from collections import defaultdict

OBJECTS = ["Banana", "TomatoSoupCan", "Pear", "MustardBottle",
           "Scissors", "CrackerBox", "PowerDrill"]

CONDITIONS = [
    ("s3",       3, "0.0"),
    ("gate_0.0", 4, "0.0"),
    ("gate_0.1", 4, "0.1"),
    ("gate_0.2", 4, "0.2"),
]


def run_trial(stage: int, obj: str, seed: int, gate_delta: str, timeout: int) -> str:
    """Returns 'success', 'fail', or 'timeout'."""
    env = os.environ.copy()
    env["OWG_GATE_DELTA"] = gate_delta
    env["MUJOCO_GL"] = "egl"
    cmd = [
        "conda", "run", "-n", "owg-mujoco", "--no-capture-output",
        "python", "demo.py",
        "--stage", str(stage),
        "--prompt", obj,
        "--seed", str(seed),
        "--once",
        "--verbose", "0",
        "--gate-delta", gate_delta,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        out = result.stdout + result.stderr
        if "Done pick" in out:
            return "success"
        return "fail"
    except subprocess.TimeoutExpired:
        return "timeout"


def parse_seeds(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1-25")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--out", default="results/gate_uncertainty_ablation.txt")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    n_seeds = len(seeds)
    total_trials = len(CONDITIONS) * len(OBJECTS) * n_seeds

    # results[condition][object] = list of bools
    results = {cond: defaultdict(list) for cond, _, _ in CONDITIONS}

    done = 0
    print(f"Running {total_trials} trials "
          f"({len(CONDITIONS)} conds × {len(OBJECTS)} obj × {n_seeds} seeds) …\n")

    for seed in seeds:
        for obj in OBJECTS:
            for cond, stage, gate in CONDITIONS:
                outcome = run_trial(stage, obj, seed, gate, args.timeout)
                ok = outcome == "success"
                results[cond][obj].append(ok)
                done += 1
                sym = "✓" if ok else ("T" if outcome == "timeout" else "✗")
                print(f"  [{sym}] {cond:<10} {obj:<16} seed={seed:2d}  ({done}/{total_trials})",
                      flush=True)

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("Score-spread uncertainty gating ablation")
    lines.append(f"Seeds: {seeds[0]}–{seeds[-1]}  ({n_seeds} seeds × {len(OBJECTS)} objects)")
    lines.append("=" * 70)

    cond_names = [c for c, _, _ in CONDITIONS]
    col_w = 12

    # Header
    hdr = f"{'Object':<16}" + "".join(f"{c:>{col_w}}" for c in cond_names)
    lines.append(hdr)
    lines.append("-" * len(hdr))

    # Per-object rows: show "S4 / S3" and net for each S4 condition
    s3_totals = {}
    gate_totals = {c: 0 for c in cond_names if c != "s3"}

    for obj in OBJECTS:
        s3 = sum(results["s3"][obj])
        s3_totals[obj] = s3
        cells = [f"{s3}/{n_seeds}"]
        for cond in cond_names[1:]:
            s4 = sum(results[cond][obj])
            net = s4 - s3
            sign = "+" if net >= 0 else ""
            cells.append(f"{s4}/{n_seeds}({sign}{net})")
            gate_totals[cond] += s4
        row = f"  {obj:<14}" + "".join(f"{c:>{col_w}}" for c in cells)
        lines.append(row)

    lines.append("-" * len(hdr))

    # Totals row
    s3_tot = sum(s3_totals.values())
    total_cells = [f"{s3_tot}/{n_seeds*len(OBJECTS)}"]
    for cond in cond_names[1:]:
        s4_tot = gate_totals[cond]
        net_tot = s4_tot - s3_tot
        sign = "+" if net_tot >= 0 else ""
        total_cells.append(f"{s4_tot}/{n_seeds*len(OBJECTS)}({sign}{net_tot})")
    lines.append(f"  {'TOTAL':<14}" + "".join(f"{c:>{col_w}}" for c in total_cells))

    lines.append("=" * 70)
    lines.append("")
    lines.append("Net improvement = S4(condition) − S3 successes (paired per seed/object).")
    lines.append("gate_0.0 = no gating (baseline Stage 4).")
    lines.append("gate_0.1 = skip LGGSN reranking when score spread < 0.1.")
    lines.append("gate_0.2 = skip LGGSN reranking when score spread < 0.2.")

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
