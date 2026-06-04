"""
Parse quick_eval.sh output files from gate ablation runs and produce
a comparison table.

Usage:
    python scripts/parse_gate_results.py \
        --s3    results/gate_s3.txt \
        --g0    results/gate_nogate.txt \
        --g01   results/gate_01.txt \
        --g02   results/gate_02.txt \
        --out   results/gate_uncertainty_ablation.txt
"""

import argparse
import re
from collections import defaultdict

OBJECTS = ["Banana", "TomatoSoupCan", "Pear", "MustardBottle",
           "Scissors", "CrackerBox", "PowerDrill"]


def parse_file(path: str):
    """
    Returns:
        per_obj_ok   : dict[obj] -> int  (successes)
        per_obj_tot  : dict[obj] -> int  (trials)
        per_obj_gate : dict[obj] -> int  (gate fires; 0 for S3)
        total_seeds  : int
    """
    per_obj_ok   = defaultdict(int)
    per_obj_tot  = defaultdict(int)
    per_obj_gate = defaultdict(int)

    # Match trial result lines: [✓]/[✗]/[T] + object name + seed=N
    trial_re  = re.compile(r'\[(✓|✗|T)\]\s+(\w+)\s+seed=(\d+)')
    # Match summary line per object: --- ObjName: S/N  gate_fired=G/N
    summary_re = re.compile(r'---\s+(\w+):\s+(\d+)/(\d+)\s+gate_fired=(\d+)/(\d+)')

    with open(path) as f:
        for line in f:
            m = trial_re.search(line)
            if m:
                sym, obj, _ = m.group(1), m.group(2), m.group(3)
                per_obj_tot[obj] += 1
                if sym == '✓':
                    per_obj_ok[obj] += 1
                continue
            m = summary_re.search(line)
            if m:
                obj = m.group(1)
                per_obj_gate[obj] = int(m.group(4))

    # infer total seeds from any object
    total_seeds = max(per_obj_tot.values()) if per_obj_tot else 25
    return per_obj_ok, per_obj_tot, per_obj_gate, total_seeds


def format_net(n: int) -> str:
    return f"+{n}" if n >= 0 else str(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3",  required=True)
    parser.add_argument("--g0",  required=True, help="gate-delta=0.0 (no gate)")
    parser.add_argument("--g01", required=True, help="gate-delta=0.1")
    parser.add_argument("--g02", required=True, help="gate-delta=0.2")
    parser.add_argument("--out", default="results/gate_uncertainty_ablation.txt")
    args = parser.parse_args()

    s3_ok,   s3_tot,   _,       n_seeds = parse_file(args.s3)
    g0_ok,   g0_tot,   g0_gate, _       = parse_file(args.g0)
    g01_ok,  g01_tot,  g01_gate, _      = parse_file(args.g01)
    g02_ok,  g02_tot,  g02_gate, _      = parse_file(args.g02)

    lines = []
    lines.append("=" * 85)
    lines.append("Score-spread uncertainty gating ablation  (Stage 4, v5d checkpoint)")
    lines.append(f"{n_seeds} seeds × {len(OBJECTS)} objects = {n_seeds*len(OBJECTS)} trials per condition")
    lines.append("=" * 85)
    lines.append(
        f"{'Object':<16} {'S3':>5} {'S4(Δ=0)':>8} {'S4(Δ=.1)':>9} {'S4(Δ=.2)':>9}"
        f" {'net(0)':>7} {'net(.1)':>7} {'net(.2)':>7}"
        f" {'gate%(.1)':>10} {'gate%(.2)':>10}"
    )
    lines.append("-" * 85)

    tot_s3 = tot_g0 = tot_g01 = tot_g02 = 0
    tot_gate01 = tot_gate02 = 0

    for obj in OBJECTS:
        n  = n_seeds
        s3 = s3_ok.get(obj, 0)
        g0 = g0_ok.get(obj, 0)
        g1 = g01_ok.get(obj, 0)
        g2 = g02_ok.get(obj, 0)
        gf1 = g01_gate.get(obj, 0)
        gf2 = g02_gate.get(obj, 0)
        gp1 = f"{100*gf1/n:.0f}%" if n else "—"
        gp2 = f"{100*gf2/n:.0f}%" if n else "—"

        tot_s3 += s3; tot_g0 += g0; tot_g01 += g1; tot_g02 += g2
        tot_gate01 += gf1; tot_gate02 += gf2

        lines.append(
            f"  {obj:<14} {s3:>3}/{n} {g0:>4}/{n}   {g1:>4}/{n}   {g2:>4}/{n}"
            f"  {format_net(g0-s3):>6}  {format_net(g1-s3):>6}  {format_net(g2-s3):>6}"
            f"  {gp1:>9}  {gp2:>9}"
        )

    lines.append("-" * 85)
    N = n_seeds * len(OBJECTS)
    gp1t = f"{100*tot_gate01/N:.0f}%" if N else "—"
    gp2t = f"{100*tot_gate02/N:.0f}%" if N else "—"
    lines.append(
        f"  {'TOTAL':<14} {tot_s3:>3}/{N} {tot_g0:>4}/{N}   {tot_g01:>4}/{N}   {tot_g02:>4}/{N}"
        f"  {format_net(tot_g0-tot_s3):>6}  {format_net(tot_g01-tot_s3):>6}  {format_net(tot_g02-tot_s3):>6}"
        f"  {gp1t:>9}  {gp2t:>9}"
    )
    lines.append("=" * 85)
    lines.append("")
    lines.append("Columns:")
    lines.append("  S3        — Stage 3 baseline (GR-ConvNet top-1, no LGGSN)")
    lines.append("  S4(Δ=0)   — Stage 4, no score-spread gate")
    lines.append("  S4(Δ=.1)  — Stage 4, skip LGGSN when max−min score < 0.1")
    lines.append("  S4(Δ=.2)  — Stage 4, skip LGGSN when max−min score < 0.2")
    lines.append("  net(*)    — S4(condition) − S3 (positive = LGGSN helped)")
    lines.append("  gate%(*)  — fraction of episodes where gate fired (fell back to S3 order)")

    report = "\n".join(lines)
    print(report)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
