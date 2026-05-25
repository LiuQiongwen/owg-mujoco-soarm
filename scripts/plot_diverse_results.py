#!/usr/bin/env python3
"""Generate all benchmark plots without importing MuJoCo.

Can be run from any Python environment with matplotlib installed.
Does NOT require the bridge/owg2 conda environment.

Usage
-----
    conda run -n ai python scripts/plot_diverse_results.py
    conda run -n ai python scripts/plot_diverse_results.py --run-dir results/diverse_medium
    conda run -n ai python scripts/plot_diverse_results.py --all
    conda run -n ai python scripts/plot_diverse_results.py --combined
    conda run -n ai python scripts/plot_diverse_results.py --side-by-side
"""

import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── register a stub `benchmark` package so __init__.py is never executed ──────
import importlib.util, importlib.machinery, math
import matplotlib
matplotlib.use("Agg")

# 1. stub the package itself (prevents __init__.py from running)
_benchmark_pkg = types.ModuleType("benchmark")
_benchmark_pkg.__path__ = [str(ROOT / "benchmark")]
_benchmark_pkg.__package__ = "benchmark"
sys.modules["benchmark"] = _benchmark_pkg

# 2. load logger directly (no env_soarm dependency)
def _load_direct(pkg_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(pkg_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    mod.__package__ = "benchmark"
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod

_logger  = _load_direct("benchmark.logger",  str(ROOT / "benchmark" / "logger.py"))
_plotter = _load_direct("benchmark.plotter", str(ROOT / "benchmark" / "plotter.py"))

BenchmarkPlotter = _plotter.BenchmarkPlotter
wilson_ci        = _logger.wilson_ci


# ── combined plotter (all difficulties in one figure set) ─────────────────────

def plot_combined(run_dirs: list, plots_dir: Path) -> None:
    """Merge trials from multiple runs and produce combined plots."""
    import json, collections
    all_records = []
    for rd in run_dirs:
        p = rd / "trials.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except Exception:
                        pass
    if not all_records:
        print("[plot_combined] no records found")
        return

    # write a temporary jsonl for BenchmarkPlotter to read
    tmp = plots_dir / "_combined_trials.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    plotter = BenchmarkPlotter(run_dir=plots_dir, plots_dir=plots_dir)
    # override the internal _records since we can't change run_dir after init
    plotter._records = all_records
    plotter._methods = plotter._sorted_methods()
    plotter._objects = plotter._sorted_objects()

    saved = plotter.plot_all()
    tmp.unlink(missing_ok=True)
    print(f"[plot_combined] {len(saved)} plots → {plots_dir}")
    for p in saved[:8]:
        print(f"  {Path(p).name}")


# ── side-by-side difficulty comparison ───────────────────────────────────────

def plot_difficulty_comparison(run_dirs_by_diff: dict, plots_dir: Path) -> list:
    """Grouped bar chart: objects on X, one bar-cluster per object coloured by difficulty.

    Each bar is the success rate averaged across methods (geometry + random).
    Error bars are Wilson 95% CI on the pooled (method-averaged) counts.

    Parameters
    ----------
    run_dirs_by_diff : {difficulty_label: Path}  e.g. {"easy": Path(...), ...}
    plots_dir        : output directory
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    DIFF_COLORS = {
        "easy":   "#4878d0",
        "medium": "#ee854a",
        "hard":   "#d65f5f",
    }

    # ── load & aggregate ──────────────────────────────────────────────────────
    # stats[difficulty][object] = (rate, ci_lo, ci_hi, n_valid)
    stats: dict = {}
    all_objects: set = set()

    for diff, run_dir in run_dirs_by_diff.items():
        p = run_dir / "trials.jsonl"
        if not p.exists():
            continue
        records = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        # dedupe by (object, seed, method) – keep last entry
        seen: dict = {}
        for r in records:
            key = (r["object"], r["seed"], r["method"])
            seen[key] = r
        records = list(seen.values())

        # pool across methods per object
        counts: dict = {}   # object → [n_success, n_valid]
        for r in records:
            obj = r["object"]
            if obj not in counts:
                counts[obj] = [0, 0]
            if r.get("stability_valid") is False:
                continue
            counts[obj][1] += 1
            if r.get("success"):
                counts[obj][0] += 1

        stats[diff] = {}
        for obj, (n_ok, n_val) in counts.items():
            if n_val == 0:
                stats[diff][obj] = (0.0, 0.0, 0.0, 0)
            else:
                rate = n_ok / n_val
                _, lo, hi = wilson_ci(n_ok, n_val)
                stats[diff][obj] = (rate, lo, hi, n_val)
            all_objects.add(obj)

    if not stats:
        print("[plot_difficulty_comparison] no data found")
        return []

    # consistent object order: sort, put cylinder last if present
    objects = sorted(o for o in all_objects if o != "cylinder")
    if "cylinder" in all_objects:
        objects.append("cylinder")

    difficulties = [d for d in ("easy", "medium", "hard") if d in stats]
    n_diff   = len(difficulties)
    n_obj    = len(objects)
    bar_w    = 0.7 / n_diff
    x        = np.arange(n_obj)

    fig, ax = plt.subplots(figsize=(max(9, n_obj * 1.8), 5))

    for di, diff in enumerate(difficulties):
        offset = (di - n_diff / 2 + 0.5) * bar_w
        rates, lo_errs, hi_errs, labels = [], [], [], []
        for obj in objects:
            r, lo, hi, n_val = stats[diff].get(obj, (0.0, 0.0, 0.0, 0))
            rates.append(r)
            lo_errs.append(r - lo)
            hi_errs.append(hi - r)
            labels.append(f"{r:.0%}" if n_val > 0 else "–")

        color = DIFF_COLORS.get(diff, "#888888")
        bars = ax.bar(x + offset, rates, bar_w * 0.92,
                      label=diff.capitalize(), color=color, alpha=0.85,
                      zorder=3, linewidth=0)
        ax.errorbar(x + offset, rates,
                    yerr=[lo_errs, hi_errs],
                    fmt="none", color="black", capsize=3, linewidth=1, zorder=4)

        # value labels above bars
        for bar, lbl in zip(bars, labels):
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.025,
                        lbl, ha="center", va="bottom", fontsize=7.5,
                        color="black", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([o.capitalize() for o in objects], fontsize=11)
    ax.set_ylabel("Success rate", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9,
              title="Difficulty", title_fontsize=9)
    ax.set_title("Grasp Success Rate vs. Difficulty — 50 seeds × 5 objects\n"
                 "(bars averaged across geometry + random methods; 95% Wilson CI)",
                 fontsize=11)

    fig.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ("pdf", "png", "svg"):
        out = plots_dir / f"difficulty_comparison.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        saved.append(out)
    plt.close(fig)
    print(f"[difficulty_comparison] saved to {plots_dir}/difficulty_comparison.{{pdf,png,svg}}")
    return saved


# ── single-run plotter ────────────────────────────────────────────────────────

def plot_run(run_dir: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plotter = BenchmarkPlotter(run_dir=run_dir, plots_dir=plots_dir)
    saved   = plotter.plot_all()
    print(f"[{run_dir.name}] {len(saved)} plots → {plots_dir}")
    for p in saved[:6]:
        print(f"  {Path(p).name}")
    if len(saved) > 6:
        print(f"  ... and {len(saved)-6} more")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate diverse benchmark plots (no MuJoCo required)")
    ap.add_argument("--run-dir",  default=None, help="Single run dir to plot")
    ap.add_argument("--all",      action="store_true", help="Plot all results/diverse_* dirs")
    ap.add_argument("--combined", action="store_true",
                    help="Also generate combined cross-difficulty plots")
    ap.add_argument("--side-by-side", action="store_true",
                    help="Generate difficulty comparison bar chart (easy/medium/hard side by side)")
    args = ap.parse_args()

    results_root = ROOT / "results"
    plots_root   = ROOT / "plots"

    if args.side_by_side:
        run_dirs_by_diff = {
            d: results_root / f"diverse_{d}"
            for d in ("easy", "medium", "hard")
            if (results_root / f"diverse_{d}" / "trials.jsonl").exists()
        }
        if not run_dirs_by_diff:
            print("No results/diverse_{easy,medium,hard} directories found.")
            return
        plot_difficulty_comparison(run_dirs_by_diff, plots_root / "combined")
        return

    if args.all or (not args.run_dir):
        run_dirs = sorted(results_root.glob("diverse_*"))
        if not run_dirs:
            print("No results/diverse_* directories found.")
            return
        for rd in run_dirs:
            if (rd / "trials.jsonl").exists():
                plot_run(rd, plots_root / rd.name)
        if args.combined or not args.run_dir:
            plot_combined(run_dirs, plots_root / "combined")
    elif args.run_dir:
        rd = Path(args.run_dir)
        plot_run(rd, plots_root / rd.name)


if __name__ == "__main__":
    main()
