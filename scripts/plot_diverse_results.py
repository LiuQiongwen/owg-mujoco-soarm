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
    args = ap.parse_args()

    results_root = ROOT / "results"
    plots_root   = ROOT / "plots"

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
