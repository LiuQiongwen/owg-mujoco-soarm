#!/usr/bin/env python3
"""Run the diverse MuJoCo grasp benchmark for SO-ARM101.

Supports three difficulty modes (easy / medium / hard) with randomised object
yaw, XY spawn positions, and optional clutter objects.  Uses
physics_weld_after_bilateral execution — no fake successes.

Usage
-----
Single difficulty:
    MUJOCO_GL=egl conda run -n owg2 python scripts/run_diverse_benchmark.py \\
        --config configs/benchmark/diverse_medium.yaml

All three difficulties in sequence:
    MUJOCO_GL=egl conda run -n owg2 python scripts/run_diverse_benchmark.py --all

Override any config field:
    MUJOCO_GL=egl conda run -n owg2 python scripts/run_diverse_benchmark.py \\
        --config configs/benchmark/diverse_hard.yaml \\
        --seeds 1-10 --methods geometry,random --n-trials 10

Plot only (no new trials):
    conda run -n owg2 python scripts/run_diverse_benchmark.py \\
        --plot-only --run-dir results/diverse_medium
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from benchmark.diverse_runner import DiverseBenchmarkConfig, DiverseBenchmarkRunner
from benchmark.logger import TrialLogger
from benchmark.methods import build_method
from benchmark.plotter import BenchmarkPlotter


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Diverse MuJoCo grasp benchmark for SO-ARM101",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=None,
                    help="Path to YAML config  (default: diverse_medium.yaml)")
    ap.add_argument("--all", action="store_true",
                    help="Run easy + medium + hard in sequence")
    ap.add_argument("--difficulty",
                    choices=["easy", "medium", "hard"], default=None,
                    help="Override the difficulty field in the config")
    ap.add_argument("--seeds", default=None,
                    help="Seed range or list  (e.g. '1-10' or '1,3,5')")
    ap.add_argument("--methods", default=None,
                    help="Comma-separated method names  (e.g. 'geometry,random')")
    ap.add_argument("--objects", default=None,
                    help="Comma-separated object short names  (e.g. 'banana,pear')")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="Number of seeds to use (overrides seeds list)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-logged (obj, seed, method) triples")
    ap.add_argument("--plot-only", action="store_true",
                    help="Load existing trials.jsonl and regenerate plots only")
    ap.add_argument("--run-dir", default=None,
                    help="Run dir for --plot-only mode")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip plot generation after run")
    return ap.parse_args()


def _parse_seeds(seed_str: str):
    if "-" in seed_str and not seed_str.startswith("-"):
        lo, hi = seed_str.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in seed_str.split(",")]


# ── single run ────────────────────────────────────────────────────────────────

def _run_one(config_path: str, args: argparse.Namespace) -> Path:
    cfg = DiverseBenchmarkConfig.from_yaml(config_path)

    if args.difficulty:
        cfg.difficulty = args.difficulty
    if args.seeds:
        cfg.seeds = _parse_seeds(args.seeds)
    if args.n_trials:
        cfg.seeds = cfg.seeds[: args.n_trials]
    if args.methods:
        cfg.methods = [m.strip() for m in args.methods.split(",")]
    if args.objects:
        cfg.objects = [o.strip() for o in args.objects.split(",")]

    run_dir = cfg.results_dir / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[diverse_bench] difficulty={cfg.difficulty}  run_id={cfg.run_id}")
    print(f"  objects  : {cfg.objects}")
    print(f"  seeds    : {cfg.seeds[:5]}{'...' if len(cfg.seeds) > 5 else ''}")
    print(f"  methods  : {cfg.methods}")
    print(f"  run_dir  : {run_dir}")
    n_total = len(cfg.objects) * len(cfg.seeds) * len(cfg.methods)
    print(f"  trials   : {n_total}")

    methods = [build_method(m) for m in cfg.methods]
    logger  = TrialLogger(run_dir, config_path=config_path, resume=args.resume)
    runner  = DiverseBenchmarkRunner(cfg, methods, logger, resume=args.resume)
    runner.run()

    return run_dir


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot(run_dir: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plotter = BenchmarkPlotter(run_dir=run_dir, plots_dir=plots_dir)
    saved   = plotter.plot_all()
    if saved:
        print(f"\n[diverse_bench] {len(saved)} plot files written to {plots_dir}")
        for p in saved[:6]:
            print(f"  {p}")
        if len(saved) > 6:
            print(f"  ... and {len(saved)-6} more")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── plot-only mode ────────────────────────────────────────────────────────
    if args.plot_only:
        run_dir = Path(args.run_dir) if args.run_dir else Path("results/diverse_medium")
        _plot(run_dir, run_dir.parent / "plots" / run_dir.name)
        return

    # ── run all three difficulties ────────────────────────────────────────────
    if args.all:
        configs = [
            ROOT / "configs" / "benchmark" / "diverse_easy.yaml",
            ROOT / "configs" / "benchmark" / "diverse_medium.yaml",
            ROOT / "configs" / "benchmark" / "diverse_hard.yaml",
        ]
        run_dirs = []
        for c in configs:
            rd = _run_one(str(c), args)
            run_dirs.append(rd)
            if not args.no_plots:
                _plot(rd, rd.parent / "plots" / rd.name)
        print(f"\n[diverse_bench] all difficulties complete.")
        print(f"  Run dirs: {[str(r) for r in run_dirs]}")
        return

    # ── single config ─────────────────────────────────────────────────────────
    config_path = args.config or str(
        ROOT / "configs" / "benchmark" / "diverse_medium.yaml"
    )
    run_dir = _run_one(config_path, args)

    if not args.no_plots:
        cfg_tmp = DiverseBenchmarkConfig.from_yaml(config_path)
        plots_dir = cfg_tmp.plots_dir / cfg_tmp.run_id
        _plot(run_dir, plots_dir)


if __name__ == "__main__":
    main()
