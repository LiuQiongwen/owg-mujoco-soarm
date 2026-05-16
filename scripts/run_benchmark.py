#!/usr/bin/env python3
"""
OWG Grasp Benchmark — mj-grasp-sim style evaluation.

Runs a clean embodied grasp benchmark across all (object × seed × method)
combinations in the MuJoCo SO-ARM101 simulation.

Usage
-----
# Default: 6 objects × 25 seeds × geometry + lggsn (physics mode)
MUJOCO_GL=egl conda run -n bridge python scripts/run_benchmark.py

# Quick smoke test (5 seeds, 2 objects)
MUJOCO_GL=egl conda run -n bridge python scripts/run_benchmark.py \\
    --objects banana,pear --seeds 1-5

# Specific methods
MUJOCO_GL=egl conda run -n bridge python scripts/run_benchmark.py \\
    --methods geometry,lggsn,random

# Resume interrupted run
MUJOCO_GL=egl conda run -n bridge python scripts/run_benchmark.py \\
    --run-id my_run --resume

# Only plot (from existing results)
MUJOCO_GL=egl conda run -n bridge python scripts/run_benchmark.py \\
    --run-id my_run --plot-only

Output
------
  results/<run_id>/
    trials.jsonl          — per-trial metadata (one JSON per line)
    summary.csv           — aggregated success rates + Wilson CI
    config.yaml           — config snapshot for reproducibility
    benchmark_summary.md  — auto-generated markdown report

  plots/
    success_rate.{pdf,png}
    dz_histogram.{pdf,png}
    per_object_heatmap.{pdf,png}
    overview.{pdf,png}
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")


def parse_seeds(s: str):
    """Parse '1-10' or '1,2,3,4,5' into a list of ints."""
    if "-" in s and "," not in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(",")]


def parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def run(args):
    from benchmark.runner import BenchmarkConfig, BenchmarkRunner, OBJECT_REGISTRY
    from benchmark.methods import build_method
    from benchmark.logger import TrialLogger
    from benchmark.plotter import BenchmarkPlotter
    from benchmark.summarizer import generate_summary

    # ── config ────────────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = BenchmarkConfig.from_yaml(cfg_path)
    else:
        from dataclasses import replace
        cfg = BenchmarkConfig(
            run_id          = "default",
            objects         = list(OBJECT_REGISTRY.keys()),
            seeds           = list(range(1, 26)),
            methods         = ["geometry", "lggsn"],
            execution_mode  = "physics",
            n_grasp_candidates = 10,
            n_grasp_attempts   = 3,
            settle_steps    = 300,
        )

    # ── CLI overrides ─────────────────────────────────────────────────────────
    if args.run_id:
        cfg.run_id = args.run_id
    if args.objects:
        cfg.objects = parse_list(args.objects)
    if args.seeds:
        cfg.seeds = parse_seeds(args.seeds)
    if args.methods:
        cfg.methods = parse_list(args.methods)
    if args.mode:
        cfg.execution_mode = args.mode
    if args.n_candidates:
        cfg.n_grasp_candidates = args.n_candidates
    if args.n_attempts:
        cfg.n_grasp_attempts = args.n_attempts
    if args.results_dir:
        cfg.results_dir = Path(args.results_dir)
    if args.plots_dir:
        cfg.plots_dir = Path(args.plots_dir)
    cfg.verbose = not args.quiet

    # validate objects
    unknown = [o for o in cfg.objects if o not in OBJECT_REGISTRY]
    if unknown:
        print(f"[error] unknown objects: {unknown}")
        print(f"  available: {sorted(OBJECT_REGISTRY)}")
        sys.exit(1)

    # validate methods
    from benchmark.methods import available_methods
    unknown_m = [m for m in cfg.methods if m not in available_methods()]
    if unknown_m:
        print(f"[error] unknown methods: {unknown_m}")
        print(f"  available: {available_methods()}")
        sys.exit(1)

    run_dir = cfg.results_dir / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(cfg.objects) * len(cfg.seeds) * len(cfg.methods)
    print(f"\n[benchmark] {cfg.run_id}")
    print(f"  objects  : {cfg.objects}")
    print(f"  seeds    : {cfg.seeds[0]}–{cfg.seeds[-1]} ({len(cfg.seeds)} seeds)")
    print(f"  methods  : {cfg.methods}")
    print(f"  mode     : {cfg.execution_mode}")
    print(f"  trials   : {n_total}")
    print(f"  results  : {run_dir}")
    print(f"  plots    : {cfg.plots_dir}")

    # ── plot-only shortcut ────────────────────────────────────────────────────
    if args.plot_only:
        _run_plots(run_dir, cfg.plots_dir)
        summary_path = generate_summary(run_dir)
        print(f"[benchmark] summary written: {summary_path}")
        return

    # ── build methods ─────────────────────────────────────────────────────────
    lggsn_ckpt = args.lggsn_ckpt or os.environ.get("LGGSN_CKPT")
    wm_ckpt    = args.wm_ckpt or "data/world_model_ckpt.pt"

    method_kwargs = {
        "lggsn":       {"ckpt_path": lggsn_ckpt, "fallback_on_error": True},
        "world_model": {"ckpt_path": wm_ckpt,    "fallback_on_error": True},
        "hybrid":      {"lggsn_ckpt": lggsn_ckpt, "wm_ckpt": wm_ckpt},
    }
    methods = []
    for m in cfg.methods:
        kw = method_kwargs.get(m, {})
        method = build_method(m, **kw)
        if hasattr(method, "load_error") and method.load_error:
            print(f"  [warn] {m}: {method.load_error}")
        methods.append(method)

    # ── logger ────────────────────────────────────────────────────────────────
    logger = TrialLogger(
        run_dir     = run_dir,
        config_path = cfg_path if cfg_path.exists() else None,
        resume      = args.resume,
    )

    # ── run ───────────────────────────────────────────────────────────────────
    runner = BenchmarkRunner(cfg, methods, logger, resume=args.resume)
    t0 = time.time()
    runner.run()
    elapsed = time.time() - t0
    print(f"\n[benchmark] finished in {elapsed/60:.1f} min")

    # ── plots + summary ───────────────────────────────────────────────────────
    if not args.no_plots:
        _run_plots(run_dir, cfg.plots_dir)

    summary_path = generate_summary(run_dir)
    print(f"[benchmark] summary: {summary_path}")


def _run_plots(run_dir: Path, plots_dir: Path):
    try:
        from benchmark.plotter import BenchmarkPlotter
        plotter = BenchmarkPlotter(run_dir=run_dir, plots_dir=plots_dir)
        saved   = plotter.plot_all()
        print(f"[benchmark] plots saved ({len(saved)} files): {plots_dir}/")
    except ImportError as e:
        print(f"[warn] plotting skipped (missing dependency: {e})")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",
                   default="configs/benchmark/default.yaml",
                   help="Path to benchmark YAML config (default: configs/benchmark/default.yaml)")
    p.add_argument("--run-id",       default="",
                   help="Run identifier used as subdirectory name in results/")
    p.add_argument("--objects",      default="",
                   help="Comma-separated object list (overrides config)")
    p.add_argument("--seeds",        default="",
                   help="Seed range '1-25' or comma list '1,2,3' (overrides config)")
    p.add_argument("--methods",      default="",
                   help="Comma-separated method list: geometry,lggsn,random,world_model,hybrid")
    p.add_argument("--mode",         default="",
                   choices=["", "physics", "demo_attach"],
                   help="Execution mode (default: physics from config)")
    p.add_argument("--n-candidates", type=int, default=0,
                   help="Number of grasp candidates per trial (overrides config)")
    p.add_argument("--n-attempts",   type=int, default=0,
                   help="Max grasp attempts per trial (overrides config)")
    p.add_argument("--results-dir",  default="",
                   help="Override results output directory")
    p.add_argument("--plots-dir",    default="",
                   help="Override plots output directory")
    p.add_argument("--lggsn-ckpt",   default="",
                   help="Path to LGGSN checkpoint (default: env var LGGSN_CKPT)")
    p.add_argument("--wm-ckpt",      default="",
                   help="Path to world model checkpoint")
    p.add_argument("--resume",       action="store_true",
                   help="Skip already-completed (obj, seed, method) triples")
    p.add_argument("--plot-only",    action="store_true",
                   help="Skip execution; only regenerate plots + summary from existing results")
    p.add_argument("--no-plots",     action="store_true",
                   help="Skip plot generation after benchmark completes")
    p.add_argument("--quiet",        action="store_true",
                   help="Suppress per-trial output")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
