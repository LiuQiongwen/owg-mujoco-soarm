"""Publication-quality benchmark plots.

Generates four figures saved to the plots/ directory:

  success_rate.pdf      — grouped bar chart: method × object, Wilson 95% CI error bars
  dz_histogram.pdf      — per-method dz distribution (successful grasps only)
  per_object_heatmap.pdf— success rate colour matrix (method rows, object cols)
  overview.pdf          — 2×2 grid combining the three main views

All plots use a consistent colour palette and are saved in both PDF (for
paper inclusion) and PNG (for README / quick preview).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmark.logger import wilson_ci


# ── colour palette (colour-blind friendly) ────────────────────────────────────
_METHOD_COLORS = {
    "random":      "#aaaaaa",
    "geometry":    "#4878d0",
    "lggsn":       "#ee854a",
    "world_model": "#6acc65",
    "hybrid":      "#d65f5f",
}
_OBJECT_COLORS = {
    "banana":   "#ffe135",
    "pear":     "#a8d8a8",
    "mustard":  "#f4a261",
    "cracker":  "#e76f51",
    "drill":    "#457b9d",
    "cylinder": "#a8c5da",
}
_DEFAULT_COLOR = "#888888"


class BenchmarkPlotter:
    """
    Load benchmark results and produce publication-quality figures.

    Parameters
    ----------
    run_dir  : Path to the run directory (contains trials.jsonl)
    plots_dir: Where to save output figures
    dpi      : Resolution for PNG output
    """

    def __init__(
        self,
        run_dir:   Path | str,
        plots_dir: Path | str = Path("plots"),
        dpi:       int = 150,
    ):
        self.run_dir   = Path(run_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        self._records = self._load()
        self._methods = self._sorted_methods()
        self._objects = self._sorted_objects()

    # ── public ────────────────────────────────────────────────────────────────

    def plot_all(self) -> List[Path]:
        """Generate all figures.  Returns list of saved file paths."""
        saved = []
        saved += self.plot_success_rate()
        saved += self.plot_dz_histogram()
        saved += self.plot_per_object_heatmap()
        saved += self.plot_overview()
        return saved

    def plot_success_rate(self) -> List[Path]:
        """Grouped bar chart: success rate ± Wilson 95% CI per method."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        stats = self._aggregate()   # {method: {object: (rate, lo, hi, n)}}
        methods = self._methods
        objects = self._objects + ["overall"]

        n_methods = len(methods)
        n_objects = len(objects)
        x = np.arange(n_objects)
        bar_w = 0.8 / max(n_methods, 1)

        fig, ax = plt.subplots(figsize=(max(8, n_objects * 1.4), 5))

        for mi, method in enumerate(methods):
            offset = (mi - n_methods / 2 + 0.5) * bar_w
            rates, lo_errs, hi_errs = [], [], []
            for obj in objects:
                r, lo, hi = stats.get(method, {}).get(obj, (0.0, 0.0, 0.0))[:3]
                rates.append(r)
                lo_errs.append(r - lo)
                hi_errs.append(hi - r)

            color = _METHOD_COLORS.get(method, _DEFAULT_COLOR)
            ax.bar(x + offset, rates, bar_w * 0.9,
                   label=method, color=color, alpha=0.85, zorder=3)
            ax.errorbar(x + offset, rates,
                        yerr=[lo_errs, hi_errs],
                        fmt="none", color="black", capsize=3, linewidth=1, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(objects, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Success rate", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_title("Grasp Success Rate by Method (95% Wilson CI)", fontsize=12)

        fig.tight_layout()
        paths = self._save(fig, "success_rate")
        plt.close(fig)
        return paths

    def plot_dz_histogram(self) -> List[Path]:
        """Per-method distribution of dz for successful grasps."""
        import matplotlib.pyplot as plt

        dz_by_method: Dict[str, list] = defaultdict(list)
        for rec in self._records:
            if rec.get("success") and rec.get("dz") is not None:
                dz_by_method[rec["method"]].append(rec["dz"])

        if not any(dz_by_method.values()):
            return []

        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(-0.05, 0.35, 25)

        for method in self._methods:
            vals = dz_by_method.get(method, [])
            if not vals:
                continue
            color = _METHOD_COLORS.get(method, _DEFAULT_COLOR)
            ax.hist(vals, bins=bins, alpha=0.55, color=color,
                    label=f"{method} (n={len(vals)})", edgecolor="none")

        ax.set_xlabel("dz (m) — object height gain after lift", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title("dz Distribution for Successful Grasps", fontsize=12)

        fig.tight_layout()
        paths = self._save(fig, "dz_histogram")
        plt.close(fig)
        return paths

    def plot_per_object_heatmap(self) -> List[Path]:
        """Success rate colour matrix: methods (rows) × objects (cols)."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        stats = self._aggregate()
        methods = self._methods
        objects = self._objects

        matrix = np.full((len(methods), len(objects)), np.nan)
        for mi, m in enumerate(methods):
            for oi, o in enumerate(objects):
                entry = stats.get(m, {}).get(o)
                if entry:
                    matrix[mi, oi] = entry[0]   # point estimate

        fig, ax = plt.subplots(figsize=(max(6, len(objects) * 1.2), max(3, len(methods) * 0.9)))
        cmap = plt.cm.RdYlGn
        cmap.set_bad("lightgrey")

        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(objects)))
        ax.set_xticklabels(objects, rotation=25, ha="right", fontsize=10)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=10)

        for mi in range(len(methods)):
            for oi in range(len(objects)):
                v = matrix[mi, oi]
                if np.isnan(v):
                    continue
                text_color = "white" if v < 0.35 or v > 0.75 else "black"
                ax.text(oi, mi, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                     format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title("Success Rate per Object × Method", fontsize=12)
        fig.tight_layout()
        paths = self._save(fig, "per_object_heatmap")
        plt.close(fig)
        return paths

    def plot_overview(self) -> List[Path]:
        """2×2 overview grid combining success rate, dz histogram, heatmap."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        stats = self._aggregate()
        methods = self._methods
        objects = self._objects

        fig = plt.figure(figsize=(14, 10))
        gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # ── top-left: overall success rate bar ───────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        overall_rates = []
        overall_cis   = []
        colors = []
        for m in methods:
            entry = stats.get(m, {}).get("overall", (0, 0, 0, 0))
            rate, lo, hi = entry[:3]
            overall_rates.append(rate)
            overall_cis.append([rate - lo, hi - rate])
            colors.append(_METHOD_COLORS.get(m, _DEFAULT_COLOR))
        xs = np.arange(len(methods))
        ax1.bar(xs, overall_rates, color=colors, alpha=0.85, zorder=3)
        lo_e = [v[0] for v in overall_cis]
        hi_e = [v[1] for v in overall_cis]
        ax1.errorbar(xs, overall_rates, yerr=[lo_e, hi_e],
                     fmt="none", color="black", capsize=4, zorder=4)
        ax1.set_xticks(xs)
        ax1.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
        ax1.set_ylabel("Success rate")
        ax1.set_ylim(0, 1.05)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax1.set_title("Overall Success Rate")

        # ── top-right: dz histogram ───────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        dz_by_method: Dict[str, list] = defaultdict(list)
        for rec in self._records:
            if rec.get("success") and rec.get("dz") is not None:
                dz_by_method[rec["method"]].append(rec["dz"])
        bins = np.linspace(-0.05, 0.35, 20)
        for m in methods:
            vals = dz_by_method.get(m, [])
            if vals:
                ax2.hist(vals, bins=bins, alpha=0.55,
                         color=_METHOD_COLORS.get(m, _DEFAULT_COLOR),
                         label=m, edgecolor="none")
        ax2.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.set_xlabel("dz (m)")
        ax2.set_ylabel("Count")
        ax2.legend(fontsize=8)
        ax2.set_title("dz Distribution (Successes)")

        # ── bottom: per-object heatmap ────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, :])
        matrix = np.full((len(methods), len(objects)), np.nan)
        for mi, m in enumerate(methods):
            for oi, o in enumerate(objects):
                e = stats.get(m, {}).get(o)
                if e:
                    matrix[mi, oi] = e[0]
        masked = np.ma.masked_invalid(matrix)
        im = ax3.imshow(masked, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect="auto")
        ax3.set_xticks(range(len(objects)))
        ax3.set_xticklabels(objects, rotation=15, ha="right", fontsize=9)
        ax3.set_yticks(range(len(methods)))
        ax3.set_yticklabels(methods, fontsize=9)
        for mi in range(len(methods)):
            for oi in range(len(objects)):
                v = matrix[mi, oi]
                if not np.isnan(v):
                    tc = "white" if v < 0.35 or v > 0.75 else "black"
                    ax3.text(oi, mi, f"{v:.0%}", ha="center", va="center",
                             fontsize=9, color=tc, fontweight="bold")
        plt.colorbar(im, ax=ax3, fraction=0.015, pad=0.02,
                     format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax3.set_title("Per-Object Success Rate")

        fig.suptitle("OWG Grasp Benchmark — Method Comparison", fontsize=13, y=1.01)
        paths = self._save(fig, "overview")
        plt.close(fig)
        return paths

    # ── internals ─────────────────────────────────────────────────────────────

    def _load(self) -> List[dict]:
        path = self.run_dir / "trials.jsonl"
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

    def _sorted_methods(self) -> List[str]:
        preferred = ["random", "geometry", "lggsn", "world_model", "hybrid"]
        seen = {r["method"] for r in self._records}
        ordered = [m for m in preferred if m in seen]
        ordered += sorted(seen - set(ordered))
        return ordered

    def _sorted_objects(self) -> List[str]:
        preferred = ["banana", "pear", "mustard", "cracker", "drill", "cylinder"]
        seen = {r["object"] for r in self._records}
        ordered = [o for o in preferred if o in seen]
        ordered += sorted(seen - set(ordered))
        return ordered

    def _aggregate(self) -> Dict[str, Dict[str, tuple]]:
        """Compute (rate, ci_lo, ci_hi, n_valid) per (method, object) + 'overall'."""
        counts: Dict[str, Dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"k": 0, "n": 0})
        )
        for rec in self._records:
            if not rec.get("stability_valid"):
                continue
            m = rec["method"]
            o = rec["object"]
            counts[m][o]["n"] += 1
            counts[m]["overall"]["n"] += 1
            if rec.get("success"):
                counts[m][o]["k"] += 1
                counts[m]["overall"]["k"] += 1

        result = {}
        for m, objs in counts.items():
            result[m] = {}
            for o, d in objs.items():
                rate, lo, hi = wilson_ci(d["k"], d["n"])
                result[m][o] = (rate, lo, hi, d["n"])
        return result

    def _save(self, fig, name: str) -> List[Path]:
        paths = []
        for ext in ("pdf", "png"):
            p = self.plots_dir / f"{name}.{ext}"
            fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
            paths.append(p)
        return paths
