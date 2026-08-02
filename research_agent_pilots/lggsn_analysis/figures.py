#!/usr/bin/env python
"""Phase 4: PNG/PDF figures from Phase 3's already-computed pairwise
comparison records.

Needs matplotlib, which is NOT installed in the research-agent venv used
for `python -m compileall`/`pytest`/`real_analysis.py` (see
experiments/lggsn_statistical_analysis.yaml). Because of that, this module
is:
  - never imported by loader.py/alignment.py/statistics.py/real_analysis.py
    /reporting.py/latex_tables.py, which all stay dependency-free and are
    exercised by the standard `python -m pytest` command;
  - never re-exported from research_agent_pilots/lggsn_analysis/__init__.py
    (importing the package must not require matplotlib);
  - run directly via the `tango` conda env (which does have matplotlib),
    exactly like Phase 2's research_agent_pilots/lggsn_suite/evaluator.py
    regeneration -- and, like that command, deliberately NOT one of this
    spec's allowed_executor_commands, for the same reason: a heavier
    dependency than the restricted harness's own environment provides.
  - tests/test_lggsn_statistical_analysis.py's Phase 4 figures section uses
    `pytest.importorskip("matplotlib")`, so the standard test command still
    passes cleanly without matplotlib installed; those tests only actually
    run when pytest is invoked under an environment that has it.

This module never recomputes a statistic; every value it plots is read
verbatim from research_agent_pilots/lggsn_analysis/outputs/
pairwise_comparisons.json (real_analysis.py's already-tested output).

PDF/PNG metadata is pinned to a fixed value (never datetime.now()) so that
repeated runs in the same environment produce byte-identical files -- see
_FIXED_METADATA_DATE below. This is a same-environment determinism claim
only: matplotlib version/font/backend differences across machines can still
change rendered bytes, which plain string-formatted LaTeX/JSON/CSV output
does not have to worry about.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# matplotlib's PDF backend requires a real datetime object here (a string
# is silently rejected with a UserWarning) -- fixed, never datetime.now(),
# so repeated runs in the same environment embed the same CreationDate.
_FIXED_METADATA_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _comparison_labels(comparisons: Sequence[Mapping[str, Any]]) -> list[str]:
    return [f"{c['checkpoint_a']} vs {c['checkpoint_b']}" for c in comparisons]


def plot_win_tie_loss(comparisons: Sequence[Mapping[str, Any]], png_path: Path, pdf_path: Path) -> None:
    labels = _comparison_labels(comparisons)
    win = [c["win"] for c in comparisons]
    tie = [c["tie"] for c in comparisons]
    loss = [c["loss"] for c in comparisons]
    bottom_loss = [w + t for w, t in zip(win, tie)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(comparisons))
    ax.bar(x, win, label="win (B correct, A wrong)", color="#4C72B0")
    ax.bar(x, tie, bottom=win, label="tie", color="#B0B0B0")
    ax.bar(x, loss, bottom=bottom_loss, label="loss (A correct, B wrong)", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("pair count (of 582 aligned pairs)")
    ax.set_title("LGGSN core-matrix win/tie/loss per comparison")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, metadata={"Software": "research_agent_pilots.lggsn_analysis.figures"})
    fig.savefig(
        pdf_path,
        metadata={"Creator": "research_agent_pilots.lggsn_analysis.figures", "CreationDate": _FIXED_METADATA_DATE},
    )
    plt.close(fig)


def plot_bootstrap_ci_forest(comparisons: Sequence[Mapping[str, Any]], png_path: Path, pdf_path: Path) -> None:
    labels = _comparison_labels(comparisons)
    diffs = [c["accuracy_diff_b_minus_a"] for c in comparisons]
    los = [c["bootstrap"]["ci_lower"] for c in comparisons]
    his = [c["bootstrap"]["ci_upper"] for c in comparisons]
    y = list(range(len(comparisons)))
    err_lower = [d - lo for d, lo in zip(diffs, los)]
    err_upper = [hi - d for hi, d in zip(his, diffs)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(diffs, y, xerr=[err_lower, err_upper], fmt="o", color="#4C72B0", capsize=4)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy difference (B - A); cluster bootstrap 95% CI, unit=query")
    ax.set_title("LGGSN core-matrix accuracy difference with 95% CI")
    fig.tight_layout()

    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, metadata={"Software": "research_agent_pilots.lggsn_analysis.figures"})
    fig.savefig(
        pdf_path,
        metadata={"Creator": "research_agent_pilots.lggsn_analysis.figures", "CreationDate": _FIXED_METADATA_DATE},
    )
    plt.close(fig)


def generate_all_figures(comparisons: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_win_tie_loss(comparisons, output_dir / "win_tie_loss.png", output_dir / "win_tie_loss.pdf")
    plot_bootstrap_ci_forest(
        comparisons, output_dir / "bootstrap_ci_forest.png", output_dir / "bootstrap_ci_forest.pdf"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    comparisons = json.loads(Path(args.comparisons_json).read_text(encoding="utf-8"))["comparisons"]
    generate_all_figures(comparisons, Path(args.output_dir))

    print(f"[lggsn_figures] wrote 2 figures (PNG+PDF each) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
