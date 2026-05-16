"""Auto-generate benchmark_summary.md from trial results.

The summary is written to <run_dir>/benchmark_summary.md and includes:
  - Run configuration
  - Overall results table (method, n, success_rate, 95% Wilson CI)
  - Per-object breakdown table
  - Stability / validity statistics
  - Method comparison notes
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from benchmark.logger import wilson_ci


def generate_summary(run_dir: Path | str, out_path: Optional[Path | str] = None) -> Path:
    """
    Read trials.jsonl from run_dir and write benchmark_summary.md.

    Parameters
    ----------
    run_dir  : directory containing trials.jsonl (and optionally config.yaml)
    out_path : where to write the markdown (default: run_dir/benchmark_summary.md)

    Returns the path to the written file.
    """
    run_dir  = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "benchmark_summary.md"

    records = _load(run_dir / "trials.jsonl")
    config  = _load_config(run_dir / "config.yaml")

    methods = _sorted_methods(records)
    objects = _sorted_objects(records)
    overall = _aggregate(records, methods, objects)
    skipped = _count_skipped(records)

    lines: List[str] = []

    # ── header ────────────────────────────────────────────────────────────────
    lines += [
        "# OWG Grasp Benchmark Summary",
        "",
        f"**Run directory**: `{run_dir}`  ",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total trials**: {len(records)}  ",
        f"**Skipped (unstable scene)**: {skipped}  ",
        "",
    ]

    # ── configuration block ───────────────────────────────────────────────────
    if config:
        lines += ["## Configuration", "", "```yaml"]
        for k, v in config.items():
            lines.append(f"{k}: {v}")
        lines += ["```", ""]

    # ── overall results table ──────────────────────────────────────────────────
    lines += [
        "## Overall Results",
        "",
        "| Method | n valid | n success | Success Rate | 95% CI |",
        "|--------|---------|-----------|-------------|--------|",
    ]
    for m in methods:
        d    = overall.get(m, {}).get("__overall__", {"k": 0, "n": 0})
        k, n = d["k"], d["n"]
        rate, lo, hi = wilson_ci(k, n)
        lines.append(
            f"| `{m}` | {n} | {k} | {rate:.1%} | [{lo:.1%}, {hi:.1%}] |"
        )
    lines.append("")

    # ── per-object breakdown ───────────────────────────────────────────────────
    if objects:
        lines += [
            "## Per-Object Breakdown",
            "",
        ]
        # header
        hdr = "| Method |"
        sep = "|--------|"
        for o in objects:
            hdr += f" {o} |"
            sep += "--------|"
        lines += [hdr, sep]

        for m in methods:
            row = f"| `{m}` |"
            for o in objects:
                d    = overall.get(m, {}).get(o, {"k": 0, "n": 0})
                k, n = d["k"], d["n"]
                if n == 0:
                    row += " — |"
                else:
                    rate, lo, hi = wilson_ci(k, n)
                    row += f" {rate:.0%} ({k}/{n}) |"
            lines.append(row)
        lines.append("")

    # ── stability statistics ───────────────────────────────────────────────────
    valid_count   = sum(1 for r in records if r.get("stability_valid"))
    invalid_count = len(records) - valid_count
    skip_reasons: dict = defaultdict(int)
    for r in records:
        if not r.get("stability_valid") and r.get("skip_reason"):
            reason = r["skip_reason"].split("=")[0]   # strip numeric suffix
            skip_reasons[reason] += 1

    lines += [
        "## Stability Filtering",
        "",
        f"- Valid scenes: **{valid_count}** / {len(records)}",
        f"- Skipped scenes: **{invalid_count}**",
    ]
    for reason, count in sorted(skip_reasons.items()):
        lines.append(f"  - `{reason}`: {count}")
    lines.append("")

    # ── failure analysis ───────────────────────────────────────────────────────
    failure_by_method: Dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r.get("stability_valid") and not r.get("success"):
            fr = r.get("failure_reason", "unknown")
            failure_by_method[r["method"]][fr] += 1

    if any(failure_by_method.values()):
        lines += ["## Failure Analysis", ""]
        for m in methods:
            fb = failure_by_method.get(m)
            if fb:
                lines.append(f"**{m}**")
                for reason, count in sorted(fb.items(), key=lambda x: -x[1]):
                    lines.append(f"  - `{reason}`: {count}")
        lines.append("")

    # ── dz stats ──────────────────────────────────────────────────────────────
    lines += ["## dz Statistics (Successful Grasps)", ""]
    lines += ["| Method | n | mean dz (m) | std dz (m) |", "|--------|---|------------|-----------|"]
    for m in methods:
        dzs = [r["dz"] for r in records if r.get("method") == m
               and r.get("success") and r.get("dz") is not None]
        if dzs:
            lines.append(
                f"| `{m}` | {len(dzs)} | {_mean(dzs):.4f} | {_std(dzs):.4f} |"
            )
        else:
            lines.append(f"| `{m}` | 0 | — | — |")
    lines.append("")

    # ── footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "_Generated by `benchmark/summarizer.py`_",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> List[dict]:
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


def _load_config(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _sorted_methods(records: List[dict]) -> List[str]:
    preferred = ["random", "geometry", "lggsn", "world_model", "hybrid"]
    seen = {r.get("method", "") for r in records}
    return [m for m in preferred if m in seen] + sorted(seen - set(preferred))


def _sorted_objects(records: List[dict]) -> List[str]:
    preferred = ["banana", "pear", "mustard", "cracker", "drill", "cylinder"]
    seen = {r.get("object", "") for r in records}
    return [o for o in preferred if o in seen] + sorted(seen - set(preferred))


def _aggregate(records, methods, objects) -> Dict[str, Dict[str, dict]]:
    counts: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"k": 0, "n": 0})
    )
    for r in records:
        if not r.get("stability_valid"):
            continue
        m = r.get("method", "")
        o = r.get("object", "")
        counts[m][o]["n"]          += 1
        counts[m]["__overall__"]["n"] += 1
        if r.get("success"):
            counts[m][o]["k"]          += 1
            counts[m]["__overall__"]["k"] += 1
    return counts


def _count_skipped(records: List[dict]) -> int:
    return sum(1 for r in records if not r.get("stability_valid"))


def _mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list) -> float:
    import math
    if len(vals) < 2:
        return 0.0
    m   = _mean(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)
