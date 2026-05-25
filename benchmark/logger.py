"""Trial logger for the grasp benchmark.

Writes one JSONL record per trial and maintains a CSV summary.
Both files are append-safe: interrupted runs can be resumed by
checking which (object, seed, method) triples already have records.

Output layout::

    results/
        <run_id>/
            trials.jsonl      — one record per trial (primary log)
            summary.csv       — aggregated per (object, method) stats
            config.yaml       — copy of the run config for reproducibility
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class TrialRecord:
    """One benchmark trial."""
    trial_id:          int
    object:            str
    seed:              int
    method:            str
    execution_mode:    str

    # validity
    stability_valid:   bool
    skip_reason:       Optional[str]   # non-None only when stability_valid=False

    # grasp details (None when skipped)
    n_candidates:      Optional[int]
    grasp_rank_order:  Optional[List[int]]   # which candidate indices were tried, in order
    grasp_used_rank:   Optional[int]         # rank position of successful grasp (0-based)
    grasp_index:       Optional[int]         # raw candidate array index used

    # contact metrics (measured after gripper closes, before lift)
    contact_count:     Optional[int]         # total jaw-object contact points
    bilateral_contact: Optional[bool]        # both fixed and moving jaw in contact

    # outcome
    success:           Optional[bool]
    dz:                Optional[float]       # object Z gain (m) relative to pre-grasp
    slip:              Optional[float]       # |XY displacement| during lift (0 if lifted clean)
    fell_off:          Optional[bool]
    failure_reason:    Optional[str]

    # weld / contact detail (physics_weld_after_bilateral mode)
    weld_triggered:    Optional[bool]  = None  # kinematic weld was activated
    table_contact:     Optional[bool]  = None  # fixed jaw sphere touched table
    final_z:           Optional[float] = None  # object z after lift attempt (m)
    lifted:            Optional[bool]  = None  # obj_z > TABLE_TOP_Z + 0.07

    # diversity fields (set by DiverseBenchmarkRunner)
    difficulty:        Optional[str]   = None  # "easy" | "medium" | "hard"
    spawn_x:           Optional[float] = None  # m
    spawn_y:           Optional[float] = None  # m
    spawn_yaw:         Optional[float] = None  # rad — initial object yaw at spawn
    clutter_count:     Optional[int]   = None  # number of distractor objects
    grasp_yaw:         Optional[float] = None  # rad — yaw of the attempted grasp

    # metadata
    scene_file:        Optional[str] = None  # path to saved scene JSON for replay
    timestamp:         str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = d["timestamp"] or time.strftime("%Y-%m-%dT%H:%M:%S")
        return d


class TrialLogger:
    """
    Append-only benchmark trial logger.

    Parameters
    ----------
    run_dir : Path | str
        Directory for this run's output (created if absent).
    config_path : Path | str | None
        If given, copied into run_dir/config.yaml at init.
    resume : bool
        If True, don't overwrite existing trials.jsonl (resume interrupted run).
    """

    def __init__(
        self,
        run_dir:     Path | str,
        config_path: Optional[Path | str] = None,
        resume:      bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self.run_dir / "trials.jsonl"
        self._csv_path   = self.run_dir / "summary.csv"
        self._trial_id   = 0

        if config_path and Path(config_path).exists():
            shutil.copy(config_path, self.run_dir / "config.yaml")

        if resume and self._jsonl_path.exists():
            existing = list(self._iter_jsonl())
            self._trial_id = len(existing)
            print(f"[logger] resuming: {self._trial_id} existing trials found")

    # ── logging ───────────────────────────────────────────────────────────────

    def log(self, rec: TrialRecord) -> None:
        rec.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(rec.as_dict()) + "\n")
        self._trial_id += 1

    def log_skipped(
        self,
        obj:    str,
        seed:   int,
        method: str,
        mode:   str,
        reason: str,
    ) -> TrialRecord:
        rec = TrialRecord(
            trial_id=self._trial_id,
            object=obj, seed=seed, method=method,
            execution_mode=mode,
            stability_valid=False,
            skip_reason=reason,
            n_candidates=None,
            grasp_rank_order=None,
            grasp_used_rank=None,
            grasp_index=None,
            contact_count=None,
            bilateral_contact=None,
            success=None,
            dz=None,
            slip=None,
            fell_off=None,
            failure_reason=reason,
        )
        self.log(rec)
        return rec

    # ── summary CSV ───────────────────────────────────────────────────────────

    def write_summary(self) -> Path:
        """Aggregate all trials and write summary.csv.  Safe to call repeatedly."""
        from collections import defaultdict

        rows: dict = defaultdict(lambda: {
            "n_total": 0, "n_valid": 0, "n_success": 0,
            "dz_vals": [], "slip_vals": [],
        })

        for rec in self._iter_jsonl():
            key = (rec["object"], rec["method"])
            rows[key]["n_total"] += 1
            if rec.get("stability_valid"):
                rows[key]["n_valid"] += 1
            if rec.get("success"):
                rows[key]["n_success"] += 1
                if rec.get("dz") is not None:
                    rows[key]["dz_vals"].append(rec["dz"])
                if rec.get("slip") is not None:
                    rows[key]["slip_vals"].append(rec["slip"])

        bilateral: dict = defaultdict(lambda: {"n": 0, "k": 0})
        for rec in self._iter_jsonl():
            if not rec.get("stability_valid"):
                continue
            key = (rec["object"], rec["method"])
            if rec.get("bilateral_contact") is not None:
                bilateral[key]["n"] += 1
                if rec["bilateral_contact"]:
                    bilateral[key]["k"] += 1

        fieldnames = [
            "object", "method",
            "n_total", "n_valid", "n_success",
            "success_rate", "ci_lo", "ci_hi",
            "bilateral_rate",
            "dz_mean", "dz_std",
            "slip_mean",
        ]
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for (obj, method), d in sorted(rows.items()):
                k, n = d["n_success"], d["n_valid"]
                rate, lo, hi = wilson_ci(k, n)
                dz_vals   = d["dz_vals"]
                slip_vals = d.get("slip_vals", [])
                bi        = bilateral.get((obj, method), {})
                bi_rate   = bi["k"] / bi["n"] if bi.get("n") else ""
                writer.writerow({
                    "object":         obj,
                    "method":         method,
                    "n_total":        d["n_total"],
                    "n_valid":        d["n_valid"],
                    "n_success":      k,
                    "success_rate":   round(rate, 4),
                    "ci_lo":          round(lo,   4),
                    "ci_hi":          round(hi,   4),
                    "bilateral_rate": round(bi_rate, 4) if bi_rate != "" else "",
                    "dz_mean":        round(float(sum(dz_vals) / len(dz_vals)), 4) if dz_vals else "",
                    "dz_std":         round(float(_std(dz_vals)), 4) if dz_vals else "",
                    "slip_mean":      round(float(sum(slip_vals) / len(slip_vals)), 4) if slip_vals else "",
                })

        # optional per-difficulty summary when diversity fields are present
        self._write_difficulty_summary()

        return self._csv_path

    def _write_difficulty_summary(self) -> None:
        """Write difficulty_summary.csv when trials include difficulty labels."""
        from collections import defaultdict

        has_difficulty = any(
            rec.get("difficulty") for rec in self._iter_jsonl()
        )
        if not has_difficulty:
            return

        rows: dict = defaultdict(lambda: {"n_total": 0, "n_valid": 0, "n_success": 0})
        for rec in self._iter_jsonl():
            diff = rec.get("difficulty") or "unknown"
            key  = (diff, rec["method"])
            rows[key]["n_total"] += 1
            if rec.get("stability_valid"):
                rows[key]["n_valid"] += 1
            if rec.get("success"):
                rows[key]["n_success"] += 1

        diff_csv = self.run_dir / "difficulty_summary.csv"
        fieldnames = ["difficulty", "method", "n_total", "n_valid", "n_success",
                      "success_rate", "ci_lo", "ci_hi"]
        with open(diff_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for (diff, method), d in sorted(rows.items()):
                k, n = d["n_success"], d["n_valid"]
                rate, lo, hi = wilson_ci(k, n)
                writer.writerow({
                    "difficulty":   diff,
                    "method":       method,
                    "n_total":      d["n_total"],
                    "n_valid":      d["n_valid"],
                    "n_success":    k,
                    "success_rate": round(rate, 4),
                    "ci_lo":        round(lo,   4),
                    "ci_hi":        round(hi,   4),
                })

    # ── read-back ─────────────────────────────────────────────────────────────

    def load_records(self) -> List[dict]:
        return list(self._iter_jsonl())

    def already_done(self, obj: str, seed: int, method: str) -> bool:
        """Return True if this (obj, seed, method) triple already has a record."""
        for rec in self._iter_jsonl():
            if rec["object"] == obj and rec["seed"] == seed and rec["method"] == method:
                return True
        return False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _iter_jsonl(self) -> Iterator[dict]:
        if not self._jsonl_path.exists():
            return
        with open(self._jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass


# ── statistics ────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score confidence interval.

    Returns (point_estimate, lower, upper).
    """
    import math
    if n == 0:
        return 0.0, 0.0, 0.0
    p      = k / n
    denom  = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def _std(vals: list) -> float:
    import math
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var  = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)
