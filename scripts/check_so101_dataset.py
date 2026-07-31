#!/usr/bin/env python3
"""Read-only structural validator for the SO-101 (SO-ARM101) demonstration
dataset format.

Targets the "Format B / LeRobot-style" on-disk layout written by
``datasets.writer.EpisodeWriter`` and read by ``datasets.reader.EpisodeReader``
(see ``datasets/episode.py`` for the canonical Episode/EpisodeStep schema):

    <dataset_path>/
        meta.jsonl                 one JSON object per line (episode summary)
        dataset_info.json          optional dataset-level metadata
        episodes/
            ep_{episode_id:05d}/
                steps.npz           obs (T, 10), action (T, 6), timestamp (T,)
                frames/              optional per-frame images
                    step_{idx:04d}_rgb.npy
                    step_{idx:04d}_depth.npy

This script only reads files under ``dataset_path``; it never writes into it.
See docs/so101_dataset_checker.md for the full format description, field
definitions, and known limitations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCHEMA_VERSION = "1.0"
DATASET_FORMAT = "so101_lerobot_style_v1"

# Canonical field names, mirrored from datasets/episode.py so this checker
# cannot silently drift from the repository's own Episode/EpisodeStep schema.
#   obs_vector()    = JointState(5) + GripperState(2) + eef_pos(3) = 10 dims
#   GraspAction.as_vector() = eef_pos(3) + yaw + opening_m + obj_height = 6 dims
OBSERVATION_FIELDS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
    "gripper_opening_m", "gripper_is_closed",
    "eef_x", "eef_y", "eef_z",
]
ACTION_FIELDS = ["eef_x", "eef_y", "eef_z", "yaw", "opening_m", "obj_height"]

EXPECTED_OBS_DIM = len(OBSERVATION_FIELDS)
EXPECTED_ACTION_DIM = len(ACTION_FIELDS)

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_PATH_MISSING = 2


def _finite_check(arr: np.ndarray, name: str, errors: List[str]) -> None:
    if arr.size == 0:
        return
    if not np.issubdtype(arr.dtype, np.number):
        return
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        errors.append(f"{name}: {n_bad} non-finite value(s) (NaN/Inf)")


def _load_meta_lines(meta_path: Path, errors: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen_ids = set()
    text = meta_path.read_text()
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"meta.jsonl line {i}: malformed JSON ({e})")
            continue
        if not isinstance(entry, dict) or "episode_id" not in entry:
            errors.append(f"meta.jsonl line {i}: missing required field 'episode_id'")
            continue
        ep_id = entry["episode_id"]
        if ep_id in seen_ids:
            errors.append(f"meta.jsonl line {i}: duplicate episode_id {ep_id}")
            continue
        seen_ids.add(ep_id)
        entries.append(entry)
    return entries


def check_episode(
    dataset_root: Path,
    entry: Dict[str, Any],
    expected_action_dim_holder: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    """Validate a single episode. Returns a per-episode summary dict."""
    ep_id = entry["episode_id"]
    ep_errors: List[str] = []
    ep_warnings: List[str] = []
    summary: Dict[str, Any] = {
        "episode_id": ep_id,
        "n_frames": 0,
        "has_steps_file": False,
        "observation_dim": None,
        "action_dim": None,
        "errors": ep_errors,
        "warnings": ep_warnings,
    }

    try:
        ep_dir_name = f"ep_{int(ep_id):05d}"
    except (TypeError, ValueError):
        ep_errors.append(f"episode_id {ep_id!r} is not an integer")
        return summary

    steps_path = dataset_root / "episodes" / ep_dir_name / "steps.npz"
    if not steps_path.exists():
        ep_errors.append(f"missing steps file: {steps_path}")
        return summary
    summary["has_steps_file"] = True

    try:
        data = np.load(steps_path)
    except Exception as e:  # corrupted/unreadable npz
        ep_errors.append(f"failed to load {steps_path}: {e}")
        return summary

    required_keys = ("obs", "action", "timestamp")
    missing = [k for k in required_keys if k not in data.files]
    if missing:
        ep_errors.append(f"steps.npz missing required array(s): {missing}")
        return summary

    obs = data["obs"]
    action = data["action"]
    timestamp = data["timestamp"]

    n_obs, n_action, n_ts = len(obs), len(action), len(timestamp)
    summary["n_frames"] = n_obs
    if not (n_obs == n_action == n_ts):
        ep_errors.append(
            f"inconsistent lengths: obs={n_obs} action={n_action} timestamp={n_ts}"
        )

    if n_obs == 0:
        ep_errors.append("episode has zero frames")
        return summary

    obs_dim = obs.shape[1] if obs.ndim == 2 else None
    action_dim = action.shape[1] if action.ndim == 2 else None
    summary["observation_dim"] = obs_dim
    summary["action_dim"] = action_dim

    if obs_dim != EXPECTED_OBS_DIM:
        ep_errors.append(f"observation_dim={obs_dim}, expected {EXPECTED_OBS_DIM}")
    if action_dim != EXPECTED_ACTION_DIM:
        ep_errors.append(f"action_dim={action_dim}, expected {EXPECTED_ACTION_DIM}")

    if expected_action_dim_holder["value"] is None:
        expected_action_dim_holder["value"] = action_dim
    elif action_dim != expected_action_dim_holder["value"]:
        ep_errors.append(
            f"action_dim={action_dim} inconsistent with earlier episode "
            f"action_dim={expected_action_dim_holder['value']}"
        )

    _finite_check(obs, "obs", ep_errors)
    _finite_check(action, "action", ep_errors)
    _finite_check(timestamp, "timestamp", ep_errors)

    if n_ts > 1 and np.any(np.diff(timestamp) < 0):
        ep_errors.append("timestamps are not ordered (non-monotonic)")

    # Row order in obs/action/timestamp is the frame-index order by
    # construction (EpisodeWriter appends steps sequentially). When optional
    # per-frame image files are present, cross-check their indices too.
    frames_dir = dataset_root / "episodes" / ep_dir_name / "frames"
    if frames_dir.is_dir():
        for label in ("rgb", "depth"):
            files = sorted(frames_dir.glob(f"step_*_{label}.npy"))
            if not files:
                continue
            indices = []
            for f in files:
                try:
                    indices.append(int(f.name.split("_")[1]))
                except (IndexError, ValueError):
                    ep_warnings.append(f"unparseable frame filename: {f.name}")
            if indices != sorted(indices):
                ep_errors.append(f"{label} frame indices are not ordered")

            missing_refs = [i for i in range(n_obs)
                             if not (frames_dir / f"step_{i:04d}_{label}.npy").exists()]
            if missing_refs:
                ep_warnings.append(
                    f"{label}: {len(missing_refs)}/{n_obs} expected frame file(s) "
                    f"missing (e.g. step_{missing_refs[0]:04d}_{label}.npy)"
                )

    return summary


def check_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Run all structural checks against dataset_path. Never writes to disk."""
    dataset_path = Path(dataset_path)
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "valid": False,
        "dataset_format": DATASET_FORMAT,
        "number_of_episodes": 0,
        "number_of_frames": 0,
        "observation_fields": OBSERVATION_FIELDS,
        "action_fields": ACTION_FIELDS,
        "action_dimension": EXPECTED_ACTION_DIM,
        "errors": [],
        "warnings": [],
        "per_episode_summary": [],
    }

    if not dataset_path.exists():
        report["errors"].append(f"dataset path does not exist: {dataset_path}")
        return report
    if not dataset_path.is_dir():
        report["errors"].append(f"dataset path is not a directory: {dataset_path}")
        return report

    meta_path = dataset_path / "meta.jsonl"
    if not meta_path.exists():
        report["errors"].append(f"missing meta.jsonl (expected at {meta_path})")
        return report

    entries = _load_meta_lines(meta_path, report["errors"])

    if not entries:
        report["errors"].append("dataset contains no episodes")
        return report

    report["number_of_episodes"] = len(entries)

    expected_action_dim_holder: Dict[str, Optional[int]] = {"value": None}
    total_frames = 0
    for entry in entries:
        summary = check_episode(dataset_path, entry, expected_action_dim_holder)
        report["per_episode_summary"].append(summary)
        total_frames += summary["n_frames"]
        report["errors"].extend(
            f"episode {summary['episode_id']}: {e}" for e in summary["errors"]
        )
        report["warnings"].extend(
            f"episode {summary['episode_id']}: {w}" for w in summary["warnings"]
        )

    report["number_of_frames"] = total_frames

    info_path = dataset_path / "dataset_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            if isinstance(info, dict) and "n_episodes" in info \
                    and info["n_episodes"] != len(entries):
                report["warnings"].append(
                    f"dataset_info.json n_episodes={info['n_episodes']} "
                    f"!= meta.jsonl episode count={len(entries)}"
                )
        except json.JSONDecodeError as e:
            report["warnings"].append(f"dataset_info.json malformed: {e}")
    else:
        report["warnings"].append("dataset_info.json not found (optional)")

    report["valid"] = len(report["errors"]) == 0
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only structural validator for the SO-101 (SO-ARM101) "
                     "demonstration dataset format (see docs/so101_dataset_checker.md)."
    )
    parser.add_argument("--dataset-path", required=True,
                         help="Path to the dataset root directory")
    parser.add_argument("--output", default=None,
                         help="Optional path to write the JSON report")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset_path)
    path_missing = not dataset_path.exists()

    report = check_dataset(dataset_path)
    report_json = json.dumps(report, indent=2)

    print(report_json)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_json + "\n")

    if path_missing:
        print(f"ERROR: dataset path does not exist: {dataset_path}", file=sys.stderr)
        return EXIT_PATH_MISSING
    if not report["valid"]:
        return EXIT_INVALID
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
