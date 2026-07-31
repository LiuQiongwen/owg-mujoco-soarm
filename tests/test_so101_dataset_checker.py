"""Tests for scripts/check_so101_dataset.py.

All tests are local, offline, and do not require GPU, network, Docker, sudo,
or a real robot. Every dataset used by these tests — valid and broken — is
built at runtime under pytest's tmp_path via the fixture-builder helpers
below, using numpy.savez_compressed for steps.npz. Nothing is checked into
tests/fixtures/: the repository root .gitignore excludes *.npz, so a
checked-in dataset directory would silently lose its steps.npz files on a
fresh clone. Building everything at runtime avoids that trap entirely.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.check_so101_dataset import (
    check_dataset, main, EXIT_OK, EXIT_INVALID, EXIT_PATH_MISSING,
)


# ── fixture-builder helpers ─────────────────────────────────────────────────

def _write_meta(root, episode_ids):
    lines = [json.dumps({"episode_id": ep_id, "obj_name": "banana"}) for ep_id in episode_ids]
    (root / "meta.jsonl").write_text("\n".join(lines) + "\n")


def _write_episode(root, ep_id, obs, action, timestamp):
    """Write one episode's steps.npz at runtime and return its directory."""
    ep_dir = root / "episodes" / f"ep_{ep_id:05d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ep_dir / "steps.npz",
        obs=np.asarray(obs, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        timestamp=np.asarray(timestamp, dtype=np.float64),
    )
    return ep_dir


def _write_frame_images(ep_dir, n_frames, label="rgb", seed=0):
    """Write tiny per-frame image fixtures (e.g. step_0000_rgb.npy) at runtime."""
    frames_dir = ep_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_frames):
        img = rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8)
        np.save(frames_dir / f"step_{i:04d}_{label}.npy", img)


def _write_dataset_info(root, n_episodes):
    (root / "dataset_info.json").write_text(json.dumps({
        "robot": "soarm101",
        "camera": "overhead",
        "obs_dim": 10,
        "action_dim": 6,
        "n_episodes": n_episodes,
        "updated_at": "2026-07-31T00:00:00",
    }, indent=2))


def _build_valid_dataset(root):
    """Two-episode valid Format B dataset, generated entirely at runtime:
    episode 0 has 3 frames with rgb image references, episode 1 has 2 frames
    with no frames/ directory (image recording disabled, a legitimate mode)."""
    root.mkdir(parents=True, exist_ok=True)
    _write_meta(root, [0, 1])

    obs0 = [
        [0.0, -0.4, 0.8, -0.4, 0.0, 0.08, 0.0, 0.30, -0.30, 0.85],
        [0.01, -0.39, 0.79, -0.39, 0.0, 0.07, 0.0, 0.30, -0.30, 0.86],
        [0.02, -0.38, 0.78, -0.38, 0.0, 0.05, 1.0, 0.31, -0.30, 0.86],
    ]
    action0 = [
        [0.30, -0.30, 0.85, 0.0, 0.08, 0.05],
        [0.30, -0.30, 0.86, 0.0, 0.07, 0.05],
        [0.31, -0.30, 0.86, 0.0, 0.05, 0.05],
    ]
    ts0 = [0.0, 0.05, 0.10]
    ep0_dir = _write_episode(root, 0, obs0, action0, ts0)
    _write_frame_images(ep0_dir, n_frames=3, label="rgb")

    obs1 = [
        [0.0, -0.4, 0.8, -0.4, 0.0, 0.08, 0.0, 0.28, -0.31, 0.85],
        [0.01, -0.41, 0.81, -0.41, 0.0, 0.05, 1.0, 0.28, -0.31, 0.85],
    ]
    action1 = [
        [0.28, -0.31, 0.85, 0.1, 0.08, 0.04],
        [0.28, -0.31, 0.85, 0.1, 0.05, 0.04],
    ]
    ts1 = [0.0, 0.05]
    _write_episode(root, 1, obs1, action1, ts1)

    _write_dataset_info(root, n_episodes=2)
    return root


def _minimal_valid_dataset(root):
    root.mkdir(parents=True, exist_ok=True)
    _write_meta(root, [0])
    _write_episode(
        root, 0,
        obs=np.zeros((2, 10)),
        action=np.zeros((2, 6)),
        timestamp=[0.0, 0.1],
    )
    return root


# ── valid synthetic dataset ─────────────────────────────────────────────────

def test_valid_dataset_is_valid_and_exits_zero(tmp_path):
    root = _build_valid_dataset(tmp_path / "valid_ds")
    report = check_dataset(root)
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["number_of_episodes"] == 2
    assert report["number_of_frames"] == 5
    assert report["action_dimension"] == 6
    assert report["dataset_format"] == "so101_lerobot_style_v1"
    assert len(report["per_episode_summary"]) == 2

    exit_code = main(["--dataset-path", str(root)])
    assert exit_code == EXIT_OK


def test_valid_dataset_writes_output_report(tmp_path):
    root = _build_valid_dataset(tmp_path / "valid_ds")
    out_path = tmp_path / "report.json"
    exit_code = main(["--dataset-path", str(root), "--output", str(out_path)])
    assert exit_code == EXIT_OK
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written["valid"] is True


def test_valid_minimal_dataset_passes(tmp_path):
    root = _minimal_valid_dataset(tmp_path / "minimal_ok_ds")
    report = check_dataset(root)
    assert report["valid"] is True
    assert report["errors"] == []


# ── missing dataset path ────────────────────────────────────────────────────

def test_missing_dataset_path_returns_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist"
    report = check_dataset(missing)
    assert report["valid"] is False
    assert any("does not exist" in e for e in report["errors"])

    exit_code = main(["--dataset-path", str(missing)])
    assert exit_code == EXIT_PATH_MISSING


# ── empty dataset ────────────────────────────────────────────────────────────

def test_empty_dataset_no_meta_file(tmp_path):
    root = tmp_path / "empty_ds"
    root.mkdir()
    report = check_dataset(root)
    assert report["valid"] is False
    assert any("meta.jsonl" in e for e in report["errors"])


def test_empty_dataset_zero_episodes(tmp_path):
    root = tmp_path / "zero_ep_ds"
    root.mkdir()
    (root / "meta.jsonl").write_text("")
    report = check_dataset(root)
    assert report["valid"] is False
    assert any("no episodes" in e for e in report["errors"])
    assert report["number_of_episodes"] == 0


# ── inconsistent action dimensions ──────────────────────────────────────────

def test_inconsistent_action_dimensions_across_episodes(tmp_path):
    root = tmp_path / "bad_action_dim_ds"
    root.mkdir()
    _write_meta(root, [0, 1])
    _write_episode(root, 0, obs=np.zeros((2, 10)), action=np.zeros((2, 6)),
                    timestamp=[0.0, 0.1])
    # episode 1 has a 5-dim action instead of 6-dim
    _write_episode(root, 1, obs=np.zeros((2, 10)), action=np.zeros((2, 5)),
                    timestamp=[0.0, 0.1])

    report = check_dataset(root)
    assert report["valid"] is False
    assert any("action_dim=5" in e for e in report["errors"])


# ── NaN / infinite numeric values ───────────────────────────────────────────

def test_nan_values_are_reported(tmp_path):
    root = tmp_path / "nan_ds"
    root.mkdir()
    _write_meta(root, [0])
    obs = np.zeros((2, 10))
    obs[0, 0] = np.nan
    _write_episode(root, 0, obs=obs, action=np.zeros((2, 6)), timestamp=[0.0, 0.1])

    report = check_dataset(root)
    assert report["valid"] is False
    assert any("non-finite" in e for e in report["errors"])


def test_infinite_values_are_reported(tmp_path):
    root = tmp_path / "inf_ds"
    root.mkdir()
    _write_meta(root, [0])
    action = np.zeros((2, 6))
    action[1, 2] = np.inf
    _write_episode(root, 0, obs=np.zeros((2, 10)), action=action, timestamp=[0.0, 0.1])

    report = check_dataset(root)
    assert report["valid"] is False
    assert any("non-finite" in e for e in report["errors"])


# ── missing required fields ─────────────────────────────────────────────────

def test_missing_episode_id_in_meta(tmp_path):
    root = tmp_path / "missing_field_ds"
    root.mkdir()
    (root / "meta.jsonl").write_text(json.dumps({"obj_name": "banana"}) + "\n")
    report = check_dataset(root)
    assert report["valid"] is False
    assert any("episode_id" in e for e in report["errors"])


def test_missing_steps_arrays(tmp_path):
    root = tmp_path / "missing_array_ds"
    root.mkdir()
    _write_meta(root, [0])
    ep_dir = root / "episodes" / "ep_00000"
    ep_dir.mkdir(parents=True)
    # steps.npz present but missing the required 'action' array
    np.savez_compressed(ep_dir / "steps.npz",
                         obs=np.zeros((2, 10), dtype=np.float32),
                         timestamp=np.array([0.0, 0.1]))
    report = check_dataset(root)
    assert report["valid"] is False
    assert any("missing required array" in e for e in report["errors"])


# ── malformed / incomplete episodes ─────────────────────────────────────────

def test_missing_steps_file(tmp_path):
    root = tmp_path / "no_steps_ds"
    root.mkdir()
    _write_meta(root, [0])
    report = check_dataset(root)
    assert report["valid"] is False
    assert any("missing steps file" in e for e in report["errors"])


def test_unordered_timestamps_reported(tmp_path):
    root = tmp_path / "bad_ts_ds"
    root.mkdir()
    _write_meta(root, [0])
    _write_episode(root, 0, obs=np.zeros((3, 10)), action=np.zeros((3, 6)),
                    timestamp=[0.0, 0.2, 0.1])
    report = check_dataset(root)
    assert report["valid"] is False
    assert any("not ordered" in e for e in report["errors"])


def test_inconsistent_obs_action_lengths(tmp_path):
    root = tmp_path / "mismatch_len_ds"
    root.mkdir()
    _write_meta(root, [0])
    ep_dir = root / "episodes" / "ep_00000"
    ep_dir.mkdir(parents=True)
    np.savez_compressed(
        ep_dir / "steps.npz",
        obs=np.zeros((3, 10), dtype=np.float32),
        action=np.zeros((2, 6), dtype=np.float32),
        timestamp=np.array([0.0, 0.1, 0.2]),
    )
    report = check_dataset(root)
    assert report["valid"] is False
    assert any("inconsistent lengths" in e for e in report["errors"])


def test_zero_frame_episode_reported(tmp_path):
    root = tmp_path / "zero_frame_ds"
    root.mkdir()
    _write_meta(root, [0])
    _write_episode(root, 0, obs=np.zeros((0, 10)), action=np.zeros((0, 6)), timestamp=[])
    report = check_dataset(root)
    assert report["valid"] is False
    assert any("zero frames" in e for e in report["errors"])


def test_does_not_modify_dataset(tmp_path):
    """The checker must be read-only with respect to the dataset directory."""
    root = tmp_path / "readonly_ds"
    root.mkdir()
    _minimal_valid_dataset(root)

    before = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            before[str(p)] = (p.stat().st_mtime_ns, p.read_bytes())

    check_dataset(root)

    after = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            after[str(p)] = (p.stat().st_mtime_ns, p.read_bytes())

    assert before == after
