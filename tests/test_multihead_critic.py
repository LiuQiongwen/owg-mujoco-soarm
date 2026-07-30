"""
Unit tests for world_model/train_multihead_critic.py -- split equivalence
(Option B duplication, verified against the real, unmodified original) and
model/loss structural correctness.

Run: conda run -n tango python -m pytest tests/test_multihead_critic.py -v
"""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from world_model.train_multihead_critic import (
    scene_grouped_split, load_scenes_multihead, MultiHeadCritic, compute_loss,
)

_TRAIN_PATH = Path("results/risk_gated_vla/counterfactual_train_n40_20260730/scenes.jsonl")


# ── Split equivalence: duplicated (Option B) logic vs. the real, unmodified
#    original in train_counterfactual_critic.py -- the file itself is never
#    imported for modification, only called as a black box for comparison ──

def test_split_matches_existing_train_one():
    if not _TRAIN_PATH.exists():
        pytest.skip("no train-100 scenes.jsonl found")
    from world_model.train_counterfactual_critic import load_scenes, train_one

    scenes = load_scenes(_TRAIN_PATH, relative=True)
    # epochs=1: cheap, only need train_one()'s own split, not a trained model
    _, _, _, orig_train, orig_val, _ = train_one(scenes, "object_bce", seed=0, epochs=1)
    orig_train_keys = {s["key"] for s in orig_train}
    orig_val_keys = {s["key"] for s in orig_val}

    my_train, my_val = scene_grouped_split(scenes, seed=0)
    my_train_keys = {s["key"] for s in my_train}
    my_val_keys = {s["key"] for s in my_val}

    assert my_train_keys == orig_train_keys, "duplicated split disagrees with the original train_one()"
    assert my_val_keys == orig_val_keys
    assert my_train_keys.isdisjoint(my_val_keys)


def test_split_matches_existing_train_one_multiple_seeds():
    if not _TRAIN_PATH.exists():
        pytest.skip("no train-100 scenes.jsonl found")
    from world_model.train_counterfactual_critic import load_scenes, train_one

    scenes = load_scenes(_TRAIN_PATH, relative=True)
    for seed in (1, 2, 3):
        _, _, _, orig_train, orig_val, _ = train_one(scenes, "object_bce", seed=seed, epochs=1)
        my_train, my_val = scene_grouped_split(scenes, seed=seed)
        assert {s["key"] for s in my_train} == {s["key"] for s in orig_train}, f"seed={seed}"
        assert {s["key"] for s in my_val} == {s["key"] for s in orig_val}, f"seed={seed}"


# ── Data loading: multi-head labels wired correctly ─────────────────────────

def test_load_scenes_multihead_label_shapes():
    if not _TRAIN_PATH.exists():
        pytest.skip("no train-100 scenes.jsonl found")
    scenes = load_scenes_multihead(_TRAIN_PATH)
    assert len(scenes) > 0
    s0 = scenes[0]
    assert len(s0["x"]) == len(s0["y"])
    y0 = s0["y"][0]
    assert set(y0.keys()) == {"bilateral_contact", "lifted", "success", "failure_type_idx"}
    assert y0["failure_type_idx"] in (0.0, 1.0, 2.0)


def test_load_scenes_multihead_feature_dim_matches_baseline():
    if not _TRAIN_PATH.exists():
        pytest.skip("no train-100 scenes.jsonl found")
    from world_model.train_counterfactual_critic import load_scenes
    baseline_scenes = load_scenes(_TRAIN_PATH, relative=True)
    multihead_scenes = load_scenes_multihead(_TRAIN_PATH)
    assert len(baseline_scenes[0]["x"][0]) == len(multihead_scenes[0]["x"][0]), (
        "multi-head critic must use the exact same feature vector as the baseline "
        "(same causally-admissible inputs, only the label side differs)"
    )


# ── Model structure ──────────────────────────────────────────────────────────

def test_multihead_critic_output_shapes():
    model = MultiHeadCritic(dim=19)
    x = torch.randn(5, 19)
    out = model(x)
    assert out["bilateral_contact"].shape == (5,)
    assert out["lifted"].shape == (5,)
    assert out["success"].shape == (5,)
    assert out["failure_type"].shape == (5, 3)


def test_compute_loss_weightings_are_both_valid_and_differ():
    model = MultiHeadCritic(dim=19)
    x = torch.randn(8, 19)
    preds = model(x)
    targets = {
        "bilateral_contact": torch.randint(0, 2, (8,)).float(),
        "lifted": torch.randint(0, 2, (8,)).float(),
        "success": torch.randint(0, 2, (8,)).float(),
        "failure_type_idx": torch.randint(0, 3, (8,)).float(),
    }
    loss_equal = compute_loss(preds, targets, "equal")
    loss_weighted = compute_loss(preds, targets, "success_weighted")
    assert torch.isfinite(loss_equal) and loss_equal.item() > 0
    assert torch.isfinite(loss_weighted) and loss_weighted.item() > 0
    assert loss_equal.item() != loss_weighted.item()


def test_compute_loss_rejects_unknown_weighting():
    model = MultiHeadCritic(dim=19)
    preds = model(torch.randn(2, 19))
    targets = {k: torch.zeros(2) for k in
               ("bilateral_contact", "lifted", "success", "failure_type_idx")}
    with pytest.raises(ValueError):
        compute_loss(preds, targets, "not_a_real_weighting")
