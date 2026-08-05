"""
Unit tests for the candidate-level point-cloud feature correction.

Background (the defect this fixes): data/transition_logger.py's
compute_pc_stats() is called once per scene, before any candidate is
sampled. world_model/train_counterfactual_critic.py's feature() then
attached that same scene-level 9-dim vector to every candidate in the
scene, so the point-cloud channel carried zero candidate-specific
geometric signal -- every candidate in a scene got an identical pc-block
regardless of its own pose. compute_pc_stats_local() (new) crops to the
candidate's own local neighborhood instead; feature() now prefers it when
present, falling back to the old shared stat for legacy records that don't
have it (feature()'s fallback path IS the pre-fix behavior, exercised
directly by test_fallback_reproduces_old_shared_behavior below).

These are pure-numpy unit tests -- no MuJoCo/tango env required, matching
this project's own convention (see tests/test_risk_gated_vla_phase1.py)
of keeping real-simulator-dependent checks separate from fast unit tests.
A live-MuJoCo end-to-end verification (real point clouds, real candidate
poses, real physics outcomes) was run separately and is not repeated here
as a pytest test since it needs the tango conda env; see the accompanying
report for that evidence.

Run: python -m pytest tests/test_candidate_pc_local_features.py -v
(works in any environment with numpy -- does not need the tango env)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.transition_logger import compute_pc_stats, compute_pc_stats_local
from causal_validity_audit.provenance import (
    audit_feature_set, CausalValidityViolation, Provenance, WORLD_MODEL_FIELDS,
)
from world_model.train_counterfactual_critic import feature, OBJECTS


# ── Synthetic fixture construction ──────────────────────────────────────────
# Two well-separated point clusters on the SAME object (obj_id=7), plus
# unrelated background points on a different id (obj_id=0) to confirm
# segmentation masking still applies in the local-crop path exactly as it
# does in the existing scene-level compute_pc_stats().

_OBJ_ID = 7
_BG_ID = 0


def _make_obs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=[0.0, 0.0, 0.90], scale=0.005, size=(30, 3))
    cluster_b = rng.normal(loc=[0.50, 0.50, 0.92], scale=0.005, size=(30, 3))
    background = rng.normal(loc=[2.0, 2.0, 0.0], scale=0.1, size=(15, 3))

    points = np.concatenate([cluster_a, cluster_b, background], axis=0).astype(np.float32)
    seg = np.concatenate([
        np.full(30, _OBJ_ID, dtype=np.int32),
        np.full(30, _OBJ_ID, dtype=np.int32),
        np.full(15, _BG_ID, dtype=np.int32),
    ])
    return {"seg": seg, "points": points}


CENTER_A = np.array([0.0, 0.0, 0.90], dtype=np.float32)
CENTER_B = np.array([0.50, 0.50, 0.92], dtype=np.float32)
CENTER_EMPTY = np.array([10.0, 10.0, 10.0], dtype=np.float32)  # far from everything


# ── Property 1: candidates with different local geometry get different features ──

def test_different_local_geometry_gives_different_stats():
    obs = _make_obs()
    stats_a = compute_pc_stats_local(obs, _OBJ_ID, CENTER_A, radius=0.05)
    stats_b = compute_pc_stats_local(obs, _OBJ_ID, CENTER_B, radius=0.05)
    assert not np.array_equal(stats_a, stats_b)
    # centroid (dims 0:3) should track the requested center, not be identical
    assert not np.allclose(stats_a[:3], stats_b[:3])


def test_different_local_geometry_gives_different_full_feature_vectors():
    obs = _make_obs()
    rec = {"object": "cracker", "obj_pos_before": [0.0, 0.0, 0.9],
           "pc_stats_before": [0.0] * 9}
    stats_a = compute_pc_stats_local(obs, _OBJ_ID, CENTER_A, radius=0.05)
    stats_b = compute_pc_stats_local(obs, _OBJ_ID, CENTER_B, radius=0.05)
    cand_a = {"candidate_pose": [0.0, 0.0, 0.9, 0.0, 0.05, 0.05],
              "pc_stats_local": stats_a.tolist()}
    cand_b = {"candidate_pose": [0.5, 0.5, 0.92, 0.0, 0.05, 0.05],
              "pc_stats_local": stats_b.tolist()}
    feat_a = feature(rec, cand_a, relative=False)
    feat_b = feature(rec, cand_b, relative=False)
    assert feat_a[7:16] != feat_b[7:16], "pc-block must differ for candidates at different local geometry"


# ── Property 2: identical candidate geometry stays deterministic ───────────────

def test_identical_geometry_is_deterministic():
    obs = _make_obs()
    s1 = compute_pc_stats_local(obs, _OBJ_ID, CENTER_A, radius=0.05)
    s2 = compute_pc_stats_local(obs, _OBJ_ID, CENTER_A, radius=0.05)
    assert np.array_equal(s1, s2)


def test_identical_candidate_dict_gives_identical_feature_vector():
    rec = {"object": "cracker", "obj_pos_before": [0.0, 0.0, 0.9],
           "pc_stats_before": [0.0] * 9}
    cand = {"candidate_pose": [0.1, 0.2, 0.9, 0.3, 0.05, 0.05],
            "pc_stats_local": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5]}
    f1 = feature(rec, cand, relative=True)
    f2 = feature(rec, cand, relative=True)
    assert f1 == f2


# ── Property 3: feature dimension and order unchanged ──────────────────────────

def test_local_stats_are_9_dimensional_matching_scene_level():
    obs = _make_obs()
    local = compute_pc_stats_local(obs, _OBJ_ID, CENTER_A, radius=0.05)
    scene = compute_pc_stats(obs, _OBJ_ID)
    assert local.shape == scene.shape == (9,)


def test_feature_vector_length_and_layout_unchanged_by_the_fix():
    rec = {"object": "cracker", "obj_pos_before": [0.0, 0.0, 0.9],
           "pc_stats_before": [0.0] * 9}
    base_pose = [0.1, 0.2, 0.9, 0.3, 0.05, 0.05]
    cand_old = {"candidate_pose": base_pose}  # no pc_stats_local -> fallback path
    cand_new = {"candidate_pose": base_pose,
                "pc_stats_local": [9.0] * 9}  # corrected path
    feat_old = feature(rec, cand_old, relative=True)
    feat_new = feature(rec, cand_new, relative=True)
    expected_len = 3 + 2 + 2 + 9 + len(OBJECTS)  # xyz + sin/cos + 2 gripper + pc(9) + onehot
    assert len(feat_old) == len(feat_new) == expected_len == 19
    # only the pc block (indices 7:16) may differ; base pose (0:7) and
    # object one-hot (16:19) must be byte-identical regardless of which pc
    # source was used
    assert feat_old[:7] == feat_new[:7]
    assert feat_old[16:] == feat_new[16:]
    assert feat_old[7:16] != feat_new[7:16]


# ── Property 4: no post-execution / outcome-derived information leaks in ──────

def test_pc_stats_local_registered_as_pre_execution_not_execution_derived():
    assert "pc_stats_local" in WORLD_MODEL_FIELDS
    assert WORLD_MODEL_FIELDS["pc_stats_local"].provenance is Provenance.PRE_EXECUTION


def test_feature_set_including_pc_stats_local_passes_the_causal_validity_gate():
    assert audit_feature_set(
        ["grasp_pose", "obj_pos_before", "obj_quat_before",
         "pc_stats_before", "pc_stats_local"],
        context="test",
    ) is True


def test_compute_pc_stats_local_signature_takes_no_outcome_fields():
    # structural guard: the function must not even be callable with anything
    # that smells like an execution outcome -- it only accepts the
    # observation, the object id, and a proposed (pre-execution) center.
    import inspect
    params = set(inspect.signature(compute_pc_stats_local).parameters)
    outcome_like = {"success", "dz", "fell_off", "outcome", "label",
                     "lifted", "grasped_id", "bilateral_contact"}
    assert params.isdisjoint(outcome_like)
    assert params == {"obs", "obj_id", "center_xyz", "radius"}


# ── Property 5: no candidate is silently dropped ────────────────────────────────

def test_all_candidates_produce_a_feature_row_even_with_sparse_local_geometry():
    obs = _make_obs()
    rec = {"object": "cracker", "obj_pos_before": [0.0, 0.0, 0.9],
           "pc_stats_before": [0.0] * 9}
    # one candidate sits on real geometry, one sits far from every point
    # (local crop will be empty -> zeros, per compute_pc_stats_local's own
    # documented fallback) -- both must still produce a full-length feature
    # row, never an exception and never a dropped entry.
    candidates_raw = [CENTER_A, CENTER_B, CENTER_EMPTY]
    rows = []
    for c in candidates_raw:
        local = compute_pc_stats_local(obs, _OBJ_ID, c, radius=0.05)
        cand = {"candidate_pose": [*c.tolist(), 0.0, 0.05, 0.05],
                "pc_stats_local": local.tolist()}
        rows.append(feature(rec, cand, relative=False))
    assert len(rows) == len(candidates_raw) == 3
    assert all(len(r) == 19 for r in rows)
    # the empty-neighborhood candidate must be zeros, not missing/NaN/dropped
    empty_local = compute_pc_stats_local(obs, _OBJ_ID, CENTER_EMPTY, radius=0.05)
    assert np.array_equal(empty_local, np.zeros(9, dtype=np.float32))
    assert not np.isnan(empty_local).any()


def test_too_few_local_points_returns_zeros_not_an_exception():
    obs = _make_obs()
    # radius so small that fewer than 5 points fall inside even at a real
    # cluster center -- must degrade to documented zeros, not raise.
    result = compute_pc_stats_local(obs, _OBJ_ID, CENTER_A, radius=1e-6)
    assert result.shape == (9,)
    assert np.array_equal(result, np.zeros(9, dtype=np.float32))


# ── Property 6: old shared-feature behavior cannot regress unnoticed ───────────

def test_fallback_reproduces_old_shared_behavior_exactly():
    """Direct regression test for the pre-fix code path: a candidate dict
    WITHOUT pc_stats_local must reproduce byte-identical output to what
    feature() always returned before this fix (reading rec["pc_stats_before"]
    unconditionally). This is what protects against someone reverting the
    fix's one-line change without noticing."""
    rec = {"object": "mustard", "obj_pos_before": [0.1, 0.1, 0.9],
           "pc_stats_before": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 0.9]}
    cand = {"candidate_pose": [0.15, 0.12, 0.9, 1.0, 0.06, 0.05]}  # no pc_stats_local
    feat = feature(rec, cand, relative=True)
    assert feat[7:16] == rec["pc_stats_before"]


def test_regression_guard_two_candidates_without_local_stats_collapse_to_shared():
    """If the fix were reverted (feature() went back to unconditionally
    reading rec["pc_stats_before"]), this test would still pass -- which is
    exactly why test_different_local_geometry_gives_different_full_feature_vectors
    above is the one that actually catches a regression: it requires
    pc_stats_local to be present AND used. This test documents, explicitly,
    what the pre-fix (shared) behavior looks like so a reviewer can see the
    contrast directly next to the regression-catching test."""
    rec = {"object": "drill", "obj_pos_before": [0.0, 0.0, 0.9],
           "pc_stats_before": [5.0] * 9}
    cand_1 = {"candidate_pose": [0.1, 0.0, 0.9, 0.0, 0.05, 0.05]}
    cand_2 = {"candidate_pose": [-0.1, 0.3, 0.9, 1.5, 0.07, 0.05]}
    feat_1 = feature(rec, cand_1, relative=True)
    feat_2 = feature(rec, cand_2, relative=True)
    # without pc_stats_local, both DO collapse to the shared stat -- this is
    # the defect, faithfully reproduced for callers that don't opt in yet.
    assert feat_1[7:16] == feat_2[7:16] == rec["pc_stats_before"]
