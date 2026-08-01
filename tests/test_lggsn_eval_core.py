# -*- coding: utf-8 -*-
"""Tests for research_agent_pilots/lggsn_suite/eval_core.py. Pure stdlib --
no torch, runs under the research-agent venv.

Covers requirement list items:
  2.  checkpoint hash mismatch refusal
  3.  input-dimension mismatch refusal
  4.  state_dict schema mismatch refusal
  5.  missing feature refusal
  7.  deterministic grouping and pair construction
  8.  malformed group accounting
  9.  no silent row dropping
  10. finite-score checks
  11. deterministic repeat digest
  20. pair-accuracy calculation matches a hand-checked fixture
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research_agent_pilots", "lggsn_suite"))
import eval_core as ec  # noqa: E402


def _row(query, scene_id, label, row_id, **extra):
    r = {"query": query, "scene_id": scene_id, "label": label, "_row_id": row_id}
    r.update(extra)
    return r


# ── 2. checkpoint hash mismatch refusal ─────────────────────────────────────

def test_checkpoint_hash_mismatch_is_refused():
    with pytest.raises(ec.SchemaVerificationError):
        ec.verify_checkpoint_identity(expected_sha256="a" * 64, actual_sha256="b" * 64, name="x")


def test_checkpoint_hash_match_does_not_raise():
    ec.verify_checkpoint_identity(expected_sha256="a" * 64, actual_sha256="a" * 64, name="x")


# ── 3. input-dimension mismatch refusal ─────────────────────────────────────

def test_input_dimension_mismatch_is_refused():
    shapes = {"mlp.0.weight": (40, 12), "mlp.2.weight": (1, 40)}
    with pytest.raises(ec.SchemaVerificationError):
        ec.verify_state_dict_schema(
            state_dict_shapes=shapes, expected_input_dim=14, expected_query_dim=0,
            expected_hidden_dim=40, name="x",
        )


def test_input_dimension_match_does_not_raise():
    shapes = {"mlp.0.weight": (40, 12), "mlp.2.weight": (1, 40)}
    ec.verify_state_dict_schema(
        state_dict_shapes=shapes, expected_input_dim=12, expected_query_dim=0,
        expected_hidden_dim=40, name="x",
    )


# ── 4. state_dict schema mismatch refusal ───────────────────────────────────

def test_missing_mlp0_key_is_refused():
    with pytest.raises(ec.SchemaVerificationError):
        ec.verify_state_dict_schema(
            state_dict_shapes={"some.other.key": (1, 1)}, expected_input_dim=12,
            expected_query_dim=0, expected_hidden_dim=40, name="x",
        )


def test_missing_mlp2_key_is_refused():
    with pytest.raises(ec.SchemaVerificationError):
        ec.verify_state_dict_schema(
            state_dict_shapes={"mlp.0.weight": (40, 12)}, expected_input_dim=12,
            expected_query_dim=0, expected_hidden_dim=40, name="x",
        )


def test_unexpected_query_embedding_is_refused():
    shapes = {"mlp.0.weight": (40, 16), "mlp.2.weight": (1, 40), "query_emb.weight": (7, 4)}
    with pytest.raises(ec.SchemaVerificationError):
        ec.verify_state_dict_schema(
            state_dict_shapes=shapes, expected_input_dim=12, expected_query_dim=0,
            expected_hidden_dim=40, name="x",
        )


def test_missing_expected_query_embedding_is_refused():
    shapes = {"mlp.0.weight": (40, 12), "mlp.2.weight": (1, 40)}
    with pytest.raises(ec.SchemaVerificationError):
        ec.verify_state_dict_schema(
            state_dict_shapes=shapes, expected_input_dim=12, expected_query_dim=4,
            expected_hidden_dim=40, name="x",
        )


# ── 5. missing feature refusal ──────────────────────────────────────────────

def test_missing_raw_required_key_raises_at_read_time(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"query": "A", "scene_id": 1}\n')  # no "label"
    with pytest.raises(ec.DatasetSchemaError):
        ec.read_jsonl_rows(str(bad))


def test_validate_feature_columns_raises_on_missing_column():
    episodes = {("A", 1): [_row("A", 1, 1, 0, x=0.0, y=0.0)]}  # no "z"
    with pytest.raises(ec.DatasetSchemaError):
        ec.validate_feature_columns(episodes, ["x", "y", "z"])


def test_validate_feature_columns_passes_when_all_present():
    episodes = {("A", 1): [_row("A", 1, 1, 0, x=0.0, y=0.0, z=0.0)]}
    ec.validate_feature_columns(episodes, ["x", "y", "z"])  # must not raise


def test_derived_features_computed_when_needed_and_required_raw_cols_missing_raises():
    episode_rows = [_row("A", 1, 1, 0, y=0.0, z=0.0)]  # no "x"
    with pytest.raises(ec.DatasetSchemaError):
        ec.compute_derived_features(episode_rows, {"dist_to_centroid"})


def test_derived_features_noop_when_not_needed():
    episode_rows = [_row("A", 1, 1, 0)]  # no x/y/z at all, but not needed
    out = ec.compute_derived_features(episode_rows, set())
    assert out == episode_rows
    assert out is not episode_rows  # copies, never mutates in place


# ── 7/8/9. grouping, pairing, malformed-group accounting, no silent drop ───

def _tiny_fixture():
    """3 episodes for query 'A': one tied (excluded), one positive
    (2 candidates), one negative (1 candidate). Hand-computed expected
    pairing: cartesian(pos, neg) = 2 x 1 = 2 pairs, all landing in val
    (val_frac=0.2 -> max(1, round(1*0.2))=1 episode each side, and there's
    only 1 episode per side, so ALL of it goes to val, none to train)."""
    rows = [
        _row("A", 1, 1, 0, x=0.0, y=0.0, z=0.0),  # tied episode
        _row("A", 1, 0, 1, x=0.1, y=0.0, z=0.0),
        _row("A", 2, 1, 2, x=0.0, y=0.0, z=0.0),  # positive episode (2 cands)
        _row("A", 2, 1, 3, x=0.2, y=0.0, z=0.0),
        _row("A", 3, 0, 4, x=0.5, y=0.0, z=0.0),  # negative episode (1 cand)
    ]
    return rows


def test_grouping_produces_expected_episode_keys():
    episodes = ec.group_episodes(_tiny_fixture())
    assert list(episodes.keys()) == [("A", 1), ("A", 2), ("A", 3)]


def test_malformed_group_accounting_matches_hand_count():
    episodes = ec.group_episodes(_tiny_fixture())
    by_query, malformed, skipped = ec.label_episodes(episodes)
    assert malformed == 1  # episode (A,1): 1 pos, 1 neg -> tied
    assert skipped == 2  # both rows of that tied episode


def test_no_silent_row_dropping_total_accounted():
    rows = _tiny_fixture()
    episodes = ec.group_episodes(rows)
    by_query, malformed, skipped = ec.label_episodes(episodes)
    eligible_rows = sum(
        len(ep) for sides in by_query.values() for side in ("pos", "neg") for ep in sides[side]
    )
    assert eligible_rows + skipped == len(rows)  # every row is either eligible or accounted-for as skipped


def test_pairing_matches_hand_computed_fixture():
    episodes = ec.group_episodes(_tiny_fixture())
    by_query, _, _ = ec.label_episodes(episodes)
    train_pairs, val_pairs = ec.build_pairs(by_query, val_frac=0.2, seed=42)
    assert train_pairs == []
    assert set(val_pairs) == {(2, 4, "A"), (3, 4, "A")}


def test_pairing_is_deterministic_across_repeated_calls():
    episodes = ec.group_episodes(_tiny_fixture())
    by_query, _, _ = ec.label_episodes(episodes)
    a = ec.build_pairs(by_query, val_frac=0.2, seed=42)
    b = ec.build_pairs(by_query, val_frac=0.2, seed=42)
    assert a == b


# ── 20. pair-accuracy calculation matches a hand-checked fixture ───────────

def test_pair_accuracy_matches_hand_calculation():
    rows = _tiny_fixture()
    episodes = ec.group_episodes(rows)
    by_query, malformed, skipped = ec.label_episodes(episodes)
    train_pairs, val_pairs = ec.build_pairs(by_query, val_frac=0.2, seed=42)
    # val_pairs = {(2,4,'A'), (3,4,'A')} (row 2 vs row 4, row 3 vs row 4)
    scores = {0: 0.5, 1: 0.5, 2: 0.9, 3: 0.1, 4: 0.3}
    # pair (2,4): score[2]=0.9 > score[4]=0.3 -> correct
    # pair (3,4): score[3]=0.1 < score[4]=0.3 -> incorrect
    # hand accuracy = 1/2 = 0.5
    metrics = ec.build_metrics(
        checkpoint_name="fixture", checkpoint_sha256="x" * 64, provenance_status="TEST",
        feature_columns=["x", "y", "z"], input_dim=3, dataset_sha256="y" * 64, seed=42, val_frac=0.2,
        by_query=by_query, train_pairs=train_pairs, val_pairs=val_pairs, scores=scores,
        malformed_group_count=malformed, skipped_row_count=skipped, runtime_seconds=0.0,
    )
    assert metrics["pair_accuracy"] == pytest.approx(0.5)
    assert metrics["eligible_pair_count"] == 2
    assert metrics["ties_count"] == 0


# ── 10. finite-score checks ─────────────────────────────────────────────────

def test_nonfinite_score_is_reported_not_crashed():
    rows = _tiny_fixture()
    episodes = ec.group_episodes(rows)
    by_query, malformed, skipped = ec.label_episodes(episodes)
    train_pairs, val_pairs = ec.build_pairs(by_query, val_frac=0.2, seed=42)
    scores = {0: 0.5, 1: 0.5, 2: float("nan"), 3: 0.1, 4: 0.3}
    metrics = ec.build_metrics(
        checkpoint_name="fixture", checkpoint_sha256="x" * 64, provenance_status="TEST",
        feature_columns=["x", "y", "z"], input_dim=3, dataset_sha256="y" * 64, seed=42, val_frac=0.2,
        by_query=by_query, train_pairs=train_pairs, val_pairs=val_pairs, scores=scores,
        malformed_group_count=malformed, skipped_row_count=skipped, runtime_seconds=0.0,
    )  # must not raise
    assert metrics["all_scores_finite"] is False
    assert metrics["pilot_ok"] is False


def test_all_finite_scores_report_true():
    rows = _tiny_fixture()
    episodes = ec.group_episodes(rows)
    by_query, malformed, skipped = ec.label_episodes(episodes)
    train_pairs, val_pairs = ec.build_pairs(by_query, val_frac=0.2, seed=42)
    scores = {0: 0.5, 1: 0.5, 2: 0.9, 3: 0.1, 4: 0.3}
    metrics = ec.build_metrics(
        checkpoint_name="fixture", checkpoint_sha256="x" * 64, provenance_status="TEST",
        feature_columns=["x", "y", "z"], input_dim=3, dataset_sha256="y" * 64, seed=42, val_frac=0.2,
        by_query=by_query, train_pairs=train_pairs, val_pairs=val_pairs, scores=scores,
        malformed_group_count=malformed, skipped_row_count=skipped, runtime_seconds=0.0,
    )
    assert metrics["all_scores_finite"] is True


# ── 11. deterministic repeat digest ─────────────────────────────────────────

def test_deterministic_digest_repeatable_for_same_input():
    rows = _tiny_fixture()
    episodes = ec.group_episodes(rows)
    by_query, malformed, skipped = ec.label_episodes(episodes)
    train_pairs, val_pairs = ec.build_pairs(by_query, val_frac=0.2, seed=42)
    scores = {0: 0.5, 1: 0.5, 2: 0.9, 3: 0.1, 4: 0.3}
    kwargs = dict(
        checkpoint_name="fixture", checkpoint_sha256="x" * 64, provenance_status="TEST",
        feature_columns=["x", "y", "z"], input_dim=3, dataset_sha256="y" * 64, seed=42, val_frac=0.2,
        by_query=by_query, train_pairs=train_pairs, val_pairs=val_pairs, scores=scores,
        malformed_group_count=malformed, skipped_row_count=skipped,
    )
    m1 = ec.build_metrics(**kwargs, runtime_seconds=0.111)
    m2 = ec.build_metrics(**kwargs, runtime_seconds=99.999)  # different wall-clock time
    assert m1["deterministic_digest"] == m2["deterministic_digest"]  # digest excludes runtime_seconds


def test_deterministic_digest_changes_with_different_scores():
    rows = _tiny_fixture()
    episodes = ec.group_episodes(rows)
    by_query, malformed, skipped = ec.label_episodes(episodes)
    train_pairs, val_pairs = ec.build_pairs(by_query, val_frac=0.2, seed=42)
    kwargs = dict(
        checkpoint_name="fixture", checkpoint_sha256="x" * 64, provenance_status="TEST",
        feature_columns=["x", "y", "z"], input_dim=3, dataset_sha256="y" * 64, seed=42, val_frac=0.2,
        by_query=by_query, train_pairs=train_pairs, val_pairs=val_pairs,
        malformed_group_count=malformed, skipped_row_count=skipped, runtime_seconds=0.0,
    )
    # First case: 1/2 pairs correct (pair_accuracy=0.5). Second: both correct
    # (pair_accuracy=1.0) -- an aggregate-level change, not just a same-stats
    # relabeling (swapping which of two symmetric pairs "wins" leaves
    # pair_accuracy/margins unchanged in a 2-pair fixture, which would be a
    # vacuous test of digest sensitivity).
    m1 = ec.build_metrics(scores={0: 0.5, 1: 0.5, 2: 0.9, 3: 0.1, 4: 0.3}, **kwargs)
    m2 = ec.build_metrics(scores={0: 0.5, 1: 0.5, 2: 0.9, 3: 0.9, 4: 0.3}, **kwargs)
    assert m1["pair_accuracy"] != m2["pair_accuracy"]
    assert m1["deterministic_digest"] != m2["deterministic_digest"]


# ── output JSON hygiene ──────────────────────────────────────────────────────

def test_metrics_output_is_valid_json_no_nan_literals():
    import json
    rows = [_row("A", 1, 1, 0, x=0.0, y=0.0, z=0.0)]  # single episode, no opposite side -> zero pairs
    episodes = ec.group_episodes(rows)
    by_query, malformed, skipped = ec.label_episodes(episodes)
    train_pairs, val_pairs = ec.build_pairs(by_query, val_frac=0.2, seed=42)
    metrics = ec.build_metrics(
        checkpoint_name="fixture", checkpoint_sha256="x" * 64, provenance_status="TEST",
        feature_columns=["x", "y", "z"], input_dim=3, dataset_sha256="y" * 64, seed=42, val_frac=0.2,
        by_query=by_query, train_pairs=train_pairs, val_pairs=val_pairs, scores={0: 0.5},
        malformed_group_count=malformed, skipped_row_count=skipped, runtime_seconds=0.0,
    )
    s = json.dumps(metrics)
    assert "NaN" not in s
    assert json.loads(s) == metrics
    assert metrics["pair_accuracy"] is None  # zero pairs -> undefined, not NaN
