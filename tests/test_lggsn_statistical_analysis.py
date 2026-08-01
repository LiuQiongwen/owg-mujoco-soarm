# -*- coding: utf-8 -*-
"""Tests for research_agent_pilots/lggsn_analysis/ Phase 1 (loader.py,
alignment.py, statistics.py). Pure stdlib -- no torch/numpy required.

Pair-level JSONL fixtures are written on the fly under tmp_path rather than
committed as separate files, since Phase 1's allowed_modify_paths covers
only the five source/test files listed in
experiments/lggsn_statistical_analysis.yaml.
"""
import json
import math
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_agent_pilots.lggsn_analysis import alignment, loader
from research_agent_pilots.lggsn_analysis import statistics as lggsn_statistics

EVAL_OUTPUTS_DIR = REPO_ROOT / "research_agent_pilots" / "lggsn_suite" / "eval_outputs"


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def _pair(query: str, pos: str, neg: str, correct: bool) -> dict:
    return {"query": query, "pos_row_id": pos, "neg_row_id": neg, "correct": correct}


# ── loader.py: aggregate metrics (real, committed data) ─────────────────────

def test_load_all_aggregate_metrics_reads_all_four_checkpoints():
    metrics = loader.load_all_aggregate_metrics(EVAL_OUTPUTS_DIR)
    assert set(metrics.keys()) == set(loader.CORE_CHECKPOINT_NAMES)
    for name, data in metrics.items():
        assert data["checkpoint_name"] == name
        assert isinstance(data["pair_accuracy"], float)


def test_load_aggregate_metrics_does_not_modify_the_input_file(tmp_path):
    src = EVAL_OUTPUTS_DIR / "base" / "metrics.json"
    before = src.read_bytes()
    before_mtime = src.stat().st_mtime_ns
    loader.load_aggregate_metrics(EVAL_OUTPUTS_DIR / "base")
    assert src.read_bytes() == before
    assert src.stat().st_mtime_ns == before_mtime


def test_load_aggregate_metrics_missing_file_fails_closed(tmp_path):
    with pytest.raises(loader.LoaderError):
        loader.load_aggregate_metrics(tmp_path / "does_not_exist")


def test_load_matrix_summary_reads_real_file():
    summary = loader.load_matrix_summary(EVAL_OUTPUTS_DIR)
    assert summary["core_matrix"] == list(loader.CORE_CHECKPOINT_NAMES)


# ── loader.py: aggregate-only mode ──────────────────────────────────────────

def test_load_dataset_aggregate_only_mode_has_no_pair_records():
    dataset = loader.load_dataset(EVAL_OUTPUTS_DIR, mode=loader.LoadMode.AGGREGATE_ONLY)
    assert dataset.mode is loader.LoadMode.AGGREGATE_ONLY
    assert dataset.pair_records == {}
    assert set(dataset.aggregate_metrics.keys()) == set(loader.CORE_CHECKPOINT_NAMES)


def test_load_dataset_aggregate_only_mode_rejects_pair_fixture_paths(tmp_path):
    fixture = _write_jsonl(tmp_path / "base_pairs.jsonl", [_pair("q1", "p1", "n1", True)])
    with pytest.raises(loader.LoaderError):
        loader.load_dataset(
            EVAL_OUTPUTS_DIR, mode=loader.LoadMode.AGGREGATE_ONLY,
            pair_fixture_paths={"base": fixture},
        )


def test_load_dataset_pair_fixtures_mode_requires_fixture_paths():
    with pytest.raises(loader.LoaderError):
        loader.load_dataset(EVAL_OUTPUTS_DIR, mode=loader.LoadMode.PAIR_FIXTURES)


# ── loader.py: pair-level JSONL fixtures ────────────────────────────────────

def test_load_pair_fixture_jsonl_parses_valid_fixture(tmp_path):
    path = _write_jsonl(tmp_path / "a.jsonl", [
        _pair("q1", "p1", "n1", True),
        _pair("q1", "p2", "n2", False),
    ])
    records = loader.load_pair_fixture_jsonl(path)
    assert len(records) == 2
    assert records[0] == loader.PairRecord("q1", "p1", "n1", True)
    assert records[0].pair_key == ("q1", "p1", "n1")


def test_load_pair_fixture_jsonl_ignores_blank_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(
        json.dumps(_pair("q1", "p1", "n1", True)) + "\n\n" +
        json.dumps(_pair("q1", "p2", "n2", False)) + "\n"
    )
    records = loader.load_pair_fixture_jsonl(path)
    assert len(records) == 2


def test_load_pair_fixture_jsonl_missing_file_fails_closed(tmp_path):
    with pytest.raises(loader.LoaderError):
        loader.load_pair_fixture_jsonl(tmp_path / "missing.jsonl")


def test_load_pair_fixture_jsonl_empty_file_fails_closed(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(loader.LoaderError):
        loader.load_pair_fixture_jsonl(path)


def test_load_pair_fixture_jsonl_invalid_json_fails_closed(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json\n")
    with pytest.raises(loader.LoaderError):
        loader.load_pair_fixture_jsonl(path)


def test_load_pair_fixture_jsonl_missing_key_fails_closed(tmp_path):
    path = tmp_path / "missing_key.jsonl"
    path.write_text(json.dumps({"query": "q1", "pos_row_id": "p1", "correct": True}) + "\n")
    with pytest.raises(loader.LoaderError):
        loader.load_pair_fixture_jsonl(path)


def test_load_pair_fixture_jsonl_wrong_type_fails_closed(tmp_path):
    path = tmp_path / "wrong_type.jsonl"
    path.write_text(json.dumps({"query": "q1", "pos_row_id": "p1", "neg_row_id": "n1", "correct": "yes"}) + "\n")
    with pytest.raises(loader.LoaderError):
        loader.load_pair_fixture_jsonl(path)


def test_load_pair_fixture_jsonl_duplicate_identity_fails_closed(tmp_path):
    path = _write_jsonl(tmp_path / "dup.jsonl", [
        _pair("q1", "p1", "n1", True),
        _pair("q1", "p1", "n1", False),
    ])
    with pytest.raises(loader.LoaderError):
        loader.load_pair_fixture_jsonl(path)


# ── loader.py: real pair_results.jsonl availability ─────────────────────────

@pytest.mark.parametrize("checkpoint_name", loader.CORE_CHECKPOINT_NAMES)
def test_load_real_pair_results_is_unavailable_for_every_committed_checkpoint(checkpoint_name):
    result = loader.load_real_pair_results(EVAL_OUTPUTS_DIR / checkpoint_name)
    assert result.status is loader.PairDataStatus.UNAVAILABLE
    assert result.records == ()


def test_load_real_pair_results_loads_when_file_is_present(tmp_path):
    checkpoint_dir = tmp_path / "some_checkpoint"
    _write_jsonl(checkpoint_dir / "pair_results.jsonl", [_pair("q1", "p1", "n1", True)])
    result = loader.load_real_pair_results(checkpoint_dir)
    assert result.status is loader.PairDataStatus.LOADED
    assert len(result.records) == 1


# ── alignment.py ─────────────────────────────────────────────────────────────

def _two_checkpoint_fixtures(tmp_path):
    a = _write_jsonl(tmp_path / "chk_a.jsonl", [
        _pair("q1", "p1", "n1", True),
        _pair("q1", "p2", "n2", False),
        _pair("q1", "p3", "n3", False),
        _pair("q2", "p1", "n1", True),
        _pair("q2", "p2", "n2", False),
        _pair("q2", "p3", "n3", False),
    ])
    b = _write_jsonl(tmp_path / "chk_b.jsonl", [
        _pair("q1", "p1", "n1", True),
        _pair("q1", "p2", "n2", False),
        _pair("q1", "p3", "n3", True),
        _pair("q2", "p1", "n1", False),
        _pair("q2", "p2", "n2", True),
        _pair("q2", "p3", "n3", True),
    ])
    return loader.load_pair_fixture_jsonl(a), loader.load_pair_fixture_jsonl(b)


def test_align_pairs_matches_identical_identities(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    aligned = alignment.align_pairs({"chk_a": records_a, "chk_b": records_b})
    assert len(aligned) == 6
    assert all(set(p.correct_by_checkpoint.keys()) == {"chk_a", "chk_b"} for p in aligned)
    # deterministic (sorted-key) ordering
    assert [p.pair_key for p in aligned] == sorted(p.pair_key for p in aligned)


def test_align_pairs_requires_at_least_two_checkpoints(tmp_path):
    records_a, _ = _two_checkpoint_fixtures(tmp_path)
    with pytest.raises(alignment.AlignmentError):
        alignment.align_pairs({"chk_a": records_a})


def test_align_pairs_fails_closed_on_missing_pair_in_one_checkpoint(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    with pytest.raises(alignment.AlignmentError):
        alignment.align_pairs({"chk_a": records_a, "chk_b": records_b[:-1]})


def test_align_pairs_fails_closed_on_extra_pair_in_one_checkpoint(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    extra = records_b + (loader.PairRecord("q3", "p1", "n1", True),)
    with pytest.raises(alignment.AlignmentError):
        alignment.align_pairs({"chk_a": records_a, "chk_b": extra})


def test_align_pairs_fails_closed_on_duplicate_identity_within_checkpoint():
    duplicated = (
        loader.PairRecord("q1", "p1", "n1", True),
        loader.PairRecord("q1", "p1", "n1", False),
    )
    other = (loader.PairRecord("q1", "p1", "n1", True),)
    with pytest.raises(alignment.AlignmentError):
        alignment.align_pairs({"chk_a": duplicated, "chk_b": other})


def test_align_pairs_fails_closed_on_empty_records():
    with pytest.raises(alignment.AlignmentError):
        alignment.align_pairs({"chk_a": (), "chk_b": ()})


# ── statistics.py: exact McNemar ────────────────────────────────────────────

def test_exact_mcnemar_matches_hand_computed_p_value(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    aligned = alignment.align_pairs({"chk_a": records_a, "chk_b": records_b})
    result = lggsn_statistics.exact_mcnemar(aligned, checkpoint_a="chk_a", checkpoint_b="chk_b")
    # n01 (a wrong, b correct) = 3: (q1,p3,n3), (q2,p2,n2), (q2,p3,n3)
    # n10 (a correct, b wrong) = 1: (q2,p1,n1)
    # n11 = 1: (q1,p1,n1); n00 = 1: (q1,p2,n2)
    assert result.n01 == 3
    assert result.n10 == 1
    assert result.n11 == 1
    assert result.n00 == 1
    assert result.discordant_pairs == 4
    # exact binomial(n=4, p=0.5) two-sided tail at k=min(3,1)=1: 2 * (C(4,0)+C(4,1))/16 = 5/8
    assert result.p_value == pytest.approx(0.625, abs=1e-12)


def test_exact_mcnemar_symmetric_discordant_counts_give_p_value_one():
    aligned = alignment.align_pairs({
        "chk_a": (
            loader.PairRecord("q1", "p1", "n1", False),
            loader.PairRecord("q1", "p2", "n2", False),
            loader.PairRecord("q1", "p3", "n3", True),
            loader.PairRecord("q1", "p4", "n4", True),
        ),
        "chk_b": (
            loader.PairRecord("q1", "p1", "n1", True),
            loader.PairRecord("q1", "p2", "n2", True),
            loader.PairRecord("q1", "p3", "n3", False),
            loader.PairRecord("q1", "p4", "n4", False),
        ),
    })
    result = lggsn_statistics.exact_mcnemar(aligned, checkpoint_a="chk_a", checkpoint_b="chk_b")
    assert result.n01 == 2
    assert result.n10 == 2
    assert result.p_value == 1.0


def test_exact_mcnemar_no_discordant_pairs_gives_p_value_one():
    aligned = alignment.align_pairs({
        "chk_a": (loader.PairRecord("q1", "p1", "n1", True),),
        "chk_b": (loader.PairRecord("q1", "p1", "n1", True),),
    })
    result = lggsn_statistics.exact_mcnemar(aligned, checkpoint_a="chk_a", checkpoint_b="chk_b")
    assert result.discordant_pairs == 0
    assert result.p_value == 1.0


def test_exact_mcnemar_rejects_unknown_checkpoint():
    aligned = alignment.align_pairs({
        "chk_a": (loader.PairRecord("q1", "p1", "n1", True),),
        "chk_b": (loader.PairRecord("q1", "p1", "n1", True),),
    })
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.exact_mcnemar(aligned, checkpoint_a="chk_a", checkpoint_b="does_not_exist")


# ── statistics.py: win/tie/loss ─────────────────────────────────────────────

def test_win_tie_loss_counts_matches_hand_computed_fixture(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    aligned = alignment.align_pairs({"chk_a": records_a, "chk_b": records_b})
    result = lggsn_statistics.win_tie_loss_counts(aligned, checkpoint_a="chk_a", checkpoint_b="chk_b")
    assert result.win == 3
    assert result.loss == 1
    assert result.tie == 2
    assert result.total == 6 == len(aligned)


def test_win_tie_loss_counts_rejects_unknown_checkpoint():
    aligned = alignment.align_pairs({
        "chk_a": (loader.PairRecord("q1", "p1", "n1", True),),
        "chk_b": (loader.PairRecord("q1", "p1", "n1", True),),
    })
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.win_tie_loss_counts(aligned, checkpoint_a="chk_a", checkpoint_b="nope")


# ── statistics.py: deterministic paired bootstrap ───────────────────────────

def test_paired_bootstrap_ci_is_deterministic_for_a_fixed_seed(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    aligned = alignment.align_pairs({"chk_a": records_a, "chk_b": records_b})
    kwargs = dict(checkpoint_a="chk_a", checkpoint_b="chk_b", seed=12345, n_resamples=500)
    first = lggsn_statistics.paired_bootstrap_ci(aligned, **kwargs)
    second = lggsn_statistics.paired_bootstrap_ci(aligned, **kwargs)
    assert first == second


def test_paired_bootstrap_ci_observed_diff_matches_hand_computed_means(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    aligned = alignment.align_pairs({"chk_a": records_a, "chk_b": records_b})
    result = lggsn_statistics.paired_bootstrap_ci(
        aligned, checkpoint_a="chk_a", checkpoint_b="chk_b", seed=1, n_resamples=200,
    )
    # a: 2/6 correct, b: 4/6 correct -> diff = 2/6 = 1/3
    assert result.observed_diff_b_minus_a == pytest.approx(1.0 / 3.0)
    assert result.ci_lower <= result.ci_upper
    assert result.seed == 1
    assert result.n_resamples == 200
    assert result.n_pairs == 6


def test_paired_bootstrap_ci_different_seeds_can_differ(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    aligned = alignment.align_pairs({"chk_a": records_a, "chk_b": records_b})
    result_1 = lggsn_statistics.paired_bootstrap_ci(
        aligned, checkpoint_a="chk_a", checkpoint_b="chk_b", seed=1, n_resamples=200,
    )
    result_2 = lggsn_statistics.paired_bootstrap_ci(
        aligned, checkpoint_a="chk_a", checkpoint_b="chk_b", seed=2, n_resamples=200,
    )
    assert (result_1.ci_lower, result_1.ci_upper) != (result_2.ci_lower, result_2.ci_upper)


def test_paired_bootstrap_ci_rejects_empty_pairs():
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.paired_bootstrap_ci((), checkpoint_a="chk_a", checkpoint_b="chk_b", seed=1)


def test_paired_bootstrap_ci_rejects_invalid_confidence(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    aligned = alignment.align_pairs({"chk_a": records_a, "chk_b": records_b})
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.paired_bootstrap_ci(
            aligned, checkpoint_a="chk_a", checkpoint_b="chk_b", seed=1, confidence=1.5,
        )


def test_paired_bootstrap_ci_rejects_zero_resamples(tmp_path):
    records_a, records_b = _two_checkpoint_fixtures(tmp_path)
    aligned = alignment.align_pairs({"chk_a": records_a, "chk_b": records_b})
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.paired_bootstrap_ci(
            aligned, checkpoint_a="chk_a", checkpoint_b="chk_b", seed=1, n_resamples=0,
        )


# ── end-to-end Phase 1 pipeline: real aggregate data + fixture pair data ────

def test_end_to_end_pair_fixtures_mode_pipeline(tmp_path):
    a = _write_jsonl(tmp_path / "base_pairs.jsonl", [
        _pair("Banana", "p1", "n1", True),
        _pair("Banana", "p2", "n2", False),
        _pair("MustardBottle", "p1", "n1", True),
    ])
    b = _write_jsonl(tmp_path / "nodist_pairs.jsonl", [
        _pair("Banana", "p1", "n1", False),
        _pair("Banana", "p2", "n2", False),
        _pair("MustardBottle", "p1", "n1", True),
    ])
    dataset = loader.load_dataset(
        EVAL_OUTPUTS_DIR, mode=loader.LoadMode.PAIR_FIXTURES,
        pair_fixture_paths={"base": a, "nodist": b},
    )
    assert dataset.mode is loader.LoadMode.PAIR_FIXTURES
    assert set(dataset.aggregate_metrics.keys()) == set(loader.CORE_CHECKPOINT_NAMES)
    assert set(dataset.pair_records.keys()) == {"base", "nodist"}

    aligned = alignment.align_pairs(dataset.pair_records)
    assert len(aligned) == 3

    mcnemar = lggsn_statistics.exact_mcnemar(aligned, checkpoint_a="base", checkpoint_b="nodist")
    wtl = lggsn_statistics.win_tie_loss_counts(aligned, checkpoint_a="base", checkpoint_b="nodist")
    bootstrap = lggsn_statistics.paired_bootstrap_ci(
        aligned, checkpoint_a="base", checkpoint_b="nodist", seed=7, n_resamples=100,
    )
    assert mcnemar.discordant_pairs == wtl.win + wtl.loss
    assert wtl.total == len(aligned)
    assert bootstrap.n_pairs == len(aligned)


# ── Phase 1 never infers pair-level observations from aggregate metrics ────

def test_aggregate_only_dataset_never_carries_pair_records():
    dataset = loader.load_dataset(EVAL_OUTPUTS_DIR, mode=loader.LoadMode.AGGREGATE_ONLY)
    assert dataset.pair_records == {}
    # nothing derivable from aggregate_metrics can stand in for pair_records:
    # align_pairs always fails closed on an empty/missing checkpoint mapping.
    with pytest.raises(alignment.AlignmentError):
        alignment.align_pairs(dataset.pair_records)
