# -*- coding: utf-8 -*-
"""Tests for research_agent_pilots/lggsn_analysis/ (loader.py, alignment.py,
statistics.py) and, from Phase 2 on, the committed real pair-level data
under research_agent_pilots/lggsn_analysis/pair_results/. Pure stdlib -- no
torch/numpy required.

Phase 1 pair-level JSONL fixtures are written on the fly under tmp_path
rather than committed as separate files. Phase 2's pair_results.jsonl files
are different: they are real, already-regenerated-and-verified data (see
each checkpoint's provenance.json), so they are read directly from
research_agent_pilots/lggsn_analysis/pair_results/ instead. Phase 3 (see the
final section of this file) computes and reports the actual paired
statistical conclusions from that real data via real_analysis.py/
reporting.py -- those tests exercise the real, committed input files, never
synthetic replacements, except where a test is specifically about an edge
case (zero-discordance, missing/non-finite scores) that the real data does
not happen to contain.
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_agent_pilots.lggsn_analysis import alignment, loader, real_analysis, reporting
from research_agent_pilots.lggsn_analysis import statistics as lggsn_statistics

EVAL_OUTPUTS_DIR = REPO_ROOT / "research_agent_pilots" / "lggsn_suite" / "eval_outputs"
PAIR_RESULTS_DIR = REPO_ROOT / "research_agent_pilots" / "lggsn_analysis" / "pair_results"


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


# ── Phase 2: real, committed pair_results.jsonl ─────────────────────────────
#
# These load the genuine data under research_agent_pilots/lggsn_analysis/
# pair_results/ (regenerated by re-running research_agent_pilots/lggsn_suite
# /evaluator.py against the committed checkpoints/dataset; see each
# checkpoint's provenance.json). They only confirm the data is well-formed
# and aligns -- computing an actual McNemar/bootstrap/win-tie-loss result
# from it is explicitly out of scope until a later phase.

def test_real_pair_results_directory_has_all_four_core_checkpoints():
    assert {p.name for p in PAIR_RESULTS_DIR.iterdir() if p.is_dir()} == set(loader.CORE_CHECKPOINT_NAMES)


@pytest.mark.parametrize("checkpoint_name", loader.CORE_CHECKPOINT_NAMES)
def test_real_pair_results_jsonl_loads_for_every_core_checkpoint(checkpoint_name):
    path = PAIR_RESULTS_DIR / checkpoint_name / "pair_results.jsonl"
    records = loader.load_pair_fixture_jsonl(path)
    assert len(records) == 582
    assert all(isinstance(r.correct, bool) for r in records)


@pytest.mark.parametrize("checkpoint_name", loader.CORE_CHECKPOINT_NAMES)
def test_real_pair_results_provenance_confirms_committed_digest_match(checkpoint_name):
    provenance = json.loads((PAIR_RESULTS_DIR / checkpoint_name / "provenance.json").read_text())
    assert provenance["checkpoint_name"] == checkpoint_name
    assert provenance["matched_committed_digest"] is True
    assert (
        provenance["regenerated_metrics_deterministic_digest"]
        == provenance["committed_metrics_deterministic_digest"]
    )
    committed_metrics = json.loads((EVAL_OUTPUTS_DIR / checkpoint_name / "metrics.json").read_text())
    assert provenance["committed_metrics_deterministic_digest"] == committed_metrics["deterministic_digest"]


def test_real_pair_results_align_across_all_four_core_checkpoints():
    pair_records = {
        name: loader.load_pair_fixture_jsonl(PAIR_RESULTS_DIR / name / "pair_results.jsonl")
        for name in loader.CORE_CHECKPOINT_NAMES
    }
    aligned = alignment.align_pairs(pair_records)
    assert len(aligned) == 582
    assert all(set(p.correct_by_checkpoint.keys()) == set(loader.CORE_CHECKPOINT_NAMES) for p in aligned)


# ── Phase 3: real paired statistical conclusions ────────────────────────────
#
# real_analysis.py/reporting.py compute and format the actual McNemar /
# clustered bootstrap / win-tie-loss / Holm-Bonferroni results across the
# five planned comparisons, from the real pair_results.jsonl files only.

@pytest.fixture(scope="module")
def real_run():
    return real_analysis.run_analysis(REPO_ROOT)


# 1. all 4 real inputs align to exactly 582 identities

def test_phase3_all_four_real_inputs_align_to_exactly_582_identities(real_run):
    assert len(real_run.aligned_pairs) == real_analysis.EXPECTED_ALIGNED_PAIR_COUNT
    for name in loader.CORE_CHECKPOINT_NAMES:
        assert len(real_run.pair_records[name]) == real_analysis.EXPECTED_ALIGNED_PAIR_COUNT


# 2. every comparison uses the same identity set

def test_phase3_every_comparison_uses_the_same_582_identities(real_run):
    reference_keys = {p.pair_key for p in real_run.aligned_pairs}
    assert len(reference_keys) == real_analysis.EXPECTED_ALIGNED_PAIR_COUNT
    assert len(real_run.comparisons) == len(real_analysis.PLANNED_COMPARISONS)
    for comparison in real_run.comparisons:
        assert comparison["n_pairs"] == len(reference_keys)
        # every comparison's win+tie+loss covers exactly the same identity count
        assert comparison["win"] + comparison["tie"] + comparison["loss"] == len(reference_keys)


def test_phase3_planned_comparisons_match_requested_five():
    assert real_analysis.PLANNED_COMPARISONS == (
        ("base", "nodist"),
        ("base", "nozrel"),
        ("base", "full_v2"),
        ("nodist", "full_v2"),
        ("nozrel", "full_v2"),
    )


# 3. hand-checked McNemar fixture (Phase-3-scoped; the underlying exact_mcnemar
#    function itself already has Phase 1 fixture coverage above)

def test_phase3_mcnemar_hand_checked_fixture():
    aligned = alignment.align_pairs({
        "a": (
            loader.PairRecord("q", "p1", "n1", True), loader.PairRecord("q", "p2", "n2", True),
            loader.PairRecord("q", "p3", "n3", False), loader.PairRecord("q", "p4", "n4", False),
            loader.PairRecord("q", "p5", "n5", False),
        ),
        "b": (
            loader.PairRecord("q", "p1", "n1", True), loader.PairRecord("q", "p2", "n2", False),
            loader.PairRecord("q", "p3", "n3", True), loader.PairRecord("q", "p4", "n4", True),
            loader.PairRecord("q", "p5", "n5", False),
        ),
    })
    result = lggsn_statistics.exact_mcnemar(aligned, checkpoint_a="a", checkpoint_b="b")
    # n01 (a wrong, b correct) = 2: p3, p4; n10 (a correct, b wrong) = 1: p2
    assert result.n01 == 2
    assert result.n10 == 1
    # exact binomial(n=3, p=0.5), k=min(2,1)=1: 2*(C(3,0)+C(3,1))/8 = 2*4/8 = 1.0
    assert result.p_value == pytest.approx(1.0, abs=1e-12)


# 4. Holm-Bonferroni fixture

def test_phase3_holm_bonferroni_hand_checked_fixture():
    # p=[0.5, 0.01], m=2: sorted (0.01 rank0, 0.5 rank1).
    # rank0: min(1, 2*0.01)=0.02, running_max=0.02
    # rank1: min(1, 1*0.5)=0.5, running_max=max(0.02,0.5)=0.5
    adjusted = lggsn_statistics.holm_bonferroni_adjust([0.5, 0.01])
    assert adjusted == pytest.approx([0.5, 0.02])


def test_phase3_holm_bonferroni_enforces_monotonicity():
    # p=[0.01, 0.02, 0.03, 0.04, 0.05], m=5, all ascending already.
    # raw: [5*0.01, 4*0.02, 3*0.03, 2*0.04, 1*0.05] = [0.05, 0.08, 0.09, 0.08, 0.05]
    # running max enforces non-decreasing: [0.05, 0.08, 0.09, 0.09, 0.09]
    adjusted = lggsn_statistics.holm_bonferroni_adjust([0.01, 0.02, 0.03, 0.04, 0.05])
    assert adjusted == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])
    assert adjusted == sorted(adjusted)  # monotone non-decreasing in original (already-sorted) order


def test_phase3_holm_bonferroni_rejects_empty():
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.holm_bonferroni_adjust([])


def test_phase3_holm_bonferroni_rejects_out_of_range_p_value():
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.holm_bonferroni_adjust([0.5, 1.5])


def test_phase3_real_run_holm_adjusted_are_never_below_raw(real_run):
    # Holm-adjusted p-values can never be smaller than their raw counterpart.
    for comparison in real_run.comparisons:
        assert comparison["mcnemar_p_value_holm_adjusted"] >= comparison["mcnemar_p_value_raw"] - 1e-15


# 5. deterministic bootstrap repeat

def test_phase3_clustered_bootstrap_is_deterministic_for_a_fixed_seed():
    aligned = alignment.align_pairs({
        "a": tuple(loader.PairRecord(f"q{i % 3}", f"p{i}", f"n{i}", i % 2 == 0) for i in range(12)),
        "b": tuple(loader.PairRecord(f"q{i % 3}", f"p{i}", f"n{i}", i % 3 == 0) for i in range(12)),
    })
    kwargs = dict(
        checkpoint_a="a", checkpoint_b="b", cluster_key_fn=lambda p: p.query,
        resampling_unit="query", seed=999, n_resamples=500,
    )
    first = lggsn_statistics.paired_bootstrap_ci_clustered(aligned, **kwargs)
    second = lggsn_statistics.paired_bootstrap_ci_clustered(aligned, **kwargs)
    assert first == second


def test_phase3_clustered_bootstrap_requires_at_least_two_clusters():
    aligned = alignment.align_pairs({
        "a": (loader.PairRecord("only_query", "p1", "n1", True),),
        "b": (loader.PairRecord("only_query", "p1", "n1", False),),
    })
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.paired_bootstrap_ci_clustered(
            aligned, checkpoint_a="a", checkpoint_b="b", cluster_key_fn=lambda p: p.query,
            resampling_unit="query", seed=1,
        )


def test_phase3_full_pipeline_is_deterministic_across_repeated_runs():
    first = real_analysis.run_analysis(REPO_ROOT)
    second = real_analysis.run_analysis(REPO_ROOT)
    assert first.comparisons == second.comparisons


# 6. zero-discordance handling

def test_phase3_zero_discordance_is_not_significant():
    aligned = alignment.align_pairs({
        "a": (loader.PairRecord("q", "p1", "n1", True), loader.PairRecord("q", "p2", "n2", False)),
        "b": (loader.PairRecord("q", "p1", "n1", True), loader.PairRecord("q", "p2", "n2", False)),
    })
    mcnemar = lggsn_statistics.exact_mcnemar(aligned, checkpoint_a="a", checkpoint_b="b")
    assert mcnemar.discordant_pairs == 0
    assert mcnemar.p_value == 1.0
    assert real_analysis._interpretation(mcnemar.p_value, 0.0, alpha=0.05) == "NOT_SIGNIFICANT"


# 7. missing/nonfinite score handling

def test_phase3_score_margin_excludes_missing_and_nonfinite_scores(tmp_path):
    path = tmp_path / "pairs_with_bad_scores.jsonl"
    rows = [
        {"query": "q", "pos_row_id": "p1", "neg_row_id": "n1", "pos_score": 0.9, "neg_score": 0.1, "correct": True},
        {"query": "q", "pos_row_id": "p2", "neg_row_id": "n2", "pos_score": float("nan"), "neg_score": 0.1, "correct": True},
        {"query": "q", "pos_row_id": "p3", "neg_row_id": "n3", "pos_score": float("inf"), "neg_score": 0.1, "correct": True},
        {"query": "q", "pos_row_id": "p4", "neg_row_id": "n4", "neg_score": 0.1, "correct": True},  # pos_score missing
        {"query": "q", "pos_row_id": "p5", "neg_row_id": "n5", "pos_score": "not-a-number", "neg_score": 0.1, "correct": True},
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    margins = real_analysis._load_score_margins(path)
    assert margins[("q", "p1", "n1")] == pytest.approx(0.8)
    assert margins[("q", "p2", "n2")] is None
    assert margins[("q", "p3", "n3")] is None
    assert margins[("q", "p4", "n4")] is None
    assert margins[("q", "p5", "n5")] is None


def test_phase3_score_margin_summary_reports_exclusions_not_zeros():
    aligned = alignment.align_pairs({
        "a": (loader.PairRecord("q", "p1", "n1", True), loader.PairRecord("q", "p2", "n2", False)),
        "b": (loader.PairRecord("q", "p1", "n1", True), loader.PairRecord("q", "p2", "n2", False)),
    })
    margins_a = {("q", "p1", "n1"): 0.5, ("q", "p2", "n2"): None}
    margins_b = {("q", "p1", "n1"): 0.2, ("q", "p2", "n2"): 0.1}
    summary = real_analysis._score_margin_summary(aligned, margins_a, margins_b)
    assert summary["n_finite_pairs"] == 1
    assert summary["n_excluded"] == 1
    assert summary["mean_diff_b_minus_a"] == pytest.approx(0.2 - 0.5)
    assert summary["median_diff_b_minus_a"] == pytest.approx(0.2 - 0.5)


def test_phase3_score_margin_summary_all_excluded_gives_none_not_zero():
    aligned = alignment.align_pairs({
        "a": (loader.PairRecord("q", "p1", "n1", True),),
        "b": (loader.PairRecord("q", "p1", "n1", True),),
    })
    summary = real_analysis._score_margin_summary(aligned, {("q", "p1", "n1"): None}, {("q", "p1", "n1"): None})
    assert summary["n_finite_pairs"] == 0
    assert summary["n_excluded"] == 1
    assert summary["mean_diff_b_minus_a"] is None
    assert summary["median_diff_b_minus_a"] is None


def test_phase3_real_data_has_finite_scores_for_every_pair(real_run):
    # Documents the real data's actual state (not assumed): every comparison's
    # score-margin excludes zero pairs, since evaluator.py always writes a
    # finite float score for both sides of every pair.
    for comparison in real_run.comparisons:
        assert comparison["score_margin"]["n_excluded"] == 0
        assert comparison["score_margin"]["n_finite_pairs"] == comparison["n_pairs"]


# 8. query-level breakdown totals reconcile

def test_phase3_per_query_breakdown_totals_reconcile(real_run):
    for comparison in real_run.comparisons:
        breakdown = comparison["per_query_breakdown"]
        assert sum(row["n_pairs"] for row in breakdown) == comparison["n_pairs"]
        assert sum(row["win"] for row in breakdown) == comparison["win"]
        assert sum(row["tie"] for row in breakdown) == comparison["tie"]
        assert sum(row["loss"] for row in breakdown) == comparison["loss"]
        assert {row["query"] for row in breakdown} == {"Banana", "CrackerBox", "MustardBottle", "PowerDrill", "Scissors", "TomatoSoupCan"}


# 9. JSON and CSV values agree

def test_phase3_json_and_csv_outputs_agree(tmp_path, real_run):
    reporting.write_pairwise_comparisons_json(real_run.comparisons, tmp_path / "pairwise_comparisons.json")
    reporting.write_pairwise_comparisons_csv(real_run.comparisons, tmp_path / "pairwise_comparisons.csv")

    json_payload = json.loads((tmp_path / "pairwise_comparisons.json").read_text())["comparisons"]
    with (tmp_path / "pairwise_comparisons.csv").open(newline="") as f:
        csv_rows = list(csv.DictReader(f))

    assert len(json_payload) == len(csv_rows) == len(real_analysis.PLANNED_COMPARISONS)
    for json_row, csv_row in zip(json_payload, csv_rows):
        assert json_row["checkpoint_a"] == csv_row["checkpoint_a"]
        assert json_row["checkpoint_b"] == csv_row["checkpoint_b"]
        assert json_row["n_pairs"] == int(csv_row["n_pairs"])
        assert json_row["accuracy_a"] == pytest.approx(float(csv_row["accuracy_a"]))
        assert json_row["accuracy_b"] == pytest.approx(float(csv_row["accuracy_b"]))
        assert json_row["accuracy_diff_b_minus_a"] == pytest.approx(float(csv_row["accuracy_diff_b_minus_a"]))
        assert json_row["win"] == int(csv_row["win"])
        assert json_row["tie"] == int(csv_row["tie"])
        assert json_row["loss"] == int(csv_row["loss"])
        assert json_row["mcnemar_p_value_raw"] == pytest.approx(float(csv_row["mcnemar_p_value_raw"]))
        assert json_row["mcnemar_p_value_holm_adjusted"] == pytest.approx(float(csv_row["mcnemar_p_value_holm_adjusted"]))
        assert json_row["interpretation_raw"] == csv_row["interpretation_raw"]
        assert json_row["interpretation_holm"] == csv_row["interpretation_holm"]


def test_phase3_per_query_csv_row_count_matches_five_comparisons_times_six_queries(tmp_path, real_run):
    reporting.write_per_query_breakdown_csv(real_run.comparisons, tmp_path / "per_query_breakdown.csv")
    with (tmp_path / "per_query_breakdown.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(real_analysis.PLANNED_COMPARISONS) * 6


# 10. outputs are byte-deterministic across repeated runs

def test_phase3_main_outputs_are_byte_deterministic_across_repeated_runs(tmp_path):
    out_1 = tmp_path / "run1"
    out_2 = tmp_path / "run2"
    assert real_analysis.main(["--repo-root", str(REPO_ROOT), "--output-dir", str(out_1)]) == 0
    assert real_analysis.main(["--repo-root", str(REPO_ROOT), "--output-dir", str(out_2)]) == 0

    for filename in ("pairwise_comparisons.json", "pairwise_comparisons.csv", "per_query_breakdown.csv", "statistical_summary.md"):
        assert (out_1 / filename).read_bytes() == (out_2 / filename).read_bytes(), filename

    manifest_1 = json.loads((out_1 / "analysis_manifest.json").read_text())
    manifest_2 = json.loads((out_2 / "analysis_manifest.json").read_text())
    _non_deterministic_manifest_fields = {"command", "generated_at", "runtime_seconds", "started_at"}
    for key in set(manifest_1) - _non_deterministic_manifest_fields:
        assert manifest_1[key] == manifest_2[key], key


# 11. input files unchanged before/after

def test_phase3_input_files_unchanged_before_and_after_full_run(tmp_path):
    paths = [
        PAIR_RESULTS_DIR / name / fname
        for name in loader.CORE_CHECKPOINT_NAMES
        for fname in ("pair_results.jsonl", "provenance.json")
    ]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    real_analysis.main(["--repo-root", str(REPO_ROOT), "--output-dir", str(tmp_path / "out")])
    for p in paths:
        content, mtime = p.read_bytes(), p.stat().st_mtime_ns
        assert content == before[p][0], p
        assert mtime == before[p][1], p


# 12. no result is labeled significant unless the declared rule is satisfied

def test_phase3_interpretation_never_significant_at_or_above_alpha():
    assert real_analysis._interpretation(0.05, 1.0, alpha=0.05) == "NOT_SIGNIFICANT"
    assert real_analysis._interpretation(0.5, 1.0, alpha=0.05) == "NOT_SIGNIFICANT"
    assert real_analysis._interpretation(0.049999, 1.0, alpha=0.05) == "SIGNIFICANT_FAVORS_B"
    assert real_analysis._interpretation(0.049999, -1.0, alpha=0.05) == "SIGNIFICANT_FAVORS_A"


def test_phase3_real_run_significance_labels_match_declared_rule(real_run):
    alpha = real_analysis.ALPHA
    for comparison in real_run.comparisons:
        if comparison["interpretation_raw"] != "NOT_SIGNIFICANT":
            assert comparison["mcnemar_p_value_raw"] < alpha
            expected = "SIGNIFICANT_FAVORS_B" if comparison["accuracy_diff_b_minus_a"] > 0 else "SIGNIFICANT_FAVORS_A"
            assert comparison["interpretation_raw"] == expected
        else:
            assert comparison["mcnemar_p_value_raw"] >= alpha

        if comparison["interpretation_holm"] != "NOT_SIGNIFICANT":
            assert comparison["mcnemar_p_value_holm_adjusted"] < alpha
            expected = "SIGNIFICANT_FAVORS_B" if comparison["accuracy_diff_b_minus_a"] > 0 else "SIGNIFICANT_FAVORS_A"
            assert comparison["interpretation_holm"] == expected
        else:
            assert comparison["mcnemar_p_value_holm_adjusted"] >= alpha


def test_phase3_manifest_records_all_required_provenance_fields(real_run):
    manifest = real_analysis.build_manifest(
        repo_root=REPO_ROOT, run=real_run, started_at="2026-01-01T00:00:00+00:00",
        runtime_seconds=0.0, command=["python", "-m", "research_agent_pilots.lggsn_analysis.real_analysis"],
    )
    required_keys = {
        "git_commit", "input_file_sha256", "checkpoint_sha256", "dataset_sha256",
        "pair_identity_digest", "bootstrap_seed", "bootstrap_resampling_unit",
        "alpha", "multiple_comparison_method", "command", "runtime_seconds",
    }
    assert required_keys <= set(manifest.keys())
    assert set(manifest["input_file_sha256"].keys()) == set(loader.CORE_CHECKPOINT_NAMES)
    assert set(manifest["checkpoint_sha256"].keys()) == set(loader.CORE_CHECKPOINT_NAMES)
    assert len(manifest["dataset_sha256"]) == 64  # sha256 hex digest
    assert len(manifest["pair_identity_digest"]) == 64


# ── Phase 4: cluster-level (query-as-unit) sensitivity analysis ─────────────
#
# statistics.cluster_sign_flip_test: an exact cluster-level sign-flip
# permutation test, added alongside -- never replacing -- the existing
# pair-level exact McNemar test and query-cluster bootstrap CI. Explicitly
# NOT McNemar's test (see its own docstring for the exchangeability
# assumption it actually relies on).


def _make_query_pairs(query_diffs: dict[str, int]) -> tuple:
    """Build AlignedPairs from a {query: direction} map, 2 pairs per query.
    direction > 0 -> that query's pairs go a=wrong/wrong, b=correct/correct
    (accuracy_diff_b_minus_a = +1.0); direction < 0 -> the reverse
    (diff = -1.0); direction == 0 -> a=correct/wrong, b=correct/wrong
    (diff = 0.0). Used only to build fixtures with an exactly
    hand-computable per-query accuracy_diff_b_minus_a -- never used to
    represent real data."""
    a_by_q: dict[str, tuple] = {}
    b_by_q: dict[str, tuple] = {}
    for q, d in query_diffs.items():
        if d > 0:
            a_by_q[q], b_by_q[q] = (False, False), (True, True)
        elif d < 0:
            a_by_q[q], b_by_q[q] = (True, True), (False, False)
        else:
            a_by_q[q], b_by_q[q] = (True, False), (True, False)
    a_records = []
    b_records = []
    for q in query_diffs:
        for i, (av, bv) in enumerate(zip(a_by_q[q], b_by_q[q])):
            a_records.append(loader.PairRecord(q, f"{q}p{i}", f"{q}n{i}", av))
            b_records.append(loader.PairRecord(q, f"{q}p{i}", f"{q}n{i}", bv))
    return alignment.align_pairs({"a": tuple(a_records), "b": tuple(b_records)})


# 1. Hand-checkable 3-cluster fixture: all three query-level diffs are
# +1.0 (unanimous). Only the "all +1" and "all -1" sign assignments (2 of
# the 2**3=8 total) can reach |mean| == 1.0 -- every mixed assignment gives
# a strictly smaller magnitude -- so the exact two-sided p-value is
# provably 2/8 = 0.25, independent of the permutation-enumeration code
# under test (verified by direct hand enumeration in this docstring, not
# just by re-running the implementation on itself):
#   +++: (1+1+1)/3=+1.00  ++-: +1/3  +-+: +1/3  -++: +1/3
#   +--: -1/3  -+-: -1/3  --+: -1/3  ---: (-1-1-1)/3=-1.00
# -> exactly 2 of 8 reach |stat| >= 1.00.
def test_phase4_cluster_sign_flip_hand_checked_3_cluster_fixture():
    aligned = _make_query_pairs({"q1": 1, "q2": 1, "q3": 1})
    result = lggsn_statistics.cluster_sign_flip_test(
        aligned, checkpoint_a="a", checkpoint_b="b", cluster_key_fn=lambda p: p.query, resampling_unit="query",
    )
    assert result.n_clusters == 3
    assert result.n_permutations == 8
    assert result.n_favor_b == 3 and result.n_favor_a == 0 and result.n_tied == 0
    assert result.mean_cluster_diff == pytest.approx(1.0)
    assert result.median_cluster_diff == pytest.approx(1.0)
    assert result.observed_statistic == pytest.approx(1.0)
    assert result.p_value == pytest.approx(0.25, abs=1e-12)


# 2. Hand-checkable 3-cluster fixture with a mixed, non-extremal result
# (2 favor B, 1 favors A, all magnitude 1.0). Observed mean = (1+1-1)/3 =
# 1/3. Hand enumeration of all 8 signed means (same 8 listed above, but
# now the "base" pattern is [+1, +1, -1] with the third cluster's sign
# itself part of what gets flipped) gives |stat| in {1/3, 1/3, 1/3, 1/3,
# 1/3, 1/3, 1.0, 1.0} -- every one of the 8 assignments has |stat| >= 1/3,
# so p = 8/8 = 1.0 (a case worth covering precisely because it is *not*
# the maximally-extreme fixture, unlike test 1 above).
def test_phase4_cluster_sign_flip_hand_checked_3_cluster_mixed_fixture():
    aligned = _make_query_pairs({"q1": 1, "q2": 1, "q3": -1})
    result = lggsn_statistics.cluster_sign_flip_test(
        aligned, checkpoint_a="a", checkpoint_b="b", cluster_key_fn=lambda p: p.query, resampling_unit="query",
    )
    assert result.n_favor_b == 2 and result.n_favor_a == 1 and result.n_tied == 0
    assert result.mean_cluster_diff == pytest.approx(1.0 / 3.0)
    assert result.median_cluster_diff == pytest.approx(1.0)
    assert result.p_value == pytest.approx(1.0, abs=1e-12)


# 3. Hand-checkable 6-cluster fixture: all six query-level diffs are +1.0
# (unanimous). Same reasoning as test 1, generalized: for n equal-magnitude
# same-sign diffs, only the 2 unanimous assignments (all+, all-) reach the
# maximal |mean|; every assignment with k in [1, n-1] signs flipped gives
# |mean| = |n-2k|/n * magnitude, strictly less than the unanimous case.
# Exact p = 2/2**6 = 2/64 = 0.03125 -- also the coarsest possible p-value
# this test can ever report for 6 clusters, referenced in the power
# caveat text.
def test_phase4_cluster_sign_flip_hand_checked_6_cluster_fixture():
    aligned = _make_query_pairs({f"q{i}": 1 for i in range(6)})
    result = lggsn_statistics.cluster_sign_flip_test(
        aligned, checkpoint_a="a", checkpoint_b="b", cluster_key_fn=lambda p: p.query, resampling_unit="query",
    )
    assert result.n_clusters == 6
    assert result.n_permutations == 64
    assert result.n_favor_b == 6 and result.n_favor_a == 0
    assert result.p_value == pytest.approx(2 / 64, abs=1e-12)


# 4. Hand-checkable 6-cluster mixed fixture (4 favor B, 2 favor A, equal
# magnitude), cross-checked against an independent brute-force
# itertools.product reference implementation written directly in this
# test -- not just against the module under test re-running itself.
def test_phase4_cluster_sign_flip_6_cluster_mixed_matches_independent_brute_force():
    import itertools

    aligned = _make_query_pairs({"q1": 1, "q2": 1, "q3": 1, "q4": 1, "q5": -1, "q6": -1})
    result = lggsn_statistics.cluster_sign_flip_test(
        aligned, checkpoint_a="a", checkpoint_b="b", cluster_key_fn=lambda p: p.query, resampling_unit="query",
    )

    diffs = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
    observed = sum(diffs) / len(diffs)
    count = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        stat = sum(s * d for s, d in zip(signs, diffs)) / len(diffs)
        if abs(stat) >= abs(observed) - 1e-9:
            count += 1
    expected_p = count / (2 ** len(diffs))

    assert result.n_favor_b == 4 and result.n_favor_a == 2
    assert result.mean_cluster_diff == pytest.approx(observed)
    assert result.median_cluster_diff == pytest.approx(1.0)
    assert result.p_value == pytest.approx(expected_p, abs=1e-12)


def test_phase4_cluster_sign_flip_is_not_named_mcnemar():
    # Explicit requirement: this test must never be described/labeled as
    # McNemar's test anywhere in its own result type or docstring.
    assert "mcnemar" not in lggsn_statistics.ClusterSignFlipResult.__doc__.lower() if \
        lggsn_statistics.ClusterSignFlipResult.__doc__ else True
    assert "McNemar" not in lggsn_statistics.cluster_sign_flip_test.__name__


def test_phase4_cluster_sign_flip_requires_at_least_two_clusters():
    aligned = _make_query_pairs({"only_query": 1})
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.cluster_sign_flip_test(
            aligned, checkpoint_a="a", checkpoint_b="b", cluster_key_fn=lambda p: p.query, resampling_unit="query",
        )


def test_phase4_cluster_sign_flip_is_deterministic_no_seed_needed():
    # Exact enumeration: repeated calls with identical input must be
    # byte-identical without ever passing a seed (there is nothing random
    # to seed).
    aligned = _make_query_pairs({"q1": 1, "q2": -1, "q3": 1, "q4": 0})
    kwargs = dict(checkpoint_a="a", checkpoint_b="b", cluster_key_fn=lambda p: p.query, resampling_unit="query")
    first = lggsn_statistics.cluster_sign_flip_test(aligned, **kwargs)
    second = lggsn_statistics.cluster_sign_flip_test(aligned, **kwargs)
    assert first == second


def test_phase4_cluster_sign_flip_rejects_unknown_checkpoint():
    aligned = _make_query_pairs({"q1": 1, "q2": -1})
    with pytest.raises(lggsn_statistics.StatisticsError):
        lggsn_statistics.cluster_sign_flip_test(
            aligned, checkpoint_a="a", checkpoint_b="nonexistent",
            cluster_key_fn=lambda p: p.query, resampling_unit="query",
        )


# ── Phase 4: conclusion_category logic (real_analysis.py) ───────────────────

def test_phase4_conclusion_category_consistent_when_both_significant_and_agree():
    category = real_analysis._conclusion_category(
        pair_interpretation_raw="SIGNIFICANT_FAVORS_B", cluster_p_value=0.01,
        accuracy_diff_b_minus_a=0.2, mean_cluster_diff=0.15, alpha=0.05,
    )
    assert category == "CONSISTENT_ACROSS_CLUSTER_AND_PAIR_INFERENCE"


def test_phase4_conclusion_category_pair_level_only_when_cluster_not_significant():
    category = real_analysis._conclusion_category(
        pair_interpretation_raw="SIGNIFICANT_FAVORS_A", cluster_p_value=0.41,
        accuracy_diff_b_minus_a=-0.13, mean_cluster_diff=-0.11, alpha=0.05,
    )
    assert category == "PAIR_LEVEL_ONLY"


def test_phase4_conclusion_category_no_clear_difference_when_pair_not_significant():
    category = real_analysis._conclusion_category(
        pair_interpretation_raw="NOT_SIGNIFICANT", cluster_p_value=0.03,
        accuracy_diff_b_minus_a=0.01, mean_cluster_diff=0.2, alpha=0.05,
    )
    assert category == "NO_CLEAR_DIFFERENCE"


def test_phase4_conclusion_category_no_clear_difference_when_both_significant_but_disagree_direction():
    category = real_analysis._conclusion_category(
        pair_interpretation_raw="SIGNIFICANT_FAVORS_B", cluster_p_value=0.01,
        accuracy_diff_b_minus_a=0.2, mean_cluster_diff=-0.15, alpha=0.05,
    )
    assert category == "NO_CLEAR_DIFFERENCE"


# ── Phase 4: regression -- existing pair-level McNemar/bootstrap results ────
# are unaltered by this addition ─────────────────────────────────────────────

def test_phase4_does_not_alter_existing_mcnemar_hand_checked_fixture():
    # Byte-for-byte the same fixture and assertions as
    # test_phase3_mcnemar_hand_checked_fixture above, re-run after the
    # Phase 4 addition to confirm exact_mcnemar's own behavior/values are
    # untouched by anything added in this section.
    aligned = alignment.align_pairs({
        "a": (
            loader.PairRecord("q", "p1", "n1", True), loader.PairRecord("q", "p2", "n2", True),
            loader.PairRecord("q", "p3", "n3", False), loader.PairRecord("q", "p4", "n4", False),
            loader.PairRecord("q", "p5", "n5", False),
        ),
        "b": (
            loader.PairRecord("q", "p1", "n1", True), loader.PairRecord("q", "p2", "n2", False),
            loader.PairRecord("q", "p3", "n3", True), loader.PairRecord("q", "p4", "n4", True),
            loader.PairRecord("q", "p5", "n5", False),
        ),
    })
    result = lggsn_statistics.exact_mcnemar(aligned, checkpoint_a="a", checkpoint_b="b")
    assert result.n01 == 2
    assert result.n10 == 1
    assert result.p_value == pytest.approx(1.0, abs=1e-12)


def test_phase4_real_run_mcnemar_and_bootstrap_fields_match_standalone_calls(real_run):
    # For every real planned comparison, the mcnemar_p_value_raw and
    # bootstrap fields embedded in _build_comparison's output must equal
    # a fresh, independent standalone call to exact_mcnemar /
    # paired_bootstrap_ci_clustered on the same aligned_pairs -- proving
    # the Phase 4 addition did not change what those existing calls
    # compute or how their results are threaded into the comparison dict.
    for comparison in real_run.comparisons:
        a, b = comparison["checkpoint_a"], comparison["checkpoint_b"]
        direct_mcnemar = lggsn_statistics.exact_mcnemar(real_run.aligned_pairs, checkpoint_a=a, checkpoint_b=b)
        assert comparison["mcnemar_p_value_raw"] == direct_mcnemar.p_value
        assert comparison["discordant_a_correct_b_wrong"] == direct_mcnemar.n10
        assert comparison["discordant_a_wrong_b_correct"] == direct_mcnemar.n01

        direct_bootstrap = lggsn_statistics.paired_bootstrap_ci_clustered(
            real_run.aligned_pairs, checkpoint_a=a, checkpoint_b=b, cluster_key_fn=lambda p: p.query,
            resampling_unit=real_analysis.BOOTSTRAP_RESAMPLING_UNIT, seed=real_analysis.BOOTSTRAP_SEED,
            n_resamples=real_analysis.BOOTSTRAP_N_RESAMPLES, confidence=real_analysis.BOOTSTRAP_CONFIDENCE,
        )
        assert comparison["bootstrap"]["ci_lower"] == direct_bootstrap.ci_lower
        assert comparison["bootstrap"]["ci_upper"] == direct_bootstrap.ci_upper
        assert comparison["bootstrap"]["observed_diff_b_minus_a"] == direct_bootstrap.observed_diff_b_minus_a


# ── Phase 4: real-data integration ───────────────────────────────────────────

def test_phase4_real_run_cluster_sign_flip_has_six_query_clusters(real_run):
    for comparison in real_run.comparisons:
        cluster_sign_flip = comparison["cluster_sign_flip"]
        assert cluster_sign_flip["n_clusters"] == 6
        assert cluster_sign_flip["n_permutations"] == 64
        assert len(cluster_sign_flip["cluster_diffs"]) == 6
        assert cluster_sign_flip["n_favor_a"] + cluster_sign_flip["n_favor_b"] + cluster_sign_flip["n_tied"] == 6
        assert 0.0 <= cluster_sign_flip["p_value"] <= 1.0


def test_phase4_real_run_conclusion_category_is_one_of_three_known_values(real_run):
    known = {
        "CONSISTENT_ACROSS_CLUSTER_AND_PAIR_INFERENCE", "PAIR_LEVEL_ONLY", "NO_CLEAR_DIFFERENCE",
    }
    for comparison in real_run.comparisons:
        assert comparison["conclusion_category"] in known


def test_phase4_real_run_primary_goal_comparisons_present_with_both_evidence_lines(real_run):
    # Primary goal (per the task spec): base-vs-nodist and nodist-vs-full_v2
    # must both be assessable through cluster-level inference, not just
    # pair-level.
    by_pair = {(c["checkpoint_a"], c["checkpoint_b"]): c for c in real_run.comparisons}
    for key in [("base", "nodist"), ("nodist", "full_v2")]:
        assert key in by_pair
        comparison = by_pair[key]
        assert comparison["cluster_sign_flip"]["n_clusters"] == 6
        assert comparison["conclusion_category"] in {
            "CONSISTENT_ACROSS_CLUSTER_AND_PAIR_INFERENCE", "PAIR_LEVEL_ONLY", "NO_CLEAR_DIFFERENCE",
        }


def test_phase4_evidence_columns_field_lists_exactly_three_names(real_run):
    for comparison in real_run.comparisons:
        assert comparison["evidence_columns"] == [
            "pair_level_mcnemar", "query_cluster_permutation", "query_cluster_bootstrap_ci",
        ]


# ── Phase 4: JSON/CSV agreement for the new fields ───────────────────────────

def test_phase4_json_and_csv_outputs_agree_for_cluster_permutation_fields(tmp_path, real_run):
    reporting.write_pairwise_comparisons_json(real_run.comparisons, tmp_path / "pairwise_comparisons.json")
    reporting.write_pairwise_comparisons_csv(real_run.comparisons, tmp_path / "pairwise_comparisons.csv")

    json_payload = json.loads((tmp_path / "pairwise_comparisons.json").read_text())["comparisons"]
    with (tmp_path / "pairwise_comparisons.csv").open(newline="") as f:
        csv_rows = list(csv.DictReader(f))

    for json_row, csv_row in zip(json_payload, csv_rows):
        cluster_sign_flip = json_row["cluster_sign_flip"]
        assert cluster_sign_flip["resampling_unit"] == csv_row["cluster_permutation_resampling_unit"]
        assert cluster_sign_flip["n_clusters"] == int(csv_row["cluster_permutation_n_clusters"])
        assert cluster_sign_flip["n_favor_a"] == int(csv_row["cluster_permutation_n_favor_a"])
        assert cluster_sign_flip["n_favor_b"] == int(csv_row["cluster_permutation_n_favor_b"])
        assert cluster_sign_flip["n_tied"] == int(csv_row["cluster_permutation_n_tied"])
        assert cluster_sign_flip["mean_cluster_diff"] == pytest.approx(float(csv_row["cluster_permutation_mean_diff"]))
        assert cluster_sign_flip["median_cluster_diff"] == pytest.approx(float(csv_row["cluster_permutation_median_diff"]))
        assert cluster_sign_flip["n_permutations"] == int(csv_row["cluster_permutation_n_permutations"])
        assert cluster_sign_flip["p_value"] == pytest.approx(float(csv_row["cluster_permutation_p_value"]))
        assert json_row["conclusion_category"] == csv_row["conclusion_category"]


def test_phase4_markdown_summary_mentions_three_evidence_columns_and_not_mcnemar_for_cluster_test(real_run):
    manifest = real_analysis.build_manifest(
        repo_root=REPO_ROOT, run=real_run, started_at="2026-01-01T00:00:00+00:00",
        runtime_seconds=0.0, command=["python", "-m", "research_agent_pilots.lggsn_analysis.real_analysis"],
    )
    md = reporting.render_statistical_summary_markdown(real_run.comparisons, manifest)
    assert "pair_level_mcnemar" in md
    assert "query_cluster_permutation" in md
    assert "query_cluster_bootstrap_ci" in md
    assert "NOT McNemar" in md
    assert "CONSISTENT_ACROSS_CLUSTER_AND_PAIR_INFERENCE" in md or "PAIR_LEVEL_ONLY" in md or "NO_CLEAR_DIFFERENCE" in md
