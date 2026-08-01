"""Deterministic LGGSN checkpoint statistical-analysis suite.

Phase 1 only: the statistical core (loader, alignment, statistics). Plots,
LaTeX/publication tables, full report generation, and real pair_results.jsonl
analysis are out of scope until a later phase -- see
experiments/lggsn_statistical_analysis.yaml for the current acceptance
criteria.
"""
from research_agent_pilots.lggsn_analysis.alignment import (
    AlignedPair,
    AlignmentError,
    align_pairs,
)
from research_agent_pilots.lggsn_analysis.loader import (
    CORE_CHECKPOINT_NAMES,
    LoadedDataset,
    LoaderError,
    LoadMode,
    PairDataResult,
    PairDataStatus,
    PairRecord,
    load_aggregate_metrics,
    load_all_aggregate_metrics,
    load_dataset,
    load_matrix_summary,
    load_pair_fixture_jsonl,
    load_real_pair_results,
)
from research_agent_pilots.lggsn_analysis.statistics import (
    McNemarResult,
    PairedBootstrapResult,
    StatisticsError,
    WinTieLossResult,
    exact_mcnemar,
    paired_bootstrap_ci,
    win_tie_loss_counts,
)

__all__ = [
    "CORE_CHECKPOINT_NAMES",
    "LoadMode",
    "LoaderError",
    "LoadedDataset",
    "PairDataResult",
    "PairDataStatus",
    "PairRecord",
    "load_aggregate_metrics",
    "load_all_aggregate_metrics",
    "load_dataset",
    "load_matrix_summary",
    "load_pair_fixture_jsonl",
    "load_real_pair_results",
    "AlignedPair",
    "AlignmentError",
    "align_pairs",
    "McNemarResult",
    "PairedBootstrapResult",
    "StatisticsError",
    "WinTieLossResult",
    "exact_mcnemar",
    "paired_bootstrap_ci",
    "win_tie_loss_counts",
]
