"""Deterministic LGGSN checkpoint statistical-analysis suite.

Phase 1: the statistical core (loader, alignment, statistics), exercised
only against aggregate metrics and hand-authored pair-level fixtures.
Phase 2: real, provenance-pinned pair_results.jsonl regenerated for the
four core checkpoints, under pair_results/. Phase 3: real_analysis.py /
reporting.py compute and report the actual paired statistical conclusions
across the LGGSN core matrix from that real data -- see
experiments/lggsn_statistical_analysis.yaml for the current acceptance
criteria and research_agent_pilots/lggsn_analysis/outputs/ for the
generated results. Plots, LaTeX/publication tables, and any conversion of
pair accuracy into a grasp success rate remain out of scope.
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
    ClusteredPairedBootstrapResult,
    McNemarResult,
    PairedBootstrapResult,
    StatisticsError,
    WinTieLossResult,
    exact_mcnemar,
    holm_bonferroni_adjust,
    pair_accuracy,
    paired_bootstrap_ci,
    paired_bootstrap_ci_clustered,
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
    "ClusteredPairedBootstrapResult",
    "McNemarResult",
    "PairedBootstrapResult",
    "StatisticsError",
    "WinTieLossResult",
    "exact_mcnemar",
    "holm_bonferroni_adjust",
    "pair_accuracy",
    "paired_bootstrap_ci",
    "paired_bootstrap_ci_clustered",
    "win_tie_loss_counts",
]
