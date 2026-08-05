#!/usr/bin/env python
"""Phase 3: real paired statistical conclusions for the LGGSN core matrix.

Reads ONLY the real, provenance-pinned, already-regenerated-and-verified
pair-level data committed under
research_agent_pilots/lggsn_analysis/pair_results/{base,nodist,nozrel,
full_v2}/pair_results.jsonl (see each checkpoint's sibling provenance.json
for how it was produced and verified -- Phase 2). Never reads
research_agent_pilots/lggsn_suite/eval_outputs/, grasp_6dof/models/, or any
dataset/checkpoint file directly, and never retrains or regenerates a
checkpoint.

All five planned comparisons (base-vs-nodist, base-vs-nozrel,
base-vs-full_v2, nodist-vs-full_v2, nozrel-vs-full_v2) are computed from
one single alignment.align_pairs(...) call over all four checkpoints at
once, so every comparison below uses the exact same 582 aligned pair
identities -- never a different subset per comparison.

`correct` (per pair, per checkpoint) is loaded via loader.load_pair_fixture_
jsonl, which already fails closed (LoaderError) if any record's `correct`
field is missing or not a plain boolean -- so by the time exact_mcnemar
below ever runs, binary correctness has always been exactly, unambiguously
reconstructed. If it could not have been, this module would already have
raised before reaching the statistics.

Score-margin (pos_score - neg_score) handling is intentionally separate
from loader.PairRecord (which only carries `correct`, by design -- see
loader.py's module docstring): this module reads pos_score/neg_score
directly from the same committed JSONL files, and any pair with a missing
or non-finite score on either checkpoint is excluded from the score-margin
statistic (never coerced to zero or dropped from n_pairs itself) with the
exclusion count reported explicitly.

Bootstrap resampling unit: see statistics.paired_bootstrap_ci_clustered's
docstring -- pairs are clustered by `query` (the finest unit reconstructable
from pair_results.jsonl's own columns, since LGGSN pairs are a cartesian
product of same-query episode pairs and are therefore correlated within a
query; episode/scene_id identity is not present in this file at all).

Nothing here converts pair_accuracy into a grasp success rate, and nothing
here claims causal feature importance from a single checkpoint per ablation
-- see reporting.py's generated "What these results do and do not prove"
section for the explicit, non-silent caveats.

Phase 4 addition: a cluster-as-independent-unit sensitivity analysis
(statistics.cluster_sign_flip_test), added alongside -- never replacing --
the existing pair-level exact McNemar test and query-cluster bootstrap CI.
Motivation: the pair-level McNemar test treats each of the 582 aligned
pairs as an independent Bernoulli trial, which understates uncertainty
given LGGSN pairs are correlated within a query (see the bootstrap
resampling-unit note above); this addition asks the same base-vs-nodist
and nodist-vs-full_v2 question again with `query` (6 clusters) as the unit
of analysis instead, using an exact sign-flip permutation test rather than
a bootstrap. All three lines of evidence -- pair_level_mcnemar,
query_cluster_permutation, query_cluster_bootstrap_ci -- are reported
separately for every comparison and are never collapsed into one
significance label; `_conclusion_category` below is a conservative,
explicitly-labeled combined read that still keeps all three visible
alongside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_agent_pilots.lggsn_analysis import alignment, loader, reporting
from research_agent_pilots.lggsn_analysis import statistics as lggsn_statistics

CORE_CHECKPOINT_NAMES: tuple[str, ...] = loader.CORE_CHECKPOINT_NAMES

# The five comparisons the user asked for, in report order. Every
# comparison is (checkpoint_a, checkpoint_b); accuracy_diff and all other
# directional quantities below are always checkpoint_b minus checkpoint_a.
PLANNED_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("base", "nodist"),
    ("base", "nozrel"),
    ("base", "full_v2"),
    ("nodist", "full_v2"),
    ("nozrel", "full_v2"),
)

EXPECTED_ALIGNED_PAIR_COUNT = 582

# Predeclared and fixed for this analysis -- never changed post-hoc based
# on what the computed results turn out to be (safeguard: "predeclared
# alpha=0.05 without post-hoc threshold changes").
ALPHA = 0.05
MULTIPLE_COMPARISON_METHOD = "holm-bonferroni"
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_N_RESAMPLES = 10000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLING_UNIT = "query"

# Phase 4: cluster-level (query-as-unit) sensitivity analysis, added
# alongside the existing pair-level McNemar and query-cluster bootstrap CI
# -- never replacing either. See statistics.cluster_sign_flip_test's
# docstring for the exact method and its exchangeability assumption; it is
# an exact enumeration (2**n_clusters), so there is no seed to declare.
CLUSTER_PERMUTATION_RESAMPLING_UNIT = "query"

PAIR_RESULTS_RELATIVE_DIR = Path("research_agent_pilots") / "lggsn_analysis" / "pair_results"


@dataclass(frozen=True)
class AnalysisRun:
    comparisons: list[dict[str, Any]]
    aligned_pairs: tuple[alignment.AlignedPair, ...]
    pair_records: dict[str, tuple[loader.PairRecord, ...]]


def _pair_results_path(repo_root: Path, checkpoint_name: str) -> Path:
    return repo_root / PAIR_RESULTS_RELATIVE_DIR / checkpoint_name / "pair_results.jsonl"


def _provenance_path(repo_root: Path, checkpoint_name: str) -> Path:
    return repo_root / PAIR_RESULTS_RELATIVE_DIR / checkpoint_name / "provenance.json"


def load_core_pair_records(repo_root: Path) -> dict[str, tuple[loader.PairRecord, ...]]:
    """Load all four core checkpoints' real pair_results.jsonl via the
    existing Phase 1 loader (fail-closed on any structural problem).
    Raises if any checkpoint does not have exactly the expected pair
    count -- the analysis only ever proceeds against a fully-formed,
    already-verified core matrix, never a partial one."""
    records: dict[str, tuple[loader.PairRecord, ...]] = {}
    for name in CORE_CHECKPOINT_NAMES:
        path = _pair_results_path(repo_root, name)
        checkpoint_records = loader.load_pair_fixture_jsonl(path)
        if len(checkpoint_records) != EXPECTED_ALIGNED_PAIR_COUNT:
            raise lggsn_statistics.StatisticsError(
                f"checkpoint {name!r}: expected exactly {EXPECTED_ALIGNED_PAIR_COUNT} pair "
                f"records, got {len(checkpoint_records)} ({path})"
            )
        records[name] = checkpoint_records
    return records


def _load_score_margins(path: Path) -> dict[tuple[str, str, str], float | None]:
    """Best-effort pos_score-neg_score margin per pair identity, read
    directly from pair_results.jsonl (independent of loader.PairRecord,
    which intentionally carries `correct` only -- see this module's
    docstring). None means the pair's margin is unavailable for this
    checkpoint (missing key, non-numeric, or non-finite) -- callers must
    count and report these exclusions explicitly, never silently drop them
    from n_pairs or treat them as zero."""
    margins: dict[tuple[str, str, str], float | None] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (str(obj["query"]), str(obj["pos_row_id"]), str(obj["neg_row_id"]))
            pos_score = obj.get("pos_score")
            neg_score = obj.get("neg_score")
            margin = None
            if (
                isinstance(pos_score, (int, float)) and not isinstance(pos_score, bool)
                and isinstance(neg_score, (int, float)) and not isinstance(neg_score, bool)
            ):
                pos_f, neg_f = float(pos_score), float(neg_score)
                if math.isfinite(pos_f) and math.isfinite(neg_f):
                    margin = pos_f - neg_f
            margins[key] = margin
    return margins


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _score_margin_summary(
    aligned_pairs: Sequence[alignment.AlignedPair],
    margins_a: Mapping[tuple[str, str, str], float | None],
    margins_b: Mapping[tuple[str, str, str], float | None],
) -> dict[str, Any]:
    diffs: list[float] = []
    n_excluded = 0
    for pair in aligned_pairs:
        margin_a = margins_a.get(pair.pair_key)
        margin_b = margins_b.get(pair.pair_key)
        if margin_a is None or margin_b is None:
            n_excluded += 1
            continue
        diffs.append(margin_b - margin_a)
    return {
        "n_finite_pairs": len(diffs),
        "n_excluded": n_excluded,
        "mean_diff_b_minus_a": _mean(diffs) if diffs else None,
        "median_diff_b_minus_a": _median(diffs) if diffs else None,
    }


def _per_query_breakdown(
    aligned_pairs: Sequence[alignment.AlignedPair], *, checkpoint_a: str, checkpoint_b: str
) -> list[dict[str, Any]]:
    by_query: dict[str, list[alignment.AlignedPair]] = {}
    for pair in aligned_pairs:
        by_query.setdefault(pair.query, []).append(pair)

    rows: list[dict[str, Any]] = []
    for query in sorted(by_query):
        pairs = by_query[query]
        accuracy_a = lggsn_statistics.pair_accuracy(pairs, checkpoint=checkpoint_a)
        accuracy_b = lggsn_statistics.pair_accuracy(pairs, checkpoint=checkpoint_b)
        wtl = lggsn_statistics.win_tie_loss_counts(pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)
        rows.append({
            "query": query,
            "n_pairs": len(pairs),
            "accuracy_a": accuracy_a,
            "accuracy_b": accuracy_b,
            "accuracy_diff_b_minus_a": accuracy_b - accuracy_a,
            "win": wtl.win, "tie": wtl.tie, "loss": wtl.loss,
        })
    return rows


def _interpretation(p_value: float, accuracy_diff_b_minus_a: float, *, alpha: float) -> str:
    if p_value >= alpha:
        return "NOT_SIGNIFICANT"
    if accuracy_diff_b_minus_a > 0:
        return "SIGNIFICANT_FAVORS_B"
    if accuracy_diff_b_minus_a < 0:
        return "SIGNIFICANT_FAVORS_A"
    return "NOT_SIGNIFICANT"  # p < alpha with a zero observed difference cannot occur under exact McNemar


def _conclusion_category(
    *, pair_interpretation_raw: str, cluster_p_value: float,
    accuracy_diff_b_minus_a: float, mean_cluster_diff: float, alpha: float,
) -> str:
    """Conservative combined read across the two independent significance
    tests (pair-level exact McNemar, raw/unadjusted; cluster-level exact
    sign-flip permutation, see statistics.cluster_sign_flip_test) -- never
    a replacement for either, always reported alongside both in full (see
    the three separate evidence columns in reporting.py). Uses the RAW,
    not Holm-adjusted, pair-level p-value: Holm-Bonferroni is a family-wise
    correction across the five planned comparisons, a separate concern
    from this per-comparison cluster-vs-pair agreement check.

    CONSISTENT_ACROSS_CLUSTER_AND_PAIR_INFERENCE: both tests reject the
      null at `alpha`, AND they agree on direction (the pooled pair-level
      accuracy_diff_b_minus_a and the unweighted mean_cluster_diff have
      the same sign) -- the strongest defensible read.
    PAIR_LEVEL_ONLY: the pair-level test rejects but the cluster-level
      test does not -- exactly the pattern real within-cluster correlation
      would produce (the pair-level test overstates confidence because it
      assumes independence the data does not have; see
      statistics.paired_bootstrap_ci_clustered's docstring for why LGGSN
      pairs are correlated within a query).
    NO_CLEAR_DIFFERENCE: everything else -- including the pair-level test
      failing to reject (regardless of the cluster-level result), and the
      edge case where both tests reject but disagree on direction (never
      labeled "consistent" even though both are individually
      significant)."""
    pair_significant = pair_interpretation_raw != "NOT_SIGNIFICANT"
    cluster_significant = cluster_p_value < alpha
    same_direction = (accuracy_diff_b_minus_a > 0) == (mean_cluster_diff > 0)

    if pair_significant and cluster_significant and same_direction:
        return "CONSISTENT_ACROSS_CLUSTER_AND_PAIR_INFERENCE"
    if pair_significant and not cluster_significant:
        return "PAIR_LEVEL_ONLY"
    return "NO_CLEAR_DIFFERENCE"


def _build_comparison(
    aligned_pairs: Sequence[alignment.AlignedPair],
    *,
    checkpoint_a: str,
    checkpoint_b: str,
    margins: Mapping[str, Mapping[tuple[str, str, str], float | None]],
    alpha: float,
) -> dict[str, Any]:
    accuracy_a = lggsn_statistics.pair_accuracy(aligned_pairs, checkpoint=checkpoint_a)
    accuracy_b = lggsn_statistics.pair_accuracy(aligned_pairs, checkpoint=checkpoint_b)
    accuracy_diff = accuracy_b - accuracy_a

    wtl = lggsn_statistics.win_tie_loss_counts(aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)
    mcnemar = lggsn_statistics.exact_mcnemar(aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)
    bootstrap = lggsn_statistics.paired_bootstrap_ci_clustered(
        aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b,
        cluster_key_fn=lambda p: p.query, resampling_unit=BOOTSTRAP_RESAMPLING_UNIT,
        seed=BOOTSTRAP_SEED, n_resamples=BOOTSTRAP_N_RESAMPLES, confidence=BOOTSTRAP_CONFIDENCE,
    )
    cluster_sign_flip = lggsn_statistics.cluster_sign_flip_test(
        aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b,
        cluster_key_fn=lambda p: p.query, resampling_unit=CLUSTER_PERMUTATION_RESAMPLING_UNIT,
    )
    score_margin = _score_margin_summary(aligned_pairs, margins[checkpoint_a], margins[checkpoint_b])
    per_query = _per_query_breakdown(aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)

    interpretation_raw = _interpretation(mcnemar.p_value, accuracy_diff, alpha=alpha)
    conclusion_category = _conclusion_category(
        pair_interpretation_raw=interpretation_raw,
        cluster_p_value=cluster_sign_flip.p_value,
        accuracy_diff_b_minus_a=accuracy_diff,
        mean_cluster_diff=cluster_sign_flip.mean_cluster_diff,
        alpha=alpha,
    )

    return {
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "n_pairs": len(aligned_pairs),
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "accuracy_diff_b_minus_a": accuracy_diff,
        "win": wtl.win, "tie": wtl.tie, "loss": wtl.loss,
        "discordant_a_correct_b_wrong": mcnemar.n10,
        "discordant_a_wrong_b_correct": mcnemar.n01,
        "mcnemar_p_value_raw": mcnemar.p_value,
        # Filled in by run_analysis once all planned comparisons' raw
        # p-values are available (Holm-Bonferroni needs the whole family).
        "mcnemar_p_value_holm_adjusted": None,
        "interpretation_raw": interpretation_raw,
        "interpretation_holm": None,
        "bootstrap": {
            "seed": bootstrap.seed,
            "n_resamples": bootstrap.n_resamples,
            "resampling_unit": bootstrap.resampling_unit,
            "n_clusters": bootstrap.n_clusters,
            "confidence": bootstrap.confidence,
            "observed_diff_b_minus_a": bootstrap.observed_diff_b_minus_a,
            "ci_lower": bootstrap.ci_lower,
            "ci_upper": bootstrap.ci_upper,
        },
        "score_margin": score_margin,
        "per_query_breakdown": per_query,
        # Phase 4: cluster-as-unit sensitivity analysis (see
        # statistics.cluster_sign_flip_test) -- explicitly NOT McNemar, and
        # never combined with pair_level_mcnemar/query_cluster_bootstrap_ci
        # into a single number. All three evidence columns are reported in
        # full; `evidence_columns` below is only a display-order pointer,
        # never a computed/derived value.
        "cluster_sign_flip": {
            "resampling_unit": cluster_sign_flip.resampling_unit,
            "n_clusters": cluster_sign_flip.n_clusters,
            "cluster_diffs": [list(item) for item in cluster_sign_flip.cluster_diffs],
            "n_favor_a": cluster_sign_flip.n_favor_a,
            "n_favor_b": cluster_sign_flip.n_favor_b,
            "n_tied": cluster_sign_flip.n_tied,
            "mean_cluster_diff": cluster_sign_flip.mean_cluster_diff,
            "median_cluster_diff": cluster_sign_flip.median_cluster_diff,
            "n_permutations": cluster_sign_flip.n_permutations,
            "p_value": cluster_sign_flip.p_value,
            "exchangeability_assumption": (
                "Under the null of no true difference between the two checkpoints, the SIGN "
                "of each query's own accuracy_diff_b_minus_a is exchangeable (equally likely "
                "positive or negative), independently across queries. This does not assume "
                "pairs within a query are independent, and does not assume anything about the "
                "magnitude of each query's difference."
            ),
            "power_caveat": (
                f"Only {cluster_sign_flip.n_clusters} query clusters: p-values from this test "
                f"are discrete (coarsest possible two-sided p = 2/{cluster_sign_flip.n_permutations} "
                f"= {2 / cluster_sign_flip.n_permutations:.5f}) and power is low. Failing to reject "
                "the null here is not evidence of equivalence between the two checkpoints."
            ),
        },
        "evidence_columns": ["pair_level_mcnemar", "query_cluster_permutation", "query_cluster_bootstrap_ci"],
        "conclusion_category": conclusion_category,
    }


def _pair_identity_digest(aligned_pairs: Sequence[alignment.AlignedPair]) -> str:
    lines = sorted(f"{p.query}|{p.pos_row_id}|{p.neg_row_id}" for p in aligned_pairs)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def run_analysis(repo_root: Path) -> AnalysisRun:
    """Compute all five planned comparisons in memory -- no file is written
    by this function. Every comparison reuses the exact same aligned_pairs
    (a single alignment.align_pairs(...) call over all four checkpoints),
    so every comparison uses the identical 582 pair identities."""
    repo_root = Path(repo_root).resolve()
    pair_records = load_core_pair_records(repo_root)

    aligned_pairs = alignment.align_pairs(pair_records)
    if len(aligned_pairs) != EXPECTED_ALIGNED_PAIR_COUNT:
        raise lggsn_statistics.StatisticsError(
            f"expected exactly {EXPECTED_ALIGNED_PAIR_COUNT} aligned pairs across the core "
            f"matrix, got {len(aligned_pairs)}"
        )

    margins = {
        name: _load_score_margins(_pair_results_path(repo_root, name)) for name in CORE_CHECKPOINT_NAMES
    }

    comparisons = [
        _build_comparison(aligned_pairs, checkpoint_a=a, checkpoint_b=b, margins=margins, alpha=ALPHA)
        for a, b in PLANNED_COMPARISONS
    ]

    raw_p_values = [c["mcnemar_p_value_raw"] for c in comparisons]
    holm_adjusted_p_values = lggsn_statistics.holm_bonferroni_adjust(raw_p_values)
    for comparison, adjusted_p in zip(comparisons, holm_adjusted_p_values):
        comparison["mcnemar_p_value_holm_adjusted"] = adjusted_p
        comparison["interpretation_holm"] = _interpretation(
            adjusted_p, comparison["accuracy_diff_b_minus_a"], alpha=ALPHA,
        )

    return AnalysisRun(comparisons=comparisons, aligned_pairs=aligned_pairs, pair_records=pair_records)


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip()
    except Exception as e:  # pragma: no cover - environment-dependent
        return f"<unavailable: {e}>"


def build_manifest(
    *, repo_root: Path, run: AnalysisRun, started_at: str, runtime_seconds: float, command: list[str],
) -> dict[str, Any]:
    input_file_sha256: dict[str, str] = {}
    checkpoint_sha256: dict[str, str] = {}
    dataset_sha256_values: set[str] = set()

    for name in CORE_CHECKPOINT_NAMES:
        pr_path = _pair_results_path(repo_root, name)
        input_file_sha256[name] = hashlib.sha256(pr_path.read_bytes()).hexdigest()
        provenance = json.loads(_provenance_path(repo_root, name).read_text(encoding="utf-8"))
        if provenance.get("checkpoint_name") != name:
            raise lggsn_statistics.StatisticsError(
                f"provenance.json for {name!r} has checkpoint_name={provenance.get('checkpoint_name')!r}"
            )
        checkpoint_sha256[name] = provenance["checkpoint_sha256"]
        dataset_sha256_values.add(provenance["dataset_sha256"])

    if len(dataset_sha256_values) != 1:
        raise lggsn_statistics.StatisticsError(
            "expected a single shared dataset_sha256 across all core checkpoints' provenance.json, "
            f"got {sorted(dataset_sha256_values)}"
        )

    return {
        "schema_version": "lggsn_analysis_phase4_v1",
        "task_id": "lggsn_statistical_analysis",
        "phase": 4,
        "git_commit": _git_commit(repo_root),
        "core_checkpoint_names": list(CORE_CHECKPOINT_NAMES),
        "planned_comparisons": [list(pair) for pair in PLANNED_COMPARISONS],
        "input_file_sha256": input_file_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": next(iter(dataset_sha256_values)),
        "pair_identity_digest": _pair_identity_digest(run.aligned_pairs),
        "n_aligned_pairs": len(run.aligned_pairs),
        "alpha": ALPHA,
        "multiple_comparison_method": MULTIPLE_COMPARISON_METHOD,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_resampling_unit": BOOTSTRAP_RESAMPLING_UNIT,
        "bootstrap_resampling_unit_note": (
            "Cluster/block bootstrap over `query` (6 clusters in this dataset): the finest "
            "clustering unit reconstructable from pair_results.jsonl's own columns. LGGSN "
            "pairs are a cartesian product of (positive episode, negative episode) rows "
            "within one query (research_agent_pilots/lggsn_suite/eval_core.py's build_pairs), "
            "so pairs sharing a query are correlated, not independent -- resampling "
            "individual pairs i.i.d. would understate the true sampling variance."
        ),
        "cluster_permutation_resampling_unit": CLUSTER_PERMUTATION_RESAMPLING_UNIT,
        "cluster_permutation_note": (
            "Phase 4 addition: exact cluster-level (query-as-unit) sign-flip permutation test, "
            "added alongside -- never replacing -- the pair-level exact McNemar test and the "
            "query-cluster bootstrap CI above. Enumerates all 2**n_clusters sign assignments "
            "exactly (no seed: nothing is randomly sampled). Explicitly not McNemar's test and "
            "not reported as one. See statistics.cluster_sign_flip_test's docstring for the "
            "exchangeability assumption and each comparison's `cluster_sign_flip` block for the "
            "per-comparison power caveat."
        ),
        "command": command,
        "started_at": started_at,
        "runtime_seconds": runtime_seconds,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pair_accuracy_is_not_grasp_success_rate": True,
        "no_causal_feature_importance_claimed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    run = run_analysis(repo_root)

    reporting.write_pairwise_comparisons_json(run.comparisons, output_dir / "pairwise_comparisons.json")
    reporting.write_pairwise_comparisons_csv(run.comparisons, output_dir / "pairwise_comparisons.csv")
    reporting.write_per_query_breakdown_csv(run.comparisons, output_dir / "per_query_breakdown.csv")

    manifest = build_manifest(
        repo_root=repo_root, run=run, started_at=started_at,
        runtime_seconds=round(time.monotonic() - t0, 4), command=list(sys.argv),
    )
    reporting.write_statistical_summary_markdown(run.comparisons, manifest, output_dir / "statistical_summary.md")
    reporting.write_analysis_manifest_json(manifest, output_dir / "analysis_manifest.json")

    print(
        f"[lggsn_real_analysis] wrote {len(run.comparisons)} comparisons "
        f"({EXPECTED_ALIGNED_PAIR_COUNT} aligned pairs each) to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
