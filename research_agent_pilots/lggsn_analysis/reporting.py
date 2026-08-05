"""Phase 3 (+ Phase 4 cluster-permutation addition) output formatting:
JSON/CSV/Markdown writers for real_analysis.py's already-computed
pairwise comparison records.

Pure formatting only -- no statistics are computed here (see statistics.py)
and no input file under research_agent_pilots/lggsn_analysis/pair_results/
is ever opened for writing. Every writer here is deterministic given
deterministic input: fixed column order, fixed row order (whatever order
the caller supplies, never re-sorted or re-grouped here), JSON emitted with
sort_keys=True. The one genuinely non-deterministic field this module
formats -- generated_at / runtime_seconds in the manifest -- is confined to
write_manifest_json; every other writer's output is byte-identical across
repeated runs on the same input.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

PAIRWISE_CSV_FIELDS: tuple[str, ...] = (
    "checkpoint_a",
    "checkpoint_b",
    "n_pairs",
    "accuracy_a",
    "accuracy_b",
    "accuracy_diff_b_minus_a",
    "win",
    "tie",
    "loss",
    "discordant_a_correct_b_wrong",
    "discordant_a_wrong_b_correct",
    "mcnemar_p_value_raw",
    "mcnemar_p_value_holm_adjusted",
    "interpretation_raw",
    "interpretation_holm",
    "bootstrap_seed",
    "bootstrap_n_resamples",
    "bootstrap_resampling_unit",
    "bootstrap_n_clusters",
    "bootstrap_confidence",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "score_margin_n_finite_pairs",
    "score_margin_n_excluded",
    "score_margin_mean_diff_b_minus_a",
    "score_margin_median_diff_b_minus_a",
    # Phase 4: cluster-as-unit sign-flip permutation test -- a separate
    # evidence column from mcnemar_p_value_raw/holm above, never combined
    # with it into one number.
    "cluster_permutation_resampling_unit",
    "cluster_permutation_n_clusters",
    "cluster_permutation_n_favor_a",
    "cluster_permutation_n_favor_b",
    "cluster_permutation_n_tied",
    "cluster_permutation_mean_diff",
    "cluster_permutation_median_diff",
    "cluster_permutation_n_permutations",
    "cluster_permutation_p_value",
    "conclusion_category",
)

PER_QUERY_CSV_FIELDS: tuple[str, ...] = (
    "checkpoint_a",
    "checkpoint_b",
    "query",
    "n_pairs",
    "accuracy_a",
    "accuracy_b",
    "accuracy_diff_b_minus_a",
    "win",
    "tie",
    "loss",
)


def write_pairwise_comparisons_json(comparisons: Sequence[Mapping[str, Any]], path: Path) -> None:
    payload = {"comparisons": list(comparisons)}
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _pairwise_csv_row(comparison: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = comparison["bootstrap"]
    score_margin = comparison["score_margin"]
    cluster_sign_flip = comparison["cluster_sign_flip"]
    return {
        "checkpoint_a": comparison["checkpoint_a"],
        "checkpoint_b": comparison["checkpoint_b"],
        "n_pairs": comparison["n_pairs"],
        "accuracy_a": comparison["accuracy_a"],
        "accuracy_b": comparison["accuracy_b"],
        "accuracy_diff_b_minus_a": comparison["accuracy_diff_b_minus_a"],
        "win": comparison["win"],
        "tie": comparison["tie"],
        "loss": comparison["loss"],
        "discordant_a_correct_b_wrong": comparison["discordant_a_correct_b_wrong"],
        "discordant_a_wrong_b_correct": comparison["discordant_a_wrong_b_correct"],
        "mcnemar_p_value_raw": comparison["mcnemar_p_value_raw"],
        "mcnemar_p_value_holm_adjusted": comparison["mcnemar_p_value_holm_adjusted"],
        "interpretation_raw": comparison["interpretation_raw"],
        "interpretation_holm": comparison["interpretation_holm"],
        "bootstrap_seed": bootstrap["seed"],
        "bootstrap_n_resamples": bootstrap["n_resamples"],
        "bootstrap_resampling_unit": bootstrap["resampling_unit"],
        "bootstrap_n_clusters": bootstrap["n_clusters"],
        "bootstrap_confidence": bootstrap["confidence"],
        "bootstrap_ci_lower": bootstrap["ci_lower"],
        "bootstrap_ci_upper": bootstrap["ci_upper"],
        "score_margin_n_finite_pairs": score_margin["n_finite_pairs"],
        "score_margin_n_excluded": score_margin["n_excluded"],
        "score_margin_mean_diff_b_minus_a": score_margin["mean_diff_b_minus_a"],
        "score_margin_median_diff_b_minus_a": score_margin["median_diff_b_minus_a"],
        "cluster_permutation_resampling_unit": cluster_sign_flip["resampling_unit"],
        "cluster_permutation_n_clusters": cluster_sign_flip["n_clusters"],
        "cluster_permutation_n_favor_a": cluster_sign_flip["n_favor_a"],
        "cluster_permutation_n_favor_b": cluster_sign_flip["n_favor_b"],
        "cluster_permutation_n_tied": cluster_sign_flip["n_tied"],
        "cluster_permutation_mean_diff": cluster_sign_flip["mean_cluster_diff"],
        "cluster_permutation_median_diff": cluster_sign_flip["median_cluster_diff"],
        "cluster_permutation_n_permutations": cluster_sign_flip["n_permutations"],
        "cluster_permutation_p_value": cluster_sign_flip["p_value"],
        "conclusion_category": comparison["conclusion_category"],
    }


def write_pairwise_comparisons_csv(comparisons: Sequence[Mapping[str, Any]], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PAIRWISE_CSV_FIELDS), lineterminator="\n")
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(_pairwise_csv_row(comparison))


def write_per_query_breakdown_csv(comparisons: Sequence[Mapping[str, Any]], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PER_QUERY_CSV_FIELDS), lineterminator="\n")
        writer.writeheader()
        for comparison in comparisons:
            for row in comparison["per_query_breakdown"]:
                writer.writerow({
                    "checkpoint_a": comparison["checkpoint_a"],
                    "checkpoint_b": comparison["checkpoint_b"],
                    "query": row["query"],
                    "n_pairs": row["n_pairs"],
                    "accuracy_a": row["accuracy_a"],
                    "accuracy_b": row["accuracy_b"],
                    "accuracy_diff_b_minus_a": row["accuracy_diff_b_minus_a"],
                    "win": row["win"],
                    "tie": row["tie"],
                    "loss": row["loss"],
                })


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_statistical_summary_markdown(
    comparisons: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> str:
    lines: list[str] = []
    lines.append("# LGGSN Core-Matrix Pairwise Statistical Analysis (Phase 3 + Phase 4)")
    lines.append("")
    lines.append(
        "Computed only from the real, provenance-pinned "
        "`research_agent_pilots/lggsn_analysis/pair_results/*/pair_results.jsonl` "
        "files regenerated in Phase 2. See `analysis_manifest.json` for full "
        "provenance (git commit, input/checkpoint/dataset SHA-256 values, "
        "pair-identity digest, bootstrap seed, and the exact command used)."
    )
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(f"- Predeclared significance level: alpha = {manifest['alpha']} (not changed post-hoc).")
    lines.append(
        "- Per-comparison significance test: exact two-sided McNemar test "
        "(binomial(n, 0.5) tail on the discordant pairs, computed with exact "
        "rational arithmetic -- not the chi-squared approximation)."
    )
    lines.append(
        f"- Multiple-comparison correction: {manifest['multiple_comparison_method']}, "
        f"applied jointly across all {len(comparisons)} planned comparisons below. "
        "Both the raw and the Holm-Bonferroni-adjusted p-value/interpretation are "
        "reported for every comparison -- the correction is never applied silently."
    )
    lines.append(
        f"- Bootstrap: deterministic cluster (block) bootstrap, resampling unit = "
        f"`{manifest['bootstrap_resampling_unit']}`, seed = {manifest['bootstrap_seed']} "
        f"(explicit, fixed, recorded here and in analysis_manifest.json)."
    )
    lines.append(
        "  Resampling unit justification: LGGSN pairs are constructed as a cartesian "
        "product of (positive episode, negative episode) row pairs within one query "
        "(research_agent_pilots/lggsn_suite/eval_core.py's build_pairs), so pairs "
        "sharing a query are correlated, not independent draws. The committed "
        "pair_results.jsonl columns carry `query` but not the finer episode/scene_id "
        "identity, so `query` is the finest clustering unit reconstructable from the "
        "real, provenance-pinned inputs this analysis is restricted to -- a per-pair "
        "i.i.d. bootstrap would understate the true sampling variance."
    )
    lines.append(
        "  Caveat: there are only 6 query clusters in this dataset, so the bootstrap "
        "has coarse resolution -- treat the resulting confidence intervals as "
        "conservative/wide, not precise."
    )
    lines.append(
        "- **Phase 4 addition, reported as a separate evidence column, never combined with "
        "the above:** an exact cluster-level (query-as-independent-unit) sign-flip "
        "permutation test. All `2**n_clusters` sign assignments are enumerated exactly (no "
        "seed -- nothing is randomly sampled). This is explicitly **not McNemar's test**: "
        "McNemar treats each pair as an independent Bernoulli trial; this test's only "
        "assumption is that, under the null, the *sign* of each query's own accuracy "
        "difference is exchangeable, independent of every other query's sign -- it makes no "
        "claim about independence of pairs within a query, and no claim about the "
        "*magnitude* of any query's difference."
    )
    lines.append(
        "  Power caveat: with only 6 query clusters, the coarsest possible two-sided exact "
        "p-value is 2/64 = 0.03125. p-values from this test are discrete and power is low -- "
        "**failing to reject the null is not evidence of equivalence** between two "
        "checkpoints, only that 6 clusters cannot resolve the question further."
    )
    lines.append("")

    lines.append("## Summary table")
    lines.append("")
    lines.append(
        "| A | B | n | acc(A) | acc(B) | diff(B-A) | win | tie | loss | "
        "p_raw | p_holm | raw | holm |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in comparisons:
        lines.append(
            f"| {c['checkpoint_a']} | {c['checkpoint_b']} | {c['n_pairs']} | "
            f"{_fmt(c['accuracy_a'])} | {_fmt(c['accuracy_b'])} | "
            f"{_fmt(c['accuracy_diff_b_minus_a'])} | {c['win']} | {c['tie']} | {c['loss']} | "
            f"{_fmt(c['mcnemar_p_value_raw'])} | {_fmt(c['mcnemar_p_value_holm_adjusted'])} | "
            f"{c['interpretation_raw']} | {c['interpretation_holm']} |"
        )
    lines.append("")

    lines.append(
        "## Three separate evidence columns (never combined into one significance label)"
    )
    lines.append("")
    lines.append(
        "Each comparison below is read through three independent lenses. They are reported "
        "side by side; no single overall p-value or verdict is computed from them."
    )
    lines.append("")
    lines.append(
        "| A | B | pair_level_mcnemar (raw p) | query_cluster_permutation (p, n_clusters) | "
        "query_cluster_bootstrap_ci (95%) | conclusion_category |"
    )
    lines.append("|---|---|---|---|---|---|")
    for c in comparisons:
        cluster_sign_flip = c["cluster_sign_flip"]
        bootstrap = c["bootstrap"]
        lines.append(
            f"| {c['checkpoint_a']} | {c['checkpoint_b']} | {_fmt(c['mcnemar_p_value_raw'])} | "
            f"{_fmt(cluster_sign_flip['p_value'])} (n={cluster_sign_flip['n_clusters']}) | "
            f"[{_fmt(bootstrap['ci_lower'])}, {_fmt(bootstrap['ci_upper'])}] | "
            f"{c['conclusion_category']} |"
        )
    lines.append("")

    for c in comparisons:
        bootstrap = c["bootstrap"]
        score_margin = c["score_margin"]
        lines.append(f"## {c['checkpoint_a']} vs {c['checkpoint_b']}")
        lines.append("")
        lines.append(f"- Aligned pairs: {c['n_pairs']}")
        lines.append(f"- Accuracy: {c['checkpoint_a']}={_fmt(c['accuracy_a'])}, {c['checkpoint_b']}={_fmt(c['accuracy_b'])}")
        lines.append(f"- Accuracy difference (B - A): {_fmt(c['accuracy_diff_b_minus_a'])}")
        lines.append(f"- Win / tie / loss: {c['win']} / {c['tie']} / {c['loss']}")
        lines.append(
            f"- Discordant pairs: A correct & B wrong = {c['discordant_a_correct_b_wrong']}, "
            f"A wrong & B correct = {c['discordant_a_wrong_b_correct']}"
        )
        lines.append(
            f"- Exact McNemar p-value: raw = {_fmt(c['mcnemar_p_value_raw'])}, "
            f"Holm-adjusted = {_fmt(c['mcnemar_p_value_holm_adjusted'])}"
        )
        lines.append(
            f"- Interpretation (alpha={manifest['alpha']}): raw = {c['interpretation_raw']}, "
            f"Holm-adjusted = {c['interpretation_holm']}"
        )
        lines.append(
            f"- Cluster bootstrap {int(bootstrap['confidence'] * 100)}% CI for accuracy "
            f"difference (B-A): [{_fmt(bootstrap['ci_lower'])}, {_fmt(bootstrap['ci_upper'])}] "
            f"(unit={bootstrap['resampling_unit']}, n_clusters={bootstrap['n_clusters']}, "
            f"n_resamples={bootstrap['n_resamples']}, seed={bootstrap['seed']})"
        )
        cluster_sign_flip = c["cluster_sign_flip"]
        lines.append("")
        lines.append(
            f"- **Cluster-level sign-flip permutation test** (query as independent unit, "
            f"NOT McNemar): {cluster_sign_flip['n_favor_b']} of {cluster_sign_flip['n_clusters']} "
            f"queries favor {c['checkpoint_b']}, {cluster_sign_flip['n_favor_a']} favor "
            f"{c['checkpoint_a']}, {cluster_sign_flip['n_tied']} tied. "
            f"Mean query-level diff (B-A) = {_fmt(cluster_sign_flip['mean_cluster_diff'])}, "
            f"median = {_fmt(cluster_sign_flip['median_cluster_diff'])}. "
            f"Exact two-sided p-value = {_fmt(cluster_sign_flip['p_value'])} "
            f"(enumerated all {cluster_sign_flip['n_permutations']} sign assignments exactly, "
            f"no seed)."
        )
        lines.append(f"  - Exchangeability assumption: {cluster_sign_flip['exchangeability_assumption']}")
        lines.append(f"  - Power caveat: {cluster_sign_flip['power_caveat']}")
        lines.append(f"- **Conclusion category** (conservative combined read, alpha={manifest['alpha']}): {c['conclusion_category']}")
        if score_margin["n_finite_pairs"] > 0:
            lines.append(
                f"- Score-margin difference (B-A), over {score_margin['n_finite_pairs']} pairs with "
                f"finite scores ({score_margin['n_excluded']} excluded): "
                f"mean = {_fmt(score_margin['mean_diff_b_minus_a'])}, "
                f"median = {_fmt(score_margin['median_diff_b_minus_a'])}"
            )
        else:
            lines.append(
                f"- Score-margin difference: unavailable (0 of {c['n_pairs']} pairs had finite "
                "scores for both checkpoints)"
            )
        lines.append("")
        lines.append("  Per-query breakdown:")
        lines.append("")
        lines.append("  | query | n | acc(A) | acc(B) | diff(B-A) | win | tie | loss |")
        lines.append("  |---|---|---|---|---|---|---|---|")
        for row in c["per_query_breakdown"]:
            lines.append(
                f"  | {row['query']} | {row['n_pairs']} | {_fmt(row['accuracy_a'])} | "
                f"{_fmt(row['accuracy_b'])} | {_fmt(row['accuracy_diff_b_minus_a'])} | "
                f"{row['win']} | {row['tie']} | {row['loss']} |"
            )
        lines.append("")

    lines.append("## What these results do and do not prove")
    lines.append("")
    lines.append(
        "- `pair_accuracy` measures whether the model scored the labeled-positive grasp "
        "candidate above the labeled-negative one for a given pair. It is **not** a "
        "grasp success rate and must never be reported or read as one -- no physical "
        "grasp attempt, simulated or real, was executed to produce this data."
    )
    lines.append(
        "- Each of base/nodist/nozrel/full_v2 is a single checkpoint per ablation "
        "configuration. A significant pairwise difference between two checkpoints is "
        "evidence about *those two trained models' pairwise-ranking accuracy on this "
        "fixed validation split* -- it is not evidence that any specific input feature "
        "(e.g. `dist_to_centroid`, `z_rel`) *causes* the difference. No causal "
        "feature-importance claim is made or supported by this analysis."
    )
    lines.append(
        "- The cluster bootstrap above resamples by query (6 clusters) because that is "
        "the finest grouping reconstructable from the committed pair_results.jsonl "
        "files; it does not resample by training seed or by independently retrained "
        "model replicates (there is only one checkpoint per configuration), so these "
        "intervals do not capture across-training-run variance."
    )
    lines.append(
        "- The cluster-level sign-flip permutation test (Phase 4) is **not McNemar's test** "
        "and is never labeled as one. It answers a deliberately different question at a "
        "deliberately different unit of analysis (query, not pair) than the pair-level "
        "McNemar test above -- the two are expected to disagree when within-query "
        "correlation is substantial, and a `PAIR_LEVEL_ONLY` conclusion_category is exactly "
        "that disagreement made explicit, not an error."
    )
    lines.append(
        "- With only 6 query clusters, the cluster-level permutation test's p-values are "
        "coarse and discrete (minimum possible two-sided p = 2/64 = 0.03125). "
        "**A `NO_CLEAR_DIFFERENCE` or `PAIR_LEVEL_ONLY` conclusion_category is not evidence "
        "that the two checkpoints are equivalent** -- it may simply reflect that 6 clusters "
        "cannot resolve a real but modest effect. No equivalence claim is made anywhere in "
        "this report."
    )
    lines.append(
        "- `pair_level_mcnemar`, `query_cluster_permutation`, and `query_cluster_bootstrap_ci` "
        "are reported as three separate evidence columns for every comparison and are never "
        "collapsed into a single significance label. `conclusion_category` is a conservative, "
        "explicitly-named combined read (see its docstring in real_analysis.py for the exact "
        "rule) reported *alongside*, not instead of, all three -- it is computed from the "
        "raw (not Holm-adjusted) pair-level p-value, since Holm-Bonferroni is a separate, "
        "family-wise concern across the five planned comparisons."
    )
    lines.append(
        "- Both raw and Holm-Bonferroni-adjusted p-values/interpretations are reported "
        "for every comparison above; a comparison that is significant under the raw "
        "p-value but not after the Holm adjustment (or vice versa) is reported exactly "
        "as such, not resolved into a single number."
    )
    lines.append(
        "- All five comparisons are reported regardless of outcome, including any that "
        "are NOT_SIGNIFICANT under either rule -- no null or contradictory result is "
        "omitted."
    )
    lines.append("")
    return "\n".join(lines)


def write_statistical_summary_markdown(
    comparisons: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], path: Path
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        f.write(render_statistical_summary_markdown(comparisons, manifest))


def write_analysis_manifest_json(manifest: Mapping[str, Any], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
