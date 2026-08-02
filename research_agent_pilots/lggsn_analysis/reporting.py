"""Output formatting: JSON/CSV/Markdown writers for real_analysis.py's
(Phase 3) and power_analysis.py's (Phase 5) already-computed records.

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
    lines.append("# LGGSN Core-Matrix Pairwise Statistical Analysis (Phase 3)")
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


# ── Phase 5: power analysis (power_analysis.py) ─────────────────────────────

POWER_ANALYSIS_CSV_FIELDS: tuple[str, ...] = (
    "checkpoint_a",
    "checkpoint_b",
    "n_discordant_pairs",
    "alpha",
    "target_power",
    "observed_proportion_favor_b",
    "post_hoc_power",
    "min_detectable_proportion_favor_b",
    "required_discordant_pairs_for_target_power",
    "additional_discordant_pairs_needed",
    "note",
)


def write_power_analysis_json(reports: Sequence[Mapping[str, Any]], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        json.dump({"power_analysis": list(reports)}, f, indent=2, sort_keys=True)
        f.write("\n")


def write_power_analysis_csv(reports: Sequence[Mapping[str, Any]], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(POWER_ANALYSIS_CSV_FIELDS), lineterminator="\n")
        writer.writeheader()
        for report in reports:
            writer.writerow({field: report.get(field) for field in POWER_ANALYSIS_CSV_FIELDS})


def _fmt_opt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_power_analysis_markdown(reports: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# LGGSN Core-Matrix McNemar Power Analysis (Phase 5)")
    lines.append("")
    lines.append(
        "Follow-up to the three NOT_SIGNIFICANT comparisons in "
        "`statistical_summary.md` (Phase 3): does a null result here mean "
        "\"no real effect\" or \"underpowered study\"? Computed for all five "
        "planned comparisons, not just the null ones, so nothing is "
        "selectively reported. Every number below is read from or "
        "recomputed only from `pairwise_comparisons.json`'s discordant "
        "counts -- no new data, no re-run of McNemar's test itself."
    )
    lines.append("")
    lines.append("## Method and caveats")
    lines.append("")
    lines.append(
        "- All power/sample-size math treats the n discordant pairs as "
        "independent draws -- the same assumption McNemar's test itself "
        "makes. It does not separately model the query-level clustering "
        "that the cluster bootstrap CIs (`statistical_summary.md`) "
        "account for; if discordant pairs are positively correlated "
        "within a query, true power is likely somewhat lower than "
        "reported here."
    )
    lines.append(
        "- **Post-hoc power** (power to detect the effect actually "
        "observed) is a monotonic rescaling of the p-value, not "
        "independent evidence -- reported because it is commonly "
        "requested, but should not be over-interpreted on its own. The "
        "**minimum detectable proportion** (smallest effect this study's "
        "actual sample size *could* have detected at 80% power) and "
        "**required discordant pairs** (how many would be needed to "
        "detect the observed effect at 80% power) are less circular and "
        "more actionable."
    )
    lines.append(
        "- \"Required discordant pairs\" assumes the *true* effect equals "
        "the *observed* effect exactly; it is a what-if calculation for "
        "planning a follow-up study, not a claim about what the true "
        "effect actually is."
    )
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append("| A | B | n (discordant) | observed p(favor B) | post-hoc power | min detectable p | required n | additional pairs needed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in reports:
        lines.append(
            f"| {r['checkpoint_a']} | {r['checkpoint_b']} | {r['n_discordant_pairs']} | "
            f"{_fmt_opt(r['observed_proportion_favor_b'])} | {_fmt_opt(r['post_hoc_power'])} | "
            f"{_fmt_opt(r['min_detectable_proportion_favor_b'])} | "
            f"{_fmt_opt(r['required_discordant_pairs_for_target_power'], 0)} | "
            f"{_fmt_opt(r['additional_discordant_pairs_needed'], 0)} |"
        )
    lines.append("")
    for r in reports:
        lines.append(f"## {r['checkpoint_a']} vs {r['checkpoint_b']}")
        lines.append("")
        lines.append(f"- Discordant pairs: {r['n_discordant_pairs']}")
        if r["note"] and r["n_discordant_pairs"] == 0:
            lines.append(f"- {r['note']}")
        else:
            lines.append(f"- Observed proportion favoring B (of discordant pairs): {_fmt_opt(r['observed_proportion_favor_b'])}")
            lines.append(
                f"- Post-hoc power to detect this effect at alpha={r['alpha']}: "
                f"{_fmt_opt(r['post_hoc_power'])}"
            )
            lines.append(
                f"- Minimum detectable proportion at this study's actual n, for "
                f"{int(r['target_power']*100)}% power: {_fmt_opt(r['min_detectable_proportion_favor_b'])}"
            )
            if r["required_discordant_pairs_for_target_power"] is not None:
                lines.append(
                    f"- Discordant pairs required for {int(r['target_power']*100)}% power at the "
                    f"observed effect: {r['required_discordant_pairs_for_target_power']} "
                    f"({r['additional_discordant_pairs_needed']} more than currently available)"
                )
            else:
                lines.append(f"- Required discordant pairs for {int(r['target_power']*100)}% power: {r['note']}")
        lines.append("")
    return "\n".join(lines)


def write_power_analysis_markdown(reports: Sequence[Mapping[str, Any]], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        f.write(render_power_analysis_markdown(reports))
