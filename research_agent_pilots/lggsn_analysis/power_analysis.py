#!/usr/bin/env python
"""Phase 5: power analysis for McNemar's test on Phase 3's already-computed
pairwise comparisons.

Reads ONLY research_agent_pilots/lggsn_analysis/outputs/
pairwise_comparisons.json (real_analysis.py's already-tested output) --
never recomputes discordant counts, McNemar's p-value, or any other
statistic Phase 3 already produced. For each of the five comparisons, this
answers: given the discordant-pair count we actually have, and the effect
size we actually observed, what was our statistical power to detect it --
and how much more paired data would we have needed for a conventional 80%
power?

This directly follows up on the three NOT_SIGNIFICANT comparisons from
Phase 3 (base-vs-nozrel, base-vs-full_v2, nozrel-vs-full_v2): a null result
can mean "no real effect" or "underpowered study that could not detect a
real effect" -- this module reports which is more likely for each, rather
than leaving that question unanswered. It is computed for all five
comparisons (not just the three null ones) for consistency and so nothing
is selectively reported.

See statistics.py's "Phase 5" section for the underlying exact-binomial
power math and its documented scope (independence assumption, matching
exact_mcnemar itself; float/log-space arithmetic, not exact Fractions --
see mcnemar_exact_rejection_region's docstring for why).

Post-hoc power (power to detect the observed effect size) is reported
alongside, not instead of, two less circular numbers: the minimum
detectable proportion at the fixed n this study actually has, and the
discordant-pair count that would be required to reach 80% power for the
observed effect. Post-hoc power is included because it is commonly
requested, with an explicit methodological caveat (see
reporting.render_power_analysis_markdown) that it is mathematically just a
monotonic rescaling of the p-value and should not be over-interpreted as
independent evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_agent_pilots.lggsn_analysis import reporting
from research_agent_pilots.lggsn_analysis import statistics as lggsn_statistics

ALPHA = 0.05
TARGET_POWER = 0.8
REQUIRED_N_MAX = 200_000


def build_power_report(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for c in comparisons:
        n01 = c["discordant_a_wrong_b_correct"]
        n10 = c["discordant_a_correct_b_wrong"]
        n = n01 + n10

        entry: dict[str, Any] = {
            "checkpoint_a": c["checkpoint_a"],
            "checkpoint_b": c["checkpoint_b"],
            "n_discordant_pairs": n,
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
        }

        if n == 0:
            entry.update({
                "observed_proportion_favor_b": None,
                "post_hoc_power": None,
                "min_detectable_proportion_favor_b": None,
                "required_discordant_pairs_for_target_power": None,
                "additional_discordant_pairs_needed": None,
                "note": "zero discordant pairs -- power analysis is undefined",
            })
            reports.append(entry)
            continue

        observed_proportion = n01 / n
        post_hoc_power = lggsn_statistics.mcnemar_power(n, observed_proportion, alpha=ALPHA)
        min_detectable = lggsn_statistics.mcnemar_minimum_detectable_proportion(
            n, alpha=ALPHA, target_power=TARGET_POWER
        )
        if observed_proportion == 0.5:
            required_n = None
            note = "observed effect is exactly null (n01 == n10) -- no finite sample size would detect it"
        else:
            required_n = lggsn_statistics.mcnemar_required_n_for_power(
                observed_proportion, alpha=ALPHA, target_power=TARGET_POWER, n_max=REQUIRED_N_MAX
            )
            note = None if required_n is not None else (
                f"required discordant-pair count exceeds this report's search bound "
                f"({REQUIRED_N_MAX}) -- the observed effect is too close to null to "
                f"resolve with any realistic sample size"
            )

        entry.update({
            "observed_proportion_favor_b": observed_proportion,
            "post_hoc_power": post_hoc_power,
            "min_detectable_proportion_favor_b": min_detectable,
            "required_discordant_pairs_for_target_power": required_n,
            "additional_discordant_pairs_needed": None if required_n is None else max(0, required_n - n),
            "note": note,
        })
        reports.append(entry)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    comparisons = json.loads(Path(args.comparisons_json).read_text(encoding="utf-8"))["comparisons"]
    reports = build_power_report(comparisons)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reporting.write_power_analysis_json(reports, output_dir / "power_analysis.json")
    reporting.write_power_analysis_csv(reports, output_dir / "power_analysis.csv")
    reporting.write_power_analysis_markdown(reports, output_dir / "power_analysis.md")

    print(f"[lggsn_power_analysis] wrote power analysis for {len(reports)} comparisons to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
