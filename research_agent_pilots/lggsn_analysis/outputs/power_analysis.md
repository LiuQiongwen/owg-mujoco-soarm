# LGGSN Core-Matrix McNemar Power Analysis (Phase 5)

Follow-up to the three NOT_SIGNIFICANT comparisons in `statistical_summary.md` (Phase 3): does a null result here mean "no real effect" or "underpowered study"? Computed for all five planned comparisons, not just the null ones, so nothing is selectively reported. Every number below is read from or recomputed only from `pairwise_comparisons.json`'s discordant counts -- no new data, no re-run of McNemar's test itself.

## Method and caveats

- All power/sample-size math treats the n discordant pairs as independent draws -- the same assumption McNemar's test itself makes. It does not separately model the query-level clustering that the cluster bootstrap CIs (`statistical_summary.md`) account for; if discordant pairs are positively correlated within a query, true power is likely somewhat lower than reported here.
- **Post-hoc power** (power to detect the effect actually observed) is a monotonic rescaling of the p-value, not independent evidence -- reported because it is commonly requested, but should not be over-interpreted on its own. The **minimum detectable proportion** (smallest effect this study's actual sample size *could* have detected at 80% power) and **required discordant pairs** (how many would be needed to detect the observed effect at 80% power) are less circular and more actionable.
- "Required discordant pairs" assumes the *true* effect equals the *observed* effect exactly; it is a what-if calculation for planning a follow-up study, not a claim about what the true effect actually is.

## Summary table

| A | B | n (discordant) | observed p(favor B) | post-hoc power | min detectable p | required n | additional pairs needed |
|---|---|---|---|---|---|---|---|
| base | nodist | 283 | 0.3675 | 0.9938 | 0.5847 | 112 | 0 |
| base | nozrel | 119 | 0.5798 | 0.3922 | 0.6296 | 314 | 195 |
| base | full_v2 | 140 | 0.5071 | 0.0368 | 0.6237 | 38528 | 38388 |
| nodist | full_v2 | 263 | 0.6464 | 0.9979 | 0.5864 | 99 | 0 |
| nozrel | full_v2 | 171 | 0.4503 | 0.2455 | 0.6074 | 824 | 653 |

## base vs nodist

- Discordant pairs: 283
- Observed proportion favoring B (of discordant pairs): 0.3675
- Post-hoc power to detect this effect at alpha=0.05: 0.9938
- Minimum detectable proportion at this study's actual n, for 80% power: 0.5847
- Discordant pairs required for 80% power at the observed effect: 112 (0 more than currently available)

## base vs nozrel

- Discordant pairs: 119
- Observed proportion favoring B (of discordant pairs): 0.5798
- Post-hoc power to detect this effect at alpha=0.05: 0.3922
- Minimum detectable proportion at this study's actual n, for 80% power: 0.6296
- Discordant pairs required for 80% power at the observed effect: 314 (195 more than currently available)

## base vs full_v2

- Discordant pairs: 140
- Observed proportion favoring B (of discordant pairs): 0.5071
- Post-hoc power to detect this effect at alpha=0.05: 0.0368
- Minimum detectable proportion at this study's actual n, for 80% power: 0.6237
- Discordant pairs required for 80% power at the observed effect: 38528 (38388 more than currently available)

## nodist vs full_v2

- Discordant pairs: 263
- Observed proportion favoring B (of discordant pairs): 0.6464
- Post-hoc power to detect this effect at alpha=0.05: 0.9979
- Minimum detectable proportion at this study's actual n, for 80% power: 0.5864
- Discordant pairs required for 80% power at the observed effect: 99 (0 more than currently available)

## nozrel vs full_v2

- Discordant pairs: 171
- Observed proportion favoring B (of discordant pairs): 0.4503
- Post-hoc power to detect this effect at alpha=0.05: 0.2455
- Minimum detectable proportion at this study's actual n, for 80% power: 0.6074
- Discordant pairs required for 80% power at the observed effect: 824 (653 more than currently available)
