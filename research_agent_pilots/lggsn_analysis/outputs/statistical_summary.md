# LGGSN Core-Matrix Pairwise Statistical Analysis (Phase 3 + Phase 4)

Computed only from the real, provenance-pinned `research_agent_pilots/lggsn_analysis/pair_results/*/pair_results.jsonl` files regenerated in Phase 2. See `analysis_manifest.json` for full provenance (git commit, input/checkpoint/dataset SHA-256 values, pair-identity digest, bootstrap seed, and the exact command used).

## Method

- Predeclared significance level: alpha = 0.05 (not changed post-hoc).
- Per-comparison significance test: exact two-sided McNemar test (binomial(n, 0.5) tail on the discordant pairs, computed with exact rational arithmetic -- not the chi-squared approximation).
- Multiple-comparison correction: holm-bonferroni, applied jointly across all 5 planned comparisons below. Both the raw and the Holm-Bonferroni-adjusted p-value/interpretation are reported for every comparison -- the correction is never applied silently.
- Bootstrap: deterministic cluster (block) bootstrap, resampling unit = `query`, seed = 20260803 (explicit, fixed, recorded here and in analysis_manifest.json).
  Resampling unit justification: LGGSN pairs are constructed as a cartesian product of (positive episode, negative episode) row pairs within one query (research_agent_pilots/lggsn_suite/eval_core.py's build_pairs), so pairs sharing a query are correlated, not independent draws. The committed pair_results.jsonl columns carry `query` but not the finer episode/scene_id identity, so `query` is the finest clustering unit reconstructable from the real, provenance-pinned inputs this analysis is restricted to -- a per-pair i.i.d. bootstrap would understate the true sampling variance.
  Caveat: there are only 6 query clusters in this dataset, so the bootstrap has coarse resolution -- treat the resulting confidence intervals as conservative/wide, not precise.
- **Phase 4 addition, reported as a separate evidence column, never combined with the above:** an exact cluster-level (query-as-independent-unit) sign-flip permutation test. All `2**n_clusters` sign assignments are enumerated exactly (no seed -- nothing is randomly sampled). This is explicitly **not McNemar's test**: McNemar treats each pair as an independent Bernoulli trial; this test's only assumption is that, under the null, the *sign* of each query's own accuracy difference is exchangeable, independent of every other query's sign -- it makes no claim about independence of pairs within a query, and no claim about the *magnitude* of any query's difference.
  Power caveat: with only 6 query clusters, the coarsest possible two-sided exact p-value is 2/64 = 0.03125. p-values from this test are discrete and power is low -- **failing to reject the null is not evidence of equivalence** between two checkpoints, only that 6 clusters cannot resolve the question further.

## Summary table

| A | B | n | acc(A) | acc(B) | diff(B-A) | win | tie | loss | p_raw | p_holm | raw | holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | nodist | 582 | 0.654639 | 0.525773 | -0.128866 | 104 | 299 | 179 | 0.000010 | 0.000039 | SIGNIFICANT_FAVORS_A | SIGNIFICANT_FAVORS_A |
| base | nozrel | 582 | 0.654639 | 0.687285 | 0.032646 | 69 | 463 | 50 | 0.098524 | 0.295571 | NOT_SIGNIFICANT | NOT_SIGNIFICANT |
| base | full_v2 | 582 | 0.654639 | 0.658076 | 0.003436 | 71 | 442 | 69 | 0.932687 | 0.932687 | NOT_SIGNIFICANT | NOT_SIGNIFICANT |
| nodist | full_v2 | 582 | 0.525773 | 0.658076 | 0.132302 | 170 | 319 | 93 | 0.000002 | 0.000012 | SIGNIFICANT_FAVORS_B | SIGNIFICANT_FAVORS_B |
| nozrel | full_v2 | 582 | 0.687285 | 0.658076 | -0.029210 | 77 | 411 | 94 | 0.221011 | 0.442022 | NOT_SIGNIFICANT | NOT_SIGNIFICANT |

## Three separate evidence columns (never combined into one significance label)

Each comparison below is read through three independent lenses. They are reported side by side; no single overall p-value or verdict is computed from them.

| A | B | pair_level_mcnemar (raw p) | query_cluster_permutation (p, n_clusters) | query_cluster_bootstrap_ci (95%) | conclusion_category |
|---|---|---|---|---|---|
| base | nodist | 0.000010 | 0.406250 (n=6) | [-0.342466, 0.106212] | PAIR_LEVEL_ONLY |
| base | nozrel | 0.098524 | 0.250000 (n=6) | [-0.007987, 0.086258] | NO_CLEAR_DIFFERENCE |
| base | full_v2 | 0.932687 | 0.781250 (n=6) | [-0.123596, 0.141264] | NO_CLEAR_DIFFERENCE |
| nodist | full_v2 | 0.000002 | 0.375000 (n=6) | [-0.056723, 0.338308] | PAIR_LEVEL_ONLY |
| nozrel | full_v2 | 0.221011 | 0.843750 (n=6) | [-0.129597, 0.068541] | NO_CLEAR_DIFFERENCE |

## base vs nodist

- Aligned pairs: 582
- Accuracy: base=0.654639, nodist=0.525773
- Accuracy difference (B - A): -0.128866
- Win / tie / loss: 104 / 299 / 179
- Discordant pairs: A correct & B wrong = 179, A wrong & B correct = 104
- Exact McNemar p-value: raw = 0.000010, Holm-adjusted = 0.000039
- Interpretation (alpha=0.05): raw = SIGNIFICANT_FAVORS_A, Holm-adjusted = SIGNIFICANT_FAVORS_A
- Cluster bootstrap 95% CI for accuracy difference (B-A): [-0.342466, 0.106212] (unit=query, n_clusters=6, n_resamples=10000, seed=20260803)

- **Cluster-level sign-flip permutation test** (query as independent unit, NOT McNemar): 2 of 6 queries favor nodist, 4 favor base, 0 tied. Mean query-level diff (B-A) = -0.107868, median = -0.124289. Exact two-sided p-value = 0.406250 (enumerated all 64 sign assignments exactly, no seed).
  - Exchangeability assumption: Under the null of no true difference between the two checkpoints, the SIGN of each query's own accuracy_diff_b_minus_a is exchangeable (equally likely positive or negative), independently across queries. This does not assume pairs within a query are independent, and does not assume anything about the magnitude of each query's difference.
  - Power caveat: Only 6 query clusters: p-values from this test are discrete (coarsest possible two-sided p = 2/64 = 0.03125) and power is low. Failing to reject the null here is not evidence of equivalence between the two checkpoints.
- **Conclusion category** (conservative combined read, alpha=0.05): PAIR_LEVEL_ONLY
- Score-margin difference (B-A), over 582 pairs with finite scores (0 excluded): mean = -0.258650, median = -0.208125

  Per-query breakdown:

  | query | n | acc(A) | acc(B) | diff(B-A) | win | tie | loss |
  |---|---|---|---|---|---|---|---|
  | Banana | 104 | 0.663462 | 0.519231 | -0.144231 | 18 | 53 | 33 |
  | CrackerBox | 99 | 0.707071 | 0.808081 | 0.101010 | 24 | 61 | 14 |
  | MustardBottle | 120 | 0.808333 | 0.291667 | -0.516667 | 7 | 44 | 69 |
  | PowerDrill | 68 | 0.617647 | 0.279412 | -0.338235 | 5 | 35 | 28 |
  | Scissors | 115 | 0.765217 | 0.660870 | -0.104348 | 17 | 69 | 29 |
  | TomatoSoupCan | 76 | 0.197368 | 0.552632 | 0.355263 | 33 | 37 | 6 |

## base vs nozrel

- Aligned pairs: 582
- Accuracy: base=0.654639, nozrel=0.687285
- Accuracy difference (B - A): 0.032646
- Win / tie / loss: 69 / 463 / 50
- Discordant pairs: A correct & B wrong = 50, A wrong & B correct = 69
- Exact McNemar p-value: raw = 0.098524, Holm-adjusted = 0.295571
- Interpretation (alpha=0.05): raw = NOT_SIGNIFICANT, Holm-adjusted = NOT_SIGNIFICANT
- Cluster bootstrap 95% CI for accuracy difference (B-A): [-0.007987, 0.086258] (unit=query, n_clusters=6, n_resamples=10000, seed=20260803)

- **Cluster-level sign-flip permutation test** (query as independent unit, NOT McNemar): 4 of 6 queries favor nozrel, 1 favor base, 1 tied. Mean query-level diff (B-A) = 0.039768, median = 0.018873. Exact two-sided p-value = 0.250000 (enumerated all 64 sign assignments exactly, no seed).
  - Exchangeability assumption: Under the null of no true difference between the two checkpoints, the SIGN of each query's own accuracy_diff_b_minus_a is exchangeable (equally likely positive or negative), independently across queries. This does not assume pairs within a query are independent, and does not assume anything about the magnitude of each query's difference.
  - Power caveat: Only 6 query clusters: p-values from this test are discrete (coarsest possible two-sided p = 2/64 = 0.03125) and power is low. Failing to reject the null here is not evidence of equivalence between the two checkpoints.
- **Conclusion category** (conservative combined read, alpha=0.05): NO_CLEAR_DIFFERENCE
- Score-margin difference (B-A), over 582 pairs with finite scores (0 excluded): mean = -0.136290, median = -0.132879

  Per-query breakdown:

  | query | n | acc(A) | acc(B) | diff(B-A) | win | tie | loss |
  |---|---|---|---|---|---|---|---|
  | Banana | 104 | 0.663462 | 0.663462 | 0.000000 | 17 | 70 | 17 |
  | CrackerBox | 99 | 0.707071 | 0.797980 | 0.090909 | 15 | 78 | 6 |
  | MustardBottle | 120 | 0.808333 | 0.816667 | 0.008333 | 6 | 109 | 5 |
  | PowerDrill | 68 | 0.617647 | 0.647059 | 0.029412 | 7 | 56 | 5 |
  | Scissors | 115 | 0.765217 | 0.730435 | -0.034783 | 7 | 97 | 11 |
  | TomatoSoupCan | 76 | 0.197368 | 0.342105 | 0.144737 | 17 | 53 | 6 |

## base vs full_v2

- Aligned pairs: 582
- Accuracy: base=0.654639, full_v2=0.658076
- Accuracy difference (B - A): 0.003436
- Win / tie / loss: 71 / 442 / 69
- Discordant pairs: A correct & B wrong = 69, A wrong & B correct = 71
- Exact McNemar p-value: raw = 0.932687, Holm-adjusted = 0.932687
- Interpretation (alpha=0.05): raw = NOT_SIGNIFICANT, Holm-adjusted = NOT_SIGNIFICANT
- Cluster bootstrap 95% CI for accuracy difference (B-A): [-0.123596, 0.141264] (unit=query, n_clusters=6, n_resamples=10000, seed=20260803)

- **Cluster-level sign-flip permutation test** (query as independent unit, NOT McNemar): 3 of 6 queries favor full_v2, 3 favor base, 0 tied. Mean query-level diff (B-A) = 0.023396, median = 0.042157. Exact two-sided p-value = 0.781250 (enumerated all 64 sign assignments exactly, no seed).
  - Exchangeability assumption: Under the null of no true difference between the two checkpoints, the SIGN of each query's own accuracy_diff_b_minus_a is exchangeable (equally likely positive or negative), independently across queries. This does not assume pairs within a query are independent, and does not assume anything about the magnitude of each query's difference.
  - Power caveat: Only 6 query clusters: p-values from this test are discrete (coarsest possible two-sided p = 2/64 = 0.03125) and power is low. Failing to reject the null here is not evidence of equivalence between the two checkpoints.
- **Conclusion category** (conservative combined read, alpha=0.05): NO_CLEAR_DIFFERENCE
- Score-margin difference (B-A), over 582 pairs with finite scores (0 excluded): mean = -0.151247, median = -0.133866

  Per-query breakdown:

  | query | n | acc(A) | acc(B) | diff(B-A) | win | tie | loss |
  |---|---|---|---|---|---|---|---|
  | Banana | 104 | 0.663462 | 0.423077 | -0.240385 | 6 | 67 | 31 |
  | CrackerBox | 99 | 0.707071 | 0.919192 | 0.212121 | 21 | 78 | 0 |
  | MustardBottle | 120 | 0.808333 | 0.775000 | -0.033333 | 6 | 104 | 10 |
  | PowerDrill | 68 | 0.617647 | 0.735294 | 0.117647 | 9 | 58 | 1 |
  | Scissors | 115 | 0.765217 | 0.652174 | -0.113043 | 11 | 80 | 24 |
  | TomatoSoupCan | 76 | 0.197368 | 0.394737 | 0.197368 | 18 | 55 | 3 |

## nodist vs full_v2

- Aligned pairs: 582
- Accuracy: nodist=0.525773, full_v2=0.658076
- Accuracy difference (B - A): 0.132302
- Win / tie / loss: 170 / 319 / 93
- Discordant pairs: A correct & B wrong = 93, A wrong & B correct = 170
- Exact McNemar p-value: raw = 0.000002, Holm-adjusted = 0.000012
- Interpretation (alpha=0.05): raw = SIGNIFICANT_FAVORS_B, Holm-adjusted = SIGNIFICANT_FAVORS_B
- Cluster bootstrap 95% CI for accuracy difference (B-A): [-0.056723, 0.338308] (unit=query, n_clusters=6, n_resamples=10000, seed=20260803)

- **Cluster-level sign-flip permutation test** (query as independent unit, NOT McNemar): 3 of 6 queries favor full_v2, 3 favor nodist, 0 tied. Mean query-level diff (B-A) = 0.131264, median = 0.051208. Exact two-sided p-value = 0.375000 (enumerated all 64 sign assignments exactly, no seed).
  - Exchangeability assumption: Under the null of no true difference between the two checkpoints, the SIGN of each query's own accuracy_diff_b_minus_a is exchangeable (equally likely positive or negative), independently across queries. This does not assume pairs within a query are independent, and does not assume anything about the magnitude of each query's difference.
  - Power caveat: Only 6 query clusters: p-values from this test are discrete (coarsest possible two-sided p = 2/64 = 0.03125) and power is low. Failing to reject the null here is not evidence of equivalence between the two checkpoints.
- **Conclusion category** (conservative combined read, alpha=0.05): PAIR_LEVEL_ONLY
- Score-margin difference (B-A), over 582 pairs with finite scores (0 excluded): mean = 0.107403, median = -0.000006

  Per-query breakdown:

  | query | n | acc(A) | acc(B) | diff(B-A) | win | tie | loss |
  |---|---|---|---|---|---|---|---|
  | Banana | 104 | 0.519231 | 0.423077 | -0.096154 | 17 | 60 | 27 |
  | CrackerBox | 99 | 0.808081 | 0.919192 | 0.111111 | 17 | 76 | 6 |
  | MustardBottle | 120 | 0.291667 | 0.775000 | 0.483333 | 68 | 42 | 10 |
  | PowerDrill | 68 | 0.279412 | 0.735294 | 0.455882 | 35 | 29 | 4 |
  | Scissors | 115 | 0.660870 | 0.652174 | -0.008696 | 23 | 68 | 24 |
  | TomatoSoupCan | 76 | 0.552632 | 0.394737 | -0.157895 | 10 | 44 | 22 |

## nozrel vs full_v2

- Aligned pairs: 582
- Accuracy: nozrel=0.687285, full_v2=0.658076
- Accuracy difference (B - A): -0.029210
- Win / tie / loss: 77 / 411 / 94
- Discordant pairs: A correct & B wrong = 94, A wrong & B correct = 77
- Exact McNemar p-value: raw = 0.221011, Holm-adjusted = 0.442022
- Interpretation (alpha=0.05): raw = NOT_SIGNIFICANT, Holm-adjusted = NOT_SIGNIFICANT
- Cluster bootstrap 95% CI for accuracy difference (B-A): [-0.129597, 0.068541] (unit=query, n_clusters=6, n_resamples=10000, seed=20260803)

- **Cluster-level sign-flip permutation test** (query as independent unit, NOT McNemar): 3 of 6 queries favor full_v2, 3 favor nozrel, 0 tied. Mean query-level diff (B-A) = -0.016372, median = 0.005482. Exact two-sided p-value = 0.843750 (enumerated all 64 sign assignments exactly, no seed).
  - Exchangeability assumption: Under the null of no true difference between the two checkpoints, the SIGN of each query's own accuracy_diff_b_minus_a is exchangeable (equally likely positive or negative), independently across queries. This does not assume pairs within a query are independent, and does not assume anything about the magnitude of each query's difference.
  - Power caveat: Only 6 query clusters: p-values from this test are discrete (coarsest possible two-sided p = 2/64 = 0.03125) and power is low. Failing to reject the null here is not evidence of equivalence between the two checkpoints.
- **Conclusion category** (conservative combined read, alpha=0.05): NO_CLEAR_DIFFERENCE
- Score-margin difference (B-A), over 582 pairs with finite scores (0 excluded): mean = -0.014957, median = -0.000173

  Per-query breakdown:

  | query | n | acc(A) | acc(B) | diff(B-A) | win | tie | loss |
  |---|---|---|---|---|---|---|---|
  | Banana | 104 | 0.663462 | 0.423077 | -0.240385 | 17 | 45 | 42 |
  | CrackerBox | 99 | 0.797980 | 0.919192 | 0.121212 | 13 | 85 | 1 |
  | MustardBottle | 120 | 0.816667 | 0.775000 | -0.041667 | 3 | 109 | 8 |
  | PowerDrill | 68 | 0.647059 | 0.735294 | 0.088235 | 10 | 54 | 4 |
  | Scissors | 115 | 0.730435 | 0.652174 | -0.078261 | 14 | 78 | 23 |
  | TomatoSoupCan | 76 | 0.342105 | 0.394737 | 0.052632 | 20 | 40 | 16 |

## What these results do and do not prove

- `pair_accuracy` measures whether the model scored the labeled-positive grasp candidate above the labeled-negative one for a given pair. It is **not** a grasp success rate and must never be reported or read as one -- no physical grasp attempt, simulated or real, was executed to produce this data.
- Each of base/nodist/nozrel/full_v2 is a single checkpoint per ablation configuration. A significant pairwise difference between two checkpoints is evidence about *those two trained models' pairwise-ranking accuracy on this fixed validation split* -- it is not evidence that any specific input feature (e.g. `dist_to_centroid`, `z_rel`) *causes* the difference. No causal feature-importance claim is made or supported by this analysis.
- The cluster bootstrap above resamples by query (6 clusters) because that is the finest grouping reconstructable from the committed pair_results.jsonl files; it does not resample by training seed or by independently retrained model replicates (there is only one checkpoint per configuration), so these intervals do not capture across-training-run variance.
- The cluster-level sign-flip permutation test (Phase 4) is **not McNemar's test** and is never labeled as one. It answers a deliberately different question at a deliberately different unit of analysis (query, not pair) than the pair-level McNemar test above -- the two are expected to disagree when within-query correlation is substantial, and a `PAIR_LEVEL_ONLY` conclusion_category is exactly that disagreement made explicit, not an error.
- With only 6 query clusters, the cluster-level permutation test's p-values are coarse and discrete (minimum possible two-sided p = 2/64 = 0.03125). **A `NO_CLEAR_DIFFERENCE` or `PAIR_LEVEL_ONLY` conclusion_category is not evidence that the two checkpoints are equivalent** -- it may simply reflect that 6 clusters cannot resolve a real but modest effect. No equivalence claim is made anywhere in this report.
- `pair_level_mcnemar`, `query_cluster_permutation`, and `query_cluster_bootstrap_ci` are reported as three separate evidence columns for every comparison and are never collapsed into a single significance label. `conclusion_category` is a conservative, explicitly-named combined read (see its docstring in real_analysis.py for the exact rule) reported *alongside*, not instead of, all three -- it is computed from the raw (not Holm-adjusted) pair-level p-value, since Holm-Bonferroni is a separate, family-wise concern across the five planned comparisons.
- Both raw and Holm-Bonferroni-adjusted p-values/interpretations are reported for every comparison above; a comparison that is significant under the raw p-value but not after the Holm adjustment (or vice versa) is reported exactly as such, not resolved into a single number.
- All five comparisons are reported regardless of outcome, including any that are NOT_SIGNIFICANT under either rule -- no null or contradictory result is omitted.
