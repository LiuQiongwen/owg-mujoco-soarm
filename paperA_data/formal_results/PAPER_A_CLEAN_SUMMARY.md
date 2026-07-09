# Paper A — authoritative clean-data summary (2026-07-09)

**This is the file to cite for Paper A submission.** It consolidates every
formal-results number after excluding Scissors, which is confirmed invalid
data (not a judgment call — see "Why Scissors is excluded" below). Every
number below is reproducible by running the named script against the files
in this directory; nothing here is hand-computed.

## Why Scissors is excluded

`tango_robot/ui.py`'s `_cfm_sample_candidates()` matches an object name to a
trained conditioning key via naive substring/prefix check (`k in key or
key.startswith(k)`). All three checkpoints (`cfm_allobj_ot`, `cfm_allobj`,
`ddpm_allobj`) were trained on the 7-key set `{banana, pear, mustard,
cracker, drill, can, cylinder}` — Scissors' training rows exist under the
`cylinder` key (aliased there by `scripts/collect_lggsn_data.py:443`), but
`"scissors"` has no substring/prefix relationship with `"cylinder"`, so the
match fails at inference time in `ui.py`. When it fails, `_cfm_sample_candidates`
returns `None` and `ui.py` silently falls back to uniform-random CoM sampling
— the pre-CFM baseline, not any of the three generative methods.

Confirmed by direct execution (2026-07-09), not inferred:
- Running the exact matching code against all 7 object names: only `Scissors`
  fails to match; the other 6 (Banana/Pear/MustardBottle/CrackerBox/PowerDrill/
  TomatoSoupCan) all match correctly.
- `exp1_variance/`'s three method files give Scissors an **identical 40/50
  (80.0%)** success rate under OT-CFM, CFM-noOT, and DDPM, with **0 discordant
  trials** in every pairwise McNemar comparison (50/50 trials agree exactly) —
  only possible if all three "methods" ran the same non-CFM fallback.
- `phase0_diag_extended/`'s Scissors contact features are exactly `0.0` for
  all 50 trials on all 3 features — consistent with (but not proof of) the
  fallback, and a second, independent reason the object can't be used for the
  contact-feature analysis regardless.

Full incident record: `paperA_data/README.md`, "CRITICAL" section (first
found 2026-07-08, fallback mechanism confirmed 2026-07-09).

**Excluded object**: Scissors only. Banana/Pear/MustardBottle/CrackerBox/
PowerDrill/TomatoSoupCan are all confirmed clean (checkpoint keys match
correctly; no fallback).

## 1. Contact-feature discriminability (Bonferroni/BH), 5 clean objects

Script: `scripts/run_contact_features_stats_5obj_clean.py` →
`contact_features_bonferroni_bh_5obj_clean.csv`. 3 features × 5 objects
(Pear/MustardBottle/CrackerBox/TomatoSoupCan/PowerDrill) = **15 tests**,
Bonferroni α = 0.05/15 = **0.00333** (was 0.05/18 = 0.00278 in the
now-superseded 6-object file that still included Scissors' 3 degenerate
p=1.0 rows).

**No significance conclusion changed** vs. the old 18-test file — the
looser α (0.00333 vs 0.00278) doesn't flip any borderline case, and removing
Scissors' three p=1.0 rows from the Benjamini-Hochberg ranking doesn't move
any of the other 15 adjusted p-values either. Result, read as a spectrum
(unchanged from the prior read, now on a clean 5-object footing):

| Tier | Objects / features |
|---|---|
| Bonferroni-surviving (large effect) | MustardBottle (`normal_consistency` p=0.00029, `local_point_density` p=0.00030); Pear (`normal_consistency` p=0.00068); TomatoSoupCan (`local_point_density` p=0.00123) |
| BH-only signal (p_BH < 0.05, fails Bonferroni) | MustardBottle `contact_width_ratio` (p_raw=0.0114, p_BH=0.0297); CrackerBox `local_point_density` (p_raw=0.0119, p_BH=0.0297); TomatoSoupCan `normal_consistency` (p_raw=0.0152, p_BH=0.0325) |
| No signal under either correction | CrackerBox `contact_width_ratio` (p_BH=0.060), Pear `local_point_density` (p_BH=0.139), and **all 3 PowerDrill features** (p_BH≥0.256) — PowerDrill is the only object with a clean, unqualified null across every feature |

**Can support**: object-dependence is a spectrum (robust signal →
correction-dependent signal → clean null), now demonstrated cleanly across
5 objects with no invalid data mixed in. **Cannot support**: any claim that
names a specific geometric property as *the* explanation — this was never
tested, independent of the Scissors issue.

## 2. exp1_variance method comparison (OT-CFM / CFM-noOT / DDPM), 6 clean objects

Script: `scripts/run_exp1_significance.py` →
`exp1_variance_significance.csv`, scope = **`ALL_excl_scissors`** (not
`ALL`, which still includes the invalid Scissors rows and is kept only for
provenance/comparison). n=300/method (Banana/Pear/MustardBottle/CrackerBox/
PowerDrill/TomatoSoupCan × 50).

| Comparison | rate A | rate B | Mann-Whitney p | Welch t-test p | McNemar (paired) p |
|---|---|---|---|---|---|
| OT-CFM vs CFM-noOT | 0.72 | 0.78 | 0.090 | 0.090 | 0.0356 |
| OT-CFM vs DDPM(DDIM-50) | 0.72 | 0.7933 | **0.037** | **0.036** | 0.00815 |
| CFM-noOT vs DDPM(DDIM-50) | 0.78 | 0.7933 | 0.691 | 0.691 | 0.627 |

**Note this changes one conclusion from the old (Scissors-contaminated)
`ALL` row**: OT-CFM vs DDPM's *unpaired* tests (Mann-Whitney/Welch) move from
non-significant (p≈0.051/0.051 with Scissors included) to **significant**
(p≈0.037/0.036) once Scissors is removed — Scissors' tied 80%/80%/80% rate
was diluting the pooled effect. The paired McNemar result (the statistically
preferred test for this matched design) was already significant either way
and is numerically unchanged (Scissors contributed exactly 0 discordant
pairs, so removing it can't move McNemar's discordant-pair count).

Per-object breakdown (all 6 clean objects, unaffected by this update except
that Scissors' row is now explicitly marked do-not-cite in the CSV):

| Object | OT-CFM | CFM-noOT | DDPM | Driving comparison |
|---|---|---|---|---|
| Banana | 100% | 100% | 100% | degenerate, no test |
| Pear | 56% | 76% | 76% | OT-CFM significantly worse (McNemar p=0.041 / p=0.021) |
| MustardBottle | 70% | 66% | 70% | no significant pairwise difference |
| CrackerBox | 44% | 42% | 42% | no significant pairwise difference |
| PowerDrill | 82% | 86% | 88% | no significant pairwise difference |
| TomatoSoupCan | 80% | 98% | 100% | OT-CFM significantly worse (McNemar p=0.012 / p=0.002) |

**Can support**: "OT-CFM is significantly less reliable than CFM-noOT and
DDPM(DDIM-50) specifically on Pear and TomatoSoupCan (paired test); pooled
across the 6 valid objects, OT-CFM is also significantly worse than DDPM by
the unpaired tests now that Scissors' diluting tie is removed." **Cannot
support**: any claim that pools in Scissors, or any ODE-vs-SDE / AUC claim
(out of scope for this data regardless of Scissors).

## 3. ikmargin vs. consensus (candidate-selection strategy), unaffected

Scissors was never part of this experiment (`phase1_v2/`,
`phase1_matched_n10/` only ever covered Pear/MustardBottle/CrackerBox) — no
change needed. Restated here for completeness, authoritative file remains
`formal_results/ikmargin_vs_consensus_matched_n10.csv` (matched ensemble
size 10 vs 10):

| Object | ikmargin | consensus | Fisher's exact p |
|---|---|---|---|
| Pear | 6.0% | 68.0% | p=5.8e-11 |
| MustardBottle | 50.0% | 68.0% | p=0.103 (ns) |
| CrackerBox | 44.0% | 44.0% | p=1.0 (exact tie) |

## File map — what to cite vs. what's kept for provenance only

| Cite this | Not this (kept for provenance, Scissors annotated `excluded_reason`) |
|---|---|
| `contact_features_bonferroni_bh_5obj_clean.csv` | `contact_features_bonferroni_bh_6obj.csv` |
| `exp1_variance_significance.csv`, scope=`ALL_excl_scissors` + per-object rows (not `Scissors`) | same file's `scope=ALL` and `scope=Scissors` rows |
| `contact_features_bonferroni_bh.csv` (original 3-object, never had Scissors) | — unaffected |
| `ikmargin_vs_consensus_matched_n10.csv` | `ikmargin_vs_consensus.csv` (superseded for a different reason — ensemble-size confound, see README) |
