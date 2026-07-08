# Paper A raw data archive

Copied verbatim (md5 verified) from the Claude Code job scratchpad
`/home/lina/.claude/jobs/b899ad73/tmp/` on 2026-07-08, because that directory
is not part of any git repo and is subject to cleanup. This is a straight
copy — no files were regenerated or edited.

## exp1_variance/ — sampler variance experiment (7 objects x 5 orient_seed x 10 gen_seed = 350 trials/file)

- `raw_results.jsonl` = **OT-CFM** (ckpt `cfm_allobj_ot.pt`), produced by `scripts/experiment1_otcfm_variance.sh`
- `raw_results_CFM-noOT.jsonl` = **CFM-noOT** (ckpt `cfm_allobj.pt`)
- `raw_results_DDPM.jsonl` = **DDPM checkpoint (`ddpm_allobj.pt`) sampled with `DDIM_STEPS=50`** — this is NOT
  an independently-trained "DDIM model"; it's the DDPM model sampled via 50-step DDIM. Don't relabel it as a
  third method on equal footing with the two CFM variants without noting this.

Both shell scripts are in `scripts/`.

## phase1_v2/ + phase1_pilot/ — consensus vs IK-margin candidate-selection strategies

- `phase1_v2/ikmargin_{Pear,MustardBottle,CrackerBox}.jsonl` — IK-margin strategy, **ensemble size 10**
  (`--ikmargin-n 10`, pick the candidate with lowest IK error out of 10), all 3 objects, 50 trials each
  (5 orient_seeds x 10 ensemble repetitions).
- `phase1_v2/consensus_MustardBottle.jsonl` — consensus strategy, **ensemble size 5** (`--consensus-n 5`,
  pick the candidate closest to the median of 5), MustardBottle only, 50 trials (5 orient_seeds x 10 reps).
- `phase1_pilot/consensus_trials_n10.jsonl` — consensus strategy, **ensemble size 5**, Pear+CrackerBox,
  100 trials total (50 each).
- `phase1_pilot/consensus_trials.jsonl` — consensus strategy, ensemble size 5, Pear+CrackerBox, only
  **5 repetitions** instead of 10 (earlier/smaller pilot, 25 each). Superseded by `consensus_trials_n10.jsonl`.

  **⚠️ Correction (2026-07-08, caught while writing the formal significance-test scripts below): an
  earlier version of this README said the consensus files were "ensemble_n=10, matched to ikmargin".
  That was wrong** — checked `scripts/phase1_v2_full.sh` and `scripts/phase1_consensus_n10.sh` directly:
  consensus was always run with `--consensus-n 5`. The "n10" in `consensus_trials_n10.jsonl`'s filename
  means **10 repetitions of a 5-candidate ensemble**, not "ensemble size 10". So **every consensus-vs-
  ikmargin comparison in this repo compares a 5-candidate pool against a 10-candidate pool** — ensemble
  size is confounded with strategy choice. This is now stated explicitly in
  `formal_results/ikmargin_vs_consensus.csv` and must be carried into any paper text: do not claim this isolates
  "selection rule" as the only variable.
- `phase1_v2/pear_ensemble_reconstruction.json` — diagnostic-only, Pear only: per-ensemble candidate
  `pe_ik` values + which candidate ikmargin picked, used to explain why ikmargin fails badly on Pear (6%
  success vs consensus's 52%).

Generating script: `scripts/phase1_v2_full.sh` (+ `phase1_consensus_pilot.sh`, `phase1_consensus_n10.sh`
for the pilot-stage files).

## phase0_diag/ — base diagnostic dataset (150 trials: Pear/MustardBottle/CrackerBox x 50 each)

- `trials.jsonl` — raw success/fail records, produced by `scripts/phase0_full_diagnostic.py`
  (see also `scripts/phase0_diagnostic_rerun.sh`)
- `data_with_ik.json` — `trials.jsonl` + per-trial IK reachability (`ik_ok`, `ik_pe_mm`) and pose (x/y/z/yaw)
- `ui_grasp_exec_snapshot.jsonl` — 300 records, used to backfill `width` (gripper opening) onto `data_with_ik.json`
- `data_with_contact_feats.json` — `data_with_ik.json` + contact geometry features
  (`local_point_density`, `normal_consistency`, `contact_width_ratio`), produced by
  `scripts/phase2_contact_features.py`. **This is the source of the Bonferroni-corrected
  Mann-Whitney table** (MustardBottle's normal_consistency/local_point_density survive
  Bonferroni with large effect sizes; Pear's normal_consistency survives; CrackerBox does not).

## scripts/

Analysis/generation scripts copied alongside their data for provenance:
`phase1_step2_causal.py`, `phase2_contact_features.py`, `phase1_causal_check.py`,
`phase0_full_diagnostic.py`, `experiment1_otcfm_variance.sh`, `experiment1_other_methods.sh`,
`phase1_v2_full.sh`, `phase1_consensus_pilot.sh`, `phase1_consensus_n10.sh`,
`phase0_diagnostic_rerun.sh`, `FIX_DESIGN.md` (seeding-bug fix design notes, not applied to
production code as of this copy).

**Caveat on `phase1_step2_causal.py`**: this script only prints to stdout, it does not write a
results file. Its hardcoded `OBJECTS = ["Pear", "CrackerBox"]` means it does not cover
MustardBottle. Any previously-seen printed p-values from this script exist only as free text in
an old session transcript, not as a saved, reproducible result — re-run it against the data in
this directory to get citable numbers.

## formal_results/ — formal, code-generated statistical outputs (2026-07-08)

These three files replace every number that previously existed only as free text in an old
session transcript. Each was produced by running the matching script in `scripts/` against the
raw data in this directory (not against the original job scratchpad) — anyone can regenerate them
with `python3 scripts/<name>.py` and get byte-identical numbers. **Any number quoted in the paper
from these three analyses must cite the specific row/file below, not the old transcript.**

### `formal_results/exp1_variance_significance.csv` (from `scripts/run_exp1_significance.py`)

Pairwise comparison of OT-CFM / CFM-noOT / DDPM(50-step DDIM) on the exp1_variance data. Reports,
per scope (pooled "ALL" and per-object) and per method pair: success rates, unpaired Mann-Whitney U,
unpaired Welch's t-test, and **paired McNemar's exact test** (the statistically appropriate one here,
since all three methods share the identical (object, orient_seed, gen_seed) trial grid).

- **Pooled (ALL, n=350/method)**: unpaired tests are not significant at α=0.05 (Mann-Whitney/t-test
  p≈0.11 for OT-CFM vs CFM-noOT, p≈0.05 for OT-CFM vs DDPM, p≈0.71 for CFM-noOT vs DDPM). The paired
  McNemar test, which has more power because it uses the matched trial design, **is** significant:
  OT-CFM vs CFM-noOT p=0.036, OT-CFM vs DDPM p=0.008; CFM-noOT vs DDPM remains non-significant (p=0.63).
- **Per-object breakdown** shows this pooled effect is driven almost entirely by **Pear** (OT-CFM 56%
  vs CFM-noOT/DDPM 76%, McNemar p=0.041 / p=0.021) and **TomatoSoupCan** (OT-CFM 80% vs CFM-noOT 98% /
  DDPM 100%, McNemar p=0.012 / p=0.002). CrackerBox, MustardBottle, PowerDrill, Scissors show no
  significant pairwise difference; Banana is 100% for all three methods (test undefined, see `note` column).
- **Can support**: "OT-CFM is significantly less reliable than CFM-noOT and DDPM(DDIM-50) specifically
  on Pear and TomatoSoupCan (paired test); CFM-noOT and DDPM(DDIM-50) are statistically indistinguishable
  from each other on every object tested."
- **Cannot support**: any ODE-vs-SDE claim, any AUC number, or a blanket "method X is better than method
  Y overall" — the pooled McNemar result is driven by 2 of 7 objects, not a uniform effect, and pooling
  across objects with very different baseline rates (44%–100%) needs the per-object table alongside it,
  never just the pooled row.

### `formal_results/contact_features_bonferroni_bh.csv` (from `scripts/run_contact_features_stats.py`)

9 tests (3 contact features x 3 objects) on `phase0_diag/data_with_contact_feats.json`: Mann-Whitney U,
raw p, rank-biserial effect size, Bonferroni-corrected significance (α=0.05/9=0.00556), and proper
Benjamini-Hochberg adjusted p-values (via `scipy.stats.false_discovery_control`, not just a threshold).
Numbers match the independently hand-verified table from the previous inventory pass exactly.

- **Can support**: MustardBottle's `normal_consistency` (p=0.00029, rank-biserial=−0.60) and
  `local_point_density` (p=0.00030, rank-biserial=−0.62) survive Bonferroni with large effect sizes;
  Pear's `normal_consistency` (p=0.00068, rank-biserial=−0.47) survives Bonferroni with a medium-large
  effect. These three are robust findings under the strictest correction.
- **Cannot support**: "CrackerBox has no pre-execution contact signal" as an unqualified claim —
  CrackerBox's `local_point_density` (p=0.0119) and `contact_width_ratio` (p=0.0322) fail Bonferroni but
  **pass** Benjamini-Hochberg (p_BH=0.021, 0.048). Which correction method you pick changes the CrackerBox
  conclusion; state the method explicitly whenever citing CrackerBox.

### `formal_results/ikmargin_vs_consensus.csv` (from `scripts/run_ikmargin_vs_consensus.py`)

Fisher's exact test (unpaired 2x2, appropriate since the two strategies don't share trials/seeds) on
ikmargin (ensemble 10) vs consensus (ensemble 5) success counts, per object.

| object | ikmargin (n=10 pool) | consensus (n=5 pool) | Fisher's exact p |
|---|---|---|---|
| Pear | 6.0% (3/50) | 52.0% (26/50) | **p=4.1e-7** |
| MustardBottle | 50.0% (25/50) | 56.0% (28/50) | p=0.69 (ns) |
| CrackerBox | 44.0% (22/50) | 42.0% (21/50) | p=1.0 (ns) |

- **Can support**: on Pear specifically, the ikmargin selection rule (as tested, with a 10-candidate
  pool) performs dramatically and significantly worse than the consensus rule (as tested, with a
  5-candidate pool). `phase1_v2/pear_ensemble_reconstruction.json` gives the per-candidate diagnostic
  detail for *why* (the lowest-IK-error candidate is frequently not the one that actually succeeds).
- **Cannot support**: "consensus is a better selection rule than ikmargin" as a general, ensemble-size-
  independent claim — see the confound note above and in the CSV. On MustardBottle and CrackerBox, where
  the confound is the same, there is no significant difference at all, which is also consistent with the
  gap being partly or wholly an ensemble-size effect rather than a selection-rule effect. A controlled
  re-run (same ensemble size for both strategies) would be needed to separate the two, and was not done.

## Explicitly NOT included (because it does not exist as a file anywhere)

Paper B's real-robot servo current/load data (tissue package / towel / glasses case) has no
backing data file on disk anywhere — it exists only as a hand-summarized table inside a memory
note (`~/.claude/projects/-lena-projects-lerobot/memory/project_paper_b_execution_prep.md`), and
the scripts that produced it (`live_probe_gripper.py`, `signal_watch.py`, etc.) were only ever in
`/tmp` and are gone. Do not treat this archive as covering Paper B.
