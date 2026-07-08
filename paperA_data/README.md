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

- `phase1_v2/ikmargin_{Pear,MustardBottle,CrackerBox}.jsonl` — IK-margin strategy, ensemble_n=10, all 3 objects
- `phase1_v2/consensus_MustardBottle.jsonl` — consensus strategy, ensemble_n=10, MustardBottle only (fresh run)
- `phase1_pilot/consensus_trials_n10.jsonl` — consensus strategy, ensemble_n=10, Pear+CrackerBox (per
  `phase1_v2_full.sh`'s own comment, this is the matching n=10 consensus data for those two objects —
  use this + `consensus_MustardBottle.jsonl` together for the full 3-object consensus-vs-ikmargin comparison)
- `phase1_pilot/consensus_trials.jsonl` — consensus strategy, **ensemble_n=5** (earlier/smaller pilot,
  Pear+CrackerBox only). Different ensemble size — not directly comparable to the n=10 files above.
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

## Explicitly NOT included (because it does not exist as a file anywhere)

Paper B's real-robot servo current/load data (tissue package / towel / glasses case) has no
backing data file on disk anywhere — it exists only as a hand-summarized table inside a memory
note (`~/.claude/projects/-lena-projects-lerobot/memory/project_paper_b_execution_prep.md`), and
the scripts that produced it (`live_probe_gripper.py`, `signal_watch.py`, etc.) were only ever in
`/tmp` and are gone. Do not treat this archive as covering Paper B.
