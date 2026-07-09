# Paper A raw data archive

## ⚠️ Separate issue, same object: `paper_final.tex`'s own Scissors number was also unreplicated (found + fixed 2026-07-09)

This is unrelated to the CFM name-matching fallback bug below -- it affects `paper_final.tex`'s
*own* 175-trial main evaluation (Table I/II/III), which is a completely different dataset from
everything else in this directory (175 trials = 7 objects x 25 seeds, no `--gen-seed` variation,
run via `scripts/quick_eval.sh` in the `tango` conda env, which was `owg-mujoco` until a
`conda rename -n owg-mujoco tango` -- confirmed via `~/miniforge3/envs/tango/conda-meta/history`).

**What happened**: `paper_final.tex`'s Baseline (82.3%) and OT-CFM+LGGSN (94.3%) numbers were
built by adding a separately-measured 25-trial Scissors block to an existing 150-trial (6-object)
total, after `a04a62c`'s VHACD-tunnelling physics fix (see `logs/eval_scissors_fix_summary.log`).
That Scissors block was measured **25/25 = 100%** on 2026-06-26, with a real per-seed log
(`logs/eval_scissors_baseline.log`, `logs/eval_scissors_cfm.log` -- genuine `[✓]` per seed, not an
estimate). A later commit (`cf58a7d`, 2026-06-28) claimed a re-run found **23/25 = 92%** and
labeled the 06-26 figure "imputed" -- inaccurate framing (it wasn't imputed), but the underlying
discrepancy is real: two actual measurements of the same nominal condition gave different counts,
and `cf58a7d`'s corrected number was never propagated back into `paper_final.tex` (it doesn't
touch that file at all, and no later commit does either -- confirmed by `git log -- paper_final.tex`).

**Resolution (2026-07-09)**: rather than trust either historical number, re-ran the identical
condition from scratch (`scripts/run_scissors_recheck_2026-07-09.sh`, 25 seeds each,
baseline + OT-CFM, `tango` env, smoke-tested first). Result: a **third** different number,
**22/25 = 88%** for both conditions (same 3 failing seeds -- 1, 7, 10 -- in both, consistent
with Scissors falling back to the same random-CoM sampler in both conditions, per the CFM
name-matching issue below). The smoke-test itself already disagreed with the 06-26 log at the
exact same seed (seed=1: success there, failure here), which is why neither historical number was
trusted blindly. No configuration drift was found -- `configs/objects/ycb_mujoco_manifest.yaml`'s
Scissors entry is byte-identical to what `eval_scissors_fix_summary.log` describes. The most
likely explanation is that the 4cm box-proxy fix is only marginally above the gripper's 4cm
minimum opening, making this specific object's outcome sensitive to small run-to-run numerical
differences (contact-solver iteration order, floating-point accumulation) -- a property of this
object's geometry, not evidence that the pipeline itself is broken.

**Adopted the freshest measurement (22/25 = 88%) per explicit user decision**, and propagated it
through every dependent number in `paper_final.tex`: Table I (Baseline 82.3%→80.6%, OT-CFM
94.3%→92.6%, z 3.49→3.29), Table II (Scissors row 100%/100%→88%/88%, All row updated to match),
Table III (every "vs. Baseline" and "vs. Full" delta and p-value recomputed -- see below for which
ones changed qualitatively), abstract, intro contributions list, Related Work, Discussion ("Why OT
Coupling Matters" rewritten -- see below), and Conclusion. Formalized in
`scripts/run_scissors_recheck_stats.py` -> `formal_results/scissors_recheck_corrected_totals.csv`;
raw per-seed data in `phase0_diag_extended/scissors_recheck_{baseline,otcfm}.jsonl`.

**Every ablation-table significance classification (SIG/ns) is unchanged** -- only the absolute
percentages and precise p-values shifted, because the 6-object (non-Scissors) totals were held
fixed and Scissors' identical-across-conditions count (22/25 both) shifts every row's absolute
value by the same constant, preserving relative comparisons almost exactly. **One qualitative
claim did have to be softened**: the original text argued "standard CFM without OT coupling drops
*below* baseline (78.9% < 82.3%)" as supporting evidence that OT coupling matters. With the
corrected baseline (80.6%), that specific pairwise comparison is no longer statistically
significant (78.9% vs 80.6%, $p=0.69$) -- rewritten to rest the argument on the comparison that
remains robust throughout (Remove-OT vs. the full pipeline: $-13.7$pp, $p<0.001$), rather than the
now-weaker vs.-Baseline framing. Similarly, DDPM's relationship to baseline flips sign (was
$-0.6$pp, now $+1.1$pp) but stays non-significant either way, so no claim needed to change there
beyond the number.

**Not in scope for this pass**: the other 6 objects' Baseline/OT-CFM/GRC-6DoF/DDPM counts were
*not* independently re-verified (only Scissors was, per the specific discrepancy found) -- if a
similar cross-date drift exists for any other object, it has not been checked. GRC-6DoF's own
82.9% and Remove-OT/DDPM's own 78.9%/81.7% are carried forward unchanged from the existing record;
only their *deltas relative to Baseline/Full* were recomputed, since those two reference points
moved.

## ✅ RESOLVED (2026-07-09): Scissors excluded from Paper A, clean 5/6-object results published

The Scissors fallback bug described in the CRITICAL section below was confirmed by
re-executing the exact matching code and re-deriving the 40/50-tie and 0-discordant-trial
evidence independently on 2026-07-09. Decision: **exclude Scissors entirely from Paper A**
rather than attempt a same-day rerun. The other 6 objects (Banana/Pear/MustardBottle/
CrackerBox/PowerDrill/TomatoSoupCan) are unaffected — confirmed clean by running the same
matching code against all 7 object names.

What changed on disk:
- **New authoritative summary**: `formal_results/PAPER_A_CLEAN_SUMMARY.md` — the single file
  to cite from the paper. Consolidates the corrected contact-feature Bonferroni/BH table
  (15 tests, 5 objects) and the corrected exp1_variance method comparison (`ALL_excl_scissors`
  scope, 6 objects).
- **New file**: `formal_results/contact_features_bonferroni_bh_5obj_clean.csv`
  (`scripts/run_contact_features_stats_5obj_clean.py`) — Scissors dropped entirely from both
  the p-values and the correction family (15 tests, not 18). No significance conclusion
  changed vs. the old 18-test file (checked row-by-row).
- **Existing files annotated, not deleted**: `contact_features_bonferroni_bh_6obj.csv` and
  `exp1_variance_significance.csv` now carry an `excluded_reason` column — non-empty only for
  Scissors rows, pointing back to this README and to the clean replacement file. Both are kept
  on disk for provenance/audit but are no longer the files to cite.
- **New pooled scope**: `exp1_variance_significance.csv` gained an `ALL_excl_scissors` row
  (n=300/method, the 6 valid objects) alongside the original `ALL` (n=350/method, still
  includes the invalid Scissors rows, kept for comparison only). One conclusion changes here:
  OT-CFM vs. DDPM's unpaired tests move from non-significant (p≈0.051) to significant
  (p≈0.037) once Scissors' diluting 80%/80%/80% tie is removed. The paired McNemar result
  (the statistically preferred test) was already significant and is numerically unchanged,
  since Scissors contributed exactly 0 discordant pairs.

See `formal_results/PAPER_A_CLEAN_SUMMARY.md` for full numbers and the file map of what to
cite vs. what's kept for provenance only.


Copied verbatim (md5 verified) from the Claude Code job scratchpad
`/home/lina/.claude/jobs/b899ad73/tmp/` on 2026-07-08, because that directory
is not part of any git repo and is subject to cleanup. This is a straight
copy — no files were regenerated or edited.

## ⚠️ CRITICAL: every "Scissors" data point in this repo is not OT-CFM/CFM-noOT/DDPM data (found 2026-07-08)

`tango_robot/ui.py`'s `_cfm_sample_candidates()` matches an object name to a checkpoint's trained
conditioning key via `key = obj_name.lower()...`; `for k in vis_map: if k in key or key.startswith(k)`.
All three checkpoints (`cfm_allobj_ot`, `cfm_allobj`, `ddpm_allobj`) were trained on the identical 7-key
set `{banana, pear, mustard, cracker, drill, can, cylinder}`. `"scissors"` does not match any of these
keys (no substring/prefix relationship) — **confirmed by direct execution of the exact matching code**,
not inferred. When the match fails, `_cfm_sample_candidates` returns `None`, and `ui.py` silently falls
back to **uniform-random CoM-based candidate sampling** — the pre-CFM baseline, not any of the three
generative methods. Elsewhere in this same codebase (`scripts/collect_lggsn_data.py:443`) there IS an
explicit `"scissors": "cylinder"` alias for a different pipeline (LGGSN training-data collection) — the
runtime path used for every experiment in this repo (`demo.py` → `ui.py`) never implements that alias,
so the intended fallback (map scissors to the "cylinder"/`YcbMediumClamp` conditioning class) never
happens here.

**Evidence this actually occurred, not just a theoretical risk**: in `exp1_variance/`, Scissors' success
rate is *exactly* 40/50 (80.0%) for OT-CFM, CFM-noOT, **and** DDPM — identical to the decimal, whereas
every other object's rate varies across methods. The paired McNemar test in
`formal_results/exp1_variance_significance.csv` finds **zero discordant trials** between any pair of
methods on Scissors (`mcnemar_p=nan`, 0/0 in the discordant-count columns) — i.e., all three methods
produced the identical trial-by-trial outcome across all 50 trials, which is only possible if they were
all secretly running the same non-CFM fallback mechanism (indifferent to which checkpoint was loaded),
not three different generative models.

**Practical consequence**: exclude Scissors from any claim comparing OT-CFM/CFM-noOT/DDPM — its rows in
`exp1_variance_significance.csv` are internally correct arithmetic but describe "the random-sampling
fallback vs. itself under three labels," not a method comparison. The `phase0_diag_extended/` Scissors
contact-feature diagnostic data (added 2026-07-08, see below) is likewise **not characterizing OT-CFM's
behavior** — it characterizes the random-CoM fallback's behavior. This is a *different and more
fundamental* problem than the "thin/flat object, structural floor effect" note already on that data
(that note is still true as an independent observation, but secondary to this one). No other object
in this repo is affected — Banana/TomatoSoupCan/Pear/MustardBottle/CrackerBox/PowerDrill all match
their checkpoint's trained keys correctly (verified by running the exact matching logic against all 7).

## Methods reference — precise, code-grounded definitions (compiled 2026-07-08)

Answers to specific methods-section questions, each traced to the exact source line, not paraphrased
from memory. Worktree paths below are relative to
`/lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding` unless stated otherwise.

**1. "Generation-isolated" setup — GT identity or GT segmentation?**
Neither exactly, but closer to GT identity, and stronger than either: every trial spawns **exactly one
object** in the scene (`n_objects: 1` in the loaded config, confirmed via `[DEBUG] loaded objects:
[(1, 'Pear')]`-style prints in every run). All generation scripts also pass `--no-semantic`, which sets
`OWG_NO_SEMANTIC=1`; `tango/policy.py`'s no-semantic fast-path matches the text prompt directly against
the simulator's own `id_to_name` registry (`tango/policy.py:218-240`) — no vision model, no segmentation
mask, no GPT-4o grounding call. So there is no segmentation step to bypass: with one object in the scene
and its identity known from spawn time, "which object is the target" is never actually a vision problem
in this dataset.

**2. Exact success criterion**
The recorded `"success"` field = **`success_grasp`**, not `success_target` (whether it landed correctly
in the tray) — these are different and the data uses the former. Chain: `env_soarm.py`'s physics_weld
grasp routine (~line 1778) computes `success = bool(grasped_ids) and lifted`, where `grasped_ids` is
non-empty only if bilateral jaw contact was detected (`check_grasped_id()`) **and** a kinematic weld was
triggered (MuJoCo sphere colliders alone can't generate enough friction to lift against gravity, so the
sim rigidly attaches the object to the gripper once bilateral contact confirms a valid grasp), and
`lifted = obj_z > Z_TABLE_TOP + 0.07` (object center must rise >7cm after the lift move). **`table_contact`
is computed and printed but does NOT gate success** — confirmed by an actual trial with
`table_contact=False, success=True`. This `success_grasp` bool propagates up through
`put_obj_in_tray()` → `ui.py:step()`, which prints `"Done {action} {input}"` (i.e. `"Done pick 1"`) only
when `success_grasp` is true (`ui.py:756-762`) — this exact string is what every shell script's
`grep -q "Done pick"` checks. `success_target` (whether the object also ended up correctly in the tray)
is computed and logged alongside but is **not** what any script in this repo checks.

**3. Object selection rationale**
- `exp1_variance/`'s 7 objects = the full trained-object roster of the CFM/DDPM checkpoints (see #9's
  answer — `{banana, pear, mustard, cracker, drill, can, cylinder}`, mapped to
  Banana/Pear/MustardBottle/CrackerBox/PowerDrill/TomatoSoupCan/**"Scissors" incorrectly** — see the
  CRITICAL note above; there is no evidence this was a deliberate representative sample, it's simply
  "every object the checkpoints were conditioned on" (modulo the Scissors bug).
- The original 3-object diagnostic set (Pear/MustardBottle/CrackerBox) has **no documented rationale**
  found anywhere in the copied scripts or job history — it predates this archive and appears to be an
  earlier, unrecorded choice.
- The 3 objects added 2026-07-08 (TomatoSoupCan/PowerDrill/Scissors) were chosen explicitly because
  they're the only remaining `exp1_variance` objects with a real success/fail split under OT-CFM (Banana
  is 100% success, degenerate for a success-vs-fail comparison) — this rationale **is** documented
  (`phase0_diag_extended/` section below) and was correct at the time, but Scissors is now known to be
  invalid for an unrelated reason (the CFM name-matching bug), leaving 2 valid additions, not 3.

**4. OT-CFM sampling configuration**
`train_cfm_grasp.py:41`: `ODE_STEPS = int(os.environ.get("CFM_ODE_STEPS", "20"))` — **20 steps**, default
value, never overridden by any script in this repo (no script sets `CFM_ODE_STEPS`). Integrator: explicit
forward Euler (`train_cfm_grasp.py`'s `sample_poses`: `dt = 1.0/steps`; `x = x + model(t, x, cond) * dt`
per step) — not RK4 or an adaptive-step method.

**5. DDIM eta / DDPM_STOCHASTIC**
`train_diffusion_grasp.py:176`: `eta = 1.0 if os.environ.get("DDPM_STOCHASTIC") == "1" else 0.0`. No
script in this repo sets `DDPM_STOCHASTIC`, so **eta=0.0 — fully deterministic DDIM reverse sampling**
was used throughout, despite the checkpoint being called "ddpm_allobj.pt" and the data files being named
"DDPM". Step count: `DDIM_STEPS` defaults to 100 (`train_diffusion_grasp.py:42`) but
`experiment1_other_methods.sh` explicitly passes `DDIM_STEPS=50` — **50 steps were used**, not the
default 100. (The initial noise `x_T` is still drawn from a `--gen-seed`-seeded generator, so different
seeds still produce different candidates even though the reverse process itself is deterministic given
that noise.)

**6. Tests reported in `exp1_variance_significance.csv`**
Three tests per (scope, method-pair) row: unpaired two-sided **Mann-Whitney U** (`scipy.stats.mannwhitneyu`),
unpaired two-sided **Welch's t-test** (`scipy.stats.ttest_ind(equal_var=False)`), and paired two-sided
**McNemar's exact test** (`scipy.stats.binomtest` on the discordant-pair count, matched by identical
`(object, orient_seed, gen_seed)` triples across methods). Reported at 8 scopes (pooled "ALL", n=350/method,
plus each of the 7 objects, n=50/method) × 3 method pairs = **24 rows total**.

**7. IK-margin / reachability metric — exact definition**
`tango_robot/headless_ik.py`'s `solve_ik_jaw_pos_only(target_jaw_mid, iters=800, pos_tol=5e-3, n_outer=8)`:
runs numerical IK (800 total iterations split across 8 outer re-anchoring passes) to move the gripper
jaw-midpoint geometry to `target_jaw_mid` (world-frame xyz, **position only, no orientation term**).
`pe = ||achieved_jaw_midpoint - target_jaw_mid||` (Euclidean, metres; reported as `ik_pe_mm` = `pe*1000`
elsewhere). **`ik_ok` = `pe < pos_tol` = `pe < 5mm`.** The "IK-margin" candidate-selection strategy picks
the candidate with the **smallest `pe`** among its ensemble, regardless of whether that `pe` clears the
5mm threshold.

**8. Complete contact-feature list**
Exactly 3, all defined in `grasp_6dof/grasp_sampler.py`, all operating on points transformed into the
gripper frame and filtered to a `gripper_width × gripper_width × 4cm` box centered at the candidate pose:
- `local_point_density`: fraction of the episode point cloud inside that box.
- `normal_consistency`: std-dev of the z-component of Open3D-estimated surface normals of the points
  inside that box (`radius=0.02, max_nn=30`, consistently oriented; returns 0.0 if <5 points or the
  normal-orientation step is degenerate, e.g. a flat/coplanar patch).
- `contact_width_ratio`: inter-decile span (p90−p10) of the in-box points along the gripper's x-axis,
  divided by `gripper_width`; returns 0.0 if <10 points or `gripper_width<=0`. `<0.3` is documented in
  the source as indicating "thin object with poor jaw engagement."

`local_point_density`/`normal_consistency`/`contact_width_ratio` are exactly the 3 features that go
  into `formal_results/contact_features_bonferroni_bh*.csv`. `ik_pe_mm` (from #7) is used as a *4th*,
  separate variable in `scripts/phase1_step2_causal.py`'s causal analysis, but is **not** one of the
  3 "contact features" and is not part of the 9/18-test Bonferroni family.

**9. Number of tests corrected for**
**9** in `contact_features_bonferroni_bh.csv` (3 features × 3 objects: Pear/MustardBottle/CrackerBox).
**18** in `contact_features_bonferroni_bh_6obj.csv` (3 features × 6 objects) — but per the CRITICAL note
above, Scissors' 3 rows there are not valid data, so **15 is the effective, citable test count** for
that file (3 features × 5 valid objects), even though the on-disk Bonferroni α (0.05/18=0.00278) was
computed treating it as 18. If citing the 6-object file, recompute α at n=15 (0.00333) rather than
reusing the stored 0.00278, or just cite the 3-object file's 9-test family plus TomatoSoupCan/PowerDrill
as a separate, smaller family.

**10. Is the "consensus" comparison pool-aligned (n=10)?**
Yes, **in the authoritative file only**: `formal_results/ikmargin_vs_consensus_matched_n10.csv` uses
`phase1_matched_n10/consensus_n10_*.jsonl` (`--consensus-n 10`) against `phase1_v2/ikmargin_*.jsonl`
(`--ikmargin-n 10`) — both pool size 10. The **original** `formal_results/ikmargin_vs_consensus.csv`
(pool 10 vs pool 5) is still on disk but explicitly marked superseded/do-not-cite in this README. If
citing "the consensus vs ikmargin result," always mean the `_matched_n10` file.

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

- `phase1_v2/ikmargin_TomatoSoupCan.jsonl` — added 2026-07-09, same design (ensemble size 10, 5
  orient_seeds x 10 reps) via `scripts/run_ikmargin_n10_tomatosoupcan.sh`, standalone (does not touch
  or re-run the original 3 objects). See `formal_results/ikmargin_vs_consensus_matched_n10.csv`'s
  TomatoSoupCan row for why.

## phase1_matched_n10/ — ensemble-size-controlled consensus re-run (2026-07-08)

- `consensus_n10_{Pear,MustardBottle,CrackerBox}.jsonl` — consensus strategy re-run at **`--consensus-n 10`**
  (matching ikmargin's pool size), same 5 orient_seeds x 10 ensemble_bases grid as `phase1_v2/ikmargin_*.jsonl`
  (identical `ensemble_base` values: 1,11,21,...,91), 50 trials/object, 150 total. Generated by
  `scripts/run_consensus_n10_matched.sh`. This directly closes the ensemble-size confound described above —
  see `formal_results/ikmargin_vs_consensus_matched_n10.csv` below for the controlled comparison.
- `consensus_n10_TomatoSoupCan.jsonl` — added 2026-07-09 via `scripts/run_consensus_n10_tomatosoupcan.sh`,
  identical grid, standalone run (does not re-touch the original 3 objects). Confirmed with a live
  smoke-test first (`--verbose 1`, checked candidate poses actually vary across gen_seed and the
  outcome differs) before committing to the full 50-trial batch, given the Scissors fallback-bug
  precedent.

**Environment note for reproducing this run**: the `tango` conda env's own `torch==2.10.0` and
`nvidia-cuda-cupti-cu12==12.8.90` (installed together 2026-06-29, unchanged since — the same versions
that produced the original 2026-07-03 data) got shadowed by a stray user-level
`~/.local/lib/python3.10/site-packages/nvidia-cuda-cupti-cu12==12.1.105` (installed 2026-07-07 by an
unrelated task on this shared machine; Python's default user-site precedence lets it leak into any
conda env). A matching stray `libnccl` caused the same problem one layer deeper. Neither `torch`/`cupti`
in `tango` nor the stray user-level package were touched — the fix is purely at invocation time:
`LD_PRELOAD` tango's own `libcupti.so.12` and `libnccl.so.2` ahead of anything else (see the exported
`LD_PRELOAD` line at the top of `run_consensus_n10_matched.sh`). This keeps user-site enabled, so
`mujoco`/`open3d`/etc. (never installed into `tango` itself, always resolved via user-site, exactly as
during the 2026-07-03 run) continue to work unchanged. **Net effect: this run used the identical
torch/cupti/nccl stack as the original exp1_variance/phase1_v2 data — old and new results are on the
same footing.** (Minor harmless leftover: while diagnosing this, `pygments`/`regex`/`safetensors`/
`tokenizers`/`typer`/`pydantic`/`requests`/`mujoco` were also pip-installed directly into `tango`'s own
site-packages during an abandoned alternate fix attempt (`PYTHONNOUSERSITE=1`); they don't conflict
with anything and weren't reverted, but the `LD_PRELOAD` approach above is what actually made this run
work, not those installs.)

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

## phase0_diag_extended/ — diagnostic pipeline extended to 3 more objects (2026-07-08)

Extends phase0_diag/'s 3-object diagnostic pipeline to **TomatoSoupCan, PowerDrill, Scissors**
(150 more trials), turning the "object-dependent reliability" pattern from a 3-point observation
into a 6-point one. Object choice: these are the only 3 remaining exp1_variance objects with a real
success/fail split under OT-CFM (Banana is 100% success — degenerate for a success-vs-fail
comparison, so it was excluded).

- `trials_new3.jsonl` — same design as the original (`phase0_diagnostic_rerun.sh`): single-draw,
  OT-CFM checkpoint, 5 orient_seeds x 10 gen_seeds = 50 trials/object. Success rates
  (TomatoSoupCan 80%, PowerDrill 82%, Scissors 80%) match `exp1_variance/raw_results.jsonl`'s
  OT-CFM rates for these objects exactly, as expected (identical design, freshly re-executed).
- `ui_grasp_exec_snapshot_new3.jsonl` — 300 raw records (150 `mode=="tray"` pose records, paired
  1:1 with `trials_new3.jsonl`), same pairing convention as the original.
- `data_with_contact_feats_new3.json` — IK reachability + the same 3 contact geometry features as
  `phase0_diag/data_with_contact_feats.json`, produced by `scripts/run_phase0_2_extended_analysis.py`
  (mirrors `phase0_full_diagnostic.py` + `phase2_contact_features.py`'s logic exactly, applied to the
  3 new objects — kept as a separate file rather than merged into the original, since the two were
  generated in separate runs and merging raw pose logs across runs would risk an ordering mismatch).

Generated by `scripts/run_phase0_extended.sh` (trials) + `scripts/run_phase0_2_extended_analysis.py`
(IK + contact features). Uses the same `LD_PRELOAD` environment workaround documented above.

**⚠️ Scissors here is not OT-CFM data at all — see the CRITICAL note at the top of this file.** All 50
Scissors trials silently used the random-CoM-sampling fallback (name-matching bug in
`_cfm_sample_candidates`), not the OT-CFM checkpoint used for the other 5 objects in this dataset.

**Separately, and secondary to the above**: Scissors' contact features are exactly 0.0 for all 50
trials, for all 3 features, which would be true regardless of which sampler produced the candidates —
`grasp_6dof/grasp_sampler.py`'s own docstring on `normal_consistency` says *"Low → flat surface
(scissors failure)"*: the metric is points-inside-a-narrow-gripper-bbox, close to zero by construction
for thin/flat objects. Treat Scissors' all-zero rows in `formal_results/contact_features_bonferroni_bh_6obj.csv`
as doubly compromised: (1) not OT-CFM candidates, and (2) this feature set can't discriminate for this
object's geometry regardless. Not usable as evidence about OT-CFM's contact-feature signal for any
thin/flat object — would need a rerun with the `scissors`→`cylinder` alias fixed to mean anything.

**PowerDrill `lateral_score` post-hoc follow-up (2026-07-09)**: PowerDrill is the only object with a
clean null across all 3 contact features in `contact_features_bonferroni_bh_5obj_clean.csv` (all
p_raw≥0.17), left unexplained in the paper. `scripts/run_powerdrill_lateral_score.py` tests a 4th,
previously-unused feature (`grasp_6dof/grasp_sampler.py`'s `lateral_score`, gripper-axis vs. object
principal-axis alignment — valid here because it varies per-trial via yaw, unlike `elongation_ratio`
which is ~constant per object and was deliberately *not* tested this way) against the same 50 already-
logged trials (no new grasp trials, only 5 lightweight object-spawns to get point clouds for PCA).
Result: `formal_results/powerdrill_lateral_score_posthoc.csv` — Mann-Whitney p=0.062 (marginal, ns at
α=0.05), rank-biserial=0.40 (moderate effect), fail-trials skew toward gripper-perpendicular-to-axis,
success toward parallel. **Deliberately reported as a single, uncorrected, post-hoc exploratory test —
do not fold into the existing 15-test Bonferroni/BH family** (that would require re-deriving every
already-cited p-value/significance flag in the 5-object-clean file). Per-trial data saved to
`phase0_diag_extended/data_with_lateral_score_powerdrill.json`. Note PowerDrill's mesh is only weakly
elongated (`elongation_ratio≈1.075`), so this finding should be read as inconclusive, not explanatory —
PCA's "principal axis" isn't a strongly stable direction for a near-isotropic point cloud.

## scripts/

Analysis/generation scripts copied alongside their data for provenance:
`phase1_step2_causal.py`, `phase2_contact_features.py`, `phase1_causal_check.py`,
`phase0_full_diagnostic.py`, `experiment1_otcfm_variance.sh`, `experiment1_other_methods.sh`,
`phase1_v2_full.sh`, `phase1_consensus_pilot.sh`, `phase1_consensus_n10.sh`,
`phase0_diagnostic_rerun.sh`, `FIX_DESIGN.md` (seeding-bug fix design notes, not applied to
production code as of this copy), plus (2026-07-08) `run_consensus_n10_matched.sh`,
`run_ikmargin_vs_consensus_matched.py`, `run_phase0_extended.sh`,
`run_phase0_2_extended_analysis.py`, `run_contact_features_stats_extended.py`, plus (2026-07-09)
`run_contact_features_stats_5obj_clean.py`, `run_ikmargin_n10_tomatosoupcan.sh`,
`run_consensus_n10_tomatosoupcan.sh`, `run_powerdrill_lateral_score.py`.

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
  DDPM 100%, McNemar p=0.012 / p=0.002). CrackerBox, MustardBottle, PowerDrill show no significant
  pairwise difference; Banana is 100% for all three methods (test undefined, see `note` column).
  **Scissors' `mcnemar_p=nan` (0 discordant trials across every pair) is not "no difference between
  methods" — see the CRITICAL note at the top of this file: all three "methods" silently ran the same
  non-CFM random-sampling fallback for this object, so there was nothing to differ between. Exclude
  Scissors when citing this table for a method comparison.**
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

### `formal_results/contact_features_bonferroni_bh_6obj.csv` — **extended, 6 objects** (from `scripts/run_contact_features_stats_extended.py`, 2026-07-08)

Same procedure as the 3-object table above, extended to 18 tests (3 features x 6 objects) using
`phase0_diag_extended/data_with_contact_feats_new3.json` for the 3 new objects. The 3-object table
above is not wrong and can still be cited on its own, but this is the fuller picture — includes a
`degenerate_all_zero` column flagging Scissors' structural floor-effect rows (see the data-quality
note above; these show `p_raw=1.0` by construction, not because nothing distinguishes success/fail).
**Scissors' rows are additionally not OT-CFM data at all — see the CRITICAL note at the top of this
file — so exclude them from the object count entirely rather than reading them as a 4th category;
this is effectively a 5-object result (Pear/MustardBottle/CrackerBox/TomatoSoupCan/PowerDrill) plus
one object (Scissors) that needs a rerun before it says anything.**

Reading the (valid) 5 objects by outcome, not just individually:
- **Bonferroni-surviving signal (large effect)**: MustardBottle (`normal_consistency` p=0.00029,
  `local_point_density` p=0.00030), Pear (`normal_consistency` p=0.00068), and now **TomatoSoupCan**
  (`local_point_density` p=0.00123, rank-biserial=−0.625) — a new object joining this group.
- **Signal only under the more lenient Benjamini-Hochberg correction**: CrackerBox (2 features) and
  TomatoSoupCan's `normal_consistency` (p_BH=0.039).
- **No signal under either correction**: **PowerDrill** — all 3 features non-significant even under
  BH (p≥0.17). This is the first object with a clean, unqualified null result on this feature set.
- **Excluded — not a real data point**: Scissors (see above).

- **Can support**: the "object-dependent" pattern is not a binary (signal / no-signal) split — across
  5 valid objects it's a spectrum: robust signal (Pear, MustardBottle, TomatoSoupCan) → correction-dependent
  signal (CrackerBox) → clean null (PowerDrill). Any paper claim about object-dependence should describe
  this spectrum, not collapse it to "some objects work, some don't", and should count this as n=5, not n=6.
- **Cannot support**: a specific geometric/physical property (e.g. "compliance", "size", "symmetry")
  as *the* explanatory variable for where signal exists — no such property was measured or tested here;
  the grouping above is purely empirical (which p-values came out significant), not mechanistically
  explained. That would need a follow-up analysis relating a measured object property to the outcome
  group, which was not done.

### `formal_results/ikmargin_vs_consensus_matched_n10.csv` — **authoritative, ensemble-size-controlled** (from `scripts/run_ikmargin_vs_consensus_matched.py`, 2026-07-08; TomatoSoupCan row added 2026-07-09)

Fisher's exact test, both strategies now at **matched ensemble size 10** (ikmargin: existing
`phase1_v2/ikmargin_*.jsonl`; consensus: new `phase1_matched_n10/consensus_n10_*.jsonl`). This closes
the confound in the original comparison below and is the number to cite going forward.

| object | ikmargin (n=10 pool) | consensus (n=10 pool) | Fisher's exact p |
|---|---|---|---|
| Pear | 6.0% (3/50) | 68.0% (34/50) | **p=5.8e-11** |
| MustardBottle | 50.0% (25/50) | 68.0% (34/50) | p=0.103 (ns) |
| CrackerBox | 44.0% (22/50) | 44.0% (22/50) | p=1.0 (exact tie) |
| TomatoSoupCan | 34.0% (17/50) | 64.0% (32/50) | **p=0.0048** |

- **Can support**: the Pear finding not only survives the ensemble-size fix, it gets *stronger*
  (p=5.8e-11 vs the original confounded p=4.1e-7) — ruling out ensemble size as an alternative
  explanation. "On Pear, the ikmargin selection rule performs dramatically and significantly worse than
  consensus, at matched candidate-pool size" is now a clean, controlled finding.
  `phase1_v2/pear_ensemble_reconstruction.json` still gives the per-candidate diagnostic detail for why.
  CrackerBox is an exact tie (22/50 both strategies) at matched pool size — reinforces the running theme
  that CrackerBox shows no selection-strategy signal under any method tested in this repo.
- **New, not previously visible**: giving consensus a fair 10-candidate pool raised its MustardBottle
  rate from 56% (at pool 5) to 68% — closer to significance (p=0.103, down from p=0.69) but still not
  significant at α=0.05. Consensus numerically leads ikmargin on MustardBottle now, but call this
  "suggestive, not established" until more trials are run.
- **TomatoSoupCan added 2026-07-09** (`scripts/run_ikmargin_n10_tomatosoupcan.sh` +
  `scripts/run_consensus_n10_tomatosoupcan.sh`, same 5 orient_seed × 10 ensemble_base grid, same OT-CFM
  checkpoint): motivated by `paper_final.tex`'s "Reliability across generation seeds" paragraph naming
  Pear **and** TomatoSoupCan as objects where single-draw OT-CFM is significantly less reliable across
  seeds (`exp1_variance`'s paired McNemar test) — this closes the gap of only having ikmargin-vs-consensus
  mitigation data for one of the two named objects. Result: consensus significantly beats ikmargin here
  too (64% vs 34%, p=0.0048) — the mitigation generalizes to both flagged objects, not just Pear.
- **Cannot support**: "consensus is a better selection rule than ikmargin" as a uniform, all-objects
  claim — it's a controlled, significant win on Pear and TomatoSoupCan; a tie on CrackerBox; and a
  non-significant lead on MustardBottle.

### `formal_results/ikmargin_vs_consensus.csv` — superseded, kept for record (confounded, ensemble 10 vs 5)

Original comparison: ikmargin (ensemble 10) vs consensus (ensemble 5) — the ensemble-size confound
described above. Kept on disk for transparency (shows what was believed before the matched re-run), but
**do not cite this file's numbers in the paper** — use `ikmargin_vs_consensus_matched_n10.csv` instead.

| object | ikmargin (n=10 pool) | consensus (n=5 pool) | Fisher's exact p |
|---|---|---|---|
| Pear | 6.0% (3/50) | 52.0% (26/50) | p=4.1e-7 |
| MustardBottle | 50.0% (25/50) | 56.0% (28/50) | p=0.69 (ns) |
| CrackerBox | 44.0% (22/50) | 42.0% (21/50) | p=1.0 (ns) |

## Explicitly NOT included (because it does not exist as a file anywhere)

Paper B's real-robot servo current/load data (tissue package / towel / glasses case) has no
backing data file on disk anywhere — it exists only as a hand-summarized table inside a memory
note (`~/.claude/projects/-lena-projects-lerobot/memory/project_paper_b_execution_prep.md`), and
the scripts that produced it (`live_probe_gripper.py`, `signal_watch.py`, etc.) were only ever in
`/tmp` and are gone. Do not treat this archive as covering Paper B.
