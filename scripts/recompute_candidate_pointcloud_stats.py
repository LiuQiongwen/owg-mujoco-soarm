#!/usr/bin/env python3
"""
Recompute per-candidate (not per-scene) point-cloud statistics for an
existing scenes.jsonl-format dataset, WITHOUT re-executing any grasp
attempts and WITHOUT changing any stored success/failure labels.

Why this exists
----------------
`data.transition_logger.compute_pc_stats` is called once per scene, before
candidates are even sampled (see `scripts/risk_gated_vla_phase1_eval.py`'s
`build_pool()` and `scripts/collect_recovery_data.py`'s
`build_pool_recovery()`). The resulting 9-dim stat used to be attached
identically to every candidate in that scene via
`world_model/train_counterfactual_critic.py`'s `feature()`. This is now
fixed for FRESH data collected via `risk_gated_vla_phase1_eval.py`'s
`run_scene()` (it computes a real per-candidate local stat as it collects).
This script is for retroactively backfilling that same fix onto ALREADY
COLLECTED data, without re-running any physics.

2026-08-05 update: this script now calls
`data.transition_logger.compute_pc_stats_local()` directly -- the same
function `run_scene()` uses for fresh collection -- instead of a private
reimplementation. There is now exactly one implementation of "crop to a
candidate's local neighborhood," not two that could silently drift apart.

2026-08-05 fix (real bug, not just cleanup): this script previously wrote
its output to a NEW record-level field `pc_stats_per_candidate` (a list,
parallel to `oracle_per_candidate`). `feature()` does not read that field --
it reads `cand.get("pc_stats_local", ...)`, a field on EACH INDIVIDUAL
candidate dict inside `oracle_per_candidate`, matching exactly what
`run_scene()` writes for fresh collection. The old output schema was
silently incompatible: pointing training at it would have fallen through
to the old shared-stat behavior on every candidate, with no error and no
warning. Fixed to write `cand["pc_stats_local"]` in place, matching the
live-collection schema exactly, so retroactively-fixed and freshly-
collected data are interchangeable to every downstream consumer.

How this avoids re-collecting data
-----------------------------------
Object placement in this pipeline is fully determined by (obj_key, seed):
`rng = np.random.default_rng(seed); cx = rng.uniform(...); cy = ... + rng.uniform(...)`
happen BEFORE any candidate is sampled, so replaying just those two draws
reproduces the same scene regardless of what happens to the rng afterward.
This script therefore only needs to replay scene *placement* (object drop +
settle -- no grasp execution, no physics contact) and re-render one point
cloud per scene, then crop it per stored candidate. Existing "success"
labels are copied through unchanged.

IMPORTANT -- this script has NOT been run against real data
-------------------------------------------------------------
`results/` is gitignored and no scenes.jsonl-format file exists in this
checkout, so this could not be executed or verified here (the fix itself,
and the crop-radius choice below, WERE verified with real MuJoCo data via
fresh collection in this same session -- see
scripts/risk_gated_vla_phase1_eval.py's run_scene() and the accompanying
report). Also: two different scene-building conventions were found in this
codebase with DIFFERENT `centre_y` constants --
  scripts/risk_gated_vla_phase1_eval.py:   EVAL_CENTRE_Y = -0.30  (default here)
  scripts/eval_wm_reranking_full.py:       _CENTRE_Y      = -0.40
  scripts/collect_recovery_data.py's build_pool_recovery() imports
    EVAL_CENTRE_Y from risk_gated_vla_phase1_eval (so it matches the default).
Confirm which convention actually produced your specific input file before
trusting this script's output -- do not assume. The --centre-y flag lets
you override the default if your data used a different convention.

Built-in safety check: after replaying each scene, this script compares the
freshly-computed obj_pos against the record's stored obj_pos_before. Any
scene whose replay disagrees beyond --pos-tol is reported and, by default,
excluded from the output rather than silently written with wrong geometry
-- use --allow-mismatch to keep them anyway (not recommended without first
understanding why they disagree).

Usage (in the tango conda env):
    conda run -n tango python scripts/recompute_candidate_pointcloud_stats.py \\
        --in results/risk_gated_vla/<split>/scenes.jsonl \\
        --out results/risk_gated_vla/<split>/scenes_with_candidate_pc.jsonl \\
        --crop-radius 0.04

Output format: same records as the input, with each entry in
`oracle_per_candidate` gaining a new `pc_stats_local` field (9-dim, in
place, same schema fresh collection produces). `pc_stats_before` (the old
scene-level stat) is left in place, unmodified, for backward compatibility
-- callers that want the fix get it automatically via feature()'s existing
`cand.get("pc_stats_local", rec["pc_stats_before"])` fallback logic.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from tango_robot.env_soarm import EnvironmentSoArm
from data.transition_logger import compute_pc_stats_local
from scripts.eval_wm_reranking_full import OBJECTS, _SPREAD_XY, _DROP_Z, _SETTLE_STEPS


def _load_scene(env: EnvironmentSoArm, obj_name: str, obj_key: str, cx: float, cy: float) -> int:
    """Mirrors scripts/risk_gated_vla_phase1_eval.py's _load_scene() exactly,
    including the env._detach_obj() call -- omitting it reproduced a
    documented determinism bug there (stale weld state leaking across scenes
    reused in the same env instance)."""
    env.reset_robot()
    env.remove_all_obj()
    env._detach_obj()
    oid = env.load_obj(obj_name, name=obj_key, pos=[cx, cy, _DROP_Z])
    env._steps(_SETTLE_STEPS)
    return oid


def replay_scene_and_get_points(env: EnvironmentSoArm, obj_key: str, seed: int,
                                 centre_y: float) -> tuple[int, np.ndarray, np.ndarray, dict]:
    """Replays ONLY object placement (no candidates sampled, no grasp
    executed) and returns (oid, obj_pos, obj_quat, obs). Must draw exactly
    the same two rng.uniform() calls, in the same order, that the original
    collection script drew before it sampled any candidate -- see module
    docstring."""
    obj_name = OBJECTS[obj_key]
    rng = np.random.default_rng(seed)
    cx = float(rng.uniform(-_SPREAD_XY, _SPREAD_XY))
    cy = centre_y + float(rng.uniform(-0.04, 0.04))
    oid = _load_scene(env, obj_name, obj_key, cx, cy)
    obj_pos = env.get_obj_pos(oid).copy()
    obj_quat = env.get_obj_pose(oid)["quaternion"].copy()
    obs = env.get_obs(pointcloud=True)
    return oid, obj_pos, obj_quat, obs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--crop-radius", type=float, default=0.04,
                     help="metres; empirically validated in a separate session pass "
                          "(swept 0.02-0.10m against real point clouds) -- 0.04 matches "
                          "tango_robot/env_soarm.py's finger_length constant and avoids both "
                          "the zero-point degeneracy seen at 0.02 and the over-smoothing seen "
                          "approaching 0.10. Still worth spot-checking against this dataset's "
                          "own point density before trusting it at scale (object/sensor setup "
                          "may differ from what was swept).")
    ap.add_argument("--centre-y", type=float, default=-0.30,
                     help="must match whatever generated --in; see module docstring "
                          "for the two known conventions found in this codebase.")
    ap.add_argument("--pos-tol", type=float, default=1e-4,
                     help="max allowed L2 disagreement (metres) between replayed and "
                          "stored obj_pos before a scene is flagged as non-reproducing.")
    ap.add_argument("--allow-mismatch", action="store_true",
                     help="keep scenes whose replay disagrees beyond --pos-tol instead "
                          "of dropping them (not recommended without investigating why).")
    args = ap.parse_args()

    env = EnvironmentSoArm(vis=False, debug=False)

    n_total = n_mismatch = n_written = 0
    with open(args.in_path) as fin, open(args.out_path, "w") as fout:
        for line in fin:
            n_total += 1
            rec = json.loads(line)
            obj_key = rec["object"]
            seed = rec["seed"]

            oid, obj_pos, obj_quat, obs = replay_scene_and_get_points(
                env, obj_key, seed, args.centre_y)

            stored_pos = np.asarray(rec["obj_pos_before"], dtype=np.float32)
            disagreement = float(np.linalg.norm(obj_pos - stored_pos))
            if disagreement > args.pos_tol:
                n_mismatch += 1
                print(f"[MISMATCH] seed={seed} object={obj_key} "
                      f"replayed-vs-stored obj_pos L2={disagreement:.6f} "
                      f"(tol={args.pos_tol}) -- {'keeping anyway' if args.allow_mismatch else 'DROPPING'}",
                      file=sys.stderr)
                if not args.allow_mismatch:
                    continue

            for cand in rec["oracle_per_candidate"]:
                pose = np.asarray(cand["candidate_pose"], dtype=np.float32)
                stats = compute_pc_stats_local(obs, oid, pose[:3], radius=args.crop_radius)
                # In place, on the candidate dict itself -- matches exactly what
                # scripts/risk_gated_vla_phase1_eval.py::run_scene() writes for fresh
                # collection, and exactly what feature()'s cand.get("pc_stats_local", ...)
                # reads. NOT a parallel record-level list (that was the bug fixed
                # 2026-08-05 -- see module docstring).
                cand["pc_stats_local"] = [float(v) for v in stats]

            fout.write(json.dumps(rec) + "\n")
            n_written += 1

    print(f"\n{n_written}/{n_total} scenes written "
          f"({n_mismatch} replay mismatches, "
          f"{'kept' if args.allow_mismatch else 'dropped'}).", file=sys.stderr)
    if n_mismatch and not args.allow_mismatch:
        print("Investigate mismatches before trusting the output -- they may mean "
              "--centre-y (or another scene-building constant) doesn't match what "
              "actually generated --in.", file=sys.stderr)


if __name__ == "__main__":
    main()
