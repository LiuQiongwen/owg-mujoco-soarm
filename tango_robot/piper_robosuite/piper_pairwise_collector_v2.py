"""
Re-collection of the same 25 Cracker scenes as pairwise_results_cracker_*.json,
now that run_pick_and_place logs the new `candidate_grasp_yaw` field (the
TRUE pre-commit candidate orientation -- see piper_pick_and_place.py's
2026-07-16 addition and AUTO_TAGGER_ALGORITHM.md's diagnostic finding that
motivated it). Saved under a distinct filename (pairwise_results_v2_*) so
the original data is preserved for audit/reproducibility rather than
overwritten.

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_pairwise_collector_v2 \
      [obj_name] [comma,separated,scene,ids] [n_candidates]
"""
import sys
import json
import os

from tango_robot.piper_robosuite.piper_pairwise_collector import collect

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    obj_name = sys.argv[1] if len(sys.argv) > 1 else "cracker"
    scene_trial_ids = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [900, 901, 902]
    n_candidates = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    results = collect(obj_name, scene_trial_ids, n_candidates)

    n_success = sum(r["success"] for r in results)
    print(f"\n{obj_name}: {n_success}/{len(results)} success ({100*n_success/len(results):.0f}%)")

    tag = f"{min(scene_trial_ids)}-{max(scene_trial_ids)}"
    out_path = os.path.join(RESULTS_DIR, f"pairwise_results_v2_{obj_name}_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
