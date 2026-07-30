"""
Collect genuinely pairwise-labeled Piper data: for a given set of scenes
(trial_id fixes the object spawn pose), execute EVERY candidate in the
sampled pool individually (not just whichever one "best"/"consensus"
would pick), via run_pick_and_place's candidate_selection=<int> mode --
so each scene yields multiple individually-labeled candidates, the
structure LGGSN's real BPR pairwise training needs.

Promoted from scratchpad/collect_pairwise_piper.py (2026-07-15) into the
package proper per EXPERIMENT_PLAN.md Stage 0/1 -- this is the Stage 1
data-collection workhorse, expected to be run repeatedly at scale, not a
one-off script.

Same rng seed (np.random.default_rng(scene_id)) re-created fresh before
each candidate execution within a scene, so every candidate in that
scene's pool is drawn from the IDENTICAL sampled pool (fair within-scene
comparison) -- only which pool index gets executed differs.

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_pairwise_collector \
      [obj_name] [comma,separated,scene,trial,ids] [n_candidates]
"""
import sys
import json
import os

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite.piper_pick_and_place import run_pick_and_place

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def collect(obj_name, scene_trial_ids, n_candidates=10, verbose_print=True):
    results = []
    for scene_id in scene_trial_ids:
        for cand_idx in range(n_candidates):
            np.random.seed(scene_id)  # fixes object spawn pose identically across all candidates in this scene
            env = PiperMultiObjectScene(
                robots="Piper", ycb_objects=[obj_name],
                has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, control_freq=20,
            )
            env.reset()
            rng = np.random.default_rng(scene_id)  # fresh, same seed each candidate -> identical sampled pool
            result = run_pick_and_place(
                env, obj_name, use_oriented_grasp=True, verbose=False,
                candidate_selection=cand_idx, noise_model="kinematic", n_candidates=n_candidates,
                rng=rng,
            )
            result["scene_id"] = scene_id
            result["candidate_idx"] = cand_idx
            results.append(result)
            if verbose_print:
                print(f"scene {scene_id} candidate {cand_idx}: success={result['success']} "
                      f"dist_to_tray={result['dist_to_tray']:.3f}")
            env.close()
    return results


def main():
    obj_name = sys.argv[1] if len(sys.argv) > 1 else "cracker"
    scene_trial_ids = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [900, 901, 902]
    n_candidates = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    results = collect(obj_name, scene_trial_ids, n_candidates)

    n_success = sum(r["success"] for r in results)
    print(f"\n{obj_name}: {n_success}/{len(results)} success ({100*n_success/len(results):.0f}%)")

    tag = f"{min(scene_trial_ids)}-{max(scene_trial_ids)}"
    out_path = os.path.join(RESULTS_DIR, f"pairwise_results_{obj_name}_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
