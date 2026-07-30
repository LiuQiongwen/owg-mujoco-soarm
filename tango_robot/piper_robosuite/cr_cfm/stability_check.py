"""CR-CFM run-to-run stability check (2026-07-18): repeats a handful of
trial_ids multiple times each under cr_cfm_descend and reports how often
the outcome disagrees across repeats -- the direct test for whether a
single run per trial_id is a valid signal for McNemar's paired design, per
the README's "Trial 1007 audited" entry. If a meaningful fraction of
trials show disagreement, single-run paired comparisons against baseline
are not trustworthy without repeats.
"""
import sys

import numpy as np
import torch

from tango_robot.piper_robosuite.cr_cfm.data import ACTION_DIM, HORIZON, DescendDataset
from tango_robot.piper_robosuite.cr_cfm.model import CRFlowNet
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite.piper_pick_and_place import run_pick_and_place


def run_once(obj_name, trial_id, model, template):
    np.random.seed(trial_id)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=[obj_name],
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    r = run_pick_and_place(env, obj_name, use_oriented_grasp=True, verbose=False,
                            cr_cfm_descend=True, cr_cfm_model=model, cr_cfm_mean_start=None,
                            cr_cfm_template=template, cr_cfm_horizon=HORIZON, cr_cfm_num_steps=6,
                            cr_cfm_device="cpu", cr_cfm_execute_steps=2, cr_cfm_max_iterations=12)
    return r["success"], r["dist_to_tray"], r["final_pos"][2]  # z, to catch table-launch cases


def main():
    obj_name = sys.argv[1] if len(sys.argv) > 1 else "cracker"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    ckpt = (sys.argv[5] if len(sys.argv) > 5 else
            "/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/cr_cfm_cracker_n155_v5_subseg.pt")

    dataset = DescendDataset.load(obj_name=obj_name, horizon=HORIZON, augment_subsegments=True)
    template = dataset.mean_template()
    model = CRFlowNet(action_dim=ACTION_DIM, horizon=HORIZON, cond_in_dim=ACTION_DIM)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    disagreements = 0
    table_launches = 0
    for i in range(n_trials):
        tid = start + i
        outcomes = [run_once(obj_name, tid, model, template) for _ in range(n_repeats)]
        successes = [o[0] for o in outcomes]
        launches = sum(1 for o in outcomes if o[2] < 0.3)  # z<0.3 = off the table, floor level
        disagree = len(set(successes)) > 1
        disagreements += disagree
        table_launches += launches
        print(f"trial {tid}: {outcomes}  {'DISAGREE' if disagree else 'consistent'}"
              f"{f'  ({launches}/{n_repeats} table-launches)' if launches else ''}")

    print(f"\n{disagreements}/{n_trials} trials show run-to-run disagreement "
          f"({100*disagreements/n_trials:.0f}%)")
    print(f"{table_launches}/{n_trials * n_repeats} individual runs launched the object off the table")


if __name__ == "__main__":
    main()
