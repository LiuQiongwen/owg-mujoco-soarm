"""CR-CFM Stage A eval pilot: paired baseline-vs-cr_cfm_descend comparison,
n=20, FRESH ENV PER TRIAL (the convention confirmed correct this session --
see RULED_OUT_METHODS.md rows 12-13 for what happens when you reuse one env
object across a trial loop instead: invalid pairing, false results).
"""
import json
import sys

import numpy as np
import torch
from scipy.stats import binomtest

from tango_robot.piper_robosuite.cr_cfm.data import ACTION_DIM, HORIZON, DescendDataset
from tango_robot.piper_robosuite.cr_cfm.model import CRFlowNet
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from tango_robot.piper_robosuite.piper_pick_and_place import run_pick_and_place

OUT_PATH = "/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/cr_cfm_eval_{obj}_{a}-{b}.json"


def run_trial(obj_name, trial_id, use_cr_cfm, model=None, mean_start=None, template=None):
    np.random.seed(trial_id)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=[obj_name],
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    kwargs = {}
    if use_cr_cfm:
        kwargs = dict(cr_cfm_descend=True, cr_cfm_model=model, cr_cfm_mean_start=mean_start,
                      cr_cfm_template=template, cr_cfm_horizon=HORIZON, cr_cfm_num_steps=6,
                      cr_cfm_device="cpu")
    r = run_pick_and_place(env, obj_name, use_oriented_grasp=True, verbose=False, **kwargs)
    r["trial_id"] = trial_id
    return r


def main():
    obj_name = sys.argv[1] if len(sys.argv) > 1 else "cracker"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    ckpt = sys.argv[4] if len(sys.argv) > 4 else "/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/cr_cfm_cracker.pt"

    dataset = DescendDataset.load(obj_name=obj_name, horizon=HORIZON)
    mean_start = dataset.mean_start()  # numpy -- move_to_cr_cfm_descend does numpy arithmetic on this
    # before handing off to inference.py's torch-based sampler
    template = dataset.mean_template()
    model = CRFlowNet(action_dim=ACTION_DIM, horizon=HORIZON)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    trial_ids = list(range(start, start + n))
    baseline_results, cr_cfm_results = [], []
    for tid in trial_ids:
        rb = run_trial(obj_name, tid, use_cr_cfm=False)
        rc = run_trial(obj_name, tid, use_cr_cfm=True, model=model, mean_start=mean_start, template=template)
        baseline_results.append(rb)
        cr_cfm_results.append(rc)
        print(f"trial {tid}: baseline={rb['success']} (dist={rb['dist_to_tray']:.3f})  "
              f"cr_cfm={rc['success']} (dist={rc['dist_to_tray']:.3f})")

    b_succ = sum(r["success"] for r in baseline_results)
    c_succ = sum(r["success"] for r in cr_cfm_results)
    only_b = sum(1 for rb, rc in zip(baseline_results, cr_cfm_results) if rb["success"] and not rc["success"])
    only_c = sum(1 for rb, rc in zip(baseline_results, cr_cfm_results) if rc["success"] and not rb["success"])
    disc = only_b + only_c
    p = binomtest(min(only_b, only_c), disc, 0.5).pvalue if disc else 1.0

    print(f"\nbaseline: {b_succ}/{n} ({100*b_succ/n:.0f}%)   cr_cfm: {c_succ}/{n} ({100*c_succ/n:.0f}%)")
    print(f"discordant: baseline_only={only_b}  cr_cfm_only={only_c}  McNemar p={p:.4f}")

    out = OUT_PATH.format(obj=obj_name, a=start, b=start + n)
    json.dump({"baseline": baseline_results, "cr_cfm": cr_cfm_results}, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
