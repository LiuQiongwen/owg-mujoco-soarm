"""CR-CFM Stage A training loop -- 3060/CPU scale, meant to validate the
architecture, loss function, and data pipeline before Stage B swaps in
DinoV2+DiT on the A100 window. Not tuned for final performance; this run's
job is "does the pipeline work and does the loss go down," per this
project's own smoke-test-then-scale convention.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from tango_robot.piper_robosuite.cr_cfm.data import (
    ACTION_DIM, HORIZON, DescendDataset, contact_proximity_weight, sample_flow_pairs,
)
from tango_robot.piper_robosuite.cr_cfm.losses import cr_cfm_loss
from tango_robot.piper_robosuite.cr_cfm.model import CRFlowNet


def train(obj_name=None, steps=2000, batch_size=32, lr=1e-3, lambda_tcr=0.1,
          drift_std_m=0.02, noise_std=0.05, seed=0, log_every=100,
          ckpt_path=None, augment_subsegments=True, angle_range=None,
          lambda_lip=0.0, lipschitz_sigma=0.01):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = DescendDataset.load(obj_name=obj_name, horizon=HORIZON,
                                   augment_subsegments=augment_subsegments, angle_range=angle_range)
    print(f"loaded {len(dataset)} descend segments (obj_name={obj_name or 'all'}, "
          f"augment_subsegments={augment_subsegments}, angle_range={angle_range})")

    # cond_in_dim=6, remaining-distance-to-target ONLY (2026-07-18, v3):
    # v2 concatenated drift (x0_start - dataset_mean_start) with remaining-
    # distance (target - x0_start), 12 dims total. Ablated head-to-head on
    # the same 8-trial probe set: remaining-only (6-dim) beat the
    # concatenated version (5/8 vs 4/8) AND tamed trial 1007's Z-residual
    # from a wild 57cm outlier down to a normal ~8.6cm, consistent with
    # every other trial -- absolute-position drift was noise for a 49K
    # model, not useful signal, exactly as hypothesized before testing.
    # remaining-distance is exact at inference (target_qpos is already
    # IK-solved before RHC starts) -- no train/inference gap for this
    # feature, unlike drift's dataset-mean proxy this replaces.
    model = CRFlowNet(action_dim=ACTION_DIM, horizon=HORIZON, cond_in_dim=ACTION_DIM).to(device)
    print(f"model params: {model.n_params():,} (target: well under 2M)")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    weights_np = contact_proximity_weight(HORIZON)
    tcr_weights = torch.from_numpy(weights_np).to(device)

    for step in range(1, steps + 1):
        x0, x1, t = sample_flow_pairs(dataset, batch_size, rng,
                                       drift_std_m=drift_std_m, noise_std=noise_std)
        # remaining: x0's first waypoint's distance to the TRUE FINAL
        # target (x1's last waypoint) -- the only conditioning feature
        # now, per the ablation above.
        cond_np = (x1[:, -1, :] - x0[:, 0, :])
        x0_t = torch.from_numpy(x0).to(device)
        x1_t = torch.from_numpy(x1).to(device)
        t_t = torch.from_numpy(t).to(device)
        cond_t = torch.from_numpy(cond_np).to(device)

        loss, parts = cr_cfm_loss(model, x0_t, x1_t, t_t, cond_t, tcr_weights, lambda_tcr=lambda_tcr,
                                   lambda_lip=lambda_lip, lipschitz_sigma=lipschitz_sigma)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == 1:
            lip_str = f"  lip={parts['lipschitz_penalty']:.5f}" if "lipschitz_penalty" in parts else ""
            print(f"step {step:5d}  fm_loss={parts['fm_loss']:.5f}  "
                  f"tcr={parts['tcr_penalty']:.5f}{lip_str}  total={parts['total']:.5f}")

    if ckpt_path:
        torch.save(model.state_dict(), ckpt_path)
        print(f"saved checkpoint -> {ckpt_path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-tcr", type=float, default=0.1)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    train(obj_name=args.obj, steps=args.steps, batch_size=args.batch_size,
          lr=args.lr, lambda_tcr=args.lambda_tcr, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
