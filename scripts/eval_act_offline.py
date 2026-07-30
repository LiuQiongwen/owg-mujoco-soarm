#!/usr/bin/env python3
"""Offline one-step action diagnostics for a local ACT checkpoint."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paperA_data" / "scripts"))
import _lerobot_groot_patch  # noqa: E402,F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--samples", type=int, default=100)
    args = ap.parse_args()
    d = LeRobotDataset(args.repo_id, root=args.dataset_root)
    p = ACTPolicy.from_pretrained(args.checkpoint); p.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=p.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})
    indices = np.linspace(0, len(d) - 1, min(args.samples, len(d)), dtype=int)
    pred, true = [], []
    for idx in indices:
        s = d[int(idx)]
        obs = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
               for k, v in s.items()
               if k.startswith("observation.") or k == "task"}
        with torch.no_grad():
            action = post(p.predict_action_chunk(pre(obs))[:, 0]).squeeze(0)
        pred.append(action.numpy()); true.append(s["action"].numpy())
    pred, true = np.asarray(pred), np.asarray(true)
    result = {"n": len(indices), "mae": float(np.abs(pred - true).mean()),
              "mae_per_joint": np.abs(pred - true).mean(0).tolist(),
              "pred_std": pred.std(0).tolist(), "true_std": true.std(0).tolist()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
