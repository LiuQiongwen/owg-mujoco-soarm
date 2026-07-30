#!/usr/bin/env python3
"""Load a local LeRobot ACT checkpoint and run one real dataset observation."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paperA_data" / "scripts"))
import _lerobot_groot_patch  # noqa: E402,F401

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True)
    args = ap.parse_args()
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.eval(); policy.reset()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})
    sample = dataset[0]
    observation = {
        k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
        for k, v in sample.items()
        if k.startswith("observation.") or k == "task"
    }
    with torch.no_grad():
        chunk = policy.predict_action_chunk(pre(observation))
        action = post(chunk[:, 0])
    assert chunk.shape == (1, policy.config.chunk_size, 6)
    assert torch.isfinite(chunk).all()
    print(f"chunk_shape={tuple(chunk.shape)} finite=True")
    print(f"first_action={action.squeeze(0).tolist()}")
    print(f"demonstration_action={sample['action'].tolist()}")


if __name__ == "__main__":
    main()
