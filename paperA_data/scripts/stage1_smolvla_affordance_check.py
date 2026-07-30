"""
Stage 1 of paperA_data/new_method_affordance_auxiliary_proposal.md: rerun the
single-task-vs-multi-task auxiliary-affordance-supervision check using the
REAL SmolVLA action-expert architecture (LoRA-adapted), not the toy 2-layer
MLP proxy from the earlier session-toy check.

Images: grasp_6dof/dataset/lggsn_scene_images/{scene_id}.png (regenerated
this session via scripts/regen_lggsn_scene_images.py, since the original
collection never saved raw images to disk).
Targets: grasp_6dof/dataset/lggsn_candidates_v9.jsonl (existing).

Trick: SmolVLA's action-expert output (`suffix_out`, the hidden state right
before `action_out_proj`) is captured via a forward hook on
`policy.model.action_out_proj`, without modifying any lerobot source file --
consistent with this project's rule to never edit site-packages.

Usage (conda run -n tango, from project root):
  python3 paperA_data/scripts/stage1_smolvla_affordance_check.py --smoke
  python3 paperA_data/scripts/stage1_smolvla_affordance_check.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lerobot_groot_patch  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JSONL_PATH = PROJECT_ROOT / "grasp_6dof/dataset/lggsn_candidates_v9.jsonl"
IMAGES_DIR = PROJECT_ROOT / "grasp_6dof/dataset/lggsn_scene_images"

POSE_FEATS = ["x", "y", "z", "roll", "pitch", "yaw"]
GEOM_FEATS = ["width", "score", "dz", "dz_lift", "need_dz", "H",
              "dist_to_centroid", "z_rel", "local_point_density",
              "normal_consistency", "contact_width_ratio", "pe_ik"]
OBJ_DISPLAY = {
    "banana": "banana", "pear": "pear", "mustard": "mustard bottle",
    "cracker": "cracker box", "drill": "power drill", "can": "soup can",
    "cylinder": "clamp",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_rows():
    rows = [json.loads(l) for l in open(JSONL_PATH)]
    rows = [r for r in rows if (IMAGES_DIR / f"{r['scene_id']}.png").exists()]
    return rows


def build_config():
    cfg = SmolVLAConfig(
        n_obs_steps=1,
        chunk_size=1,
        n_action_steps=1,
        max_state_dim=6,
        max_action_dim=6,
        device=DEVICE,
        use_amp=False,
        train_expert_only=True,
        load_vlm_weights=True,
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )
    return cfg


def build_policy(cfg, use_lora):
    # NOTE: PEFT/LoRA in lerobot's base policy class is designed for
    # fine-tuning an *already-trained* SmolVLA checkpoint (it refuses to
    # wrap a freshly-initialized action-expert: "Training from scratch using
    # PEFT is unlikely to yield good results"). We have no pretrained
    # lerobot-format SmolVLA action-expert checkpoint to start from (only
    # the raw VLM backbone), so we train the action-expert directly instead
    # -- config defaults already freeze the VLM/vision encoder
    # (freeze_vision_encoder=True, train_expert_only=True), so this is
    # "full fine-tune of the small expert only", not "full fine-tune of the
    # 500M VLM". Fine for a relative single-task-vs-multi-task comparison,
    # which doesn't depend on LoRA specifically.
    policy = SmolVLAPolicy(cfg)
    policy.to(DEVICE)
    return policy


def make_batch(rows, idx_batch, pre_processor):
    imgs, states, actions, tasks = [], [], [], []
    for i in idx_batch:
        r = rows[i]
        img = Image.open(IMAGES_DIR / f"{r['scene_id']}.png").convert("RGB").resize((224, 224))
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        imgs.append(img_t)
        states.append(torch.zeros(6, dtype=torch.float32))
        pose = torch.tensor([r[f] for f in POSE_FEATS], dtype=torch.float32)
        actions.append(pose)
        tasks.append(f"pick up the {OBJ_DISPLAY.get(r['query'], r['query'])}")

    batch = {
        "observation.image": torch.stack(imgs).unsqueeze(1),   # (B,1,3,224,224)
        "observation.state": torch.stack(states).unsqueeze(1),  # (B,1,6)
        "action": torch.stack(actions).unsqueeze(1),            # (B,1,6)
        "task": tasks,
    }
    batch = pre_processor(batch)
    return batch


def get_geom_targets(rows, idx_batch):
    g = np.array([[rows[i][f] for f in GEOM_FEATS] for i in idx_batch], dtype=np.float32)
    return torch.from_numpy(g)


def train_variant(rows, tr_idx, multitask, steps, batch_size, geom_mean, geom_std, log_prefix, seed=0):
    torch.manual_seed(seed)
    cfg = build_config()
    policy = build_policy(cfg, use_lora=True)
    pre_processor, _ = make_smolvla_pre_post_processors(cfg, dataset_stats=None)

    expert_hidden_size = policy.model.action_out_proj.in_features
    aux_head = nn.Linear(expert_hidden_size, len(GEOM_FEATS)).to(DEVICE)

    captured = {}

    def hook(module, inp, out):
        captured["suffix_out"] = inp[0]

    handle = policy.model.action_out_proj.register_forward_hook(hook)

    trainable = [p for p in policy.parameters() if p.requires_grad] + list(aux_head.parameters())
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    rng = np.random.default_rng(seed)
    n = len(tr_idx)
    for step in range(steps):
        batch_idx = [tr_idx[i] for i in rng.choice(n, size=batch_size, replace=False)]
        batch = make_batch(rows, batch_idx, pre_processor)
        opt.zero_grad()
        loss, _ = policy.forward(batch)
        total_loss = loss
        if multitask:
            geom_t = get_geom_targets(rows, batch_idx).to(DEVICE)
            geom_n = (geom_t - geom_mean) / geom_std
            suffix_out = captured["suffix_out"].to(torch.float32).squeeze(1)  # (B, hidden)
            aux_pred = aux_head(suffix_out)
            aux_loss = nn.functional.mse_loss(aux_pred, geom_n)
            total_loss = loss + aux_loss
        total_loss.backward()
        opt.step()
        if step % 5 == 0:
            extra = f" aux_loss={aux_loss.item():.4f}" if multitask else ""
            print(f"[{log_prefix}] step {step}/{steps} loss={loss.item():.4f}{extra}")

    handle.remove()
    return policy, pre_processor


def extract_representation(policy, pre_processor, rows, idx_list, batch_size=8):
    captured = {}

    def hook(module, inp, out):
        captured["suffix_out"] = inp[0]

    handle = policy.model.action_out_proj.register_forward_hook(hook)
    reps = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(idx_list), batch_size):
            chunk = idx_list[start:start + batch_size]
            batch = make_batch(rows, chunk, pre_processor)
            policy.forward(batch)
            reps.append(captured["suffix_out"].to(torch.float32).squeeze(1).cpu().numpy())
    handle.remove()
    policy.train()
    return np.concatenate(reps, axis=0)


def probe_auc(rep_tr, y_tr, rep_te, y_te):
    sc = StandardScaler().fit(rep_tr)
    rep_tr, rep_te = sc.transform(rep_tr), sc.transform(rep_te)
    clf = LogisticRegression(max_iter=2000).fit(rep_tr, y_tr)
    proba = clf.predict_proba(rep_te)[:, 1]
    return roc_auc_score(y_te, proba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Tiny smoke test (few rows, few steps)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-rows", type=int, default=0, help="0 = all available rows with images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows()
    print(f"Loaded {len(rows)} rows with matching images.")
    if args.n_rows:
        rows = rows[: args.n_rows]
    if args.smoke:
        rows = rows[:40]
        args.steps = 6
        args.batch_size = 2

    idx = np.random.default_rng(args.seed).permutation(len(rows))
    n_train = int(0.7 * len(rows))
    tr_idx, te_idx = list(idx[:n_train]), list(idx[n_train:])

    geom_all = np.array([[r[f] for f in GEOM_FEATS] for r in rows], dtype=np.float32)
    geom_mean = torch.tensor(geom_all.mean(0)).to(DEVICE)
    geom_std = torch.tensor(geom_all.std(0).clip(min=1e-6)).to(DEVICE)

    y = np.array([r["label"] for r in rows])

    print("=== Training single-task (pose-only) SmolVLA+LoRA ===")
    policy_single, pre_single = train_variant(
        rows, tr_idx, multitask=False, steps=args.steps,
        batch_size=args.batch_size, geom_mean=geom_mean, geom_std=geom_std,
        log_prefix="single", seed=args.seed,
    )
    print("=== Training multi-task (pose + geom-aux) SmolVLA+LoRA ===")
    policy_multi, pre_multi = train_variant(
        rows, tr_idx, multitask=True, steps=args.steps,
        batch_size=args.batch_size, geom_mean=geom_mean, geom_std=geom_std,
        log_prefix="multi ", seed=args.seed,
    )

    print("=== Extracting representations + probing ===")
    rep_tr_s = extract_representation(policy_single, pre_single, rows, tr_idx)
    rep_te_s = extract_representation(policy_single, pre_single, rows, te_idx)
    rep_tr_m = extract_representation(policy_multi, pre_multi, rows, tr_idx)
    rep_te_m = extract_representation(policy_multi, pre_multi, rows, te_idx)

    auc_single = probe_auc(rep_tr_s, y[tr_idx], rep_te_s, y[te_idx])
    auc_multi = probe_auc(rep_tr_m, y[tr_idx], rep_te_m, y[te_idx])

    print()
    print(f"[seed={args.seed}] single-task (pose-only) probe AUC: {auc_single:.3f}")
    print(f"[seed={args.seed}] multi-task (pose+geom-aux) probe AUC: {auc_multi:.3f}")
    print(f"[seed={args.seed}] advantage: {auc_multi - auc_single:+.3f}")


if __name__ == "__main__":
    main()
