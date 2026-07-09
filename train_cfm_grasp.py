#!/usr/bin/env python3
"""
Minimal Conditional Flow Matching for single-step grasp pose generation.

Data   : all-object label=1 rows from lggsn_candidates_v9.jsonl
Cond   : L2-normalised SAM 256-dim visual feature (encodes object identity + scene)
Target : 6-DoF pose  (x, y, z, roll, pitch, yaw)
           z / roll / pitch are constant top-down grasps — learned trivially.
           Meaningful dimensions: x, y, yaw  (object position + orientation).

Training : ConditionalFlowMatcher (linear path, sigma=0) + 4-layer MLP, 1000 epochs
Inference: 20-step Euler ODE, conditioned on a chosen vis_feat
Output   : PNG per-object scatter + yaw histograms
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher as OTFlowMatcher,
)

# ── Config ────────────────────────────────────────────────────────────────────
JSONL_PATH   = os.environ.get("CFM_JSONL",  "grasp_6dof/dataset/lggsn_candidates_v9.jsonl")
OUT_PNG      = os.environ.get("CFM_PNG",    "cfm_all_poses_ot.png")
CKPT_PATH    = os.environ.get("CFM_CKPT",   "grasp_6dof/models/cfm_allobj_ot.pt")
OBJECTS      = ["banana","can","cracker","cylinder","drill","mustard","pear"]
N_EPOCHS     = int(os.environ.get("CFM_EPOCHS",    "1000"))
BATCH_SIZE   = int(os.environ.get("CFM_BATCH",     "128"))
LR           = float(os.environ.get("CFM_LR",      "3e-4"))
WEIGHT_DECAY = float(os.environ.get("CFM_WD",      "1e-4"))
N_SAMPLES    = int(os.environ.get("CFM_N_SAMPLES", "20"))
ODE_STEPS    = int(os.environ.get("CFM_ODE_STEPS", "20"))
SEED         = 42
POSE_COLS    = ["x", "y", "z", "roll", "pitch", "yaw"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Dataset ────────────────────────────────────────────────────────────────

class GraspPoseDataset(Dataset):
    """All-object (vis_feat, pose) pairs from label=1 candidates."""

    def __init__(self, jsonl_path: str):
        all_rows = [json.loads(l) for l in open(jsonl_path)]
        rows = [r for r in all_rows if r["label"] == 1]

        poses   = np.array([[r[c] for c in POSE_COLS] for r in rows], dtype=np.float32)
        vis_raw = np.array([r["visual_feat"] for r in rows], dtype=np.float32)
        queries = [r["query"] for r in rows]

        # Per-dim z-score of poses (global across all objects)
        self.pose_mean = poses.mean(0)
        self.pose_std  = poses.std(0).clip(min=1e-6)
        poses_norm     = (poses - self.pose_mean) / self.pose_std

        # L2-normalise vis to unit sphere
        norms   = np.linalg.norm(vis_raw, axis=1, keepdims=True).clip(min=1e-8)
        vis_norm = vis_raw / norms

        self.poses   = torch.from_numpy(poses_norm)   # (N, 6)
        self.vis     = torch.from_numpy(vis_norm)     # (N, 256)
        self.queries = queries
        self.N       = len(rows)

        # Per-object stats for reporting
        by_obj = {}
        for i, q in enumerate(queries):
            by_obj.setdefault(q, []).append(i)

        print(f"\n[Dataset] label=1 samples: {self.N} total")
        print(f"  {'object':<12} {'n':>5}  {'x_range':>18}  {'y_range':>18}  {'yaw_range':>20}")
        for obj in OBJECTS:
            idx = by_obj.get(obj, [])
            if not idx:
                print(f"  {obj:<12} {'0':>5}")
                continue
            p = poses[idx]
            print(f"  {obj:<12} {len(idx):>5}  "
                  f"[{p[:,0].min():.3f},{p[:,0].max():.3f}]  "
                  f"[{p[:,1].min():.3f},{p[:,1].max():.3f}]  "
                  f"[{p[:,5].min():.3f},{p[:,5].max():.3f}]")
        print()

        # Cache raw poses per object for visualisation
        self._raw_poses = poses
        self._by_obj    = by_obj

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.vis[idx], self.poses[idx]


# ── 2. Velocity network ───────────────────────────────────────────────────────

T_DIM    = 3
POSE_DIM = 6
VIS_DIM  = 256
IN_DIM   = T_DIM + POSE_DIM + VIS_DIM   # 265

class SinusoidalTimeEmbed(nn.Module):
    def forward(self, t):
        t = t.unsqueeze(-1)
        return torch.cat([t, torch.sin(np.pi*t), torch.cos(np.pi*t)], dim=-1)

class VelocityNet(nn.Module):
    """4-layer MLP: v_θ(t, x_t, cond) → velocity ∈ R^6."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbed()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, hidden),  nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, hidden//2), nn.SiLU(),
            nn.Linear(hidden//2, POSE_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, xt, cond):
        te  = self.t_embed(t)
        inp = torch.cat([te, xt, cond], dim=-1)
        return self.net(inp)


# ── 3. Training ───────────────────────────────────────────────────────────────

def train(dataset: GraspPoseDataset) -> VelocityNet:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model  = VelocityNet(hidden=512).to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS, eta_min=1e-5)
    FM     = OTFlowMatcher(sigma=0.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] VelocityNet params: {n_params:,}  device: {device}")
    print(f"[Train] {N_EPOCHS} epochs  batch={BATCH_SIZE}  lr={LR}  wd={WEIGHT_DECAY}")

    best_loss, best_state = float("inf"), None
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for cond_b, x1_b in loader:
            cond_b = cond_b.to(device)
            x1_b   = x1_b.to(device)
            x0     = torch.randn_like(x1_b)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1_b)
            loss = F.mse_loss(model(t, xt, cond_b), ut)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * len(x1_b)

        sched.step()
        avg_loss = total_loss / len(dataset)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 200 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{N_EPOCHS}  loss={avg_loss:.5f}  best={best_loss:.5f}")

    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)
    torch.save(best_state, CKPT_PATH)
    print(f"[Train] Best loss: {best_loss:.5f}  → {CKPT_PATH}")
    model.load_state_dict(best_state)
    return model


# ── 4. ODE inference (Euler) ──────────────────────────────────────────────────

@torch.no_grad()
def sample_poses(model: VelocityNet, cond: torch.Tensor,
                 n: int = N_SAMPLES, steps: int = ODE_STEPS,
                 seed: int | None = None) -> np.ndarray:
    """seed: if given, draws the initial noise from a local torch.Generator
    seeded with this value, independent of global RNG state. Callers doing
    per-trial inference (e.g. ui.py._cfm_sample_candidates) should always
    pass a per-trial seed here — otherwise every call in a freshly-started
    process draws from whatever the global RNG's current state happens to
    be, which is NOT randomized per trial on its own."""
    model.eval()
    cond = cond.unsqueeze(0).expand(n, -1).to(device) if cond.dim()==1 else cond.to(device)
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        x = torch.randn(n, POSE_DIM, device=device, generator=gen)
    else:
        x = torch.randn(n, POSE_DIM, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = torch.full((n,), i * dt, device=device)
        x = x + model(t_val, x, cond) * dt
    return x.cpu().numpy()


# ── 5. Visualisation ─────────────────────────────────────────────────────────

COLORS = {
    "banana":"#f4d03f","can":"#2980b9","cracker":"#e67e22",
    "cylinder":"#8e44ad","drill":"#e74c3c","mustard":"#f39c12","pear":"#27ae60",
}

def plot_results(model: VelocityNet, dataset: GraspPoseDataset):
    n_obj   = len(OBJECTS)
    fig, axes = plt.subplots(3, n_obj, figsize=(4*n_obj, 12))
    fig.suptitle(f"CFM All-Object Grasp Poses  (n_train={dataset.N})", fontsize=14)

    all_gen_poses = {}

    for col, obj in enumerate(OBJECTS):
        idx = dataset._by_obj.get(obj, [])
        if not idx:
            for row in range(3):
                axes[row, col].set_visible(False)
            continue

        # Condition: mean vis_feat of this object's training samples
        mean_cond = dataset.vis[idx].mean(0)          # (256,)
        gen_norm  = sample_poses(model, mean_cond, n=N_SAMPLES)
        gen       = gen_norm * dataset.pose_std + dataset.pose_mean

        train_raw = dataset._raw_poses[idx]           # (n_obj_train, 6)
        all_gen_poses[obj] = gen
        color = COLORS[obj]

        # ── Row 0: XY scatter ─────────────────────────────────────────────
        ax = axes[0, col]
        ax.scatter(train_raw[:,0], train_raw[:,1], c=train_raw[:,5],
                   cmap="RdYlGn", vmin=-np.pi/2, vmax=np.pi/2,
                   s=40, alpha=0.5, marker="o", label="train")
        sc = ax.scatter(gen[:,0], gen[:,1], c=gen[:,5],
                        cmap="RdYlGn", vmin=-np.pi/2, vmax=np.pi/2,
                        s=80, alpha=0.9, marker="*", label="gen", edgecolors="k", lw=0.4)
        ax.set_title(f"{obj}  (train={len(idx)})", fontsize=9)
        ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
        if col == n_obj - 1:
            plt.colorbar(sc, ax=ax, label="yaw", fraction=0.04)

        # ── Row 1: yaw histogram ──────────────────────────────────────────
        ax = axes[1, col]
        bins = np.linspace(-np.pi/2, np.pi/2, 15)
        ax.hist(train_raw[:,5], bins=bins, alpha=0.55, density=True,
                color="steelblue", label="train")
        ax.hist(gen[:,5],       bins=bins, alpha=0.55, density=True,
                color="coral",    label="gen")
        ax.set_xlabel("yaw (rad)", fontsize=8); ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # ── Row 2: coverage metrics text ─────────────────────────────────
        ax = axes[2, col]
        ax.axis("off")
        tr_xr = f"[{train_raw[:,0].min():.3f},{train_raw[:,0].max():.3f}]"
        ge_xr = f"[{gen[:,0].min():.3f},{gen[:,0].max():.3f}]"
        tr_yr = f"[{train_raw[:,5].min():.3f},{train_raw[:,5].max():.3f}]"
        ge_yr = f"[{gen[:,5].min():.3f},{gen[:,5].max():.3f}]"
        txt = (f"x range\n  train: {tr_xr}\n  gen:   {ge_xr}\n\n"
               f"yaw range\n  train: {tr_yr}\n  gen:   {ge_yr}\n\n"
               f"yaw std\n  train: {train_raw[:,5].std():.3f}\n"
               f"  gen:   {gen[:,5].std():.3f}")
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=7,
                va="top", family="monospace",
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"[Plot] Saved → {OUT_PNG}")

    # Compact per-object summary
    print("\n[Coverage summary]  (train yaw_std  vs  gen yaw_std)")
    for obj in OBJECTS:
        idx = dataset._by_obj.get(obj, [])
        if not idx or obj not in all_gen_poses:
            continue
        tr = dataset._raw_poses[idx]
        ge = all_gen_poses[obj]
        print(f"  {obj:<12}  train yaw_std={tr[:,5].std():.3f}  "
              f"gen yaw_std={ge[:,5].std():.3f}  "
              f"x_in_range={((ge[:,0]>=tr[:,0].min()-0.01)&(ge[:,0]<=tr[:,0].max()+0.01)).mean():.0%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def save_inference_stats(dataset: GraspPoseDataset, ckpt_path: str):
    """Save pose normalisation stats and per-object mean vis_feat alongside checkpoint."""
    import json as _json
    stats = {
        "pose_mean": dataset.pose_mean.tolist(),
        "pose_std":  dataset.pose_std.tolist(),
        "pose_cols": POSE_COLS,
        # L2-normalised mean vis_feat per object (used as CFM condition at inference)
        "mean_vis_per_obj": {
            obj: dataset.vis[idx].mean(0).numpy().tolist()
            for obj, idx in dataset._by_obj.items()
        },
    }
    stats_path = ckpt_path.replace(".pt", "_stats.json")
    with open(stats_path, "w") as f:
        _json.dump(stats, f)
    print(f"[Stats] Saved → {stats_path}")
    return stats_path


def main():
    dataset = GraspPoseDataset(JSONL_PATH)
    if dataset.N < 10:
        print(f"[ERROR] Only {dataset.N} label=1 samples found. Need more data.")
        return

    model = train(dataset)
    save_inference_stats(dataset, CKPT_PATH)
    plot_results(model, dataset)


if __name__ == "__main__":
    # Seeding training reproducibility only — must NOT run on import, since
    # sample_poses() is imported for inference (once per eval-trial subprocess)
    # and a module-level reset here would make every trial draw identical
    # noise regardless of the --seed CLI arg. See sample_poses()'s own `seed`
    # parameter for inference-time randomness instead.
    torch.manual_seed(SEED); np.random.seed(SEED)
    main()
