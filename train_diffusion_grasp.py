#!/usr/bin/env python3
"""
DDPM grasp pose generator — controlled baseline for OT-CFM comparison.

All variables identical to train_cfm_grasp.py except the generative paradigm:
  OT-CFM  : velocity-field prediction (ExactOT coupling), 20-step Euler ODE
  DDPM    : ε-prediction, cosine noise schedule (T=1000), DDIM inference

Architecture : NoiseNet — identical to VelocityNet (265→512→512→256→6, SiLU)
Conditioning : 256-dim L2-normalised SAM visual feature (same as CFM)
Data         : all-object label=1 rows from lggsn_candidates_v9.jsonl (2,853)
Optimiser    : AdamW  lr=3e-4  wd=1e-4  cosine LR  epochs=1000  batch=128
Inference    : DDIM deterministic sampler, default 100 steps (env DDIM_STEPS=N)
               Set DDPM_STOCHASTIC=1 for full DDPM (same step count, adds noise)

Checkpoint   : grasp_6dof/models/ddpm_allobj.pt
Stats file   : grasp_6dof/models/ddpm_allobj_stats.json  (same schema as CFM + model_type)
"""

import json
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ── Config (mirrors train_cfm_grasp.py env vars) ──────────────────────────────
JSONL_PATH   = os.environ.get("CFM_JSONL",    "grasp_6dof/dataset/lggsn_candidates_v9.jsonl")
OUT_PNG      = os.environ.get("DDPM_PNG",     "ddpm_all_poses.png")
CKPT_PATH    = os.environ.get("DDPM_CKPT",    "grasp_6dof/models/ddpm_allobj.pt")
OBJECTS      = ["banana","can","cracker","cylinder","drill","mustard","pear"]
N_EPOCHS     = int(os.environ.get("CFM_EPOCHS",    "1000"))
BATCH_SIZE   = int(os.environ.get("CFM_BATCH",     "128"))
LR           = float(os.environ.get("CFM_LR",      "3e-4"))
WEIGHT_DECAY = float(os.environ.get("CFM_WD",      "1e-4"))
DDPM_T       = int(os.environ.get("DDPM_T",        "1000"))  # training diffusion steps
DDIM_STEPS   = int(os.environ.get("DDIM_STEPS",    "100"))   # inference steps (DDIM)
N_SAMPLES    = int(os.environ.get("CFM_N_SAMPLES",  "20"))
SEED         = 42
POSE_COLS    = ["x", "y", "z", "roll", "pitch", "yaw"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Cosine noise schedule (Nichol & Dhariwal 2021) ─────────────────────────

def _make_cosine_schedule(T: int = DDPM_T, s: float = 0.008):
    steps = T + 1
    ts = torch.linspace(0, T, steps, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((ts + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = betas.clamp(1e-4, 0.999).float()
    alphas     = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)          # ᾱ_t, shape (T,)
    return betas, alphas, alphas_bar

_BETAS, _ALPHAS, _ALPHAS_BAR = _make_cosine_schedule(DDPM_T)

# ── 2. Dataset: re-use from CFM script (same data, same normalisation) ─────────
from train_cfm_grasp import GraspPoseDataset

# ── 3. NoiseNet — structurally identical to VelocityNet ───────────────────────
T_DIM    = 3
POSE_DIM = 6
VIS_DIM  = 256
IN_DIM   = T_DIM + POSE_DIM + VIS_DIM   # 265

class SinusoidalTimeEmbed(nn.Module):
    def forward(self, t):
        t = t.unsqueeze(-1)
        return torch.cat([t, torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)

class NoiseNet(nn.Module):
    """4-layer MLP: ε_θ(t, x_t, cond) → predicted noise ∈ R^6.

    Identical architecture to VelocityNet; only the training target differs.
    """
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbed()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, hidden),    nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.SiLU(),
            nn.Linear(hidden, hidden//2), nn.SiLU(),
            nn.Linear(hidden//2, POSE_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, xt, cond):
        te  = self.t_embed(t)
        inp = torch.cat([te, xt, cond], dim=-1)
        return self.net(inp)

# ── 4. Training: DDPM ε-prediction ───────────────────────────────────────────

def train(dataset: GraspPoseDataset) -> NoiseNet:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model  = NoiseNet(hidden=512).to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS, eta_min=1e-5)
    abar   = _ALPHAS_BAR.to(device)   # (T,) precomputed ᾱ_t

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train/DDPM] NoiseNet params: {n_params:,}  T={DDPM_T}  device: {device}")
    print(f"[Train/DDPM] {N_EPOCHS} epochs  batch={BATCH_SIZE}  lr={LR}  wd={WEIGHT_DECAY}")

    best_loss, best_state = float("inf"), None
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for cond_b, x0_b in loader:
            cond_b = cond_b.to(device)
            x0_b   = x0_b.to(device)
            B      = x0_b.size(0)

            # Sample t ~ U{0,...,T-1}
            t_int  = torch.randint(0, DDPM_T, (B,), device=device)
            t_norm = t_int.float() / DDPM_T          # [0,1] for sinusoidal embed

            # Forward diffusion: x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε
            epsilon = torch.randn_like(x0_b)
            abar_t  = abar[t_int].view(-1, 1)
            xt      = abar_t.sqrt() * x0_b + (1.0 - abar_t).sqrt() * epsilon

            # ε-prediction loss
            eps_pred = model(t_norm, xt, cond_b)
            loss = F.mse_loss(eps_pred, epsilon)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * B

        sched.step()
        avg_loss = total_loss / len(dataset)
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 200 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{N_EPOCHS}  loss={avg_loss:.5f}  best={best_loss:.5f}")

    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)
    torch.save(best_state, CKPT_PATH)
    print(f"[Train/DDPM] Best loss: {best_loss:.5f}  → {CKPT_PATH}")
    model.load_state_dict(best_state)
    return model

# ── 5. Inference: DDIM deterministic sampler ──────────────────────────────────

@torch.no_grad()
def sample_poses_ddpm(model: NoiseNet, cond: torch.Tensor,
                      n: int = 5, steps: int = DDIM_STEPS,
                      seed: int | None = None) -> np.ndarray:
    """DDIM deterministic reverse sampler (eta=0).

    Uses a uniformly-spaced subsequence of length `steps` from the full T-step
    schedule.  Returns (n, 6) normalised poses (same unit as CFM output).

    Set env DDPM_STOCHASTIC=1 to add posterior noise (full DDPM, same step count).

    seed: if given, draws all randomness (initial noise, and posterior noise
    if DDPM_STOCHASTIC=1) from a local torch.Generator seeded with this value
    — see sample_poses()'s identical note in train_cfm_grasp.py.
    """
    model.eval()
    dev   = next(model.parameters()).device
    cond  = cond.unsqueeze(0).expand(n, -1).to(dev) if cond.dim() == 1 else cond.to(dev)
    abar  = _ALPHAS_BAR.to(dev)   # (T,)
    eta   = 1.0 if os.environ.get("DDPM_STOCHASTIC") == "1" else 0.0

    gen = None
    if seed is not None:
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)

    # Descending subsequence: T-1 → ... → 0  (length steps+1 for paired (t, t_prev))
    tau = torch.linspace(DDPM_T - 1, 0, steps + 1, dtype=torch.long).tolist()

    x = torch.randn(n, POSE_DIM, device=dev, generator=gen) if gen is not None \
        else torch.randn(n, POSE_DIM, device=dev)
    for i in range(steps):
        t_cur  = int(tau[i])
        t_prev = int(tau[i + 1])

        t_norm = torch.full((n,), t_cur / DDPM_T, device=dev)
        eps    = model(t_norm, x, cond)

        ab_t    = abar[t_cur]
        ab_prev = abar[t_prev]                      # abar[0] ≈ 0.9999 for cosine

        # x0 prediction, clamped for stability
        x0_pred = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x0_pred = x0_pred.clamp(-10.0, 10.0)

        # Direction toward x_t (epsilon component)
        dir_xt = (1.0 - ab_prev - eta**2 * (1.0 - ab_t) / (1.0 - ab_prev).clamp_min(1e-8)).clamp_min(0.0).sqrt() * eps

        # DDIM step
        x = ab_prev.sqrt() * x0_pred + dir_xt
        if eta > 0.0 and t_prev > 0:
            sigma = eta * ((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)).clamp_min(0.0).sqrt()
            noise = torch.randn(x.shape, device=dev, generator=gen) if gen is not None \
                    else torch.randn_like(x)
            x = x + sigma * noise

    return x.cpu().numpy()

# ── 6. Visualisation ─────────────────────────────────────────────────────────

COLORS = {
    "banana":"#f4d03f","can":"#2980b9","cracker":"#e67e22",
    "cylinder":"#8e44ad","drill":"#e74c3c","mustard":"#f39c12","pear":"#27ae60",
}

def plot_results(model: NoiseNet, dataset: GraspPoseDataset):
    n_obj = len(OBJECTS)
    fig, axes = plt.subplots(2, n_obj, figsize=(4*n_obj, 8))
    fig.suptitle(f"DDPM All-Object Grasp Poses  (n_train={dataset.N}  DDIM-{DDIM_STEPS})",
                 fontsize=14)
    for col, obj in enumerate(OBJECTS):
        idx = dataset._by_obj.get(obj, [])
        if not idx:
            for row in range(2):
                axes[row, col].set_visible(False)
            continue

        mean_cond = dataset.vis[idx].mean(0)
        gen_norm  = sample_poses_ddpm(model, mean_cond, n=N_SAMPLES, steps=DDIM_STEPS)
        gen       = gen_norm * dataset.pose_std + dataset.pose_mean
        train_raw = dataset._raw_poses[idx]

        ax = axes[0, col]
        ax.scatter(train_raw[:,0], train_raw[:,1], c=train_raw[:,5],
                   cmap="RdYlGn", vmin=-math.pi/2, vmax=math.pi/2,
                   s=40, alpha=0.5, marker="o", label="train")
        ax.scatter(gen[:,0], gen[:,1], c=gen[:,5],
                   cmap="RdYlGn", vmin=-math.pi/2, vmax=math.pi/2,
                   s=80, alpha=0.9, marker="*", label="gen", edgecolors="k", lw=0.4)
        ax.set_title(f"{obj}  (n={len(idx)})", fontsize=9)
        ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        bins = np.linspace(-math.pi/2, math.pi/2, 15)
        ax.hist(train_raw[:,5], bins=bins, alpha=0.55, density=True, color="steelblue", label="train")
        ax.hist(gen[:,5],       bins=bins, alpha=0.55, density=True, color="coral",    label="gen")
        ax.set_xlabel("yaw (rad)", fontsize=8); ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"[Plot] Saved → {OUT_PNG}")

# ── 7. Stats (same schema as CFM + model_type / ddpm_T / infer_steps) ─────────

def save_inference_stats(dataset: GraspPoseDataset, ckpt_path: str) -> str:
    stats = {
        "model_type":   "ddpm",
        "ddpm_T":       DDPM_T,
        "infer_steps":  DDIM_STEPS,
        "pose_mean":    dataset.pose_mean.tolist(),
        "pose_std":     dataset.pose_std.tolist(),
        "pose_cols":    POSE_COLS,
        "mean_vis_per_obj": {
            obj: dataset.vis[idx].mean(0).numpy().tolist()
            for obj, idx in dataset._by_obj.items()
        },
    }
    stats_path = ckpt_path.replace(".pt", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    print(f"[Stats] Saved → {stats_path}")
    return stats_path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dataset = GraspPoseDataset(JSONL_PATH)
    if dataset.N < 10:
        print(f"[ERROR] Only {dataset.N} label=1 samples found.")
        return
    model = train(dataset)
    save_inference_stats(dataset, CKPT_PATH)
    plot_results(model, dataset)

if __name__ == "__main__":
    # Seeding training reproducibility only — see train_cfm_grasp.py's identical
    # note. sample_poses_ddpm() takes its own `seed` param for inference-time use.
    torch.manual_seed(SEED); np.random.seed(SEED)
    main()
