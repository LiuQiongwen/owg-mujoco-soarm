"""CR-CFM Stage A losses: standard conditional flow matching (linear/OT
interpolation path) plus the asymmetric Temporal Consistency Regularizer
(TCR) from the design doc.

TCR is applied to the model's one-step clean-trajectory ESTIMATE
(x1_hat = x_t + (1-t) * v_pred, derived from the linear path's own
x_t = (1-t)x0 + t*x1 identity => x1 = x_t + (1-t)*(x1-x0) = x_t + (1-t)*v),
not to the raw predicted velocity field -- the design's intent is to
penalize roughness in the TRAJECTORY the model is converging toward, and
x1_hat is the right proxy for that at any intermediate flow time t,
whereas v_pred alone doesn't have trajectory-shape units.
"""
from __future__ import annotations

import torch


def sample_x_t(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear (optimal-transport) interpolation path. t: (B,)."""
    t_ = t[:, None, None]
    return (1.0 - t_) * x0 + t_ * x1


def flow_matching_loss(model, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor,
                        cond: torch.Tensor):
    """Standard CFM regression loss: ||v_theta(x_t, t, cond) - (x1 - x0)||^2.
    Returns (loss, v_pred, x1_hat) -- x1_hat is reused by tcr_penalty so the
    caller doesn't need to recompute the (1-t) algebra twice."""
    x_t = sample_x_t(x0, x1, t)
    v_target = x1 - x0
    v_pred = model(x_t, t, cond)
    fm_loss = ((v_pred - v_target) ** 2).mean()

    t_ = t[:, None, None].clamp(min=0.0, max=1.0)
    x1_hat = x_t + (1.0 - t_) * v_pred
    return fm_loss, v_pred, x1_hat


def tcr_penalty(traj: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Asymmetric temporal-consistency regularizer over a predicted action
    chunk (B, horizon, action_dim). `weights` (horizon,) is HIGH in free
    space (suppress high-frequency wobble) and LOW near the segment's end
    (contact-risk zone -- permit large, fast corrective motion), per
    data.py's contact_proximity_weight(). Penalizes both first-difference
    (velocity) and second-difference (jerk) roughness."""
    vel = traj[:, 1:, :] - traj[:, :-1, :]          # (B, H-1, D)
    jerk = vel[:, 1:, :] - vel[:, :-1, :]            # (B, H-2, D)
    w_vel = weights[:-1]
    w_jerk = weights[:-2]
    vel_pen = (w_vel[None, :, None] * vel.pow(2)).mean()
    jerk_pen = (w_jerk[None, :, None] * jerk.pow(2)).mean()
    return vel_pen + jerk_pen


def lipschitz_penalty(model, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                       sigma: float = 0.01) -> torch.Tensor:
    """IMPROVEMENT_PLAN.md Stage 2: penalizes the model's output sensitivity
    to small perturbations of its own real inference-time inputs, per
    arXiv:2506.19250's Lipschitz regularization for behavior cloning --
    here targeting the confirmed run-to-run chaotic instability (small
    physics-level perturbations in the sensed current arm state amplified
    by RHC into macroscopically different outcomes), not their paper's
    observation-noise setting.

    eps is correlated between x0 and cond exactly as the real system's own
    cond = target_qpos - x0[:, 0, :] relationship requires (see
    move_to_cr_cfm_descend) -- an INDEPENDENT perturbation of x0 and cond
    would test an input combination the model never actually receives at
    inference, understating or misdirecting the penalty."""
    eps = torch.randn_like(x0) * sigma
    x0_pert = x0 + eps
    cond_pert = cond - eps[:, 0, :]
    v_pred = model(x0, t, cond)
    v_pred_pert = model(x0_pert, t, cond_pert)
    diff_sq = (v_pred_pert - v_pred).pow(2).sum(dim=(1, 2))
    eps_norm_sq = eps.pow(2).sum(dim=(1, 2)) + 1e-8
    return (diff_sq / eps_norm_sq).mean()


def cr_cfm_loss(model, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor,
                 cond: torch.Tensor, tcr_weights: torch.Tensor, lambda_tcr: float = 0.1,
                 lambda_lip: float = 0.0, lipschitz_sigma: float = 0.01):
    """Total training objective: flow matching + lambda_tcr * TCR(x1_hat)
    + lambda_lip * Lipschitz penalty (Stage 2, off by default via
    lambda_lip=0.0 for exact backward compatibility with every v1-v6
    checkpoint trained before this was added).
    Returns (total_loss, dict of component losses for logging)."""
    fm_loss, v_pred, x1_hat = flow_matching_loss(model, x0, x1, t, cond)
    tcr = tcr_penalty(x1_hat, tcr_weights)
    total = fm_loss + lambda_tcr * tcr
    parts = {"fm_loss": fm_loss.item(), "tcr_penalty": tcr.item()}
    if lambda_lip > 0.0:
        lip = lipschitz_penalty(model, x0, t, cond, sigma=lipschitz_sigma)
        total = total + lambda_lip * lip
        parts["lipschitz_penalty"] = lip.item()
    parts["total"] = total.item()
    return total, parts
