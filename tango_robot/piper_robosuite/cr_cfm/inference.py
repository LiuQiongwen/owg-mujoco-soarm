"""CR-CFM inference: Euler-integrate the trained flow field from an
OBSERVED (naive/disturbed) trajectory guess to a corrected one.

Deliberately NOT sampling from pure Gaussian noise (a pseudocode draft
proposed this): sample_flow_pairs (data.py) trains the model on pairs
where x0 is a CLEAN trajectory plus injected drift plus small noise, not
unstructured noise -- starting Euler integration from noise would be
out-of-distribution relative to what the model was actually trained on.
x0 at inference must be a real, structured trajectory guess (see
piper_pick_and_place.py's move_to_cr_cfm_descend for how it's built: a
linear interpolation from the arm's actual current joint_pos to the
nominal, uncorrected descend target -- exactly the same interpolation
move_to_interpolated already does, just refined by the model afterward
instead of executed as-is).

Conditioning caveat (real design decision, not silently assumed): training
conditions on (x0_start - x1_start) using the TRUE clean target x1, which
is only available at training time. At inference we don't know x1 in
advance -- that is what we are generating -- so cond is instead
(x0_start - dataset_mean_x1_start), a proxy: "how disturbed does this
start look relative to a typical clean descend start." This is a real
train/inference distribution mismatch worth flagging, not a bug -- it is
the standard way conditional generative models handle this when the
condition would otherwise require the label.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def sample_corrected_trajectory(model, x0: torch.Tensor, cond: torch.Tensor,
                                 num_steps: int = 6, device: str = "cpu",
                                 adaptive_subdivide: bool = False,
                                 velocity_norm_threshold: float = 0.5,
                                 max_subdivide: int = 8,
                                 return_diagnostics: bool = False):
    """Euler-integrate v_theta from x0 (B, horizon, action_dim) to a
    corrected x1 estimate. num_steps=6 per the design's 4-8 step budget for
    a >=50Hz closed-loop target.

    adaptive_subdivide (2026-07-20, Stage 10, opt-in, after AdaFlow --
    arXiv:2402.04292 -- and "From Euler to Dormand-Prince" -- arXiv:2605.00836):
    Stage 7/calibration found the model's per-substep velocity magnitude is a
    near-perfect, EARLY-DETECTABLE signal for numerical divergence -- a known-
    stable trial (1001) stays bounded at ~0.08-0.13 (max per-waypoint L2 norm)
    across all 6 substeps, while a known-unstable trial (1007) is already
    6.0 (roughly 50x higher) at the VERY FIRST substep, then compounds
    exponentially (6 -> 13 -> 23 -> 74 -> 319 -> 1387) purely from explicit
    Euler's fixed step size being too coarse for that region of the learned
    field (matching the literature: discretization error connects to model
    uncertainty/field roughness, and undertrained/OOD regions need finer
    resolution, not a fixed step count). When a substep's max per-waypoint
    velocity norm exceeds `velocity_norm_threshold`, that ONE substep is
    subdivided into up to `max_subdivide` smaller sub-steps (re-evaluating
    v_theta at each), instead of taking one large, error-compounding jump --
    refining resolution rather than capping the output value (Stage 8) or the
    per-waypoint OUTPUT displacement (Stage 9, which fixed the instability but
    broke convergence within the fixed RHC iteration budget). This changes
    NOTHING for well-behaved inputs (never triggers for trial 1001-like
    cases) and only spends extra model evaluations exactly where Stage 7's
    diagnosis says they are needed."""
    model.eval()
    x_t = x0.to(device)
    cond = cond.to(device)
    B = x_t.shape[0]
    dt = 1.0 / num_steps
    t_val = 0.0
    triggered = False  # 2026-07-20, Stage 11: whether ANY substep in this call
    # exceeded velocity_norm_threshold -- the same signal that decides
    # subdivision doubles, for free, as the difficulty-detection trigger
    # `move_to_cr_cfm_descend`'s difficulty_aware mode reads back.
    for i in range(num_steps):
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v_pred = model(x_t, t_tensor, cond)
        if adaptive_subdivide:
            row_norm_max = v_pred.norm(dim=-1).max().item()
            if row_norm_max > velocity_norm_threshold:
                triggered = True
                k = min(max_subdivide, max(2, int(np.ceil(row_norm_max / velocity_norm_threshold))))
                sub_dt = dt / k
                x_t = x_t + v_pred * sub_dt  # reuse the already-computed v_pred for the first sub-step
                sub_t = t_val + sub_dt
                for _ in range(1, k):
                    t_sub_tensor = torch.full((B,), sub_t, device=device, dtype=torch.float32)
                    v_sub = model(x_t, t_sub_tensor, cond)
                    x_t = x_t + v_sub * sub_dt
                    sub_t += sub_dt
                t_val += dt
                continue
        x_t = x_t + v_pred * dt
        t_val += dt
    if return_diagnostics:
        return x_t, {"triggered_subdivision": triggered}
    return x_t


def pace_execute_length(waypoints: np.ndarray, min_steps: int = 1, max_steps: int = 8) -> int:
    """PACE-style adaptive execution length (2026-07-20, IMPROVEMENT_PLAN.md
    Stage 4, after Nie et al. "Phase-Aware Chunk Execution," arXiv:2606.00537)
    -- training-free, deployment-time-only: instead of a FIXED
    execute_steps (v1-v7 all used a constant, e.g. 2), analyze the ALREADY-
    GENERATED chunk's own per-step speed profile (joint-space delta norm
    between consecutive waypoints -- literally the same quantity
    move_to_cr_cfm_descend's terminal_velocity diagnostic already computes,
    just applied here at every step instead of only the last 3) and commit
    only up to the first low-speed valley, treating it as a natural
    replanning boundary (their finding: these valleys correspond to phase
    transitions -- contact preparation, alignment -- exactly the moments a
    stale plan is most likely to be wrong and most needs a fresh replan).

    Their paper targets task success rate; here the target is reducing
    RUN-TO-RUN INSTABILITY specifically -- a genuinely different objective,
    not yet tested in their work as far as this session's novelty check
    found (see README's Stage 4 entry). Returns an int in [min_steps,
    max_steps], bounded so this can never degenerate to 0 (no progress) or
    the full horizon (defeats RHC's whole purpose)."""
    speed = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)  # (horizon-1,)
    search_end = min(max_steps, len(speed) - 1)
    if search_end <= min_steps:
        return min_steps
    # first local minimum (a genuine valley: lower than both neighbors),
    # searched only within [min_steps, search_end] so it can't collapse to
    # step 0 (always a "valley" relative to nothing) or run past max_steps.
    for i in range(min_steps, search_end):
        if speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]:
            return i + 1  # execute THROUGH the valley waypoint, not up to it
    return max_steps  # no clear valley found in range -- fall back to the cap


def build_naive_x0(current_joint_pos: np.ndarray, target_joint_pos: np.ndarray,
                    horizon: int) -> np.ndarray:
    """DEPRECATED for cr_cfm_descend -- kept only as a documented negative
    example. Straight-line linear interpolation (constant velocity, i.e. a
    square-wave velocity profile with infinite acceleration at both ends)
    is nothing like the PD-converged trajectories flow_matching_loss was
    trained on (fast start, damped deceleration, smooth jerk). Confirmed
    2026-07-18: this produced a genuinely out-of-distribution x0 at
    inference and broke cases the plain baseline already solved (trial
    1001: baseline succeeds, cr_cfm with this x0 fails at drift=5.86cm;
    trial 1003: baseline succeeds, cr_cfm fails at drift=0.07cm -- the
    near-zero drift on a FAILING trial is the tell that the generated
    trajectory itself was bad, not just miscentered). Use
    build_template_x0 instead."""
    alphas = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None]
    return (1 - alphas) * current_joint_pos[None, :] + alphas * target_joint_pos[None, :]


def build_template_x0(current_joint_pos: np.ndarray, target_joint_pos: np.ndarray,
                       template: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """x0 built by affinely re-targeting the dataset's own MEAN trajectory
    shape (per-joint progress curve -- fast start, damped convergence,
    matching real PD dynamics) onto the actual current->target displacement,
    instead of a straight line. For each joint d, the template's own
    normalized progress fraction at waypoint i,
        frac[i, d] = (template[i, d] - template[0, d]) / (template[-1, d] - template[0, d]),
    is reused to place waypoint i between current[d] and target[d]:
        x0[i, d] = current[d] + frac[i, d] * (target[d] - current[d]).
    This keeps x0's SHAPE (non-linear pacing, realistic jerk) close to
    what flow_matching_loss actually trained on, while still starting
    exactly at the arm's real current position and ending at the real
    target -- the two boundary conditions that must hold regardless of
    template shape. Falls back to linear progress (frac[i,d] = i/(H-1))
    for any joint where the template barely moves (|denominator| < eps) --
    the template's shape carries no information for a near-static joint,
    and dividing by a near-zero span would blow up numerically."""
    horizon = template.shape[0]
    span = template[-1] - template[0]                      # (D,)
    denom = np.where(np.abs(span) < eps, 1.0, span)         # avoid div-by-~0
    frac = (template - template[0][None, :]) / denom[None, :]  # (H, D)
    linear_frac = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None]
    frac = np.where(np.abs(span)[None, :] < eps, linear_frac, frac)

    delta = (target_joint_pos - current_joint_pos)[None, :]  # (1, D)
    return current_joint_pos[None, :] + frac * delta          # (H, D)
