"""Risk gate for grasp/action proposals produced by a policy or VLA.

The gate is deliberately policy-agnostic: an upstream policy proposes grasp
end poses, then the critic either accepts its best proposal or falls back to a
known baseline proposal.  A joint-action VLA can use the same interface after
FK-projecting each action chunk to its terminal end-effector grasp pose.
"""
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from world_model.train_counterfactual_critic import score_candidates


@dataclass(frozen=True)
class GateDecision:
    candidate_idx: int
    policy_idx: int
    fallback_idx: int
    source: str
    critic_score: float
    fallback_score: float
    uncertainty: float
    accepted: bool


class CriticRiskGate:
    """Accept a policy proposal only when its critic estimate is trustworthy.

    ``uncertainty_threshold`` is calibrated on a development set and frozen
    before confirmatory evaluation. ``min_advantage`` prevents switching away
    from the fallback unless the critic predicts an actual improvement.
    """

    def __init__(self, ensemble, uncertainty_threshold: float,
                 min_advantage: float = 0.0, relative: bool = True):
        self.ensemble = ensemble
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.min_advantage = float(min_advantage)
        self.relative = bool(relative)

    def decide(self, scene: dict, candidate_poses: Sequence[Sequence[float]],
               fallback_scores: Sequence[float]) -> GateDecision:
        candidates = [{"candidate_pose": list(p)} for p in candidate_poses]
        mean, uncertainty = score_candidates(
            scene, candidates, self.ensemble, relative=self.relative)
        fallback_scores = np.asarray(fallback_scores, dtype=float)
        if len(candidates) == 0 or fallback_scores.shape != (len(candidates),):
            raise ValueError("candidate_poses and fallback_scores must be non-empty and aligned")
        policy_idx = int(np.argmax(mean))
        fallback_idx = int(np.argmax(fallback_scores))
        advantage = float(mean[policy_idx] - mean[fallback_idx])
        accepted = (float(uncertainty[policy_idx]) <= self.uncertainty_threshold
                    and advantage >= self.min_advantage)
        chosen = policy_idx if accepted else fallback_idx
        return GateDecision(
            candidate_idx=chosen, policy_idx=policy_idx,
            fallback_idx=fallback_idx,
            source="critic_policy" if accepted else "fallback",
            critic_score=float(mean[policy_idx]),
            fallback_score=float(mean[fallback_idx]),
            uncertainty=float(uncertainty[policy_idx]), accepted=accepted,
        )


class ActionPolicyRiskGate:
    """Adapter that places :class:`CriticRiskGate` after an action policy.

    ``proposal_fn(observation)`` must return candidate grasp poses and baseline
    scores. This covers OWG/LGGSN directly. For an ACT/VLA that emits action
    chunks, pass ``action_to_grasp`` to FK-project each chunk to its terminal
    ``[x, y, z, yaw, opening, object_height]`` pose before critic scoring.
    The original action proposal is returned for execution, so the gate never
    silently changes the policy's action representation.
    """

    def __init__(self, gate: CriticRiskGate,
                 proposal_fn: Callable,
                 action_to_grasp: Callable | None = None):
        self.gate = gate
        self.proposal_fn = proposal_fn
        self.action_to_grasp = action_to_grasp

    def predict(self, observation, scene_context: dict):
        proposal = self.proposal_fn(observation)
        actions = proposal["actions"]
        poses = proposal.get("grasp_poses")
        if poses is None:
            if self.action_to_grasp is None:
                raise ValueError("action proposals require action_to_grasp (e.g. FK)")
            poses = [self.action_to_grasp(a, observation) for a in actions]
        decision = self.gate.decide(
            scene_context, poses, proposal["fallback_scores"])
        audit = {**decision.__dict__, "n_proposals": len(actions),
                 "gate_type": "counterfactual_world_critic"}
        return actions[decision.candidate_idx], audit
