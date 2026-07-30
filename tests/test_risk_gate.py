"""
Unit tests for world_model/risk_gate.py's threshold and fallback behavior.

score_candidates() (real critic ensemble scoring) is monkeypatched throughout --
these tests are about the GATE's decision logic (threshold/fallback/advantage),
not about model quality, so they run fast and deterministically without loading
any real .pt checkpoint.

Run: conda run -n tango python -m pytest tests/test_risk_gate.py -v
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import world_model.risk_gate as risk_gate_module
from world_model.risk_gate import CriticRiskGate, ActionPolicyRiskGate, GateDecision


def _patch_scores(monkeypatch, mean, uncertainty):
    """Force score_candidates() to return fixed (mean, uncertainty) arrays,
    regardless of the ensemble/scene/candidates passed in."""
    mean = np.asarray(mean, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def fake_score_candidates(scene, candidates, ensemble, relative=True):
        assert len(candidates) == len(mean)
        return mean, uncertainty

    monkeypatch.setattr(risk_gate_module, "score_candidates", fake_score_candidates)


def _gate(threshold=0.5, min_advantage=0.0):
    return CriticRiskGate(ensemble=None, uncertainty_threshold=threshold,
                          min_advantage=min_advantage)


# ── Threshold behavior ──────────────────────────────────────────────────────

def test_gate_accepts_when_uncertainty_low_and_advantage_positive(monkeypatch):
    # policy_idx=0 has the highest mean score and low uncertainty -> should be accepted
    _patch_scores(monkeypatch, mean=[0.9, 0.2, 0.1], uncertainty=[0.1, 0.9, 0.9])
    gate = _gate(threshold=0.5)
    decision = gate.decide({}, [[0, 0, 0, 0, 0, 0]] * 3, fallback_scores=[0.3, 0.8, 0.1])
    assert decision.accepted is True
    assert decision.source == "critic_policy"
    assert decision.candidate_idx == decision.policy_idx == 0


def test_gate_rejects_and_falls_back_when_uncertainty_exceeds_threshold(monkeypatch):
    # policy_idx=0 has the highest mean score but uncertainty EXCEEDS the threshold
    _patch_scores(monkeypatch, mean=[0.9, 0.2, 0.1], uncertainty=[0.99, 0.1, 0.1])
    gate = _gate(threshold=0.5)
    decision = gate.decide({}, [[0, 0, 0, 0, 0, 0]] * 3, fallback_scores=[0.1, 0.8, 0.1])
    assert decision.accepted is False
    assert decision.source == "fallback"
    # falls back to argmax(fallback_scores), NOT the critic's own top pick
    assert decision.candidate_idx == decision.fallback_idx == 1
    assert decision.candidate_idx != decision.policy_idx


def test_gate_boundary_uncertainty_equal_to_threshold_is_accepted(monkeypatch):
    # decide() uses <=, so exactly-at-threshold must be accepted
    _patch_scores(monkeypatch, mean=[0.9, 0.1], uncertainty=[0.5, 0.1])
    gate = _gate(threshold=0.5)
    decision = gate.decide({}, [[0]*6, [0]*6], fallback_scores=[0.1, 0.1])
    assert decision.accepted is True


# ── min_advantage behavior ──────────────────────────────────────────────────

def test_gate_rejects_when_advantage_below_min_advantage_even_if_confident(monkeypatch):
    # policy_idx=0 is confident (low uncertainty) but its score is barely above
    # the fallback's own score -- advantage too small given min_advantage
    _patch_scores(monkeypatch, mean=[0.51, 0.50], uncertainty=[0.01, 0.01])
    gate = _gate(threshold=0.5, min_advantage=0.1)
    decision = gate.decide({}, [[0]*6, [0]*6], fallback_scores=[0.2, 0.9])
    assert decision.accepted is False
    assert decision.source == "fallback"
    assert decision.candidate_idx == decision.fallback_idx == 1


def test_gate_accepts_when_advantage_clears_min_advantage(monkeypatch):
    # advantage = 0.65 - 0.5 = 0.15, comfortably above min_advantage=0.1
    # (avoids asserting exact floating-point boundary equality, which the
    # underlying `advantage >= min_advantage` comparison does not guarantee
    # for arbitrary decimal literals -- e.g. 0.6 - 0.5 != 0.1 in IEEE754)
    _patch_scores(monkeypatch, mean=[0.65, 0.5], uncertainty=[0.01, 0.01])
    gate = _gate(threshold=0.5, min_advantage=0.1)
    decision = gate.decide({}, [[0]*6, [0]*6], fallback_scores=[0.2, 0.9])
    assert decision.accepted is True
    assert decision.candidate_idx == 0


# ── Decision payload correctness ────────────────────────────────────────────

def test_gate_decision_fields(monkeypatch):
    _patch_scores(monkeypatch, mean=[0.8, 0.3], uncertainty=[0.05, 0.4])
    gate = _gate(threshold=0.5)
    d = gate.decide({}, [[1, 2, 3, 0, 0, 0], [4, 5, 6, 0, 0, 0]], fallback_scores=[0.1, 0.9])
    assert isinstance(d, GateDecision)
    assert d.policy_idx == 0
    assert d.fallback_idx == 1
    assert d.critic_score == pytest.approx(0.8)
    assert d.fallback_score == pytest.approx(0.3)  # critic's OWN score for the fallback candidate
    assert d.uncertainty == pytest.approx(0.05)


# ── Input validation ─────────────────────────────────────────────────────────

def test_gate_raises_on_empty_candidates(monkeypatch):
    _patch_scores(monkeypatch, mean=[], uncertainty=[])
    gate = _gate()
    with pytest.raises(ValueError):
        gate.decide({}, [], fallback_scores=[])


def test_gate_raises_on_misaligned_fallback_scores(monkeypatch):
    _patch_scores(monkeypatch, mean=[0.5, 0.5], uncertainty=[0.1, 0.1])
    gate = _gate()
    with pytest.raises(ValueError):
        gate.decide({}, [[0]*6, [0]*6], fallback_scores=[0.1, 0.2, 0.3])  # length 3 != 2


# ── ActionPolicyRiskGate adapter ────────────────────────────────────────────

def test_action_policy_gate_returns_original_action_not_projected_pose(monkeypatch):
    _patch_scores(monkeypatch, mean=[0.9, 0.1], uncertainty=[0.01, 0.01])
    gate = _gate(threshold=0.5)

    original_actions = [{"chunk": "A"}, {"chunk": "B"}]

    def proposal_fn(obs):
        return {"actions": original_actions,
                "grasp_poses": [[0]*6, [1]*6],
                "fallback_scores": [0.1, 0.9]}

    apg = ActionPolicyRiskGate(gate, proposal_fn)
    action, audit = apg.predict(observation=None, scene_context={})
    # must return the ORIGINAL action object for execution, not a pose
    assert action is original_actions[0]
    assert audit["accepted"] is True
    assert audit["n_proposals"] == 2
    assert audit["gate_type"] == "counterfactual_world_critic"


def test_action_policy_gate_uses_action_to_grasp_when_poses_absent(monkeypatch):
    _patch_scores(monkeypatch, mean=[0.2, 0.9], uncertainty=[0.01, 0.01])
    gate = _gate(threshold=0.5)

    original_actions = [{"chunk": "A"}, {"chunk": "B"}]
    calls = []

    def proposal_fn(obs):
        return {"actions": original_actions, "fallback_scores": [0.9, 0.1]}  # no grasp_poses

    def action_to_grasp(action, obs):
        calls.append(action)
        return [0.0] * 6

    apg = ActionPolicyRiskGate(gate, proposal_fn, action_to_grasp=action_to_grasp)
    action, audit = apg.predict(observation=None, scene_context={})
    assert len(calls) == 2, "action_to_grasp must be called once per proposed action"
    assert action is original_actions[1]  # policy_idx=1 has the highest mean and is confident


def test_action_policy_gate_raises_without_action_to_grasp_when_needed(monkeypatch):
    _patch_scores(monkeypatch, mean=[0.5, 0.5], uncertainty=[0.01, 0.01])
    gate = _gate(threshold=0.5)

    def proposal_fn(obs):
        return {"actions": [{"chunk": "A"}, {"chunk": "B"}], "fallback_scores": [0.1, 0.1]}

    apg = ActionPolicyRiskGate(gate, proposal_fn, action_to_grasp=None)
    with pytest.raises(ValueError):
        apg.predict(observation=None, scene_context={})
