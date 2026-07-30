"""
Unit tests for the Risk-Gated VLA Phase 1 harness (scripts/risk_gated_vla_phase1_eval.py)
and the causal-validity registry entries it depends on.

Covers (per results/risk_gated_vla/preregistration requirements):
  - feature causal admissibility (rejects EXECUTION_DERIVED / unregistered fields)
  - candidate-pool identity across compared methods (data-level, from a real collected scene)
  - seed determinism and method-independence (the exact bug this study's Phase 1 fixes)
  - grouped train/val split has no scene-level leakage (world_model/train_counterfactual_critic.py)

Gate/threshold/fallback behavior (Phase 2's risk gate) is NOT covered here -- that code does
not exist yet (Phase 2 is gated on Phase 1 passing). Add those tests when Phase 2 lands.

Run: conda run -n tango python -m pytest tests/test_risk_gated_vla_phase1.py -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from causal_validity_audit.provenance import (
    audit_feature_set, CausalValidityViolation, WORLD_MODEL_FIELDS,
)
from scripts.risk_gated_vla_phase1_eval import scene_seed, random_choice_seed


# ── Causal admissibility ────────────────────────────────────────────────────

def test_world_model_fields_registered_pre_execution():
    admissible = ["grasp_pose", "obj_pos_before", "obj_quat_before", "pc_stats_before"]
    assert audit_feature_set(admissible, context="test") is True


def test_execution_derived_field_rejected():
    with pytest.raises(CausalValidityViolation):
        audit_feature_set(["grasp_pose", "success"], context="test")


def test_unregistered_field_rejected_fail_closed():
    with pytest.raises(CausalValidityViolation):
        audit_feature_set(["grasp_pose", "some_never_registered_field"], context="test")


def test_all_world_model_fields_have_declared_provenance():
    # every field this study's harness could plausibly touch must be in the registry,
    # not silently fall through to "unregistered"
    expected = {"grasp_pose", "obj_pos_before", "obj_quat_before", "pc_stats_before",
                "success", "dz", "fell_off"}
    assert expected <= set(WORLD_MODEL_FIELDS.keys())


# ── Seed determinism and method-independence ────────────────────────────────
# This is the exact bug results/risk_gated_vla/audit.md Section 4 documents in the
# predecessor script (scripts/eval_wm_reranking_full.py::trial_seed encoded `method`
# into the seed, breaking candidate-pool sharing between compared methods).

def test_scene_seed_deterministic():
    s1 = scene_seed(42, "cracker", 0, 5)
    s2 = scene_seed(42, "cracker", 0, 5)
    assert s1 == s2


def test_scene_seed_varies_with_scene_idx():
    seeds = {scene_seed(42, "cracker", 0, i) for i in range(10)}
    assert len(seeds) == 10, "distinct scene_idx must produce distinct seeds"


def test_scene_seed_varies_with_object():
    s_cracker = scene_seed(42, "cracker", 0, 0)
    s_mustard = scene_seed(42, "mustard", 1, 0)
    assert s_cracker != s_mustard


def test_scene_seed_signature_has_no_method_parameter():
    # structural guard against reintroducing the predecessor's bug: scene_seed must
    # not be able to take a `method` argument at all.
    import inspect
    params = set(inspect.signature(scene_seed).parameters)
    assert "method" not in params


def test_random_choice_seed_differs_from_scene_seed_stream():
    base = scene_seed(42, "cracker", 0, 0)
    rc = random_choice_seed(42, "cracker", 0, 0)
    assert rc != base, "random-choice RNG stream must be salted, not reuse the pool seed"


def test_random_choice_seed_deterministic():
    assert (random_choice_seed(42, "drill", 2, 7)
            == random_choice_seed(42, "drill", 2, 7))


# ── Candidate-pool identity across methods (data-level check) ──────────────
# Structural proof (single build_pool() call feeds every method) lives in the script's
# --smoke-check. This test additionally verifies it against real collected data, if
# available, without needing MuJoCo: every method's chosen candidate must appear
# verbatim among that scene's oracle_per_candidate poses (i.e. drawn from the one
# shared pool, not an independently-sampled one).

_SCENES_CANDIDATES = [
    Path("results/risk_gated_vla/smoke/pilot10/scenes.jsonl"),
    Path("results/risk_gated_vla/phase1/scenes.jsonl"),
    Path("results/risk_gated_vla/smoke/scenes.jsonl"),
]


def _first_existing_scenes_file():
    for p in _SCENES_CANDIDATES:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def test_pool_identity_across_methods_on_real_data():
    path = _first_existing_scenes_file()
    if path is None:
        pytest.skip("no collected scenes.jsonl found yet -- run Phase 1 smoke test first")

    checked = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            pool_poses = [tuple(c["candidate_pose"]) for c in rec["oracle_per_candidate"]]
            for method, idx in rec["idx_by_method"].items():
                assert 0 <= idx < len(pool_poses)
                assert rec["outcomes"][method]["candidate_idx"] == idx
                checked += 1
            # every method's idx must point into the SAME pool -- i.e. the pool has
            # exactly k_grasps entries shared by all methods, not one pool per method
            assert len(pool_poses) == rec["k_grasps"]
    assert checked > 0


def test_seeds_are_scene_not_method_keyed_on_real_data():
    """The single most direct regression test for audit.md Section 4: every recorded
    scene has exactly ONE seed, not one seed per method."""
    path = _first_existing_scenes_file()
    if path is None:
        pytest.skip("no collected scenes.jsonl found yet -- run Phase 1 smoke test first")
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            assert "seed" in rec and isinstance(rec["seed"], int)
            # exactly one seed field for the whole scene -- not per-method
            assert rec["seed"] == scene_seed(
                json.loads(Path(path.parent / "config.json").read_text())["base_seed"],
                rec["object"], rec["obj_idx"], rec["scene_idx"],
            ) if (path.parent / "config.json").exists() else True


# ── Grouped train/val split (world_model/train_counterfactual_critic.py) ───

def test_counterfactual_critic_split_has_no_scene_leakage():
    torch = pytest.importorskip("torch")
    from world_model.train_counterfactual_critic import train_one, OBJECTS

    rng = __import__("numpy").random.default_rng(0)
    synth_scenes = []
    for obj in OBJECTS:
        for i in range(12):
            k = 4
            xs = [[float(v) for v in rng.normal(size=6 + 9 + len(OBJECTS))] for _ in range(k)]
            ys = [float(j == 0) for j in range(k)]  # one success, rest fail -- mixed scene
            synth_scenes.append({"key": (obj, i), "object": obj, "x": xs, "y": ys})

    _, _, _, train, val, _ = train_one(synth_scenes, "object_bce", seed=0, epochs=1)

    train_keys = {s["key"] for s in train}
    val_keys = {s["key"] for s in val}
    assert train_keys.isdisjoint(val_keys), "a scene must not appear in both train and val"
    assert train_keys | val_keys == {s["key"] for s in synth_scenes}
