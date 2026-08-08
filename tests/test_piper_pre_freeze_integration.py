import json

import pytest

from pathlib import Path

import yaml

from piper_integration.contracts import (
    PHASE2Y_GATE_STATUS, Candidate, ExecutionConfig, Provenance)
from piper_integration.feasibility import IKResult, PiperFeasibilityAdapter
from piper_integration.metadata import load_embodiment_metadata, validate_metadata_against_assets
from piper_integration.training import (
    REQUIRED_PROVENANCE_FIELDS, STRING_PROVENANCE_FIELDS, CriticDataRejected,
    require_frozen_sample)


def candidate():
    return Candidate(
        candidate_id="scene-1:candidate-0",
        target_instance_id="instance-4",
        pose=[0.1, 0.2, 0.3, 0, 0, 0, 1],
        pose_frame="camera_depth_optical_frame",
        pose_convention="xyz+xyzw",
        score=0.8,
        local_point_indices=[2, 8],
    )


def test_candidate_requires_explicit_pose_frame():
    with pytest.raises(ValueError, match="frame and convention"):
        Candidate("c", "i", [0, 0, 0, 0, 0, 0, 1], "", "", 1.0)


def test_adapter_is_object_agnostic_and_preserves_unknowns():
    adapter = PiperFeasibilityAdapter([[-1, 1], [-2, 2]], max_opening_m=0.104)
    seen = []

    def solve(pose, frame, convention):
        seen.append((pose, frame, convention))
        return IKResult(True, [0.0, 1.0])

    features = adapter.evaluate(candidate(), solve)
    assert seen[0][1:] == ("camera_depth_optical_frame", "xyz+xyzw")
    assert features.ik_feasible is True
    assert features.joint_margin == pytest.approx(0.25)
    assert features.opening_feasible is None
    assert features.path_clearance_m is None


def test_execution_config_hash_is_order_independent():
    kwargs = dict(
        backend="piper-sim", model_variant_id="variant-a",
        capture_semantics={"frame": "tcp"}, close_lift_semantics={"close": "unset"},
        success_definition={"lift_m": None},
    )
    a = ExecutionConfig(planner={"z": 1, "a": 2}, **kwargs)
    b = ExecutionConfig(planner={"a": 2, "z": 1}, **kwargs)
    assert a.config_hash == b.config_hash


def test_embodiment_metadata_is_versioned_and_hashed():
    metadata = load_embodiment_metadata("configs/piper/pre_freeze_embodiment.yaml")
    assert metadata.values["schema_version"] == "piper-embodiment-v1alpha1"
    assert len(metadata.config_hash) == 64
    validate_metadata_against_assets(metadata)


def test_pre_freeze_provenance_cannot_be_training_eligible():
    with pytest.raises(ValueError, match="training-ineligible"):
        Provenance("e", "b", "commit", "variant", 1, "pear", "c", eligible_for_critic_training=True)


def complete_provenance(**overrides):
    """Provenance with every field the schema promises, for the positive case."""
    base = {
        "label_status": "frozen",
        "eligible_for_critic_training": True,
        "execution_semantics_version": "piper-execution-v1",
        "legacy_execution_confounded": False,
        "execution_config_hash": "a" * 64,
        "embodiment_config_hash": "b" * 64,
        "source_commit": "0123456789abcdef",
        "model_variant_id": "variant-a",
        "seed": 5001,
        "object_id": "pear",
        "candidate_id": "scene-1:candidate-0",
        "candidate_schema_version": "owg-piper-candidate-v1alpha1",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("provenance", [
    {},
    # pre-freeze / provisional data stays rejected
    {"label_status": "provisional", "eligible_for_critic_training": False,
     "execution_semantics_version": "pre-freeze"},
    # legacy execution-confounded data stays rejected
    complete_provenance(legacy_execution_confounded=True),
    # execution-version mismatch stays rejected
    complete_provenance(execution_semantics_version="pre-freeze"),
])
def test_formal_loader_rejects_unsafe_data(provenance):
    with pytest.raises(CriticDataRejected):
        require_frozen_sample({"provenance": provenance}, "piper-execution-v1")


@pytest.mark.parametrize("missing", REQUIRED_PROVENANCE_FIELDS)
def test_formal_loader_fails_closed_on_incomplete_provenance(missing):
    """Label/eligibility/version alone must not admit a sample."""
    provenance = complete_provenance()
    del provenance[missing]
    with pytest.raises(CriticDataRejected, match="missing required provenance field"):
        require_frozen_sample({"provenance": provenance}, "piper-execution-v1")


@pytest.mark.parametrize("field,value", [
    ("execution_config_hash", ""),
    ("embodiment_config_hash", "   "),
    ("source_commit", None),
    ("model_variant_id", ""),
    ("object_id", ""),
    ("candidate_id", ""),
    ("candidate_schema_version", ""),
    ("seed", "5001"),
    ("seed", True),
])
def test_formal_loader_rejects_empty_or_mistyped_provenance(field, value):
    with pytest.raises(CriticDataRejected):
        require_frozen_sample(
            {"provenance": complete_provenance(**{field: value})}, "piper-execution-v1")


@pytest.mark.parametrize("field", STRING_PROVENANCE_FIELDS)
@pytest.mark.parametrize("value", [
    1234567890,                 # int
    ["a" * 64],                 # list
    {"sha": "abc"},             # dict
    object(),                   # arbitrary object
    3.14,                       # float
    ("a",),                     # tuple
], ids=["int", "list", "dict", "object", "float", "tuple"])
def test_formal_loader_rejects_truthy_non_string_provenance(field, value):
    """A truthy non-string is not a valid hash or identifier.

    Regression guard: an earlier implementation type-checked only strings and
    let every other truthy value fall through the emptiness branch, silently
    admitting int hashes, list hashes, mapping commits and arbitrary objects.
    """
    with pytest.raises(CriticDataRejected, match="must be a string"):
        require_frozen_sample(
            {"provenance": complete_provenance(**{field: value})}, "piper-execution-v1")


@pytest.mark.parametrize("eligible", [
    1,          # Python: 1 == True, so equality admitted this
    1.0,        # Python: 1.0 == True
    "True",
    [1],
    {"a": 1},
    object(),
    False,
    None,
], ids=["int_1", "float_1", "str_True", "list", "dict", "object", "False", "None"])
def test_eligibility_flag_requires_identity_not_equality(eligible):
    """eligible_for_critic_training must be exactly True.

    Regression guard: an earlier implementation compared with ==, and because
    Python evaluates 1 == True the loader admitted eligible=1.
    """
    with pytest.raises(CriticDataRejected, match="must be exactly True"):
        require_frozen_sample(
            {"provenance": complete_provenance(eligible_for_critic_training=eligible)},
            "piper-execution-v1")


def test_eligibility_flag_accepts_boolean_true():
    require_frozen_sample(
        {"provenance": complete_provenance(eligible_for_critic_training=True)},
        "piper-execution-v1")


def test_eligibility_flag_must_be_present():
    provenance = complete_provenance()
    del provenance["eligible_for_critic_training"]
    with pytest.raises(CriticDataRejected, match="eligible_for_critic_training"):
        require_frozen_sample({"provenance": provenance}, "piper-execution-v1")


def test_string_fields_cover_every_non_seed_required_field():
    assert set(STRING_PROVENANCE_FIELDS) == set(REQUIRED_PROVENANCE_FIELDS) - {"seed"}


def test_seed_zero_is_a_legitimate_value():
    """seed=0 must not be rejected by a truthiness check."""
    require_frozen_sample({"provenance": complete_provenance(seed=0)}, "piper-execution-v1")


@pytest.mark.parametrize("bad_seed", ["5001", True, False, 1.0, None, [1], {"s": 1}])
def test_formal_loader_rejects_non_integer_seed(bad_seed):
    with pytest.raises(CriticDataRejected):
        require_frozen_sample(
            {"provenance": complete_provenance(seed=bad_seed)}, "piper-execution-v1")


def test_formal_loader_accepts_only_matching_frozen_data():
    require_frozen_sample({"provenance": complete_provenance()}, "piper-execution-v1")


def test_gate3_state_is_failed_and_blocked_not_suspended():
    """Serialized gate state must match the authoritative repo verdict."""
    assert PHASE2Y_GATE_STATUS == "gate3_failed_sweep_blocked"
    assert "suspend" not in PHASE2Y_GATE_STATUS
    assert Provenance("e", "b", "commit", "variant", 1, "pear", "c").phase2y_gate_status == (
        PHASE2Y_GATE_STATUS)
    cfg = yaml.safe_load(Path("configs/piper/pre_freeze_execution.yaml").read_text())
    assert cfg["phase2y_gate_status"] == PHASE2Y_GATE_STATUS
    assert cfg["phase2y_gate3_verdict"] == "failed"
    assert cfg["phase2y_sweep_status"] == "blocked"
    assert "finger-table" in cfg["phase2y_gate3_failure_reason"]
