import json

import pytest

from piper_integration.contracts import Candidate, ExecutionConfig, Provenance
from piper_integration.feasibility import IKResult, PiperFeasibilityAdapter
from piper_integration.metadata import load_embodiment_metadata, validate_metadata_against_assets
from piper_integration.training import CriticDataRejected, require_frozen_sample


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


@pytest.mark.parametrize("provenance", [
    {},
    {"label_status": "provisional", "eligible_for_critic_training": False,
     "execution_semantics_version": "pre-freeze"},
    {"label_status": "frozen", "eligible_for_critic_training": True,
     "execution_semantics_version": "piper-execution-v1", "legacy_execution_confounded": True},
])
def test_formal_loader_rejects_unsafe_data(provenance):
    with pytest.raises(CriticDataRejected):
        require_frozen_sample({"provenance": provenance}, "piper-execution-v1")


def test_formal_loader_accepts_only_matching_frozen_data():
    sample = {"provenance": {
        "label_status": "frozen",
        "eligible_for_critic_training": True,
        "execution_semantics_version": "piper-execution-v1",
        "legacy_execution_confounded": False,
    }}
    require_frozen_sample(sample, "piper-execution-v1")
