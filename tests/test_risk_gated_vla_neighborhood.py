import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.risk_gated_vla_neighborhood_eval import (
    PERTURBATIONS,
    build_local_perturbations,
    score_neighborhood,
    select_candidate,
)


def test_local_perturbations_preserve_nominal_and_have_expected_count():
    pose = [0.1, -0.2, 0.8, 0.3, 0.04, 0.05]
    rows = build_local_perturbations(pose)
    assert len(rows) == len(PERTURBATIONS) == 5
    assert rows[0]["perturbation_type"] == "nominal"
    assert rows[0]["candidate_pose"] == pose
    assert rows[1]["candidate_pose"][0] == pytest.approx(pose[0] + 0.002)
    assert rows[3]["candidate_pose"][3] == pytest.approx(pose[3] + np.deg2rad(2.0))


def test_local_perturbations_reject_wrong_pose_shape():
    with pytest.raises(ValueError):
        build_local_perturbations([0.0, 0.0, 0.8])


def test_select_candidate_modes():
    rows = [
        {"point_score": 0.9, "neighborhood_mean": 0.4, "neighborhood_worst_case": 0.1},
        {"point_score": 0.8, "neighborhood_mean": 0.7, "neighborhood_worst_case": 0.6},
    ]
    assert select_candidate(rows, "point") == 0
    assert select_candidate(rows, "mean") == 1
    assert select_candidate(rows, "worst_case") == 1


def test_score_neighborhood_uses_local_candidates(monkeypatch):
    def fake_score(scene, candidates, bundles, relative):
        assert len(candidates) == 5
        return np.array([0.8, 0.6, 0.7, 0.5, 0.9]), np.zeros(5)

    import scripts.risk_gated_vla_neighborhood_eval as mod
    monkeypatch.setattr(mod, "score_candidates", fake_score)
    out = score_neighborhood({"object": "cracker"}, [0, 0, 0.8, 0, 0.04, 0.05], [], True)
    assert out["point_score"] == pytest.approx(0.8)
    assert out["neighborhood_mean"] == pytest.approx(0.7)
    assert out["neighborhood_worst_case"] == pytest.approx(0.5)
