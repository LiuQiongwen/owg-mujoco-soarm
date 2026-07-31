import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.models import ExperimentSpec, RequiredMetric, TimeoutSpec


def _minimal_spec_kwargs(**overrides):
    base = dict(
        task_id="demo_task",
        goal="a harmless demo",
        allowed_paths=["experiments"],
        smoke_command=["python", "-c", "print('ok')"],
    )
    base.update(overrides)
    return base


def test_valid_minimal_spec_applies_defaults():
    spec = ExperimentSpec(**_minimal_spec_kwargs())
    assert spec.forbidden_paths == []
    assert spec.confirmatory_command is None
    assert spec.seeds == [0]
    assert spec.max_run_count == 1
    assert isinstance(spec.timeouts, TimeoutSpec)
    assert spec.timeouts.smoke_seconds > 0
    assert spec.required_metrics == []
    assert spec.approval.required_for_confirmatory is True


def test_rejects_unknown_top_level_field():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(not_a_real_field="oops"))


def test_rejects_unknown_nested_timeout_field():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(timeouts={"smoke_seconds": 10, "bogus_field": 1}))


def test_rejects_unknown_required_metric_field():
    with pytest.raises(ValidationError):
        RequiredMetric(name="acc", type="float", bogus="x")


def test_rejects_empty_allowed_paths():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(allowed_paths=[]))


def test_rejects_missing_required_field():
    with pytest.raises(ValidationError):
        ExperimentSpec(task_id="demo", allowed_paths=["x"], smoke_command=["python"])  # missing goal


def test_rejects_path_traversal_in_allowed_paths():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(allowed_paths=["../outside"]))


def test_rejects_absolute_allowed_path():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(allowed_paths=["/etc/passwd"]))


def test_rejects_overlapping_allowed_and_forbidden_paths():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(allowed_paths=["experiments"], forbidden_paths=["experiments"]))


def test_rejects_smoke_command_as_shell_string():
    """Commands must be arrays, never shell strings (see subprocess_runner)."""
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(smoke_command="python -c \"print('ok')\""))


def test_rejects_empty_smoke_command():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(smoke_command=[]))


def test_rejects_smoke_command_with_empty_token():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(smoke_command=["python", ""]))


def test_rejects_confirmatory_command_as_shell_string():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(confirmatory_command="python train.py --full"))


def test_accepts_confirmatory_command_as_array():
    spec = ExperimentSpec(**_minimal_spec_kwargs(confirmatory_command=["python", "train_full.py"]))
    assert spec.confirmatory_command == ["python", "train_full.py"]


def test_rejects_empty_seeds():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(seeds=[]))


def test_rejects_max_run_count_below_one():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(max_run_count=0))


def test_rejects_max_run_count_above_ten():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(max_run_count=11))


def test_rejects_non_positive_timeout():
    with pytest.raises(ValidationError):
        ExperimentSpec(**_minimal_spec_kwargs(timeouts={"smoke_seconds": 0}))


def test_required_metrics_round_trip():
    spec = ExperimentSpec(
        **_minimal_spec_kwargs(
            required_metrics=[
                {"name": "accuracy", "type": "float", "min_value": 0.0, "max_value": 1.0},
                {"name": "smoke_ok", "type": "bool"},
            ]
        )
    )
    assert len(spec.required_metrics) == 2
    assert spec.required_metrics[0].name == "accuracy"
    assert spec.required_metrics[0].max_value == 1.0
    assert spec.required_metrics[1].type == "bool"


def test_approval_requirements_round_trip():
    spec = ExperimentSpec(
        **_minimal_spec_kwargs(approval={"required_for_confirmatory": False, "approved_by": "alice"})
    )
    assert spec.approval.required_for_confirmatory is False
    assert spec.approval.approved_by == "alice"
