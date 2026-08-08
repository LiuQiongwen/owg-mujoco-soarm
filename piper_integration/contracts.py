"""Versioned, backend-neutral data contracts for partial Piper migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

CANDIDATE_SCHEMA_VERSION = "owg-piper-candidate-v1alpha1"
PRE_FREEZE_EXECUTION_VERSION = "pre-freeze"


def stable_hash(value: Any) -> str:
    """Return a deterministic sha256 for JSON-compatible configuration."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_vector(name: str, values: Sequence[float], length: int) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    target_instance_id: str
    pose: Sequence[float]
    pose_frame: str
    pose_convention: str
    score: float
    local_point_indices: Sequence[int] = field(default_factory=tuple)
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.target_instance_id:
            raise ValueError("candidate_id and target_instance_id are required")
        if not self.pose_frame or not self.pose_convention:
            raise ValueError("pose frame and convention must be explicit")
        object.__setattr__(self, "pose", _finite_vector("pose", self.pose, 7))
        object.__setattr__(self, "score", float(self.score))
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        indices = tuple(int(i) for i in self.local_point_indices)
        if any(i < 0 for i in indices):
            raise ValueError("local point indices must be non-negative")
        object.__setattr__(self, "local_point_indices", indices)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateFeatures:
    """Missing pre-freeze measurements remain ``None``; zero is never imputed."""

    ik_feasible: bool | None = None
    joint_margin: float | None = None
    opening_feasible: bool | None = None
    path_clearance_m: float | None = None
    gripper_frame_relation: Mapping[str, float] | None = None
    contact_frame_features: Mapping[str, float] | None = None
    feature_status: str = "provisional"
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("joint_margin", "path_clearance_m"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
        if self.feature_status != "provisional":
            raise ValueError("pre-freeze adapter may only emit provisional features")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str
    model_variant_id: str
    planner: Mapping[str, Any]
    capture_semantics: Mapping[str, Any]
    close_lift_semantics: Mapping[str, Any]
    success_definition: Mapping[str, Any]
    execution_semantics_version: str = PRE_FREEZE_EXECUTION_VERSION

    def __post_init__(self) -> None:
        if not self.backend or not self.model_variant_id:
            raise ValueError("backend and model_variant_id are required")

    @property
    def config_hash(self) -> str:
        return stable_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Provenance:
    execution_config_hash: str
    embodiment_config_hash: str
    source_commit: str
    model_variant_id: str
    seed: int
    object_id: str
    candidate_id: str
    candidate_schema_version: str = CANDIDATE_SCHEMA_VERSION
    execution_semantics_version: str = PRE_FREEZE_EXECUTION_VERSION
    label_status: str = "provisional"
    eligible_for_critic_training: bool = False
    phase2y_gate_status: str = "suspended"
    legacy_execution_confounded: bool = False

    def __post_init__(self) -> None:
        if self.execution_semantics_version == PRE_FREEZE_EXECUTION_VERSION:
            if self.label_status != "provisional" or self.eligible_for_critic_training:
                raise ValueError("pre-freeze records must be provisional and training-ineligible")
        required = (
            self.execution_config_hash, self.embodiment_config_hash,
            self.source_commit, self.model_variant_id, self.object_id, self.candidate_id,
        )
        if not all(required):
            raise ValueError("complete provenance is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    success: bool | None
    failure_stage: str | None
    phase_logs: Sequence[Mapping[str, Any]]
    provenance: Provenance
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failure_stage": self.failure_stage,
            "phase_logs": list(self.phase_logs),
            "provenance": self.provenance.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }
