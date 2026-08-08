"""Pre-freeze Piper integration contracts.

This package is intentionally disconnected from production critic training.
See :mod:`piper_integration.training` for the fail-closed admission gate.
"""

from .contracts import (
    Candidate,
    CandidateFeatures,
    ExecutionConfig,
    ExecutionResult,
    Provenance,
    stable_hash,
)
from .metadata import EmbodimentMetadata, load_embodiment_metadata, validate_metadata_against_assets

__all__ = [
    "Candidate",
    "CandidateFeatures",
    "ExecutionConfig",
    "ExecutionResult",
    "Provenance",
    "stable_hash",
    "EmbodimentMetadata",
    "load_embodiment_metadata",
    "validate_metadata_against_assets",
]
