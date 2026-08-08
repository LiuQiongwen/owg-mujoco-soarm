"""Structured provisional JSONL storage."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import Candidate, CandidateFeatures, ExecutionResult


class ProvisionalOutcomeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, candidate: Candidate, features: CandidateFeatures, result: ExecutionResult) -> None:
        provenance = result.provenance
        if provenance.label_status != "provisional" or provenance.eligible_for_critic_training:
            raise ValueError("provisional store only accepts training-ineligible records")
        if candidate.candidate_id != provenance.candidate_id:
            raise ValueError("candidate/provenance ID mismatch")
        record = {
            "candidate": candidate.to_dict(),
            "features": features.to_dict(),
            "execution": result.to_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
