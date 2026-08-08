"""Fail-closed admission gate for formal candidate-critic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class CriticDataRejected(ValueError):
    pass


def require_frozen_sample(sample: Mapping[str, Any], expected_execution_version: str) -> None:
    provenance = sample.get("provenance") or sample.get("execution", {}).get("provenance")
    if not isinstance(provenance, Mapping):
        raise CriticDataRejected("missing provenance")
    checks = {
        "label_status": "frozen",
        "eligible_for_critic_training": True,
        "execution_semantics_version": expected_execution_version,
    }
    for key, expected in checks.items():
        if key not in provenance:
            raise CriticDataRejected(f"missing required training field: {key}")
        if provenance[key] != expected:
            raise CriticDataRejected(
                f"{key}={provenance[key]!r}; expected {expected!r}"
            )
    if provenance.get("legacy_execution_confounded", False):
        raise CriticDataRejected("legacy execution-confounded data is diagnostic-only")


def iter_frozen_jsonl(path: str | Path, expected_execution_version: str) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            try:
                require_frozen_sample(sample, expected_execution_version)
            except CriticDataRejected as exc:
                raise CriticDataRejected(f"{path}:{line_number}: {exc}") from exc
            yield sample


def admit_frozen_samples(samples: Iterable[Mapping[str, Any]], expected_execution_version: str) -> list[Mapping[str, Any]]:
    admitted = []
    for sample in samples:
        require_frozen_sample(sample, expected_execution_version)
        admitted.append(sample)
    return admitted
