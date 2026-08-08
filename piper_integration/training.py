"""Fail-closed admission gate for formal candidate-critic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class CriticDataRejected(ValueError):
    pass


REQUIRED_PROVENANCE_FIELDS = (
    "execution_config_hash",
    "embodiment_config_hash",
    "source_commit",
    "model_variant_id",
    "seed",
    "object_id",
    "candidate_id",
    "candidate_schema_version",
)


def require_frozen_sample(sample: Mapping[str, Any], expected_execution_version: str) -> None:
    """Admit a sample into formal critic training, or refuse it.

    Fail-closed: a sample is rejected unless it carries COMPLETE provenance.
    Checking only label/eligibility/version is insufficient -- those can be set
    on a record that cannot be traced back to the execution config, embodiment
    config, source commit or candidate that produced it, which would make any
    downstream result unauditable.
    """
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

    for key in REQUIRED_PROVENANCE_FIELDS:
        if key not in provenance:
            raise CriticDataRejected(f"missing required provenance field: {key}")
        value = provenance[key]
        if value is None:
            raise CriticDataRejected(f"provenance field {key} is null")
        if key == "seed":
            # bool is an int subclass; a boolean seed is a schema error, and
            # seed 0 is legitimate so truthiness must not be used here.
            if isinstance(value, bool) or not isinstance(value, int):
                raise CriticDataRejected(f"provenance field seed must be an int, got {value!r}")
        elif isinstance(value, str):
            if not value.strip():
                raise CriticDataRejected(f"provenance field {key} is empty")
        elif not value:
            raise CriticDataRejected(f"provenance field {key} is empty")

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
