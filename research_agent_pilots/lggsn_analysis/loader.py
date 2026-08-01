"""Phase 1 data loading for the LGGSN checkpoint statistical-analysis suite.

Two independent, explicit data sources, never blended or inferred from one
another (see alignment.py / statistics.py for what is built on top of these):

  * aggregate metrics: the four already-committed per-checkpoint
    metrics.json files plus matrix_summary.json under
    research_agent_pilots/lggsn_suite/eval_outputs/ -- read-only, never
    written to by this module.
  * pair-level records: deterministic JSONL fixtures (Phase 1 test data) or
    a real, evaluator-produced pair_results.jsonl (Phase 1 only reports
    whether one exists -- see load_real_pair_results -- it is never
    analyzed here; that is out of scope until a later phase).

Every failure mode here is fail-closed: a missing, malformed, empty, or
duplicate-identity input raises LoaderError rather than returning a
degraded, zeroed, or fabricated result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

CORE_CHECKPOINT_NAMES: tuple[str, ...] = ("base", "nodist", "nozrel", "full_v2")

_REQUIRED_PAIR_RECORD_KEYS: tuple[str, ...] = ("query", "pos_row_id", "neg_row_id", "correct")


class LoaderError(RuntimeError):
    """Any fail-closed loader violation: missing, malformed, empty, or
    duplicate-identity input. Never raised for a merely-absent *real*
    pair_results.jsonl -- see load_real_pair_results, which reports that
    case as an explicit PairDataStatus.UNAVAILABLE instead."""


class PairDataStatus(str, Enum):
    UNAVAILABLE = "unavailable"
    LOADED = "loaded"


class LoadMode(str, Enum):
    AGGREGATE_ONLY = "aggregate_only"
    PAIR_FIXTURES = "pair_fixtures"


@dataclass(frozen=True)
class PairRecord:
    """One pair-level observation for a single checkpoint: whether it
    scored the positive grasp above the negative grasp for this exact
    (query, pos_row_id, neg_row_id) identity."""

    query: str
    pos_row_id: str
    neg_row_id: str
    correct: bool

    @property
    def pair_key(self) -> tuple[str, str, str]:
        return (self.query, self.pos_row_id, self.neg_row_id)


@dataclass(frozen=True)
class PairDataResult:
    status: PairDataStatus
    records: tuple[PairRecord, ...] = ()
    source_path: Path | None = None


@dataclass(frozen=True)
class LoadedDataset:
    mode: LoadMode
    aggregate_metrics: dict[str, dict]
    matrix_summary: dict
    pair_records: dict[str, tuple[PairRecord, ...]] = field(default_factory=dict)


def load_aggregate_metrics(checkpoint_dir: Path) -> dict:
    """Read one checkpoint's committed metrics.json verbatim, read-only."""
    checkpoint_dir = Path(checkpoint_dir)
    metrics_path = checkpoint_dir / "metrics.json"
    if not metrics_path.exists():
        raise LoaderError(f"missing aggregate metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise LoaderError(f"aggregate metrics file must contain a JSON object: {metrics_path}")
    return data


def load_all_aggregate_metrics(
    eval_outputs_dir: Path, checkpoint_names: Sequence[str] = CORE_CHECKPOINT_NAMES
) -> dict[str, dict]:
    eval_outputs_dir = Path(eval_outputs_dir)
    if not checkpoint_names:
        raise LoaderError("checkpoint_names must not be empty")
    return {name: load_aggregate_metrics(eval_outputs_dir / name) for name in checkpoint_names}


def load_matrix_summary(eval_outputs_dir: Path) -> dict:
    path = Path(eval_outputs_dir) / "matrix_summary.json"
    if not path.exists():
        raise LoaderError(f"missing matrix summary file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise LoaderError(f"matrix summary file must contain a JSON object: {path}")
    return data


def _parse_pair_record(obj: object, *, source: str) -> PairRecord:
    if not isinstance(obj, dict):
        raise LoaderError(f"{source}: pair record must be a JSON object, got {type(obj).__name__}")
    missing = [k for k in _REQUIRED_PAIR_RECORD_KEYS if k not in obj]
    if missing:
        raise LoaderError(f"{source}: pair record missing required key(s): {missing}")

    query = obj["query"]
    pos_row_id = obj["pos_row_id"]
    neg_row_id = obj["neg_row_id"]
    correct = obj["correct"]

    if not isinstance(query, str) or not query:
        raise LoaderError(f"{source}: 'query' must be a non-empty string")
    for key_name, key_value in (("pos_row_id", pos_row_id), ("neg_row_id", neg_row_id)):
        if isinstance(key_value, bool) or not isinstance(key_value, (str, int)):
            raise LoaderError(f"{source}: '{key_name}' must be a string or int")
        if isinstance(key_value, str) and not key_value:
            raise LoaderError(f"{source}: '{key_name}' must be non-empty")
    if not isinstance(correct, bool):
        raise LoaderError(f"{source}: 'correct' must be a boolean")

    return PairRecord(query=query, pos_row_id=str(pos_row_id), neg_row_id=str(neg_row_id), correct=correct)


def load_pair_fixture_jsonl(path: Path) -> tuple[PairRecord, ...]:
    """Load a deterministic pair-level JSONL fixture (Phase 1 test data --
    one JSON object per line with keys query/pos_row_id/neg_row_id/correct).
    Fails closed on a missing file, invalid JSON, a record missing a
    required key or with a wrong-typed value, an empty file, or any
    duplicate (query, pos_row_id, neg_row_id) identity within the file."""
    path = Path(path)
    if not path.exists():
        raise LoaderError(f"missing pair-level fixture file: {path}")

    records: list[PairRecord] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            source = f"{path}:{line_no}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise LoaderError(f"{source}: invalid JSON: {e}") from e
            record = _parse_pair_record(obj, source=source)
            if record.pair_key in seen:
                raise LoaderError(f"{source}: duplicate pair identity {record.pair_key!r}")
            seen.add(record.pair_key)
            records.append(record)

    if not records:
        raise LoaderError(f"pair-level fixture file contains no records: {path}")
    return tuple(records)


def load_real_pair_results(checkpoint_dir: Path) -> PairDataResult:
    """Probe for a real, evaluator-produced pair_results.jsonl for one
    checkpoint. Phase 1 never analyzes it -- this only reports whether it
    exists, explicitly, rather than silently substituting zeros or
    fabricated records when it is absent (as it currently is for every
    checkpoint in this checkout)."""
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / "pair_results.jsonl"
    if not path.exists():
        return PairDataResult(status=PairDataStatus.UNAVAILABLE, records=(), source_path=path)
    records = load_pair_fixture_jsonl(path)
    return PairDataResult(status=PairDataStatus.LOADED, records=records, source_path=path)


def load_dataset(
    eval_outputs_dir: Path,
    *,
    mode: LoadMode,
    checkpoint_names: Sequence[str] = CORE_CHECKPOINT_NAMES,
    pair_fixture_paths: Mapping[str, Path] | None = None,
) -> LoadedDataset:
    """Load one dataset for downstream alignment/statistics. Aggregate
    metrics are always loaded (read-only); pair-level records are loaded
    only in PAIR_FIXTURES mode, and only from the explicitly given fixture
    paths -- never inferred from the aggregate metrics (see statistics.py's
    module docstring for why)."""
    eval_outputs_dir = Path(eval_outputs_dir)
    aggregate_metrics = load_all_aggregate_metrics(eval_outputs_dir, checkpoint_names)
    matrix_summary = load_matrix_summary(eval_outputs_dir)

    if mode is LoadMode.AGGREGATE_ONLY:
        if pair_fixture_paths:
            raise LoaderError("aggregate-only mode must not be given pair_fixture_paths")
        return LoadedDataset(
            mode=mode, aggregate_metrics=aggregate_metrics, matrix_summary=matrix_summary, pair_records={}
        )

    if mode is LoadMode.PAIR_FIXTURES:
        if not pair_fixture_paths:
            raise LoaderError("pair_fixtures mode requires at least one entry in pair_fixture_paths")
        pair_records = {
            name: load_pair_fixture_jsonl(Path(path)) for name, path in pair_fixture_paths.items()
        }
        return LoadedDataset(
            mode=mode, aggregate_metrics=aggregate_metrics, matrix_summary=matrix_summary,
            pair_records=pair_records,
        )

    raise LoaderError(f"unknown load mode: {mode!r}")
