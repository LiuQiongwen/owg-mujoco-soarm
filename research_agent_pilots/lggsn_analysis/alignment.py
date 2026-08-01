"""Phase 1 cross-checkpoint pair alignment.

Aligns pair-level records from two or more checkpoints by the exact
identity (query, pos_row_id, neg_row_id). Fails closed -- raises
AlignmentError -- on a duplicate identity within one checkpoint's records,
or on any identity that is not present in every checkpoint being aligned.
Never drops, pads, reorders past a mismatch, or infers a missing
observation: statistics.py can only ever see pairs that genuinely exist,
with the same identity, in every checkpoint supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from research_agent_pilots.lggsn_analysis.loader import PairRecord

PairKey = tuple[str, str, str]


class AlignmentError(RuntimeError):
    """Any fail-closed alignment violation: too few checkpoints, a
    duplicate pair identity within one checkpoint, or pair identities that
    do not match exactly across all checkpoints being aligned."""


@dataclass(frozen=True)
class AlignedPair:
    query: str
    pos_row_id: str
    neg_row_id: str
    correct_by_checkpoint: Mapping[str, bool]

    @property
    def pair_key(self) -> PairKey:
        return (self.query, self.pos_row_id, self.neg_row_id)


def _index_by_key(checkpoint: str, records: Sequence[PairRecord]) -> dict[PairKey, PairRecord]:
    index: dict[PairKey, PairRecord] = {}
    for record in records:
        key = record.pair_key
        if key in index:
            raise AlignmentError(f"duplicate pair identity for checkpoint {checkpoint!r}: {key!r}")
        index[key] = record
    if not index:
        raise AlignmentError(f"checkpoint {checkpoint!r} has no pair-level records to align")
    return index


def align_pairs(pair_records: Mapping[str, Sequence[PairRecord]]) -> tuple[AlignedPair, ...]:
    """Align pair-level records across two or more checkpoints. Requires
    every checkpoint's set of (query, pos_row_id, neg_row_id) identities to
    be identical; raises AlignmentError naming the first mismatching
    checkpoint otherwise."""
    if len(pair_records) < 2:
        raise AlignmentError(
            f"alignment requires pair-level records for at least two checkpoints, got {len(pair_records)}"
        )

    indices = {name: _index_by_key(name, records) for name, records in pair_records.items()}
    key_sets = {name: set(index.keys()) for name, index in indices.items()}

    reference_name, reference_keys = next(iter(key_sets.items()))
    for name, keys in key_sets.items():
        if keys == reference_keys:
            continue
        missing_here = sorted(reference_keys - keys)
        extra_here = sorted(keys - reference_keys)
        raise AlignmentError(
            "pair identities are misaligned between checkpoints "
            f"{reference_name!r} and {name!r}: "
            f"missing_in_{name}={missing_here[:5]}{'...' if len(missing_here) > 5 else ''}, "
            f"extra_in_{name}={extra_here[:5]}{'...' if len(extra_here) > 5 else ''}"
        )

    aligned: list[AlignedPair] = []
    for key in sorted(reference_keys):
        query, pos_row_id, neg_row_id = key
        correct_by_checkpoint = {name: indices[name][key].correct for name in pair_records}
        aligned.append(
            AlignedPair(
                query=query, pos_row_id=pos_row_id, neg_row_id=neg_row_id,
                correct_by_checkpoint=correct_by_checkpoint,
            )
        )
    return tuple(aligned)
