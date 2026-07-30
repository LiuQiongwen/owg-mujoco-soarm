"""
Unit tests for world_model/multihead_labels.py -- the label-derivation layer for
the multi-head contact/lift/success critic (C.3). Covers the 4 items requested
before formal training starts:

  1. bilateral_contact == weld_triggered (synthetic + data-driven against real scenes.jsonl)
  2. fell_off always False (data-driven)
  3. failure_type_3class distribution has only 3 realized classes (data-driven)
  4. unrealized classes raise UnrealizedFailureTypeError, not silently absorbed (synthetic)

Run: conda run -n tango python -m pytest tests/test_multihead_labels.py -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from world_model.multihead_labels import (
    derive_success, derive_dropped_after_lift, derive_retained_grasp_proxy,
    failure_type_full6, failure_type_3class, UnrealizedFailureTypeError,
    FAILURE_TYPES_REALIZED, FAILURE_TYPES_FUTURE_SCHEMA,
)

_SPLITS = [
    Path("results/risk_gated_vla/counterfactual_train_n40_20260730/scenes.jsonl"),
    Path("results/risk_gated_vla/counterfactual_test_n30_20260730/scenes.jsonl"),
    Path("results/risk_gated_vla/confirmatory_n50_seed300_20260730/scenes.jsonl"),
]


def _all_candidates():
    cands = []
    for path in _SPLITS:
        if not path.exists():
            continue
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            cands.extend(rec["oracle_per_candidate"])
    return cands


def _cand(bilateral_contact, weld_triggered, lifted, fell_off, table_contact=False):
    return {
        "bilateral_contact": bilateral_contact, "weld_triggered": weld_triggered,
        "lifted": lifted, "fell_off": fell_off, "table_contact": table_contact,
        "success": lifted and weld_triggered and not fell_off,
    }


# ── Synthetic unit tests (no data dependency) ───────────────────────────────

def test_derive_success_matches_formula():
    c = _cand(True, True, True, False)
    assert derive_success(c) is True
    c2 = _cand(True, True, True, True)  # fell_off=True overrides
    assert derive_success(c2) is False


def test_derive_retained_grasp_proxy_is_weld_triggered():
    c = _cand(True, True, True, False)
    assert derive_retained_grasp_proxy(c) == c["weld_triggered"]
    c2 = _cand(True, False, False, False)
    assert derive_retained_grasp_proxy(c2) is False


def test_derive_dropped_after_lift():
    assert derive_dropped_after_lift(_cand(True, True, True, True)) is True
    assert derive_dropped_after_lift(_cand(True, True, True, False)) is False
    assert derive_dropped_after_lift(_cand(False, False, False, True)) is False


def test_failure_type_full6_success():
    assert failure_type_full6(_cand(True, True, True, False)) == "success"


def test_failure_type_full6_no_contact():
    assert failure_type_full6(_cand(False, False, False, False)) == "no_contact"


def test_failure_type_full6_contact_no_weld_synthetic():
    # Synthetic-only: not reachable in real collected data (audited separately),
    # but the full-taxonomy function must still classify it correctly if it
    # occurs -- e.g. after a future re-execution with a different weld gate.
    assert failure_type_full6(_cand(True, False, False, False)) == "contact_no_weld"


def test_failure_type_full6_weld_no_lift():
    assert failure_type_full6(_cand(True, True, False, False)) == "weld_no_lift"


def test_failure_type_full6_lifted_then_dropped_synthetic():
    # Synthetic-only: fell_off is confirmed always False in real data (see
    # test_fell_off_always_false_in_real_data), so this class is exercised
    # here only, not by real data.
    assert failure_type_full6(_cand(True, True, True, True)) == "lifted_then_dropped"


def test_failure_type_3class_raises_on_unrealized_class():
    # bilateral_contact=True, weld_triggered=False -> contact_no_weld, which
    # is one of the three classes failure_type_3class() must refuse to emit.
    with pytest.raises(UnrealizedFailureTypeError):
        failure_type_3class(_cand(True, False, False, False))


def test_failure_type_3class_raises_on_lifted_then_dropped():
    with pytest.raises(UnrealizedFailureTypeError):
        failure_type_3class(_cand(True, True, True, True))


def test_failure_type_3class_accepts_realized_classes():
    assert failure_type_3class(_cand(True, True, True, False)) == "success"
    assert failure_type_3class(_cand(False, False, False, False)) == "no_contact"
    assert failure_type_3class(_cand(True, True, False, False)) == "weld_no_lift"


def test_realized_classes_are_subset_of_future_schema():
    assert set(FAILURE_TYPES_REALIZED) <= set(FAILURE_TYPES_FUTURE_SCHEMA)
    assert len(FAILURE_TYPES_FUTURE_SCHEMA) == 6
    assert len(FAILURE_TYPES_REALIZED) == 3


def test_missing_field_raises_keyerror():
    with pytest.raises(KeyError):
        derive_success({"lifted": True})


# ── Data-driven tests against real collected scenes.jsonl ──────────────────

def _skip_if_no_data():
    if not any(p.exists() for p in _SPLITS):
        pytest.skip("no collected scenes.jsonl found -- run the Phase 1/2 harness first")


def test_bilateral_contact_equals_weld_triggered_in_real_data():
    _skip_if_no_data()
    cands = _all_candidates()
    assert len(cands) > 0
    mismatches = [c for c in cands if c["bilateral_contact"] != c["weld_triggered"]]
    assert mismatches == [], (
        f"{len(mismatches)}/{len(cands)} candidates have bilateral_contact != "
        f"weld_triggered -- this contradicts DATA_AUDIT.md's confirmed finding; "
        f"re-run scripts/audit_multihead_labels.py before trusting this test's baseline"
    )


def test_fell_off_always_false_in_real_data():
    _skip_if_no_data()
    cands = _all_candidates()
    n_true = sum(1 for c in cands if c["fell_off"])
    assert n_true == 0, f"{n_true}/{len(cands)} candidates have fell_off=True -- data has changed since DATA_AUDIT.md"


def test_success_field_matches_derived_formula_in_real_data():
    _skip_if_no_data()
    cands = _all_candidates()
    mismatches = [c for c in cands if c["success"] != derive_success(c)]
    assert mismatches == []


def test_failure_type_3class_never_raises_on_real_data():
    _skip_if_no_data()
    cands = _all_candidates()
    counts = {}
    for c in cands:
        ft = failure_type_3class(c)  # must not raise
        counts[ft] = counts.get(ft, 0) + 1
    assert set(counts.keys()) <= set(FAILURE_TYPES_REALIZED)
    assert sum(counts.values()) == len(cands)


def test_no_realized_class_has_fewer_than_5_samples_pooled():
    """Per-split, pooled-across-objects support check (not per-object) --
    matches the granularity DATA_AUDIT.md reports pooled counts at. Per-object
    support for weld_no_lift is much lower (near-zero for cracker/mustard,
    concentrated in drill) -- see DATA_AUDIT.md's per-object breakdown; this
    test intentionally checks the pooled level only, consistent with the
    3-class training scope's own unit of analysis."""
    _skip_if_no_data()
    for path in _SPLITS:
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        cands = [c for r in rows for c in r["oracle_per_candidate"]]
        counts = {}
        for c in cands:
            ft = failure_type_3class(c)
            counts[ft] = counts.get(ft, 0) + 1
        for cls in FAILURE_TYPES_REALIZED:
            assert counts.get(cls, 0) >= 5, f"{path}: class {cls!r} has only {counts.get(cls,0)} samples"
