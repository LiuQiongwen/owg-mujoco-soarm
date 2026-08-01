# -*- coding: utf-8 -*-
"""Tests for research_agent_pilots/lggsn_suite/checkpoint_registry.py.
Pure stdlib (dataclasses/enum) -- no torch, runs under the research-agent
venv.

Covers requirement list items:
  1.  exact feature order for every supported checkpoint
  12. base/nodist/nozrel/full matrix construction
  13. blocked v5/v5d/v6 entries remain visible
  14. missing v10/v11 entries remain visible
  15. provenance-incomplete checkpoints are labeled, not treated as verified
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research_agent_pilots", "lggsn_suite"))
import checkpoint_registry as cr  # noqa: E402

_BASE_12 = (
    "x", "y", "z", "roll", "pitch", "yaw", "width", "score", "dz", "dz_lift", "need_dz", "H",
)


# ── 1. exact feature order per checkpoint ───────────────────────────────────

def test_base_feature_order():
    assert cr.CHECKPOINTS_BY_NAME["base"].feature_columns == _BASE_12


def test_nodist_feature_order():
    assert cr.CHECKPOINTS_BY_NAME["nodist"].feature_columns == _BASE_12 + ("z_rel",)


def test_nozrel_feature_order():
    assert cr.CHECKPOINTS_BY_NAME["nozrel"].feature_columns == _BASE_12 + ("dist_to_centroid",)


def test_full_v2_feature_order():
    assert cr.CHECKPOINTS_BY_NAME["full_v2"].feature_columns == _BASE_12 + ("dist_to_centroid", "z_rel")


def test_every_checkpoint_declares_a_nonempty_feature_order():
    for entry in cr.CHECKPOINTS:
        assert len(entry.feature_columns) >= 1, entry.name
        assert len(entry.feature_columns) == entry.input_dim, (
            f"{entry.name}: len(feature_columns)={len(entry.feature_columns)} != input_dim={entry.input_dim}"
        )


# ── 6. dimension alone is ambiguous -- registry disambiguates by name ──────

def test_dimension_alone_is_ambiguous_between_nodist_and_nozrel():
    nodist = cr.CHECKPOINTS_BY_NAME["nodist"]
    nozrel = cr.CHECKPOINTS_BY_NAME["nozrel"]
    assert nodist.input_dim == nozrel.input_dim == 13
    assert nodist.feature_columns != nozrel.feature_columns
    assert "z_rel" in nodist.feature_columns and "dist_to_centroid" not in nodist.feature_columns
    assert "dist_to_centroid" in nozrel.feature_columns and "z_rel" not in nozrel.feature_columns


# ── 12. matrix construction ─────────────────────────────────────────────────

def test_core_matrix_is_exactly_base_nodist_nozrel_full():
    assert cr.MATRIX_NAMES == ("base", "nodist", "nozrel", "full_v2")


def test_matrix_names_are_all_verified():
    for name in cr.MATRIX_NAMES:
        assert cr.CHECKPOINTS_BY_NAME[name].provenance_status == cr.ProvenanceStatus.VERIFIED


# ── 13/14. blocked entries remain visible, with reasons ────────────────────

def test_v5_v5d_v6_are_registered_and_blocked():
    for name in ("ext_v5", "ext_v5d", "ext2_v6"):
        entry = cr.CHECKPOINTS_BY_NAME[name]
        assert entry.provenance_status == cr.ProvenanceStatus.BLOCKED_SCHEMA_MISMATCH
        assert entry.caveats, f"{name} must record a reason, not just a bare BLOCKED status"


def test_v10_v11_are_registered_and_blocked_missing():
    for name in ("v10_baseline", "v11_ik_full"):
        entry = cr.CHECKPOINTS_BY_NAME[name]
        assert entry.provenance_status == cr.ProvenanceStatus.BLOCKED_MISSING_CHECKPOINT
        assert entry.checkpoint_sha256 is None
        assert entry.caveats


def test_blocked_names_are_not_silently_omitted_from_the_registry():
    expected_blocked = {"ext_v5", "ext_v5d", "ext2_v6", "v10_baseline", "v11_ik_full"}
    assert expected_blocked.issubset(set(cr.BLOCKED_NAMES))
    assert expected_blocked.issubset(set(cr.CHECKPOINTS_BY_NAME))


# ── 15. provenance-incomplete != verified ───────────────────────────────────

def test_provenance_incomplete_checkpoints_are_not_verified():
    for name in ("v1_live", "v2_phase1"):
        entry = cr.CHECKPOINTS_BY_NAME[name]
        assert entry.provenance_status == cr.ProvenanceStatus.PROVENANCE_INCOMPLETE
        assert name not in cr.VERIFIED_NAMES
        assert name not in cr.MATRIX_NAMES


def test_provenance_incomplete_names_disjoint_from_verified_names():
    assert set(cr.PROVENANCE_INCOMPLETE_NAMES).isdisjoint(set(cr.VERIFIED_NAMES))


# ── registry integrity / immutability ───────────────────────────────────────

def test_registry_entries_are_frozen_dataclasses():
    import dataclasses
    entry = cr.CHECKPOINTS_BY_NAME["base"]
    assert dataclasses.is_dataclass(entry)
    try:
        entry.name = "tampered"
        assert False, "CheckpointEntry must be frozen (immutable)"
    except dataclasses.FrozenInstanceError:
        pass


def test_every_checkpoint_name_is_unique():
    names = [c.name for c in cr.CHECKPOINTS]
    assert len(names) == len(set(names))


def test_verified_checkpoints_have_pinned_sha256_and_arch():
    for name in cr.VERIFIED_NAMES:
        entry = cr.CHECKPOINTS_BY_NAME[name]
        assert entry.checkpoint_sha256 and len(entry.checkpoint_sha256) == 64
        assert entry.arch is not None
        assert entry.source_commit is not None
        assert entry.producing_script_blob_sha is not None
