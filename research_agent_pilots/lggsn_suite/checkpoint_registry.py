# -*- coding: utf-8 -*-
"""Explicit, immutable checkpoint registry for the LGGSN standalone
evaluator. No torch dependency (pure stdlib) -- unit-testable anywhere.

Every entry here was hand-verified against git history (see
docs/LGGSN_EVAL_SUITE.md and the provenance investigation report), never
inferred from a checkpoint's tensor shape alone. The evaluator's own
dimension probe (reading `mlp.0.weight.shape[1]` from the loaded
state_dict) exists ONLY to cross-check against this registry's declared
`input_dim` and fail closed on a mismatch -- it is never used as the sole
source of truth for feature order, which a dimension number alone cannot
determine (13-dim is ambiguous between "base+z_rel" and "base+dist_to_centroid").

FEATURE_COLS values are transcribed by hand from train_lggsn_pairwise.py's
own construction (that file is never imported by this evaluator -- see
LGGSN_EVAL_SUITE.md's "why never import train_lggsn_pairwise.py" section).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ProvenanceStatus(Enum):
    VERIFIED = "EVALUATABLE_WITH_VERIFIED_PROVENANCE"
    PROVENANCE_INCOMPLETE = "EVALUATABLE_BUT_PROVENANCE_INCOMPLETE"
    BLOCKED_SCHEMA_MISMATCH = "BLOCKED_BY_MODEL_SCHEMA_MISMATCH"
    BLOCKED_MISSING_CHECKPOINT = "BLOCKED_BY_MISSING_CHECKPOINT_FILE"


# Base 12 columns, present in every LGGSN geometry-feature checkpoint this
# registry knows about -- transcribed from train_lggsn_pairwise.py:41-44
# (and identical in tango_robot/grasp_ranker_lggsn.py's FEATURE_COLS_BASE).
_BASE_12 = (
    "x", "y", "z",
    "roll", "pitch", "yaw",
    "width", "score",
    "dz", "dz_lift", "need_dz", "H",
)


@dataclass(frozen=True)
class ArchParams:
    n_queries: int
    geom_dim: int
    query_dim: int
    hidden_dim: int
    dropout: float = 0.0


@dataclass(frozen=True)
class CheckpointEntry:
    name: str
    relative_path: str  # repo-root-relative, e.g. "grasp_6dof/models/lggsn_ablation_base.pt"
    provenance_status: ProvenanceStatus
    source_commit: Optional[str]  # git commit that introduced this checkpoint file
    producing_script: Optional[str]  # repo-relative path, as it existed at source_commit
    producing_script_blob_sha: Optional[str]  # `git rev-parse <commit>:<path>` -- exact content reference
    feature_columns: Tuple[str, ...]
    input_dim: int
    arch: Optional[ArchParams]
    dataset_identifier: str  # what this evaluator scores it against (NOT necessarily what it was trained on)
    checkpoint_sha256: Optional[str]  # None only for BLOCKED_MISSING_CHECKPOINT entries
    caveats: Tuple[str, ...] = field(default_factory=tuple)


DATASET_IDENTIFIER = "snapshots/lggsn_live_dataset_v1/lggsn_live_candidates.jsonl"
# Pinned once, from `sha256sum snapshots/lggsn_live_dataset_v1/lggsn_live_candidates.jsonl`.
# The evaluator recomputes this at runtime and fails closed on any mismatch.
DATASET_SHA256 = "30ca2c398de4bc451691ae6e802e619e7838e0b8bce12a4eba8115b6ff4c42b5"

_BULK_IMPORT_COMMIT = "2dc82b3ff2cc9554ccec5ff0bc89be5ea1ac9f01"
_PRODUCING_SCRIPT = "train_lggsn_pairwise.py"
# `git rev-parse 2dc82b3:train_lggsn_pairwise.py` -- the file's content AS
# BULK-IMPORTED, before the later audit_feature_set gate was added
# (e2bb7fb). Confirmed via `git diff 2dc82b3 HEAD -- train_lggsn_pairwise.py`
# that the ONLY change since is that gate's addition -- FEATURE_COLS
# construction is byte-identical to what shipped with these checkpoints.
_PRODUCING_SCRIPT_BLOB_AT_IMPORT = "4cde474234f828ae7889392fb9adcb9a2524d3e4"

_VERIFIED_CAVEATS = (
    "No in-checkpoint metadata exists (bare state_dict, no wrapper) -- "
    "provenance is established via git commit co-location with "
    "train_lggsn_pairwise.py in the same bulk-import commit, plus an exact "
    "input_dim match, plus a byte-identical diff of that script's "
    "FEATURE_COLS-construction code between the import commit and HEAD.",
    "Evaluated against snapshots/lggsn_live_dataset_v1/lggsn_live_candidates.jsonl, "
    "which is NOT proven to be the same dataset this checkpoint was originally "
    "trained/validated on (the original per-version training JSONL files are "
    "gitignored and absent from this worktree) -- these are offline metrics on "
    "a fixed, real, held-out-by-episode split of a real dataset, not a "
    "reproduction of any previously-reported number.",
)

CHECKPOINTS = (
    CheckpointEntry(
        name="base",
        relative_path="grasp_6dof/models/lggsn_ablation_base.pt",
        provenance_status=ProvenanceStatus.VERIFIED,
        source_commit=_BULK_IMPORT_COMMIT,
        producing_script=_PRODUCING_SCRIPT,
        producing_script_blob_sha=_PRODUCING_SCRIPT_BLOB_AT_IMPORT,
        feature_columns=_BASE_12,
        input_dim=12,
        arch=ArchParams(n_queries=1, geom_dim=12, query_dim=0, hidden_dim=40, dropout=0.0),
        dataset_identifier=DATASET_IDENTIFIER,
        checkpoint_sha256="333f792615547a293b6e7a33fe23b9ecc1366ae7406e0552f53479a719d4d499",
        caveats=_VERIFIED_CAVEATS + (
            "FEAT_DIST=0, FEAT_ZREL=0 implied by dim=12 -- neither dist_to_centroid nor z_rel present.",
        ),
    ),
    CheckpointEntry(
        name="nodist",
        relative_path="grasp_6dof/models/lggsn_ablation_nodist.pt",
        provenance_status=ProvenanceStatus.VERIFIED,
        source_commit=_BULK_IMPORT_COMMIT,
        producing_script=_PRODUCING_SCRIPT,
        producing_script_blob_sha=_PRODUCING_SCRIPT_BLOB_AT_IMPORT,
        feature_columns=_BASE_12 + ("z_rel",),
        input_dim=13,
        arch=ArchParams(n_queries=1, geom_dim=13, query_dim=0, hidden_dim=40, dropout=0.0),
        dataset_identifier=DATASET_IDENTIFIER,
        checkpoint_sha256="f85bbf0daa7c2bd0d002f7d8f4c48ce4ad2429d005e10fa62f8ba38ccc9bc3d7",
        caveats=_VERIFIED_CAVEATS + (
            "FEAT_DIST=0, FEAT_ZREL=1 implied by dim=13 + filename ('nodist' = "
            "dist_to_centroid removed, z_rel kept). 13-dim alone is ambiguous "
            "with 'nozrel' below -- disambiguated by filename, NOT by dimension.",
        ),
    ),
    CheckpointEntry(
        name="nozrel",
        relative_path="grasp_6dof/models/lggsn_ablation_nozrel.pt",
        provenance_status=ProvenanceStatus.VERIFIED,
        source_commit=_BULK_IMPORT_COMMIT,
        producing_script=_PRODUCING_SCRIPT,
        producing_script_blob_sha=_PRODUCING_SCRIPT_BLOB_AT_IMPORT,
        feature_columns=_BASE_12 + ("dist_to_centroid",),
        input_dim=13,
        arch=ArchParams(n_queries=1, geom_dim=13, query_dim=0, hidden_dim=40, dropout=0.0),
        dataset_identifier=DATASET_IDENTIFIER,
        checkpoint_sha256="9ebf71758a63646d152387d483265389b31e137b52d9b488ac9805725b24d1dc",
        caveats=_VERIFIED_CAVEATS + (
            "FEAT_DIST=1, FEAT_ZREL=0 implied by dim=13 + filename ('nozrel' = "
            "z_rel removed, dist_to_centroid kept). See 'nodist' note on "
            "dimension ambiguity.",
        ),
    ),
    CheckpointEntry(
        name="full_v2",
        relative_path="grasp_6dof/models/lggsn_pairwise_live_v2.pt",
        provenance_status=ProvenanceStatus.VERIFIED,
        source_commit=_BULK_IMPORT_COMMIT,
        producing_script=_PRODUCING_SCRIPT,
        producing_script_blob_sha=_PRODUCING_SCRIPT_BLOB_AT_IMPORT,
        feature_columns=_BASE_12 + ("dist_to_centroid", "z_rel"),
        input_dim=14,
        arch=ArchParams(n_queries=1, geom_dim=14, query_dim=0, hidden_dim=40, dropout=0.0),
        dataset_identifier=DATASET_IDENTIFIER,
        checkpoint_sha256="b94e073b7d0c36394e25fb93e708176c64ee33b0bbf00a4d0bbfaeb07888963f",
        caveats=_VERIFIED_CAVEATS + (
            "FEAT_DIST=1, FEAT_ZREL=1 -- matches train_lggsn_pairwise.py's OWN "
            "environment-variable defaults exactly (no non-default toggle "
            "assumed), the strongest provenance match of the four verified "
            "checkpoints. Documented in CLAUDE.md as 'active -- use this'.",
        ),
    ),
    CheckpointEntry(
        name="v1_live",
        relative_path="grasp_6dof/models/lggsn_pairwise_live.pt",
        provenance_status=ProvenanceStatus.PROVENANCE_INCOMPLETE,
        source_commit=_BULK_IMPORT_COMMIT,
        producing_script=_PRODUCING_SCRIPT,
        producing_script_blob_sha=_PRODUCING_SCRIPT_BLOB_AT_IMPORT,
        feature_columns=_BASE_12,
        input_dim=12,
        arch=ArchParams(n_queries=1, geom_dim=12, query_dim=0, hidden_dim=40, dropout=0.0),
        dataset_identifier=DATASET_IDENTIFIER,
        checkpoint_sha256="200e3b5a50ea430cf2d511c962b5d95ed1c1f6bc71121ab6f556d16d4a40f958",
        caveats=(
            "Same dim (12) as 'base', assumed same FEAT_DIST=0/FEAT_ZREL=0 "
            "feature set, but NOT independently corroborated the way 'base' "
            "is (no distinct ablation-study filename signal). CLAUDE.md flags "
            "this exact checkpoint: 'v1 (inflated accuracy, noisy negatives)' "
            "-- known data-quality caveat, kept for cross-reference only.",
        ),
    ),
    CheckpointEntry(
        name="v2_phase1",
        relative_path="grasp_6dof/models/lggsn_v2_phase1.pt",
        provenance_status=ProvenanceStatus.PROVENANCE_INCOMPLETE,
        source_commit=_BULK_IMPORT_COMMIT,
        producing_script=_PRODUCING_SCRIPT,
        producing_script_blob_sha=_PRODUCING_SCRIPT_BLOB_AT_IMPORT,
        feature_columns=_BASE_12 + ("dist_to_centroid", "z_rel"),
        input_dim=14,
        arch=ArchParams(n_queries=1, geom_dim=14, query_dim=0, hidden_dim=40, dropout=0.0),
        dataset_identifier=DATASET_IDENTIFIER,
        checkpoint_sha256="813c0c8ffe0f9e82917683c9d316abe6258295a93e386689ab82393110f05c7d",
        caveats=(
            "Same dim (14) as 'full_v2'; exact relationship to that checkpoint "
            "(earlier snapshot of the same run? separate run?) is not "
            "documented anywhere found in this repository.",
        ),
    ),
    # ── BLOCKED: different architecture (query embedding + point-cloud
    # features this dataset's schema does not contain) -- not evaluatable
    # by this evaluator regardless of checkpoint availability. ─────────────
    CheckpointEntry(
        name="ext_v5",
        relative_path="grasp_6dof/models/lggsn_pairwise_live_v5.pt",
        provenance_status=ProvenanceStatus.BLOCKED_SCHEMA_MISMATCH,
        source_commit=_BULK_IMPORT_COMMIT,
        producing_script="train_lggsn_v5.py",
        producing_script_blob_sha=None,
        feature_columns=_BASE_12 + ("dist_to_centroid", "z_rel", "local_point_density", "normal_consistency", "contact_width_ratio"),
        input_dim=17,
        arch=ArchParams(n_queries=7, geom_dim=17, query_dim=4, hidden_dim=40, dropout=0.0),
        dataset_identifier=DATASET_IDENTIFIER,
        checkpoint_sha256="beeaae41536219ed328eb167ef91c6a03917ac8f959ef50127fc13efb326e7b3",
        caveats=(
            "Needs local_point_density/normal_consistency/contact_width_ratio "
            "(per-candidate point-cloud features), which "
            "snapshots/lggsn_live_dataset_v1/lggsn_live_candidates.jsonl's "
            "schema does not contain and this evaluator does not compute. "
            "Also uses a query-embedding architecture (n_queries=7, "
            "query_dim=4) this evaluator does not implement.",
        ),
    ),
    CheckpointEntry(
        name="ext_v5d",
        relative_path="grasp_6dof/models/lggsn_pairwise_live_v5d.pt",
        provenance_status=ProvenanceStatus.BLOCKED_SCHEMA_MISMATCH,
        source_commit=_BULK_IMPORT_COMMIT,
        producing_script="train_lggsn_v5.py (v5d variant)",
        producing_script_blob_sha=None,
        feature_columns=_BASE_12 + ("dist_to_centroid", "z_rel", "local_point_density", "normal_consistency", "contact_width_ratio"),
        input_dim=17,
        arch=ArchParams(n_queries=7, geom_dim=17, query_dim=4, hidden_dim=40, dropout=0.0),
        dataset_identifier=DATASET_IDENTIFIER,
        checkpoint_sha256="5b75fa4bafde966c3ff8670c48920e1c1e0990938c7390ba50956ca7dfafcf7b",
        caveats=(
            "Same blocker as ext_v5 (point-cloud features + query-embedding "
            "architecture). CLAUDE.md/commit messages cite this checkpoint's "
            "historically-reported val pair_acc as ~0.750 -- NOT reproduced "
            "here; this evaluator cannot score it at all.",
        ),
    ),
    CheckpointEntry(
        name="ext2_v6",
        relative_path="grasp_6dof/models/lggsn_pairwise_live_v6.pt",
        provenance_status=ProvenanceStatus.BLOCKED_SCHEMA_MISMATCH,
        source_commit=_BULK_IMPORT_COMMIT,
        producing_script="train_lggsn_v6.py",
        producing_script_blob_sha=None,
        feature_columns=_BASE_12 + ("dist_to_centroid", "z_rel", "local_point_density", "normal_consistency", "contact_width_ratio", "pe_ik"),
        input_dim=18,
        arch=ArchParams(n_queries=7, geom_dim=18, query_dim=4, hidden_dim=40, dropout=0.0),
        dataset_identifier=DATASET_IDENTIFIER,
        checkpoint_sha256="b5f4c332dfb390b87e545a4292a862fdced5d69a31e26c1e155df77a4ccb7e40",
        caveats=(
            "Same blockers as ext_v5/ext_v5d, plus needs pe_ik (episode-level "
            "IK error), also absent from this dataset's schema.",
        ),
    ),
    # ── BLOCKED: checkpoint file does not exist in this worktree at all. ───
    CheckpointEntry(
        name="v10_baseline",
        relative_path="grasp_6dof/models/lggsn_pairwise_live_v10.pt",
        provenance_status=ProvenanceStatus.BLOCKED_MISSING_CHECKPOINT,
        source_commit=None,
        producing_script="train_lggsn_v10.py",
        producing_script_blob_sha=None,
        feature_columns=_BASE_12 + ("dist_to_centroid", "z_rel", "local_point_density", "normal_consistency", "contact_width_ratio", "pe_ik"),
        input_dim=18,
        arch=None,
        dataset_identifier="grasp_6dof/dataset/lggsn_candidates_v7.jsonl (also absent from this worktree)",
        checkpoint_sha256=None,
        caveats=(
            "train_lggsn_v10.py exists and is committed, but its output "
            "checkpoint was never committed (grasp_6dof/models/ and "
            "grasp_6dof/dataset/ per-version JSONL files are both "
            "gitignored) and is not present on this machine.",
        ),
    ),
    CheckpointEntry(
        name="v11_ik_full",
        relative_path="grasp_6dof/models/lggsn_pairwise_live_v11_ik.pt",
        provenance_status=ProvenanceStatus.BLOCKED_MISSING_CHECKPOINT,
        source_commit=None,
        producing_script="train_lggsn_v11_ik.py",
        producing_script_blob_sha=None,
        feature_columns=_BASE_12 + (
            "dist_to_centroid", "z_rel", "local_point_density", "normal_consistency",
            "contact_width_ratio", "ik_converged", "ik_residual", "max_joint_delta",
        ),
        input_dim=20,
        arch=None,
        dataset_identifier="grasp_6dof/dataset/lggsn_candidates_v11_ik.jsonl (also absent from this worktree)",
        checkpoint_sha256=None,
        caveats=(
            "Same as v10_baseline: script committed, checkpoint and dataset "
            "were never committed (commit message for the Tier-3 IK feature "
            "explicitly notes the 1250-row v11_ik dataset was 'gitignored, "
            "not committed'). This is the checkpoint that would correspond "
            "to 'v11 + IK feature' / 'full model, all features' in an "
            "aspirational reading of the matrix -- it does not exist yet.",
        ),
    ),
)

CHECKPOINTS_BY_NAME = {c.name: c for c in CHECKPOINTS}

VERIFIED_NAMES = tuple(c.name for c in CHECKPOINTS if c.provenance_status == ProvenanceStatus.VERIFIED)
PROVENANCE_INCOMPLETE_NAMES = tuple(
    c.name for c in CHECKPOINTS if c.provenance_status == ProvenanceStatus.PROVENANCE_INCOMPLETE
)
BLOCKED_NAMES = tuple(
    c.name for c in CHECKPOINTS
    if c.provenance_status in (ProvenanceStatus.BLOCKED_SCHEMA_MISMATCH, ProvenanceStatus.BLOCKED_MISSING_CHECKPOINT)
)

# The matrix this suite actually runs (task scope: base/nodist/nozrel/full).
MATRIX_NAMES = ("base", "nodist", "nozrel", "full_v2")
