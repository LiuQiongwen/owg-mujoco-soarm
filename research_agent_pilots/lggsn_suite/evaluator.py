#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone LGGSN checkpoint evaluator.

Runs OUTSIDE MVP4 / research_agent's restricted execution -- this is a
normal, unrestricted subprocess. torch cannot fit inside MVP4's fixed 1GB
RLIMIT_AS (verified directly: a bare `import torch` alone fails to mmap
libtorch_cpu.so under rlimits matching research_agent/restricted_subprocess
.py's DEFAULT_LIMITS exactly). research_agent is not modified by this
suite. See docs/LGGSN_EVAL_SUITE.md for the full architecture rationale.

TANGO's role for this suite is downstream of this script: validating the
experiment matrix/specification, and afterward verifying and reporting on
the artifacts this script already wrote. Every manifest this script
produces stamps `"execution_stage": "EXTERNAL_UNRESTRICTED_SUBPROCESS"` --
never claim, here or in any report built from this script's output, that
this computation ran through MVP4's restricted subprocess runner. It did
not, and structurally cannot (see above).

Imports ONLY lggsn_model.LGGSN for the actual model/algorithm. Never
imports train_lggsn_pairwise.py (whose causal-validity audit gate currently
fails due to an unrelated registry bug -- see
docs/CAUSAL_VALIDITY_REGISTRY_BUG.md; this script does not interact with,
bypass, or attempt to fix that gate at all) and never imports
causal_validity_audit itself. Grouping/pairing logic is a clean-room
reimplementation of documented, hand-verified semantics (eval_core.py),
not an import of any training script.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("MALLOC_ARENA_MAX", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT_DEFAULT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_THIS_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import checkpoint_registry as reg  # noqa: E402
import eval_core as ec  # noqa: E402

_PROTECTED_OUTPUT_PREFIXES = ("results/", "checkpoints/", "data/", "datasets/", "paperA_data/")
_PROTECTED_OUTPUT_BASENAME_PREFIX = "paper"


class OutputPathPolicyError(ValueError):
    pass


def _assert_output_dir_allowed(output_dir: str, repo_root: str) -> None:
    """Refuses to write into results/, checkpoints/, data/, datasets/, or
    any paper* directory -- mirrors research_agent.policies
    .execution_policy's HIGH_RISK_PREFIXES intent, reimplemented here
    (pure stdlib, no import of research_agent) since this script runs
    entirely outside that package."""
    abs_out = os.path.abspath(output_dir)
    abs_repo = os.path.abspath(repo_root)
    try:
        rel = os.path.relpath(abs_out, abs_repo)
    except ValueError:
        rel = None
    if rel is not None and not rel.startswith(".."):
        norm = rel.replace(os.sep, "/") + "/"
        for prefix in _PROTECTED_OUTPUT_PREFIXES:
            if norm.startswith(prefix):
                raise OutputPathPolicyError(f"refusing to write inside protected path {prefix!r}: {output_dir}")
        first_component = norm.split("/", 1)[0]
        if first_component.lower().startswith(_PROTECTED_OUTPUT_BASENAME_PREFIX):
            raise OutputPathPolicyError(f"refusing to write inside a paper* path: {output_dir}")


def _current_git_commit(repo_root: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip()
    except Exception as e:  # pragma: no cover - environment-dependent
        return f"<unavailable: {e}>"


def describe_blocked(name: str) -> dict:
    """Structured record for a checkpoint this evaluator will never score
    -- used so blocked entries stay visible in the matrix output instead
    of being silently omitted."""
    entry = reg.CHECKPOINTS_BY_NAME[name]
    return {
        "checkpoint_name": name,
        "status": "BLOCKED",
        "provenance_status": entry.provenance_status.value,
        "relative_path": entry.relative_path,
        "reason": "; ".join(entry.caveats) if entry.caveats else "no reason recorded",
    }


def evaluate_checkpoint(name: str, *, repo_root: str, output_dir: str) -> dict:
    entry = reg.CHECKPOINTS_BY_NAME.get(name)
    if entry is None:
        raise KeyError(f"unknown checkpoint name {name!r}; known: {sorted(reg.CHECKPOINTS_BY_NAME)}")
    if entry.provenance_status not in (reg.ProvenanceStatus.VERIFIED, reg.ProvenanceStatus.PROVENANCE_INCOMPLETE):
        raise ec.SchemaVerificationError(
            f"[{name}] provenance_status={entry.provenance_status.value} is not evaluatable by "
            f"this evaluator -- caveats: {list(entry.caveats)}"
        )
    if entry.arch is None:
        raise ec.SchemaVerificationError(f"[{name}] registry entry has no architecture parameters")

    _assert_output_dir_allowed(output_dir, repo_root)

    ckpt_path = os.path.join(repo_root, entry.relative_path)
    dataset_path = os.path.join(repo_root, entry.dataset_identifier)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[{name}] checkpoint not found: {ckpt_path}")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"[{name}] dataset not found: {dataset_path}")

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    actual_ckpt_sha = ec.file_sha256(ckpt_path)
    ec.verify_checkpoint_identity(expected_sha256=entry.checkpoint_sha256, actual_sha256=actual_ckpt_sha, name=name)

    actual_dataset_sha = ec.file_sha256(dataset_path)
    if actual_dataset_sha != reg.DATASET_SHA256:
        raise ec.SchemaVerificationError(
            f"[{name}] dataset SHA-256 mismatch: registry expects {reg.DATASET_SHA256}, "
            f"got {actual_dataset_sha} for {dataset_path}"
        )

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import torch
    torch.set_num_threads(1)
    from lggsn_model import LGGSN

    raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_shapes = {k: tuple(v.shape) for k, v in raw_state.items() if hasattr(v, "shape")}
    ec.verify_state_dict_schema(
        state_dict_shapes=state_shapes, expected_input_dim=entry.arch.geom_dim,
        expected_query_dim=entry.arch.query_dim, expected_hidden_dim=entry.arch.hidden_dim, name=name,
    )

    rows = ec.read_jsonl_rows(dataset_path)
    episodes = ec.group_episodes(rows)
    needed_derived = set(entry.feature_columns) & set(ec.DERIVABLE_FEATURES)
    episodes = {key: ec.compute_derived_features(ep_rows, needed_derived) for key, ep_rows in episodes.items()}
    ec.validate_feature_columns(episodes, entry.feature_columns)
    by_query, malformed_group_count, skipped_row_count = ec.label_episodes(episodes)
    train_pairs, val_pairs = ec.build_pairs(by_query, val_frac=0.2, seed=42)

    model = LGGSN(
        n_queries=entry.arch.n_queries, geom_dim=entry.arch.geom_dim,
        query_dim=entry.arch.query_dim, hidden_dim=entry.arch.hidden_dim, dropout=entry.arch.dropout,
    )
    model.load_state_dict(raw_state)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    all_rows = sorted((r for ep_rows in episodes.values() for r in ep_rows), key=lambda r: r["_row_id"])
    feat_matrix = [[float(r[c]) for c in entry.feature_columns] for r in all_rows]
    row_ids = [r["_row_id"] for r in all_rows]
    with torch.no_grad():
        geom = torch.tensor(feat_matrix, dtype=torch.float32)
        q_id = torch.zeros(len(all_rows), dtype=torch.long)
        logits = model(geom, q_id).view(-1)
        scores_t = torch.sigmoid(logits)
    scores = {rid: float(s) for rid, s in zip(row_ids, scores_t.tolist())}

    bootstrap_ci = ec.bootstrap_pair_accuracy_ci(by_query, scores, val_frac=0.2, seed=42)

    runtime_seconds = round(time.monotonic() - t0, 4)
    metrics = ec.build_metrics(
        checkpoint_name=name, checkpoint_sha256=actual_ckpt_sha,
        provenance_status=entry.provenance_status.value,
        feature_columns=entry.feature_columns, input_dim=entry.input_dim,
        dataset_sha256=actual_dataset_sha, seed=42, val_frac=0.2,
        by_query=by_query, train_pairs=train_pairs, val_pairs=val_pairs, scores=scores,
        malformed_group_count=malformed_group_count, skipped_row_count=skipped_row_count,
        runtime_seconds=runtime_seconds, bootstrap_ci=bootstrap_ci,
    )
    ended_at = datetime.now(timezone.utc).isoformat()

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    with open(os.path.join(output_dir, "pair_results.jsonl"), "w") as f:
        for pos_id, neg_id, query in val_pairs:
            f.write(json.dumps({
                "query": query, "pos_row_id": pos_id, "neg_row_id": neg_id,
                "pos_score": scores[pos_id], "neg_score": scores[neg_id],
                "correct": scores[pos_id] > scores[neg_id],
            }) + "\n")

    checkpoint_manifest = {
        "checkpoint_name": name,
        "relative_path": entry.relative_path,
        "checkpoint_sha256": actual_ckpt_sha,
        "provenance_status": entry.provenance_status.value,
        "source_commit": entry.source_commit,
        "producing_script": entry.producing_script,
        "producing_script_blob_sha": entry.producing_script_blob_sha,
        "feature_columns": list(entry.feature_columns),
        "input_dim": entry.input_dim,
        "architecture": {
            "n_queries": entry.arch.n_queries, "geom_dim": entry.arch.geom_dim,
            "query_dim": entry.arch.query_dim, "hidden_dim": entry.arch.hidden_dim,
            "dropout": entry.arch.dropout,
        },
        "caveats": list(entry.caveats),
    }
    with open(os.path.join(output_dir, "checkpoint_manifest.json"), "w") as f:
        json.dump(checkpoint_manifest, f, indent=2)
        f.write("\n")

    evaluation_manifest = {
        "checkpoint_name": name,
        "dataset_relative_path": entry.dataset_identifier,
        "dataset_sha256": actual_dataset_sha,
        "seed": 42,
        "val_frac": 0.2,
        "execution_stage": "EXTERNAL_UNRESTRICTED_SUBPROCESS",
        "execution_stage_note": (
            "This computation ran as a normal OS subprocess, outside MVP4's "
            "restricted execution (research_agent.restricted_subprocess). "
            "torch cannot fit inside that harness's fixed RLIMIT_AS=1GB. "
            "TANGO's role for this checkpoint is limited to validating the "
            "spec and verifying/reporting on the artifacts already written "
            "here -- see docs/LGGSN_EVAL_SUITE.md."
        ),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": "cpu",
        "gpu_used": False,
        "network_used": False,
        "training_performed": False,
        "platform": platform.platform(),
        "command": list(sys.argv),
        "git_commit": _current_git_commit(repo_root),
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": runtime_seconds,
    }
    with open(os.path.join(output_dir, "evaluation_manifest.json"), "w") as f:
        json.dump(evaluation_manifest, f, indent=2)
        f.write("\n")

    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_name", choices=sorted(reg.CHECKPOINTS_BY_NAME))
    parser.add_argument("--repo-root", default=_REPO_ROOT_DEFAULT)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    entry = reg.CHECKPOINTS_BY_NAME[args.checkpoint_name]
    if entry.provenance_status in (reg.ProvenanceStatus.BLOCKED_SCHEMA_MISMATCH, reg.ProvenanceStatus.BLOCKED_MISSING_CHECKPOINT):
        record = describe_blocked(args.checkpoint_name)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "blocked.json"), "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        print(f"[lggsn_eval:{args.checkpoint_name}] BLOCKED: {record['reason']}")
        return 3

    metrics = evaluate_checkpoint(args.checkpoint_name, repo_root=args.repo_root, output_dir=args.output_dir)
    print(
        f"[lggsn_eval:{args.checkpoint_name}] eligible_groups={metrics['eligible_group_count']} "
        f"eligible_pairs={metrics['eligible_pair_count']} pair_accuracy={metrics['pair_accuracy']} "
        f"ties={metrics['ties_count']} runtime_seconds={metrics['runtime_seconds']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
