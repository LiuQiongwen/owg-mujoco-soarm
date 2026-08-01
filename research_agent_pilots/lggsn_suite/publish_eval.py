# -*- coding: utf-8 -*-
"""TANGO-side publish/verify step -- torch-free, runs THROUGH MVP4's
restricted execution (unlike evaluator.py, which cannot -- see
docs/LGGSN_EVAL_SUITE.md).

This script does not run the LGGSN model and does not perform the
evaluation. It reads an already-computed evaluator.py output (produced
externally, as a normal unrestricted subprocess) from this repository's own
committed research_agent_pilots/lggsn_suite/eval_outputs/<name>/ directory,
verifies its `deterministic_digest` field matches the exact value this spec
pins (computed once, from the real run, over every field except wall-clock
duration -- see eval_core.build_metrics), and copies it into this run's
assigned MVP4 artifacts directory. If the pinned digest does not match --
because the checkpoint, dataset, or evaluator code changed since the
pinned run -- this fails loudly rather than publishing a stale or
tampered-with result.

Only imports eval_core (pure stdlib, no torch) for the digest recomputation
helper -- never touches torch, never touches train_lggsn_pairwise.py, never
touches causal_validity_audit.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

# Must be set before importing eval_core -- otherwise Python writes a .pyc
# into research_agent_pilots/lggsn_suite/__pycache__/, which the MVP4
# harness's mutation detection correctly flags as a POLICY_FAILURE ("main
# worktree mutated during execution") even though __pycache__/ is
# gitignored. Same fix, same root cause, as run_pilot.py's earlier
# sys.dont_write_bytecode -- see experiments/lggsn_suite/*.yaml's
# environment_overrides for the second, launch-time layer
# (PYTHONDONTWRITEBYTECODE=1).
sys.dont_write_bytecode = True

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import eval_core as ec  # noqa: E402  (path setup must run first)

_PUBLISHED_FILES = ("metrics.json", "checkpoint_manifest.json", "evaluation_manifest.json")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable {name!r} is not set")
    return value


def main() -> int:
    artifacts_dir = os.environ.get("RESEARCH_AGENT_ARTIFACTS_DIR")
    if not artifacts_dir:
        raise RuntimeError("RESEARCH_AGENT_ARTIFACTS_DIR is not set")

    checkpoint_name = _required_env("LGGSN_PUBLISH_CHECKPOINT_NAME")
    expected_digest = _required_env("LGGSN_PUBLISH_EXPECTED_DIGEST")

    source_dir = os.path.join(_THIS_DIR, "eval_outputs", checkpoint_name)
    metrics_path = os.path.join(source_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(
            f"[{checkpoint_name}] pre-computed metrics.json not found at {metrics_path} -- "
            f"the standalone evaluator (evaluator.py) must be run first, outside MVP4, to "
            f"produce this file (see docs/LGGSN_EVAL_SUITE.md)"
        )

    with open(metrics_path) as f:
        metrics = json.load(f)

    actual_digest = metrics.get("deterministic_digest")
    if actual_digest != expected_digest:
        raise ValueError(
            f"[{checkpoint_name}] deterministic_digest mismatch: this spec pins "
            f"{expected_digest!r}, but {metrics_path} currently contains "
            f"{actual_digest!r} -- refusing to publish a result that does not match "
            f"the pinned, previously-verified evaluator output"
        )
    if metrics.get("checkpoint_name") != checkpoint_name:
        raise ValueError(
            f"metrics.json checkpoint_name={metrics.get('checkpoint_name')!r} != "
            f"expected {checkpoint_name!r}"
        )

    os.makedirs(artifacts_dir, exist_ok=True)
    published = []
    for fname in _PUBLISHED_FILES:
        src = os.path.join(source_dir, fname)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(artifacts_dir, fname))
            published.append(fname)

    print(
        f"[lggsn_publish:{checkpoint_name}] verified deterministic_digest="
        f"{actual_digest[:16]}... published {published} to {artifacts_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
