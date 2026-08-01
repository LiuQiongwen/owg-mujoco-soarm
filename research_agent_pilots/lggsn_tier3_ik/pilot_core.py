# -*- coding: utf-8 -*-
"""Pure-stdlib validation/aggregation logic for the LGGSN Tier-3 per-candidate
IK reachability pilot (experiments/lggsn_tier3_ik_pilot.yaml).

Deliberately has no dependency on mujoco/numpy/tango_robot -- the only thing
that touches the real IK solver is run_pilot.py's thin adapter loop, which
calls tango_robot.headless_ik.HeadlessIKSolver.solve_ik_jaw_topdown (a
verbatim extraction of EnvironmentSoArm._solve_ik_jaw_topdown, the exact code
path behind EnvironmentSoArm.compute_ik_reachability_per_candidate -- see
tests/test_headless_ik_topdown_parity.py) and then hands the raw per-candidate
results to this module. Splitting it this way means the schema/validation/
determinism logic below can be unit tested under any Python environment (no
tango conda env required), while the real computation stays a thin,
undupli­cated call into committed repository code.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Tuple

REQUIRED_CANDIDATE_KEYS = ("x", "y", "z", "yaw")
REQUIRED_RESULT_KEYS = ("ik_converged", "ik_residual", "max_joint_delta")


class FixtureError(ValueError):
    """The candidate-pose fixture file is malformed."""


class CandidateCountMismatchError(ValueError):
    """The number of raw IK results does not match the number of candidate
    poses in the fixture -- i.e. a candidate was silently dropped (or
    duplicated) somewhere between the fixture and the solver loop."""


def load_fixture(path: str) -> Tuple[List[Dict[str, float]], str]:
    """Read the fixture JSON file, returning (candidates, sha256_hex) where
    sha256_hex is the digest of the exact bytes on disk (not of any
    re-serialized/normalized form) -- this is the provenance digest the
    pilot report cites."""
    with open(path, "rb") as f:
        raw_bytes = f.read()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    doc = json.loads(raw_bytes.decode("utf-8"))
    candidates = doc.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FixtureError(f"fixture {path!r} has no non-empty 'candidates' list")
    if not (3 <= len(candidates) <= 10):
        raise FixtureError(
            f"fixture {path!r} has {len(candidates)} candidates; expected 3-10"
        )

    normalized: List[Dict[str, float]] = []
    for i, cand in enumerate(candidates):
        missing = [k for k in REQUIRED_CANDIDATE_KEYS if k not in cand]
        if missing:
            raise FixtureError(f"candidate[{i}] missing keys {missing}: {cand!r}")
        normalized.append({k: float(cand[k]) for k in REQUIRED_CANDIDATE_KEYS})

    return normalized, digest


def build_candidate_features(
    candidates: List[Dict[str, float]], raw_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge each candidate pose with its raw solver result into one record
    per candidate. Raises CandidateCountMismatchError if any candidate was
    dropped (or an extra one appeared) -- never silently truncates/pads."""
    if len(raw_results) != len(candidates):
        raise CandidateCountMismatchError(
            f"expected {len(candidates)} IK results (one per fixture candidate), "
            f"got {len(raw_results)} -- a candidate was silently dropped or duplicated"
        )

    features: List[Dict[str, Any]] = []
    for i, (cand, result) in enumerate(zip(candidates, raw_results)):
        missing = [k for k in REQUIRED_RESULT_KEYS if k not in result]
        if missing:
            raise FixtureError(f"raw result[{i}] missing keys {missing}: {result!r}")
        features.append({
            "candidate_index": i,
            "x": cand["x"], "y": cand["y"], "z": cand["z"], "yaw": cand["yaw"],
            "ik_converged": bool(result["ik_converged"]),
            "ik_residual": float(result["ik_residual"]),
            "max_joint_delta": float(result["max_joint_delta"]),
        })
    return features


def _canonical_json_bytes(obj: Any) -> bytes:
    """Stable serialization for hashing: sorted keys, fixed separators, no
    whitespace ambiguity. Relies on the same interpreter/platform producing
    the same shortest-round-trip float repr for a given float64 value on
    every call within one environment -- true across all runs on the same
    machine/build, which is exactly the determinism guarantee this pilot
    verifies (repeat execution, not cross-platform bit-for-bit)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def deterministic_digest(candidate_features: List[Dict[str, Any]], fixture_digest: str) -> str:
    payload = {"fixture_sha256": fixture_digest, "candidate_features": candidate_features}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def compute_metrics(
    candidate_features: List[Dict[str, Any]], fixture_digest: str
) -> Dict[str, Any]:
    """Aggregate objectively-verifiable metrics. Never raises for a
    non-finite value in a candidate feature -- it records the fact (via
    all_residuals_finite / all_joint_deltas_finite / pilot_ok) rather than
    crashing or silently dropping the offending candidate. A missing/extra
    candidate is a structural fixture violation and is rejected earlier, in
    build_candidate_features, not here."""
    candidate_count = len(candidate_features)
    converged_count = sum(1 for c in candidate_features if c["ik_converged"])
    convergence_rate = converged_count / candidate_count if candidate_count else 0.0

    residuals = [c["ik_residual"] for c in candidate_features]
    joint_deltas = [c["max_joint_delta"] for c in candidate_features]

    all_residuals_finite = all(math.isfinite(v) for v in residuals)
    all_joint_deltas_finite = all(math.isfinite(v) for v in joint_deltas)

    finite_residuals = [v for v in residuals if math.isfinite(v)]
    finite_joint_deltas = [v for v in joint_deltas if math.isfinite(v)]
    max_residual = max(finite_residuals) if finite_residuals else float("nan")
    max_joint_delta = max(finite_joint_deltas) if finite_joint_deltas else float("nan")

    digest = deterministic_digest(candidate_features, fixture_digest)

    pilot_ok = (
        candidate_count > 0
        and all_residuals_finite
        and all_joint_deltas_finite
        and 0.0 <= convergence_rate <= 1.0
    )

    return {
        "candidate_count": candidate_count,
        "converged_count": converged_count,
        "convergence_rate": convergence_rate,
        "all_residuals_finite": all_residuals_finite,
        "all_joint_deltas_finite": all_joint_deltas_finite,
        "max_residual": max_residual,
        "max_joint_delta": max_joint_delta,
        "fixture_sha256": fixture_digest,
        "deterministic_digest": digest,
        "pilot_ok": pilot_ok,
    }
