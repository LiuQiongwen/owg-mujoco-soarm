# -*- coding: utf-8 -*-
"""Pure-stdlib grouping/pairing/metrics logic for the LGGSN standalone
evaluator. No torch/numpy dependency -- unit-testable under any Python
environment. The only thing that ever touches torch is evaluator.py, which
scores each dataset row exactly once (a single forward pass per row) and
hands the resulting {row_id: score} mapping back here.

Every function here reimplements, from scratch and by hand -- NOT by
importing -- the grouping/pairing semantics documented in
docs/LGGSN_EVAL_SUITE.md, transcribed from train_lggsn_pairwise.py's
load_episodes/build_pairs (read, never imported -- see that doc for why).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

DERIVABLE_FEATURES = ("dist_to_centroid", "z_rel")
_RAW_REQUIRED_ALWAYS = ("query", "scene_id", "label")


class SchemaVerificationError(ValueError):
    """A checkpoint or dataset does not match its registry entry -- fail
    closed, never silently proceed with a best-effort guess."""


class DatasetSchemaError(ValueError):
    """The dataset is missing a column a checkpoint's feature set requires
    and this evaluator cannot derive."""


class MetricsSchemaError(ValueError):
    """A raw value handed to a builder function has the wrong type/shape --
    an evaluator.py bug, never a legitimate experiment outcome."""


# ── file hashing ─────────────────────────────────────────────────────────────

def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── loading + grouping ───────────────────────────────────────────────────────

def read_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    """Assigns a stable `_row_id` (0-based, file order) to every row --
    the single source of truth for row identity used throughout this
    module and by evaluator.py's score cache."""
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in _RAW_REQUIRED_ALWAYS:
                if key not in row:
                    raise DatasetSchemaError(
                        f"row {len(rows)} is missing required key {key!r}: {row!r}"
                    )
            row = dict(row)
            row["_row_id"] = len(rows)
            rows.append(row)
    if not rows:
        raise DatasetSchemaError("dataset file contains no rows")
    return rows


def group_episodes(rows: Sequence[Dict[str, Any]]) -> "dict":
    """dict[(query, scene_id)] -> list[row]. Plain dict, populated by a
    single top-to-bottom scan -- insertion order is first-seen order,
    matching train_lggsn_pairwise.py's ep_rows construction exactly (see
    docs/LGGSN_EVAL_SUITE.md point 6)."""
    episodes: "dict" = {}
    for row in rows:
        key = (row["query"], row["scene_id"])
        episodes.setdefault(key, []).append(row)
    return episodes


def compute_derived_features(episode_rows: Sequence[Dict[str, Any]], needed: set) -> List[Dict[str, Any]]:
    """Per-episode dist_to_centroid / z_rel, computed over ALL rows in the
    episode (matching train_lggsn_pairwise.py:104-118 exactly). Returns new
    dicts; never mutates the input. No-op (returns the input list, still
    copied) if neither derived feature is needed."""
    if not (needed & set(DERIVABLE_FEATURES)):
        return [dict(r) for r in episode_rows]
    for req in ("x", "y", "z"):
        for r in episode_rows:
            if req not in r:
                raise DatasetSchemaError(
                    f"row _row_id={r.get('_row_id')} missing {req!r}, required to derive "
                    f"dist_to_centroid/z_rel"
                )
    xs = [r["x"] for r in episode_rows]
    ys = [r["y"] for r in episode_rows]
    zs = [r["z"] for r in episode_rows]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    z_min, z_max = min(zs), max(zs)
    out = []
    for r, x, y, z in zip(episode_rows, xs, ys, zs):
        r2 = dict(r)
        if "dist_to_centroid" in needed:
            r2["dist_to_centroid"] = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if "z_rel" in needed:
            r2["z_rel"] = (z - z_min) / (z_max - z_min + 1e-8)
        out.append(r2)
    return out


def validate_feature_columns(episodes: "dict", feature_columns: Sequence[str]) -> None:
    """Fail closed: every row (after derived-feature computation) must have
    every column in feature_columns. Raises naming the exact missing
    column(s) and row id -- never silently skips a row for this reason."""
    for key, rows in episodes.items():
        for row in rows:
            missing = [c for c in feature_columns if c not in row]
            if missing:
                raise DatasetSchemaError(
                    f"episode {key}, row _row_id={row.get('_row_id')}: missing feature "
                    f"column(s) {missing} required by feature_columns={list(feature_columns)}"
                )


def label_episodes(episodes: "dict") -> Tuple["dict", int, int]:
    """Returns (by_query, malformed_group_count, skipped_row_count).

    by_query: dict[query] -> {"pos": [episode_rows, ...], "neg": [...]}
    (each episode_rows is the full row list for one non-tied episode).

    Matches train_lggsn_pairwise.py:96-122 exactly: majority vote over
    row-level `label`, ties excluded. Unlike the original (which silently
    `continue`s past a tied episode), every excluded episode is counted in
    malformed_group_count and every row inside one in skipped_row_count --
    nothing is dropped unaccounted-for."""
    by_query: "dict" = {}
    malformed_group_count = 0
    skipped_row_count = 0
    for (query, _scene_id), rows in episodes.items():
        labels = [r["label"] for r in rows]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos == n_neg:
            malformed_group_count += 1
            skipped_row_count += len(rows)
            continue
        episode_label = 1 if n_pos > n_neg else 0
        side = "pos" if episode_label == 1 else "neg"
        by_query.setdefault(query, {"pos": [], "neg": []})
        by_query[query][side].append(rows)
    return by_query, malformed_group_count, skipped_row_count


PairRecord = Tuple[int, int, str]  # (pos_row_id, neg_row_id, query)


def build_pairs(
    by_query: "dict", *, val_frac: float = 0.2, seed: int = 42
) -> Tuple[List[PairRecord], List[PairRecord]]:
    """Matches train_lggsn_pairwise.py:127-168 exactly: one Random(seed)
    instance, reused in query-iteration order (by_query's own insertion
    order -- see label_episodes/group_episodes for why that's
    deterministic and first-seen, not sorted); shuffle positive episodes,
    then negative episodes, split each 80/20 by episode count (not pair
    count), then full cross product of surviving-train (resp. -val)
    positive-episode rows x negative-episode rows."""
    rng = random.Random(seed)
    train_pairs: List[PairRecord] = []
    val_pairs: List[PairRecord] = []

    for query, sides in by_query.items():
        pos_eps = list(sides["pos"])
        neg_eps = list(sides["neg"])
        if not pos_eps or not neg_eps:
            continue
        rng.shuffle(pos_eps)
        rng.shuffle(neg_eps)
        n_pos_val = max(1, round(len(pos_eps) * val_frac))
        n_neg_val = max(1, round(len(neg_eps) * val_frac))
        pos_val, pos_train = pos_eps[:n_pos_val], pos_eps[n_pos_val:]
        neg_val, neg_train = neg_eps[:n_neg_val], neg_eps[n_neg_val:]

        def cartesian(pos_list, neg_list, query=query):
            out: List[PairRecord] = []
            for p_ep in pos_list:
                for n_ep in neg_list:
                    for p_row in p_ep:
                        for n_row in n_ep:
                            out.append((p_row["_row_id"], n_row["_row_id"], query))
            return out

        train_pairs.extend(cartesian(pos_train, neg_train))
        val_pairs.extend(cartesian(pos_val, neg_val))

    return train_pairs, val_pairs


# ── checkpoint/schema verification (fail closed) ────────────────────────────

def verify_checkpoint_identity(*, expected_sha256: str, actual_sha256: str, name: str) -> None:
    if expected_sha256 != actual_sha256:
        raise SchemaVerificationError(
            f"[{name}] checkpoint SHA-256 mismatch: registry expects "
            f"{expected_sha256}, on-disk file hashes to {actual_sha256} -- "
            f"refusing to evaluate a checkpoint that does not match its "
            f"pinned registry entry"
        )


def verify_state_dict_schema(
    *, state_dict_shapes: Dict[str, Tuple[int, ...]], expected_input_dim: int,
    expected_query_dim: int, expected_hidden_dim: int, name: str,
) -> None:
    """state_dict_shapes: {key: tuple(tensor.shape)} -- plain data, no
    torch dependency here; evaluator.py extracts this from the real
    loaded state_dict before calling in."""
    if "mlp.0.weight" not in state_dict_shapes:
        raise SchemaVerificationError(
            f"[{name}] state_dict has no 'mlp.0.weight' key -- not a plain "
            f"LGGSN checkpoint (keys: {sorted(state_dict_shapes)[:10]}...)"
        )
    w0_shape = state_dict_shapes["mlp.0.weight"]
    expected_mlp_in = expected_input_dim + expected_query_dim
    if tuple(w0_shape) != (expected_hidden_dim, expected_mlp_in):
        raise SchemaVerificationError(
            f"[{name}] mlp.0.weight shape {tuple(w0_shape)} != expected "
            f"({expected_hidden_dim}, {expected_mlp_in}) (registry input_dim="
            f"{expected_input_dim} + query_dim={expected_query_dim})"
        )
    has_query_emb = "query_emb.weight" in state_dict_shapes
    if has_query_emb != (expected_query_dim > 0):
        raise SchemaVerificationError(
            f"[{name}] query_emb.weight presence ({has_query_emb}) does not match "
            f"registry expected_query_dim={expected_query_dim}"
        )
    if "mlp.2.weight" not in state_dict_shapes:
        raise SchemaVerificationError(f"[{name}] state_dict has no 'mlp.2.weight' key")
    w2_shape = tuple(state_dict_shapes["mlp.2.weight"])
    if w2_shape != (1, expected_hidden_dim):
        raise SchemaVerificationError(
            f"[{name}] mlp.2.weight shape {w2_shape} != expected (1, {expected_hidden_dim})"
        )


# ── metrics ──────────────────────────────────────────────────────────────────

def _pair_stats(pairs: Sequence[PairRecord], scores: Dict[int, float]) -> Dict[str, Any]:
    if not pairs:
        # None (not NaN): keeps every artifact strictly valid JSON, and
        # "undefined because there are zero pairs" is a distinct, cleaner
        # signal than a float that silently poisons any arithmetic done on
        # it downstream without an explicit isfinite check.
        return {
            "pair_count": 0, "pair_accuracy": None, "ties_count": 0,
            "mean_score_margin": None, "median_score_margin": None,
        }
    n_correct = 0
    n_ties = 0
    margins = []
    for pos_id, neg_id, _query in pairs:
        s_pos = scores[pos_id]
        s_neg = scores[neg_id]
        margins.append(s_pos - s_neg)
        if s_pos > s_neg:
            n_correct += 1
        elif s_pos == s_neg:
            n_ties += 1
    return {
        "pair_count": len(pairs),
        "pair_accuracy": n_correct / len(pairs),
        "ties_count": n_ties,
        "mean_score_margin": statistics.fmean(margins),
        "median_score_margin": statistics.median(margins),
    }


def _unique_row_ids_by_side(by_query: "dict", side: str) -> List[int]:
    ids: List[int] = []
    for sides in by_query.values():
        for episode_rows in sides[side]:
            ids.extend(r["_row_id"] for r in episode_rows)
    return ids


def bootstrap_pair_accuracy_ci(
    by_query: "dict", scores: Dict[int, float], *, val_frac: float, seed: int,
    n_resamples: int = 1000, alpha: float = 0.05,
) -> Optional[Tuple[float, float]]:
    """Episode-level bootstrap (resamples EPISODES with replacement, not
    pairs -- pairs sharing an episode are not independent observations, so
    a pair-level bootstrap would understate interval width via
    pseudo-replication; see docs/LGGSN_EVAL_SUITE.md). Rebuilds each
    resample's pair set from cached scores (no re-scoring). Returns None
    if there are fewer than 2 total episodes (not enough to resample)."""
    all_episodes: List[Tuple[str, str, list]] = []  # (query, side, episode_rows)
    for query, sides in by_query.items():
        for side in ("pos", "neg"):
            for episode_rows in sides[side]:
                all_episodes.append((query, side, episode_rows))
    if len(all_episodes) < 2:
        return None

    rng = random.Random(seed)
    accuracies = []
    for _ in range(n_resamples):
        resampled = [all_episodes[rng.randrange(len(all_episodes))] for _ in range(len(all_episodes))]
        resampled_by_query: "dict" = {}
        for query, side, episode_rows in resampled:
            resampled_by_query.setdefault(query, {"pos": [], "neg": []})
            resampled_by_query[query][side].append(episode_rows)
        _, val_pairs = build_pairs(resampled_by_query, val_frac=val_frac, seed=seed)
        if not val_pairs:
            continue
        stats = _pair_stats(val_pairs, scores)
        accuracies.append(stats["pair_accuracy"])

    if len(accuracies) < 2:
        return None
    accuracies.sort()
    lo_idx = int((alpha / 2) * len(accuracies))
    hi_idx = int((1 - alpha / 2) * len(accuracies)) - 1
    hi_idx = min(hi_idx, len(accuracies) - 1)
    return accuracies[lo_idx], accuracies[hi_idx]


def per_query_breakdown(by_query: "dict", val_pairs: Sequence[PairRecord], scores: Dict[int, float]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for query in by_query:
        query_pairs = [p for p in val_pairs if p[2] == query]
        result[query] = _pair_stats(query_pairs, scores)
    return result


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def deterministic_digest(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def build_metrics(
    *,
    checkpoint_name: str,
    checkpoint_sha256: str,
    provenance_status: str,
    feature_columns: Sequence[str],
    input_dim: int,
    dataset_sha256: str,
    seed: int,
    val_frac: float,
    by_query: "dict",
    train_pairs: Sequence[PairRecord],
    val_pairs: Sequence[PairRecord],
    scores: Dict[int, float],
    malformed_group_count: int,
    skipped_row_count: int,
    runtime_seconds: float,
    bootstrap_ci: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    if not all(math.isfinite(v) for v in scores.values()):
        all_scores_finite = False
    else:
        all_scores_finite = True

    eligible_group_count = sum(len(sides["pos"]) + len(sides["neg"]) for sides in by_query.values())

    val_stats = _pair_stats(val_pairs, scores)
    train_stats = _pair_stats(train_pairs, scores)

    pos_ids = _unique_row_ids_by_side(by_query, "pos")
    neg_ids = _unique_row_ids_by_side(by_query, "neg")
    positive_score_mean = statistics.fmean(scores[i] for i in pos_ids) if pos_ids else None
    negative_score_mean = statistics.fmean(scores[i] for i in neg_ids) if neg_ids else None

    per_query = per_query_breakdown(by_query, val_pairs, scores)

    digest_payload = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_columns": list(feature_columns),
        "input_dim": input_dim,
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "val_frac": val_frac,
        "eligible_group_count": eligible_group_count,
        "eligible_pair_count": val_stats["pair_count"],
        "pair_accuracy": val_stats["pair_accuracy"],
        "positive_score_mean": positive_score_mean,
        "negative_score_mean": negative_score_mean,
        "mean_score_margin": val_stats["mean_score_margin"],
        "median_score_margin": val_stats["median_score_margin"],
        "ties_count": val_stats["ties_count"],
        "malformed_group_count": malformed_group_count,
        "skipped_row_count": skipped_row_count,
        "train_pair_accuracy": train_stats["pair_accuracy"],
        "train_pair_count": train_stats["pair_count"],
    }
    digest = deterministic_digest(digest_payload)

    pilot_ok = bool(
        all_scores_finite
        and eligible_group_count > 0
        and val_stats["pair_count"] > 0
        and val_stats["pair_accuracy"] is not None
        and 0.0 <= val_stats["pair_accuracy"] <= 1.0
    )

    return {
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "provenance_status": provenance_status,
        "feature_columns": list(feature_columns),
        "input_dim": input_dim,
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "val_frac": val_frac,
        "eligible_group_count": eligible_group_count,
        "eligible_pair_count": val_stats["pair_count"],
        "pair_accuracy": val_stats["pair_accuracy"],
        "positive_score_mean": positive_score_mean,
        "negative_score_mean": negative_score_mean,
        "mean_score_margin": val_stats["mean_score_margin"],
        "median_score_margin": val_stats["median_score_margin"],
        "ties_count": val_stats["ties_count"],
        "malformed_group_count": malformed_group_count,
        "skipped_row_count": skipped_row_count,
        "all_scores_finite": all_scores_finite,
        "bootstrap_ci_95": list(bootstrap_ci) if bootstrap_ci is not None else None,
        "per_query_val_pair_accuracy": {q: s["pair_accuracy"] for q, s in per_query.items()},
        "train_split": {
            "pair_count": train_stats["pair_count"],
            "pair_accuracy": train_stats["pair_accuracy"],
            "ties_count": train_stats["ties_count"],
            "mean_score_margin": train_stats["mean_score_margin"],
            "median_score_margin": train_stats["median_score_margin"],
        },
        "deterministic_digest": digest,
        "pilot_ok": pilot_ok,
        "runtime_seconds": runtime_seconds,
    }
