"""Paired statistics over aligned pair-level records.

Every function here takes AlignedPair records (see alignment.py) that
already carry real, per-checkpoint pair-level correctness -- never an
aggregate metrics.json dict. Pair-level observations are never inferred
from an aggregate pair_accuracy (or any other aggregate figure): an
aggregate accuracy alone cannot tell you which individual pairs two
checkpoints agreed or disagreed on, which is exactly what McNemar's test,
the paired bootstrap, and win/tie/loss counts need.

Phase 3 addition: paired_bootstrap_ci (added in Phase 1) resamples
individual pairs i.i.d., which assumes the pairs are independent. They are
not: real_analysis.py's pairs come from a cartesian product of
(positive episode, negative episode) row pairs within one query (see
research_agent_pilots/lggsn_suite/eval_core.py's build_pairs), so pairs
sharing a query are correlated. paired_bootstrap_ci_clustered resamples
whole clusters (e.g. one cluster per query) with replacement instead, which
is the standard cluster/block bootstrap fix for within-cluster
correlation -- see its docstring for why "query" is the finest available
clustering unit given only pair_results.jsonl's columns.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Sequence

from research_agent_pilots.lggsn_analysis.alignment import AlignedPair


class StatisticsError(RuntimeError):
    """Any fail-closed statistics violation: an unknown checkpoint name, an
    empty pair set, or an out-of-range parameter."""


def _checkpoint_values(
    aligned_pairs: Sequence[AlignedPair], *, checkpoint_a: str, checkpoint_b: str
) -> None:
    if not aligned_pairs:
        raise StatisticsError("at least one aligned pair is required")
    for pair in aligned_pairs:
        for name in (checkpoint_a, checkpoint_b):
            if name not in pair.correct_by_checkpoint:
                raise StatisticsError(f"checkpoint {name!r} is not present in aligned pair {pair.pair_key!r}")


def pair_accuracy(aligned_pairs: Sequence[AlignedPair], *, checkpoint: str) -> float:
    if not aligned_pairs:
        raise StatisticsError("pair_accuracy requires at least one aligned pair")
    correct = 0
    for pair in aligned_pairs:
        if checkpoint not in pair.correct_by_checkpoint:
            raise StatisticsError(f"checkpoint {checkpoint!r} is not present in aligned pair {pair.pair_key!r}")
        if pair.correct_by_checkpoint[checkpoint]:
            correct += 1
    return correct / len(aligned_pairs)


@dataclass(frozen=True)
class McNemarResult:
    checkpoint_a: str
    checkpoint_b: str
    n01: int  # a wrong, b correct
    n10: int  # a correct, b wrong
    n11: int  # both correct
    n00: int  # both wrong
    discordant_pairs: int
    p_value: float


def _exact_mcnemar_p_value(n01: int, n10: int) -> float:
    """Exact two-sided McNemar p-value via the binomial(n, 0.5) tail (not
    the asymptotic chi-squared approximation), computed with exact
    rational arithmetic and only converted to float at the very end."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = Fraction(sum(math.comb(n, i) for i in range(0, k + 1)), 2**n)
    p_value = min(Fraction(1), 2 * tail)
    return float(p_value)


def exact_mcnemar(
    aligned_pairs: Sequence[AlignedPair], *, checkpoint_a: str, checkpoint_b: str
) -> McNemarResult:
    _checkpoint_values(aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)
    n01 = n10 = n11 = n00 = 0
    for pair in aligned_pairs:
        a = pair.correct_by_checkpoint[checkpoint_a]
        b = pair.correct_by_checkpoint[checkpoint_b]
        if a and b:
            n11 += 1
        elif not a and not b:
            n00 += 1
        elif not a and b:
            n01 += 1
        else:
            n10 += 1
    p_value = _exact_mcnemar_p_value(n01, n10)
    return McNemarResult(
        checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b,
        n01=n01, n10=n10, n11=n11, n00=n00,
        discordant_pairs=n01 + n10, p_value=p_value,
    )


@dataclass(frozen=True)
class WinTieLossResult:
    checkpoint_a: str
    checkpoint_b: str
    win: int   # b correct, a wrong
    tie: int   # both correct or both wrong
    loss: int  # a correct, b wrong
    total: int


def win_tie_loss_counts(
    aligned_pairs: Sequence[AlignedPair], *, checkpoint_a: str, checkpoint_b: str
) -> WinTieLossResult:
    _checkpoint_values(aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)
    win = tie = loss = 0
    for pair in aligned_pairs:
        a = pair.correct_by_checkpoint[checkpoint_a]
        b = pair.correct_by_checkpoint[checkpoint_b]
        if a == b:
            tie += 1
        elif b and not a:
            win += 1
        else:
            loss += 1
    return WinTieLossResult(
        checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b,
        win=win, tie=tie, loss=loss, total=win + tie + loss,
    )


@dataclass(frozen=True)
class PairedBootstrapResult:
    checkpoint_a: str
    checkpoint_b: str
    n_pairs: int
    n_resamples: int
    seed: int
    confidence: float
    observed_diff_b_minus_a: float
    ci_lower: float
    ci_upper: float


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise StatisticsError("cannot compute a percentile of an empty sequence")
    if not (0.0 <= q <= 1.0):
        raise StatisticsError(f"quantile must be in [0, 1]: {q}")
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    frac = pos - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def paired_bootstrap_ci(
    aligned_pairs: Sequence[AlignedPair],
    *,
    checkpoint_a: str,
    checkpoint_b: str,
    seed: int,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> PairedBootstrapResult:
    """Deterministic paired bootstrap confidence interval for the
    difference in pair-accuracy (checkpoint_b minus checkpoint_a). `seed`
    is a required, explicit parameter -- there is no default and no
    fallback to an unseeded or system-time-derived source of randomness,
    so two calls with the same inputs and the same seed always produce a
    byte-for-byte identical result."""
    _checkpoint_values(aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)
    if n_resamples < 1:
        raise StatisticsError(f"n_resamples must be >= 1: {n_resamples}")
    if not (0.0 < confidence < 1.0):
        raise StatisticsError(f"confidence must be in (0, 1): {confidence}")

    n = len(aligned_pairs)
    a_values = [1 if p.correct_by_checkpoint[checkpoint_a] else 0 for p in aligned_pairs]
    b_values = [1 if p.correct_by_checkpoint[checkpoint_b] else 0 for p in aligned_pairs]

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        a_mean = sum(a_values[i] for i in indices) / n
        b_mean = sum(b_values[i] for i in indices) / n
        diffs.append(b_mean - a_mean)
    diffs.sort()

    observed_diff = sum(b_values) / n - sum(a_values) / n
    alpha = 1.0 - confidence
    lower = _percentile(diffs, alpha / 2.0)
    upper = _percentile(diffs, 1.0 - alpha / 2.0)

    return PairedBootstrapResult(
        checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b,
        n_pairs=n, n_resamples=n_resamples, seed=seed, confidence=confidence,
        observed_diff_b_minus_a=observed_diff, ci_lower=lower, ci_upper=upper,
    )


@dataclass(frozen=True)
class ClusteredPairedBootstrapResult:
    checkpoint_a: str
    checkpoint_b: str
    n_pairs: int
    n_clusters: int
    resampling_unit: str
    n_resamples: int
    seed: int
    confidence: float
    observed_diff_b_minus_a: float
    ci_lower: float
    ci_upper: float


def _accuracy_diff_b_minus_a(pairs: Sequence[AlignedPair], *, checkpoint_a: str, checkpoint_b: str) -> float:
    n = len(pairs)
    a_correct = sum(1 for p in pairs if p.correct_by_checkpoint[checkpoint_a])
    b_correct = sum(1 for p in pairs if p.correct_by_checkpoint[checkpoint_b])
    return b_correct / n - a_correct / n


def paired_bootstrap_ci_clustered(
    aligned_pairs: Sequence[AlignedPair],
    *,
    checkpoint_a: str,
    checkpoint_b: str,
    cluster_key_fn: Callable[[AlignedPair], str],
    resampling_unit: str,
    seed: int,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> ClusteredPairedBootstrapResult:
    """Cluster (block) bootstrap confidence interval for the difference in
    pair-accuracy (checkpoint_b minus checkpoint_a), resampling whole
    clusters -- as identified by `cluster_key_fn` -- with replacement,
    rather than resampling individual pairs i.i.d. (see paired_bootstrap_ci
    for that simpler, but only valid-if-truly-independent, version).

    This is the statistically correct choice whenever pairs sharing a
    cluster key are not independent draws -- which is exactly the case for
    LGGSN pairs: each one comes from a cartesian product of
    (positive episode, negative episode) row pairs within one query (see
    research_agent_pilots/lggsn_suite/eval_core.py's build_pairs), so pairs
    sharing a query are correlated through their shared episodes, and
    episode identity (scene_id) is not present in the committed
    pair_results.jsonl columns at all -- only `query` is. "query" is
    therefore the finest clustering unit reconstructable from the real,
    provenance-pinned input files this analysis is restricted to, making it
    the highest defensible resampling unit available (a per-pair i.i.d.
    resample would understate the true sampling variance by ignoring the
    within-query correlation entirely).

    `seed` is required and explicit, exactly like paired_bootstrap_ci --
    two calls with the same inputs and seed always produce a byte-for-byte
    identical result."""
    _checkpoint_values(aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)
    if n_resamples < 1:
        raise StatisticsError(f"n_resamples must be >= 1: {n_resamples}")
    if not (0.0 < confidence < 1.0):
        raise StatisticsError(f"confidence must be in (0, 1): {confidence}")

    clusters: dict[str, list[AlignedPair]] = {}
    for pair in aligned_pairs:
        clusters.setdefault(cluster_key_fn(pair), []).append(pair)
    cluster_keys = sorted(clusters.keys())
    n_clusters = len(cluster_keys)
    if n_clusters < 2:
        raise StatisticsError(f"clustered bootstrap requires at least two clusters, got {n_clusters}")

    observed_diff = _accuracy_diff_b_minus_a(aligned_pairs, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)

    # Precompute each cluster's (n, a_correct, b_correct) once. A resample
    # is then a sum over the n_clusters sampled clusters' precomputed
    # counts (O(n_clusters) per resample) instead of re-scanning every
    # pooled AlignedPair (O(n_pairs) per resample) -- same result, just
    # avoids an O(n_resamples * n_pairs) cost for large n_resamples.
    cluster_stats: list[tuple[int, int, int]] = []
    for key in cluster_keys:
        pairs = clusters[key]
        n = len(pairs)
        a_correct = sum(1 for p in pairs if p.correct_by_checkpoint[checkpoint_a])
        b_correct = sum(1 for p in pairs if p.correct_by_checkpoint[checkpoint_b])
        cluster_stats.append((n, a_correct, b_correct))

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        total_n = total_a = total_b = 0
        for _ in range(n_clusters):
            n, a_correct, b_correct = cluster_stats[rng.randrange(n_clusters)]
            total_n += n
            total_a += a_correct
            total_b += b_correct
        diffs.append(total_b / total_n - total_a / total_n)
    diffs.sort()

    alpha = 1.0 - confidence
    lower = _percentile(diffs, alpha / 2.0)
    upper = _percentile(diffs, 1.0 - alpha / 2.0)

    return ClusteredPairedBootstrapResult(
        checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b,
        n_pairs=len(aligned_pairs), n_clusters=n_clusters, resampling_unit=resampling_unit,
        n_resamples=n_resamples, seed=seed, confidence=confidence,
        observed_diff_b_minus_a=observed_diff, ci_lower=lower, ci_upper=upper,
    )


def holm_bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (matches R's
    p.adjust(method="holm")), returned in the SAME order as the input --
    never silently applied without the caller explicitly asking for and
    labeling both the raw and the adjusted result. For p-values
    p_(1) <= ... <= p_(m) sorted ascending, the adjusted p-value at rank i
    (1-indexed) is max_{k<=i} min(1, (m - k + 1) * p_(k)) -- the running
    maximum enforces the step-down procedure's monotonicity guarantee."""
    m = len(p_values)
    if m == 0:
        raise StatisticsError("holm_bonferroni_adjust requires at least one p-value")
    for p in p_values:
        if not (0.0 <= p <= 1.0):
            raise StatisticsError(f"p-value out of [0, 1] range: {p}")

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw_adjusted = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, raw_adjusted)
        adjusted[idx] = running_max
    return adjusted
