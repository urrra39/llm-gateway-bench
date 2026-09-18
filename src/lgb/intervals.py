"""Uncertainty for published rates: Wilson score intervals for binomial
proportions, percentile bootstrap intervals for means, sums and percentiles,
and Newcombe score intervals for differences of independent proportions.

Every rate in the headline tables carries one of these, with its denominator
recomputed from parquet. Bootstrap resamples are drawn from a recorded seed;
the sha256 of the index matrix is stored alongside so the exact resamples are
pinned, not just the seed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np

INTERVAL_SEED = 20260917
BOOTSTRAP_B = 10000
Z_95 = 1.96


def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 0.0)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, {n}]")
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * float(np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def derive_seed(master: int, *parts: str) -> int:
    """One recorded seed per (frac, config, quantity); stable across runs."""
    h = hashlib.sha256(("|".join([str(master), *parts])).encode()).hexdigest()
    return int(h[:8], 16)


def bootstrap_ci(
    values: Sequence[float],
    seed: int,
    n_resamples: int = BOOTSTRAP_B,
    stat: Callable[[np.ndarray], float] | None = None,
) -> tuple[float, float, str]:
    """Percentile bootstrap CI; returns (lo, hi, sha of the index matrix)."""
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), "")
    fn = stat if stat is not None else (lambda v: float(np.mean(v)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    stats = np.asarray([fn(arr[row]) for row in idx])
    lo = float(np.percentile(stats, 2.5))
    hi = float(np.percentile(stats, 97.5))
    return (lo, hi, hashlib.sha256(idx.tobytes()).hexdigest()[:16])


def bootstrap_percentiles(
    values: Sequence[float],
    seed: int,
    levels: Sequence[float] = (50.0, 95.0, 99.0),
    n_resamples: int = BOOTSTRAP_B,
) -> tuple[dict[float, tuple[float, float]], str]:
    """One shared index matrix, one CI per percentile level."""
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return ({lv: (float("nan"), float("nan")) for lv in levels}, "")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    out: dict[float, tuple[float, float]] = {}
    for lv in levels:
        stats = np.percentile(arr[idx], lv, axis=1)
        out[lv] = (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
    return (out, hashlib.sha256(idx.tobytes()).hexdigest()[:16])


def newcombe_diff(
    k1: int, n1: int, k2: int, n2: int, z: float = Z_95
) -> tuple[float, float, float]:
    """Difference p2-p1 with the Newcombe score interval (method 10)."""
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    diff = p2 - p1
    lo = diff - float(np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2))
    hi = diff + float(np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))
    return (diff, lo, hi)


def paired_bootstrap_diff(
    x: dict[int, float],
    y: dict[int, float],
    seed: int,
    n_resamples: int = BOOTSTRAP_B,
) -> tuple[float, float, float, str, int]:
    """Mean(x)-mean(y) over shared rows, resampled in lockstep.

    Returns (diff, lo, hi, idx_sha, n_shared). Shared-row pairing is what
    gives this power over two independent intervals.
    """
    shared = sorted(set(x) & set(y))
    if not shared:
        return (float("nan"), float("nan"), float("nan"), "", 0)
    a = np.array([x[i] for i in shared])
    b = np.array([y[i] for i in shared])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_resamples, len(shared)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    sha = hashlib.sha256(idx.tobytes()).hexdigest()[:16]
    return (
        float(a.mean() - b.mean()),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
        sha,
        len(shared),
    )
