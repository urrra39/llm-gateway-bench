"""Wilson, bootstrap and Newcombe intervals behave on hand-checked values."""

from __future__ import annotations

import pytest

from lgb.intervals import (
    bootstrap_ci,
    derive_seed,
    newcombe_diff,
    paired_bootstrap_diff,
    wilson,
)


def test_wilson_known_values() -> None:
    # 0/10: interval starts at 0, ends near 0.28 (not the degenerate [0,0]).
    lo, hi = wilson(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-3)
    # 4/45, the low-fraction report-half false-hit rate.
    lo, hi = wilson(4, 45)
    assert lo == pytest.approx(0.0351, abs=1e-3)
    assert hi == pytest.approx(0.2073, abs=1e-3)
    # Empty denominator is (0,0), not a crash.
    assert wilson(0, 0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        wilson(11, 10)


def test_wilson_contains_point_estimate() -> None:
    for k, n in [(1, 24), (10, 80), (69, 238), (158, 246)]:
        lo, hi = wilson(k, n)
        assert lo <= k / n <= hi


def test_bootstrap_deterministic_for_recorded_seed() -> None:
    values = [float(i) for i in range(50)]
    first = bootstrap_ci(values, seed=7)
    second = bootstrap_ci(values, seed=7)
    assert first == second
    assert first[0] <= sum(values) / len(values) <= first[1]
    assert len(first[2]) == 16
    other = bootstrap_ci(values, seed=8)
    assert other[2] != first[2]


def test_derive_seed_stable_and_distinct() -> None:
    assert derive_seed(1, "low", "cache", "cost") == derive_seed(1, "low", "cache", "cost")
    assert derive_seed(1, "low", "cache", "cost") != derive_seed(1, "high", "cache", "cost")


def test_newcombe_contains_zero_for_overlapping_rates() -> None:
    # High-fraction cascade 6/86 vs heuristic 10/80: point diff 0.0552,
    # interval must straddle zero.
    diff, lo, hi = newcombe_diff(6, 86, 10, 80)
    assert diff == pytest.approx(0.0552, abs=1e-4)
    assert lo < 0 < hi


def test_paired_bootstrap_diff_shared_rows_only() -> None:
    x = {0: 1.0, 1: 2.0, 2: 3.0}
    y = {1: 1.0, 2: 1.0, 3: 9.0}
    diff, lo, hi, _sha, n = paired_bootstrap_diff(x, y, seed=3, n_resamples=2000)
    assert n == 2
    assert diff == pytest.approx(1.5)
    assert lo <= diff <= hi
