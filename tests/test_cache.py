"""Cache logic with hand-computed expected values."""

from __future__ import annotations

import numpy as np

from lgb.cache import (
    choose_threshold,
    evaluate_threshold,
    normalize_exact,
    simulate_threshold,
)


def test_normalize_exact_collapses_whitespace_and_case() -> None:
    assert normalize_exact("  Hello   WORLD\n") == "hello world"
    assert normalize_exact("Explain  this") == "explain this"


def test_evaluate_threshold_hand_computed() -> None:
    pos = np.array([0.9, 0.8, 0.4])
    neg = np.array([0.85, 0.3])
    stats = evaluate_threshold(pos, neg, 0.8)
    # positives >= 0.8: 0.9, 0.8 -> tp=2, fn=1; negatives >= 0.8: 0.85 -> fp=1, tn=1
    assert stats["tp"] == 2
    assert stats["fn"] == 1
    assert stats["fp"] == 1
    assert stats["tn"] == 1
    assert stats["precision"] == 2 / 3
    assert stats["recall"] == 2 / 3


def test_choose_threshold_prefers_fewer_false_hits_on_tie() -> None:
    pos = np.array([0.9, 0.9])
    neg = np.array([0.1, 0.2])
    thr, _ = choose_threshold(pos, neg, candidates=np.array([0.5, 0.9]))
    # Both thresholds give F1=1.0; the higher threshold must win.
    assert thr == 0.9


def test_simulate_threshold_replays_growing_cache() -> None:
    # Three orthogonal-ish unit vectors: row 1 paraphrases row 0, row 2 is novel.
    v0 = np.array([1.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    vecs = np.stack([v0, v1, v2])
    should_hit = np.array([False, True, False])
    eligible = np.array([True, True, True])
    stats = simulate_threshold(vecs, should_hit, eligible, 0.9)
    assert stats["tp"] == 1
    assert stats["tn"] == 2
    assert stats["fp"] == 0
    assert stats["fn"] == 0


def test_simulate_threshold_counts_false_hit() -> None:
    v0 = np.array([1.0, 0.0])
    v1 = np.array([0.0, 1.0])
    v2 = np.array([1.0, 0.0])
    vecs = np.stack([v0, v1, v2])
    # Row 2 is a trap: lexically close to row 0 in this toy space but labelled
    # should-not-hit. At threshold 0.9 the max over the store hits row 0.
    should_hit = np.array([False, False, False])
    eligible = np.array([True, True, True])
    stats = simulate_threshold(vecs, should_hit, eligible, 0.9)
    assert stats["fp"] == 1
    assert stats["false_hit_rate"] == 1.0
