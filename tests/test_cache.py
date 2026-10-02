"""Cache logic with hand-computed expected values."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lgb.cache import CacheItem, SemanticCache, normalize_exact, simulate_threshold


def test_normalize_exact_collapses_whitespace_and_case() -> None:
    assert normalize_exact("  Hello   WORLD\n") == "hello world"
    assert normalize_exact("Explain  this") == "explain this"


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


def test_exact_hit_without_embedder_survives_save_and_load(tmp_path: Path) -> None:
    cache = SemanticCache(0.9, None, exact=True)
    assert cache.lookup("Hello world").kind == "miss"
    cache.add(CacheItem(7, "Hello   world", "g1", "hi", "expensive"))
    look = cache.lookup("  hello WORLD ")
    assert (look.kind, look.hit_idx) == ("cache_exact", 7)
    assert cache.get(7).answer_text == "hi"
    # no embedder: a non-exact request misses instead of failing
    assert cache.lookup("goodbye").kind == "miss"
    cache.save(tmp_path / "items.parquet")
    loaded = SemanticCache.load(tmp_path / "items.parquet", 0.9, None)
    assert loaded.lookup("hello world").hit_idx == 7
