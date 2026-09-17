"""Content-addressed cache keys are stable and parameter-sensitive."""

from __future__ import annotations

from pathlib import Path

from lgb.content_cache import cache_key, read_entry, write_entry


def test_same_inputs_same_key() -> None:
    a = cache_key("m", "hello", {"max_tokens": 10}, seed=1)
    b = cache_key("m", "hello", {"max_tokens": 10}, seed=1)
    assert a == b


def test_different_prompt_different_key() -> None:
    a = cache_key("m", "hello", {"max_tokens": 10})
    b = cache_key("m", "world", {"max_tokens": 10})
    assert a != b


def test_different_model_or_params_different_key() -> None:
    base = cache_key("m1", "hello", {"max_tokens": 10})
    assert cache_key("m2", "hello", {"max_tokens": 10}) != base
    assert cache_key("m1", "hello", {"max_tokens": 20}) != base


def test_roundtrip_through_disk(tmp_path: Path) -> None:
    key = cache_key("m", "hello", {"t": 0.0})
    assert read_entry(tmp_path, key) is None
    write_entry(tmp_path, key, {"text": "hi", "tokens_in": 3})
    assert read_entry(tmp_path, key) == {"text": "hi", "tokens_in": 3}
