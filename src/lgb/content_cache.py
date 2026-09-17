"""Persistent content-addressed cache for gateway responses.

Keyed by model, prompt, parameters and seed, so an identical request never pays
twice when a run resumes. Stored under data/cache/ (ignored by git); run
outcomes under data/runs/ are the tracked artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(
    model: str,
    prompt: str,
    parameters: dict[str, Any],
    seed: int | None = None,
) -> str:
    canonical = json.dumps(
        {"model": model, "prompt": prompt, "parameters": parameters, "seed": seed},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def read_entry(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, ValueError):
        return None


def write_entry(cache_dir: Path, key: str, value: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{key}.tmp"
    tmp.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    tmp.replace(cache_path(cache_dir, key))
