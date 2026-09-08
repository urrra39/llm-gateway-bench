"""Atomic, resumable persistence.

Every derived table is a parquet written atomically (tmp + rename). A stage
reads what exists, works only the missing rows, and appends. A kill loses at
most the in-flight row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:  # corrupt artifact is treated as absent
        return None


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def append_rows(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Append frame's rows to the table at path (creating it if absent) and
    return the full table."""
    existing = read_parquet(path)
    if existing is None:
        write_parquet(frame, path)
        return frame.reset_index(drop=True)
    merged = pd.concat([existing, frame], ignore_index=True)
    write_parquet(merged, path)
    return merged


def done_keys(path: Path, key_col: str) -> set[str]:
    table = read_parquet(path)
    if table is None or key_col not in table.columns:
        return set()
    return {str(v) for v in table[key_col].tolist()}


def write_json(data: dict[str, Any], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    import json

    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, ValueError):
        return None
