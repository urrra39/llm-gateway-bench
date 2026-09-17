"""Repository hygiene: run artifacts are tracked, caches are not."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _check_ignored(path: str) -> bool:
    proc = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    return proc.returncode == 0


def test_run_outputs_are_tracked() -> None:
    for path in [
        "data/runs/low_cache/outcomes.parquet",
        "data/runs/low_cache/judge.parquet",
        "data/runs/low_cache/cache_items.parquet",
        "data/runs/workloads/workload_low.parquet",
        "data/runs/tuning.json",
        "results.json",
    ]:
        assert not _check_ignored(path), f"{path} must be tracked"


def test_model_caches_and_scratch_are_ignored() -> None:
    for path in [
        "data/models/all-MiniLM-L6-v2/model.safetensors",
        "data/raw/paws_train.parquet",
        "data/cache/some-entry.json",
    ]:
        assert _check_ignored(path), f"{path} must be ignored"
