"""Cost accounting and metrics math with hand-computed values."""

from __future__ import annotations

import numpy as np

from lgb.chat import TokenPrice, price_for
from lgb.config import Config
from lgb.metrics import _has_error, percentiles


def test_token_price_hand_computed() -> None:
    price = TokenPrice(input_per_mtok=1.5, output_per_mtok=6.0)
    # 1000 input tokens at $1.50/MTok = $0.0015; 500 output at $6.00/MTok = $0.003
    assert abs(price.cost_usd(1000, 500) - 0.0045) < 1e-12


def test_price_for_reads_config_table() -> None:
    cfg = Config.load("config/bench.yaml")
    price = price_for(cfg, "deepseek-v4-flash")
    assert price.input_per_mtok == 1.5
    assert price.output_per_mtok == 6.0


def test_percentiles_hand_computed() -> None:
    stats = percentiles([0.0, 50.0, 100.0])
    assert stats["p50"] == 50.0
    assert stats["p95"] == 95.0
    assert stats["mean"] == 50.0


def test_percentiles_empty_is_nan() -> None:
    stats = percentiles([])
    assert np.isnan(stats["p50"])


def test_has_error_treats_parquet_nan_as_no_error() -> None:
    assert _has_error(None) is False
    assert _has_error(float("nan")) is False
    assert _has_error("nan") is False
    assert _has_error("") is False
    assert _has_error("RuntimeError: boom") is True
