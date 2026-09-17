"""Serve configuration: bind address and upstream override."""

from __future__ import annotations

import os
from typing import Any

import pytest
import uvicorn

from lgb.__main__ import main
from lgb.config import Config


def test_gateway_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LGB_GATEWAY_BASE_URL", "http://example.invalid:9999/v1")
    cfg = Config.load("config/bench.yaml")
    assert cfg.gateway.base_url == "http://example.invalid:9999/v1"


def test_gateway_base_url_default_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LGB_GATEWAY_BASE_URL", raising=False)
    assert os.environ.get("LGB_GATEWAY_BASE_URL") is None
    cfg = Config.load("config/bench.yaml")
    assert cfg.gateway.base_url == "http://127.0.0.1:8787/v1"


def test_serve_binds_configurable_host_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_run(app: object, host: str = "0.0.0.0", port: int = 8000) -> None:
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr(uvicorn, "run", fake_run)
    assert main(["serve"]) == 0
    assert calls == {"host": "0.0.0.0", "port": 8000}
    assert main(["serve", "--host", "127.0.0.1", "--port", "8123"]) == 0
    assert calls == {"host": "127.0.0.1", "port": 8123}
