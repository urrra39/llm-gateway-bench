"""Live gateway wiring, with a stubbed upstream (no network, no cost)."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi.testclient import TestClient

import lgb.api as api_mod
from lgb.api import build_app
from lgb.chat import BASE_URL_ENV, ChatResult, Gateway, UpstreamError
from lgb.config import Config


class FakeGateway:
    """Records calls; answers briefly unless the prompt asks for reasoning."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        model: str,
        _user_content: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        _temperature: float = 0.0,
    ) -> ChatResult:
        self.calls.append(
            {"model": model, "system_prompt": system_prompt, "max_tokens": max_tokens}
        )
        return ChatResult(
            text="A short stub answer.",
            tokens_in=10,
            tokens_out=5,
            raw={},
            latency_s=0.01,
        )


def _client(monkeypatch: Any) -> tuple[TestClient, FakeGateway]:
    cfg = Config.load("config/bench.yaml")
    fake = FakeGateway(cfg)
    monkeypatch.setattr(api_mod, "Gateway", lambda _cfg: fake)
    return TestClient(build_app(cfg)), fake


def test_health_reports_cache_size(monkeypatch: Any) -> None:
    client, _ = _client(monkeypatch)
    assert client.get("/health").json() == {"ok": True, "cache_size": 0}


def test_router_mode_uses_short_recipe_and_reports_decision(monkeypatch: Any) -> None:
    client, fake = _client(monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "A short stub answer."
    assert body["x_gateway"]["served_by"] == "deepseek-v4-flash"
    assert fake.calls[0]["max_tokens"] == 512
    assert fake.calls[0]["system_prompt"] == "Answer in one sentence. Be brief. Do not refuse."
    # second identical request is an exact cache hit: no new upstream call
    resp2 = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp2.json()["x_gateway"]["hit"] is True
    assert len(fake.calls) == 1


def test_expensive_mode_skips_cache(monkeypatch: Any) -> None:
    client, fake = _client(monkeypatch)
    headers = {"x-bench-mode": "expensive"}
    payload: dict[str, Any] = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi again"}],
    }
    client.post("/v1/chat/completions", json=payload, headers=headers)
    client.post("/v1/chat/completions", json=payload, headers=headers)
    assert len(fake.calls) == 2


def _unreachable_gateway(monkeypatch: Any, exc: Exception) -> Gateway:
    """A Gateway whose transport always fails, with retry sleeps removed."""
    monkeypatch.setattr("lgb.chat.time.sleep", lambda _s: None)
    cfg = Config.load("config/bench.yaml")
    gw = Gateway(cfg)

    def die(_request: httpx.Request) -> httpx.Response:
        raise exc

    gw._client = httpx.Client(base_url=cfg.gateway.base_url, transport=httpx.MockTransport(die))
    return gw


def test_unconfigured_upstream_error_names_the_variable(monkeypatch: Any) -> None:
    """A stranger with no endpoint gets told which variable to set."""
    gw = _unreachable_gateway(monkeypatch, httpx.ConnectError("Connection refused"))
    try:
        gw.chat("deepseek-v4-flash", "hi")
    except UpstreamError as exc:
        assert exc.env_var == BASE_URL_ENV == "LGB_GATEWAY_BASE_URL"
        assert str(exc).startswith("chat failed for deepseek-v4-flash")
        assert "LGB_GATEWAY_BASE_URL" in str(exc)
        assert "No OpenAI-compatible endpoint answered" in str(exc)
    else:  # pragma: no cover - the transport cannot succeed
        raise AssertionError("an unreachable upstream must raise UpstreamError")


def test_rejected_credentials_error_names_the_key_variable(monkeypatch: Any) -> None:
    """A reachable endpoint that refuses the key is a different fix."""
    monkeypatch.setattr("lgb.chat.time.sleep", lambda _s: None)
    cfg = Config.load("config/bench.yaml")
    gw = Gateway(cfg)
    gw._client = httpx.Client(
        base_url=cfg.gateway.base_url,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(401, json={"error": "bad key"})
        ),
    )
    try:
        gw.chat("deepseek-v4-flash", "hi")
    except UpstreamError as exc:
        assert exc.env_var == cfg.gateway.api_key_env == "GSK_API_KEY"
        assert "GSK_API_KEY" in str(exc)
    else:  # pragma: no cover - a 401 never yields a completion
        raise AssertionError("a rejected key must raise UpstreamError")


def test_http_500_names_the_variable_and_keeps_the_probe_prefix(monkeypatch: Any) -> None:
    """The CI container probe greps `chat failed`; keep that prefix intact."""
    gw = _unreachable_gateway(monkeypatch, httpx.ConnectError("Connection refused"))
    monkeypatch.setattr(api_mod, "Gateway", lambda _cfg: gw)
    client = TestClient(build_app(Config.load("config/bench.yaml")), raise_server_exceptions=False)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 500
    error = resp.json()["error"]
    assert error["type"] == "upstream_unavailable"
    assert error["param"] == "LGB_GATEWAY_BASE_URL"
    assert error["message"].startswith("chat failed for deepseek-v4-flash")
    assert "LGB_GATEWAY_BASE_URL" in error["message"]
