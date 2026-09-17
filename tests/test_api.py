"""Live gateway wiring, with a stubbed upstream (no network, no cost)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import lgb.api as api_mod
from lgb.api import build_app
from lgb.chat import ChatResult
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
