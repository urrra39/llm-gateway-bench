"""HTTP client for the OpenAI-compatible gateway and a cost transform.

The gateway reports token usage; dollars are a configured price table applied
to those tokens (see config/prices basis note).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from lgb.config import Config


@dataclass
class ChatResult:
    text: str
    tokens_in: int
    tokens_out: int
    raw: dict[str, Any]
    latency_s: float = 0.0


@dataclass(frozen=True)
class TokenPrice:
    input_per_mtok: float
    output_per_mtok: float

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in / 1_000_000 * self.input_per_mtok
            + tokens_out / 1_000_000 * self.output_per_mtok
        )


def price_for(cfg: Config, model: str) -> TokenPrice:
    spec = cfg.prices.per_million_tokens[model]
    return TokenPrice(spec.input_per_mtok, spec.output_per_mtok)


def _api_key(cfg: Config) -> str | None:
    env = cfg.gateway.api_key_env
    return os.environ.get(env) if env else None


class Gateway:
    """A thin OpenAI-compatible chat client with retries."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.gateway.base_url,
            timeout=cfg.gateway.timeout_s,
        )
        key = _api_key(cfg)
        self._headers = {"Content-Type": "application/json"}
        if key:
            self._headers["Authorization"] = f"Bearer {key}"

    def chat(
        self,
        model: str,
        user_content: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> ChatResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.cfg.generation.max_tokens,
        }
        last_error: str | None = None
        started = time.perf_counter()
        for attempt in range(self.cfg.gateway.max_retries):
            try:
                resp = self._client.post("/chat/completions", json=payload, headers=self._headers)
                body = resp.json()
            except (httpx.HTTPError, ValueError) as exc:  # network or JSON
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.0 * (attempt + 1))
                continue
            if resp.status_code != 200 or "choices" not in body:
                last_error = f"http {resp.status_code}: {json.dumps(body)[:300]}"
                # budget/quota exhaustion will not clear on retry within seconds
                if resp.status_code == 429 or "quota" in str(body).lower():
                    break
                time.sleep(1.0 * (attempt + 1))
                continue
            msg = body["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            usage = body.get("usage") or {}
            elapsed = time.perf_counter() - started
            return ChatResult(
                text=text,
                tokens_in=int(usage.get("prompt_tokens", 0)),
                tokens_out=int(usage.get("completion_tokens", 0)),
                raw=body,
                latency_s=elapsed,
            )
        raise RuntimeError(f"chat failed for {model}: {last_error}")
