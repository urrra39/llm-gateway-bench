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


class UpstreamError(RuntimeError):
    """No completion could be obtained from the configured upstream.

    The message names the environment variable that controls the failing
    setting, because the common case is a stranger with no upstream at all:
    a generic "chat failed" sends them reading source to find out which knob
    they are missing. `env_var` carries the same name for the API layer.
    """

    def __init__(self, message: str, env_var: str) -> None:
        super().__init__(message)
        self.env_var = env_var


#: Environment variable that points the stack at an OpenAI-compatible upstream.
#: Declared here so the error text and config.Config.load cannot drift apart.
BASE_URL_ENV = "LGB_GATEWAY_BASE_URL"


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
        if self.cfg.generation.reasoning_effort:
            payload["reasoning_effort"] = self.cfg.generation.reasoning_effort
        last_error: str | None = None
        unreachable = False
        unauthorized = False
        started = time.perf_counter()
        for attempt in range(self.cfg.gateway.max_retries):
            try:
                resp = self._client.post("/chat/completions", json=payload, headers=self._headers)
                body = resp.json()
            except httpx.TransportError as exc:  # no route, refused, DNS, timeout
                last_error = f"{type(exc).__name__}: {exc}"
                unreachable = True
                time.sleep(1.0 * (attempt + 1))
                continue
            except (httpx.HTTPError, ValueError) as exc:  # protocol or JSON
                last_error = f"{type(exc).__name__}: {exc}"
                unreachable = False
                time.sleep(1.0 * (attempt + 1))
                continue
            unreachable = False
            if resp.status_code != 200 or "choices" not in body:
                last_error = f"http {resp.status_code}: {json.dumps(body)[:300]}"
                unauthorized = resp.status_code in (401, 403)
                # budget/quota exhaustion will not clear on retry within seconds
                if resp.status_code == 429 or "quota" in str(body).lower():
                    break
                if unauthorized:
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
        raise self._failure(model, last_error, unreachable, unauthorized)

    def _failure(
        self, model: str, last_error: str | None, unreachable: bool, unauthorized: bool
    ) -> UpstreamError:
        """Turn the last transport or HTTP error into an actionable message.

        Keeps the `chat failed for <model>` prefix the container probe and the
        README both quote, then names the variable that would fix it.
        """
        base = self.cfg.gateway.base_url
        key_env = self.cfg.gateway.api_key_env or "the gateway api_key_env"
        head = f"chat failed for {model}: {last_error}"
        if unreachable:
            return UpstreamError(
                f"{head}. No OpenAI-compatible endpoint answered at {base}. "
                f"Set {BASE_URL_ENV} to a reachable /v1 endpoint "
                f"(and {key_env} if that endpoint needs a bearer token).",
                BASE_URL_ENV,
            )
        if unauthorized:
            return UpstreamError(
                f"{head}. The endpoint at {base} rejected the credentials. "
                f"Set {key_env} to a key valid for that endpoint, "
                f"or point {BASE_URL_ENV} at one that answers keyless.",
                key_env,
            )
        return UpstreamError(
            f"{head}. The endpoint at {base} answered but served no completion. "
            f"Check {BASE_URL_ENV} names an OpenAI-compatible /v1 endpoint "
            f"that serves the model {model!r}.",
            BASE_URL_ENV,
        )
