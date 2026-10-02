"""The FastAPI gateway exposing an OpenAI-compatible /v1/chat/completions.

Live mode reuses the same components the benchmark measures, with the same
short/long recipes per tier: cascade routing and the cache. The live cache is
exact-match only. Its semantic tier is off because the benchmark's own gates
find its false-hit rate above the 5% bar on every config (README, validity
gates), and because it would need the embedding weights the image does not
ship. Per-request instrumentation is returned in the response body's
x_gateway object.
"""

from __future__ import annotations

import itertools
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from lgb.cache import CacheItem, SemanticCache
from lgb.chat import ChatResult, Gateway, UpstreamError
from lgb.config import Config
from lgb.router import CascadeRouter

#: uvicorn owns the handlers when the container runs, so an upstream failure
#: reaches `docker logs` through its logger rather than a bare traceback.
log = logging.getLogger("uvicorn.error")

#: x-bench-mode values: router = cache then cascade; cache = cache then long
#: recipe; cheap = cache then short recipe; expensive = long recipe, no cache.
MODES = ("router", "cache", "cheap", "expensive")


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _user_text(body: Any) -> str:
    """The user turns of an OpenAI chat body joined into one string.

    Accepts string content and the list-of-parts form (text parts only).
    Anything else is the client's error, so it is a 400, not a 500.
    """
    if not isinstance(body, dict):
        raise _bad_request("request body must be a JSON object")
    if body.get("stream"):
        raise _bad_request("stream=true is not supported by this gateway")
    messages = body.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise _bad_request("messages must be a list of objects")
    parts: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        elif content is not None:
            raise _bad_request("message content must be a string or a list of parts")
    text = "\n".join(parts).strip()
    if not text:
        raise _bad_request("no user message with text content")
    return text


def build_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="llm-gateway-bench", version="0.1.0")
    gw = Gateway(cfg)
    cache = SemanticCache(cfg.cache.sim_threshold, None, exact=True)
    cascade = CascadeRouter(cfg.router)
    #: Item ids for the live cache. next() on a count is atomic, so two
    #: concurrent misses never share an id (len(cache) could).
    item_ids = itertools.count()

    @app.exception_handler(HTTPException)
    async def invalid_request(_request: Request, exc: HTTPException) -> JSONResponse:
        """Client errors in the OpenAI error shape, not FastAPI's {"detail"}."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "invalid_request_error"}},
        )

    @app.exception_handler(UpstreamError)
    async def upstream_unavailable(_request: Request, exc: UpstreamError) -> JSONResponse:
        """A missing upstream is a configuration problem, so say which
        variable configures it instead of returning a bare 500."""
        log.error("%s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "upstream_unavailable",
                    "param": exc.env_var,
                }
            },
        )

    def cheap(text: str) -> ChatResult:
        return gw.chat(
            cfg.models.cheap,
            text,
            system_prompt=cfg.generation.cheap_system_prompt,
            max_tokens=cfg.generation.cheap_max_tokens,
        )

    def expensive(text: str) -> ChatResult:
        return gw.chat(cfg.models.expensive, text, system_prompt=cfg.generation.system_prompt)

    def answer(mode: str, text: str) -> tuple[str, str, int, int, dict[str, object]]:
        """Blocking: runs in a worker thread so upstream waits never stall
        the event loop (and /health with it)."""
        decision: dict[str, object] = {"mode": mode, "hit": False, "similarity": None}
        if mode == "expensive":
            res = expensive(text)
            return res.text, cfg.models.expensive, res.tokens_in, res.tokens_out, decision

        look = cache.lookup(text)
        if look.hit_idx is not None:
            hit = cache.get(look.hit_idx)
            decision.update(hit=True, kind=look.kind, similarity=look.similarity)
            return hit.answer_text, hit.model, 0, 0, decision

        if mode == "router":
            res = cheap(text)
            served = cfg.models.cheap
            if cascade.decide(text, res.text).escalate:
                first = res
                res = expensive(text)
                served = cfg.models.expensive
                # the cheap call was paid for too
                res = ChatResult(
                    res.text,
                    first.tokens_in + res.tokens_in,
                    first.tokens_out + res.tokens_out,
                    res.raw,
                )
        elif mode == "cheap":
            res, served = cheap(text), cfg.models.cheap
        else:  # cache
            res, served = expensive(text), cfg.models.expensive
        if res.text.strip():
            cache.add(CacheItem(next(item_ids), text, "live", res.text, served))
        return res.text, served, res.tokens_in, res.tokens_out, decision

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "cache_size": len(cache)}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError as exc:
            raise _bad_request("request body is not valid JSON") from exc
        text = _user_text(body)
        mode = request.headers.get("x-bench-mode", "router")
        if mode not in MODES:
            raise _bad_request(f"x-bench-mode must be one of {', '.join(MODES)}")
        started = time.perf_counter()
        content, served, tokens_in, tokens_out, decision = await run_in_threadpool(
            answer, mode, text
        )
        decision["served_by"] = served
        decision["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", cfg.models.expensive),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                },
                "x_gateway": decision,
            }
        )

    return app
