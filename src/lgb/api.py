"""The FastAPI gateway exposing an OpenAI-compatible /v1/chat/completions.

Live mode reuses the same components the benchmark measures: semantic cache
and cascade routing, with per-request instrumentation returned in the response
headers and logged to data/live/.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request

from lgb.cache import CacheItem, SemanticCache
from lgb.chat import Gateway
from lgb.config import Config
from lgb.router import CascadeRouter


def build_app(cfg: Config) -> FastAPI:
    from fastapi.responses import JSONResponse

    app = FastAPI(title="llm-gateway-bench", version="0.1.0")
    gw = Gateway(cfg)
    cache = SemanticCache(cfg.cache.sim_threshold, None, exact=True)
    cascade = CascadeRouter(cfg.router)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "cache_size": len(cache)}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        body = await request.json()
        messages = body.get("messages", [])
        user_text = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        ).strip()
        mode = request.headers.get("x-bench-mode", "router")  # router | cache | expensive
        model_name = body.get("model", cfg.models.expensive)
        started = time.perf_counter()
        decision: dict[str, object] = {"mode": mode, "hit": False, "similarity": None}

        if mode != "expensive":
            look = cache.lookup(user_text)
            if look.kind in ("cache_exact", "cache_semantic"):
                hit = next(i for i in cache.items if i.idx == look.hit_idx)
                decision = {
                    "mode": mode,
                    "hit": True,
                    "kind": look.kind,
                    "similarity": look.similarity,
                }
                answer = hit.answer_text
                served = hit.model
                tokens_in = tokens_out = 0
            else:
                if mode == "router":
                    cheap_res = gw.chat(
                        cfg.models.cheap, user_text, system_prompt=cfg.generation.system_prompt
                    )
                    cheap_text = cheap_res.text
                    dec = cascade.decide(user_text, cheap_text)
                    if dec.escalate:
                        res = gw.chat(
                            cfg.models.expensive,
                            user_text,
                            system_prompt=cfg.generation.system_prompt,
                        )
                        served = cfg.models.expensive
                    else:
                        res = cheap_res
                        served = cfg.models.cheap
                elif mode == "cheap":
                    res = gw.chat(
                        cfg.models.cheap, user_text, system_prompt=cfg.generation.system_prompt
                    )
                    served = cfg.models.cheap
                else:  # cache
                    res = gw.chat(
                        cfg.models.expensive, user_text, system_prompt=cfg.generation.system_prompt
                    )
                    served = cfg.models.expensive
                answer, tokens_in, tokens_out = res.text, res.tokens_in, res.tokens_out
                if answer.strip():
                    cache.add(CacheItem(len(cache), user_text, "live", answer, served))
        else:
            res = gw.chat(
                cfg.models.expensive, user_text, system_prompt=cfg.generation.system_prompt
            )
            answer, tokens_in, tokens_out = res.text, res.tokens_in, res.tokens_out
            served = cfg.models.expensive

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        decision["served_by"] = served
        decision["latency_ms"] = round(elapsed_ms, 1)
        payload = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
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
        return JSONResponse(payload)

    return app
