"""The benchmark engine: workload, threshold tuning, execution, judging,
metrics and validity gates.

All stages are resumable: derived tables are parquet written atomically and a
stage processes only the rows that are not yet done.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from lgb import records
from lgb.cache import (
    CacheItem,
    SemanticCache,
    choose_threshold,
    evaluate_threshold,
)
from lgb.chat import Gateway, price_for
from lgb.config import Config
from lgb.embed import Embedder
from lgb.router import CascadeRouter, HeuristicRouter, route_for
from lgb.store import append_rows, read_json, read_parquet, write_json, write_parquet
from lgb.workload import build_workload, load_pairs, split_tune_report

REPO = Path(__file__).resolve().parents[2]
SAMPLE_PATH = REPO / "data" / "workload" / "paws_sample.csv"
CONFIGS = ("baseline", "cache", "router_cascade", "router_heuristic")
HEADLINE_CONFIGS = ("baseline", "cache", "router_cascade")


# ---------------------------------------------------------------- workload


def _workload_dir(cfg: Config) -> Path:
    return cfg.data.runs_dir / "workloads"


def build_all_workloads(cfg: Config) -> dict:
    pairs = load_pairs(SAMPLE_PATH)
    out = _workload_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for frac in cfg.workload.duplicate_fractions:
        rng = np.random.default_rng(cfg.workload.seed + sum(ord(c) for c in frac.name))
        rows = build_workload(cfg, frac, pairs, rng)
        write_parquet(rows_to_frame(rows), out / f"workload_{frac.name}.parquet")
    meta = {
        "source": "PAWS (google-research-datasets/paws), labeled_final/train",
        "sample_rows": len(pairs),
        "n_per_fraction": cfg.workload.n_per_fraction,
        "request_template": cfg.workload.request_template,
        "fractions": [
            {
                "name": f.name,
                "exact": f.exact,
                "paraphrase": f.paraphrase,
                "trap": f.trap,
                "novel": f.novel,
            }
            for f in cfg.workload.duplicate_fractions
        ],
        "seed": cfg.workload.seed,
    }
    write_json(meta, out / "meta.json")
    return meta


def rows_to_frame(rows: list[records.WorkloadRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "idx": r.idx,
                "request": r.request,
                "dup_type": r.dup_type,
                "group_id": r.group_id,
            }
            for r in rows
        ]
    )


def load_workload(cfg: Config, frac: str) -> list[records.WorkloadRow]:
    frame = read_parquet(_workload_dir(cfg) / f"workload_{frac}.parquet")
    if frame is None:
        raise FileNotFoundError(f"workload {frac} not built; run `lgb workload` first")
    rows: list[records.WorkloadRow] = []
    for r in frame.itertuples(index=False):
        rows.append(
            records.WorkloadRow(
                idx=int(r.idx),
                request=str(r.request),
                dup_type=str(r.dup_type),  # type: ignore[arg-type]
                group_id=str(r.group_id),
            )
        )
    return rows


# ---------------------------------------------------------------- tuning


def _f1(stats: dict[str, float]) -> float:
    denom = stats["precision"] + stats["recall"]
    return 2 * stats["precision"] * stats["recall"] / denom if denom else 0.0


def tune_threshold(cfg: Config, embedder: Embedder) -> dict:
    """Choose the cache threshold on the tuning half of the workloads' labelled
    paraphrase/trap pairs and assert it beats a random 0.5 threshold."""
    positives: list[float] = []
    negatives: list[float] = []
    for frac in cfg.workload.duplicate_fractions:
        rows = load_workload(cfg, frac.name)
        tune, _ = split_tune_report(rows, cfg.cache.tune_fraction, cfg.cache.tune_seed)
        pos, neg = _labelled_similarities(embedder, tune)
        positives.extend(pos)
        negatives.extend(neg)
    pos_arr = np.asarray(positives, dtype=float)
    neg_arr = np.asarray(negatives, dtype=float)
    threshold, stats = choose_threshold(pos_arr, neg_arr)
    control = evaluate_threshold(pos_arr, neg_arr, 0.5)
    record = {
        "threshold": threshold,
        "tune_stats": stats,
        "control_stats": control,
        "tuned_f1": _f1(stats),
        "control_f1": _f1(control),
        "tuned_beats_control": _f1(stats) >= _f1(control),
        "n_positives": len(pos_arr),
        "n_negatives": len(neg_arr),
    }
    write_json(record, cfg.data.runs_dir / "tuning.json")
    return record


def _labelled_similarities(
    embedder: Embedder, rows: list[records.WorkloadRow]
) -> tuple[list[float], list[float]]:
    groups: dict[str, dict] = {}
    for r in rows:
        g = groups.setdefault(r.group_id, {"anchor": None, "follows": []})
        if r.dup_type == "novel":
            g["anchor"] = r.request
        else:
            g["follows"].append(r)
    pos: list[float] = []
    neg: list[float] = []
    cache: dict[str, np.ndarray] = {}

    def vec(text: str) -> np.ndarray:
        if text not in cache:
            cache[text] = embedder.encode([text])[0]
        return cache[text]

    pool = [g["anchor"] for g in groups.values() if g["anchor"]]
    for g in groups.values():
        anchor = g["anchor"]
        if anchor is None:
            continue
        va = vec(anchor)
        for f in g["follows"]:
            sim = float(va @ vec(f.request))
            if f.dup_type == "paraphrase":
                pos.append(sim)
            elif f.dup_type == "trap":
                neg.append(sim)
    if len(pool) > 1:
        rng = np.random.default_rng(20260908)
        for _ in range(max(len(pos), 40)):
            a = pool[rng.integers(0, len(pool))]
            b = pool[rng.integers(0, len(pool))]
            if a != b:
                neg.append(float(vec(a) @ vec(b)))
    return pos, neg


# ---------------------------------------------------------------- execution


def _outcome(
    config: str, frac: str, r: records.WorkloadRow, **fields: object
) -> records.DecisionRow:
    base: dict[str, object] = {
        "config": config,
        "frac": frac,
        "idx": r.idx,
        "request": r.request,
        "dup_type": r.dup_type,
        "group_id": r.group_id,
        "kind": "miss",
        "served_by": "expensive",
        "similarity": None,
        "hit_source_idx": None,
        "latency_ms": 0.0,
        "embed_ms": 0.0,
        "lookup_ms": 0.0,
        "model_ms": 0.0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "answer_text": "",
        "error": None,
    }
    base.update(fields)
    return records.DecisionRow(**base)


def _chat_nonempty(
    gw: Gateway, model: str, request: str, cfg: Config
) -> tuple[str, int, int, float]:
    res = gw.chat(model, request, system_prompt=cfg.generation.system_prompt)
    if not res.text.strip():
        res = gw.chat(
            model,
            request,
            system_prompt=cfg.generation.system_prompt,
            max_tokens=cfg.generation.max_tokens * 2,
        )
    return res.text, res.tokens_in, res.tokens_out, res.latency_s


def _outcome_row(o: records.DecisionRow) -> dict:
    return {
        "config": o.config,
        "frac": o.frac,
        "idx": o.idx,
        "request": o.request,
        "dup_type": o.dup_type,
        "group_id": o.group_id,
        "kind": o.kind,
        "served_by": o.served_by,
        "similarity": o.similarity,
        "hit_source_idx": o.hit_source_idx,
        "latency_ms": o.latency_ms,
        "embed_ms": o.embed_ms,
        "lookup_ms": o.lookup_ms,
        "model_ms": o.model_ms,
        "tokens_in": o.tokens_in,
        "tokens_out": o.tokens_out,
        "cost_usd": o.cost_usd,
        "answer_text": o.answer_text,
        "error": o.error,
    }


def execute_config(
    cfg: Config,
    config: str,
    frac: str,
    threshold: float,
    embedder: Embedder,
    run_dir: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    rows = load_workload(cfg, frac)
    if limit:
        rows = rows[:limit]
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "cache_items.parquet"
    outcomes_path = run_dir / "outcomes.parquet"
    attempts_path = run_dir / "attempts.json"
    done: set[str] = set()
    existing = read_parquet(outcomes_path)
    if existing is not None and len(existing):
        done = {f"{r.config}|{r.frac}|{r.idx}" for r in existing.itertuples(index=False)}
    attempts = read_json(attempts_path) or {}
    gw = Gateway(cfg)
    cheap = cfg.models.cheap
    expensive = cfg.models.expensive
    cache = SemanticCache.load(cache_path, threshold, embedder, exact=cfg.cache.exact_match)
    heuristic = route_for(config, cfg.router)
    cascade = CascadeRouter(cfg.router) if config == "router_cascade" else None

    for r in rows:
        key = f"{config}|{frac}|{r.idx}"
        if key in done:
            continue
        if int(attempts.get(key, 0)) >= cfg.gateway.max_retries:
            append_rows(
                pd.DataFrame(
                    [_outcome_row(_outcome(config, frac, r, error="gave up after retries"))]
                ),
                outcomes_path,
            )
            attempts[key] = 0
            write_json(attempts, attempts_path)
            continue
        try:
            out = _run_one(cfg, gw, cache, heuristic, cascade, config, frac, r, cheap, expensive)
        except Exception:
            attempts[key] = int(attempts.get(key, 0)) + 1
            write_json(attempts, attempts_path)
            raise
        if (
            config in ("cache", "router_cascade", "router_heuristic")
            and out.kind in ("model_direct", "miss")
            and out.answer_text.strip()
            and out.error is None
        ):
            cache.add(
                CacheItem(
                    idx=r.idx,
                    request=r.request,
                    group_id=r.group_id,
                    answer_text=out.answer_text,
                    model=out.served_by,
                )
            )
        append_rows(pd.DataFrame([_outcome_row(out)]), outcomes_path)
        cache.save(cache_path)
        attempts[key] = 0
    final = read_parquet(outcomes_path)
    assert final is not None
    return final.sort_values("idx").reset_index(drop=True)


def _run_one(
    cfg: Config,
    gw: Gateway,
    cache: SemanticCache,
    heuristic,
    cascade,
    config: str,
    frac: str,
    r: records.WorkloadRow,
    cheap: str,
    expensive: str,
) -> records.DecisionRow:
    t0 = time.perf_counter()

    def ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0

    if config == "baseline":
        text, ti, to, _ = _chat_nonempty(gw, expensive, r.request, cfg)
        return _outcome(
            config,
            frac,
            r,
            kind="model_direct",
            served_by="expensive",
            latency_ms=ms(t0),
            model_ms=ms(t0),
            tokens_in=ti,
            tokens_out=to,
            cost_usd=price_for(cfg, expensive).cost_usd(ti, to),
            answer_text=text,
        )

    look = cache.lookup(r.request)
    embed_ms = cache.last_embed_ms
    lookup_ms = cache.last_lookup_ms
    if look.kind in ("cache_exact", "cache_semantic"):
        hit = next(i for i in cache.items if i.idx == look.hit_idx)
        return _outcome(
            config,
            frac,
            r,
            kind=look.kind,
            served_by=hit.model,
            similarity=look.similarity,
            hit_source_idx=look.hit_idx,
            latency_ms=ms(t0),
            embed_ms=embed_ms,
            lookup_ms=lookup_ms,
            answer_text=hit.answer_text,
        )

    if config == "cache":
        text, ti, to, _ = _chat_nonempty(gw, expensive, r.request, cfg)
        return _outcome(
            config,
            frac,
            r,
            kind="miss",
            served_by="expensive",
            latency_ms=ms(t0),
            embed_ms=embed_ms,
            lookup_ms=lookup_ms,
            model_ms=ms(t0),
            tokens_in=ti,
            tokens_out=to,
            cost_usd=price_for(cfg, expensive).cost_usd(ti, to),
            answer_text=text,
        )

    if config == "router_heuristic":
        assert isinstance(heuristic, HeuristicRouter)
        dec = heuristic.decide(r.request)
        model = cheap if dec.route == "easy" else expensive
        served = "cheap" if dec.route == "easy" else "expensive"
        text, ti, to, _ = _chat_nonempty(gw, model, r.request, cfg)
        return _outcome(
            config,
            frac,
            r,
            kind="miss",
            served_by=served,
            latency_ms=ms(t0),
            embed_ms=embed_ms,
            lookup_ms=lookup_ms,
            model_ms=ms(t0),
            tokens_in=ti,
            tokens_out=to,
            cost_usd=price_for(cfg, model).cost_usd(ti, to),
            answer_text=text,
        )

    if config == "router_cascade":
        assert cascade is not None
        cheap_text, cti, cto, _ = _chat_nonempty(gw, cheap, r.request, cfg)
        model_ms = ms(t0)
        dec = cascade.decide(r.request, cheap_text)
        cost = price_for(cfg, cheap).cost_usd(cti, cto)
        if dec.escalate:
            exp_text, eti, eto, _ = _chat_nonempty(gw, expensive, r.request, cfg)
            text, ti, to = exp_text, cti + eti, cto + eto
            served, model_ms = "expensive", model_ms + ms(t0)
            cost += price_for(cfg, expensive).cost_usd(eti, eto)
        else:
            text, ti, to = cheap_text, cti, cto
            served = "cheap"
        return _outcome(
            config,
            frac,
            r,
            kind="miss",
            served_by=served,
            latency_ms=ms(t0),
            embed_ms=embed_ms,
            lookup_ms=lookup_ms,
            model_ms=model_ms,
            tokens_in=ti,
            tokens_out=to,
            cost_usd=cost,
            answer_text=text,
        )

    raise ValueError(config)  # pragma: no cover


def run_dir(cfg: Config, config: str, frac: str) -> Path:
    return cfg.data.runs_dir / f"{frac}_{config}"
