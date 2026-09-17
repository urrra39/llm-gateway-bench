"""The benchmark engine: workload, threshold tuning, execution, judging,
metrics and validity gates.

All stages are resumable: derived tables are parquet written atomically and a
stage processes only the rows that are not yet done.

Execution is concurrent but order-preserving for the cache. The workload's
meaningful structure is "an anchor request precedes its paraphrase / trap /
exact repeat", and a paraphrase must observe the anchor's stored answer to be
a hit. A naive thread pool would start a paraphrase before its anchor finished
and turn intended hits into misses, making the hit rate depend on thread
scheduling. So rows are dispatched only when their dependencies (the group's
anchor row, always an earlier novel row) have completed; independent groups
run in parallel. Cache reads, writes and the outcomes append are serialised by
one lock.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lgb import records
from lgb.cache import (
    CacheItem,
    SemanticCache,
    normalize_exact,
    simulate_threshold,
)
from lgb.chat import Gateway, price_for
from lgb.config import Config
from lgb.embed import Embedder
from lgb.router import CascadeRouter, HeuristicRouter
from lgb.store import append_rows, read_parquet, write_json, write_parquet
from lgb.workload import build_workload, load_pairs, split_tune_report

REPO = Path(__file__).resolve().parents[2]
SAMPLE_PATH = REPO / "data" / "workload" / "paws_sample.csv"
CONFIGS = ("baseline", "cache", "router_cascade", "router_heuristic")
HEADLINE_CONFIGS = ("baseline", "cache", "router_cascade")


# ---------------------------------------------------------------- workload


def _workload_dir(cfg: Config) -> Path:
    return cfg.data.runs_dir / "workloads"


def build_all_workloads(cfg: Config) -> dict[str, Any]:
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


def tune_threshold(cfg: Config, embedder: Embedder) -> dict[str, Any]:
    """Choose the cache threshold on the tuning half and compare it against a
    random-threshold control.

    The decision scored here is the one the cache actually makes: maximum
    cosine similarity against everything stored so far, not the similarity of
    one labelled pair. Tuning on isolated pairs picks a threshold that looks
    excellent and then false-hits constantly once the cache is full, because
    the maximum over a large store is much higher than a typical pair.
    """
    per_frac: dict[str, Any] = {}
    vec_parts: list[np.ndarray] = []
    hit_parts: list[np.ndarray] = []
    elig_parts: list[np.ndarray] = []
    for frac in cfg.workload.duplicate_fractions:
        rows = load_workload(cfg, frac.name)
        tune, _ = split_tune_report(rows, cfg.cache.tune_fraction, cfg.cache.tune_seed)
        vecs, should_hit, eligible = _replay_arrays(embedder, tune)
        per_frac[frac.name] = {
            "n_rows": int(len(tune)),
            "n_should_hit": int(should_hit.sum()),
            "n_eligible": int(eligible.sum()),
        }
        vec_parts.append(vecs)
        hit_parts.append(should_hit)
        elig_parts.append(eligible)

    grid = np.round(np.arange(0.30, 0.999, 0.005), 4)
    # Score every fraction's replay and sum the confusion counts, so the chosen
    # threshold is not tuned to one duplicate fraction.
    def scored(thr: float) -> dict[str, float]:
        tp = fp = fn = tn = 0.0
        for vecs, hits, elig in zip(vec_parts, hit_parts, elig_parts, strict=True):
            s = simulate_threshold(vecs, hits, elig, thr)
            tp += s["tp"]
            fp += s["fp"]
            fn += s["fn"]
            tn += s["tn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denom = precision + recall
        return {
            "threshold": thr,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / denom if denom else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "false_hit_rate": fp / (tp + fp) if tp + fp else 0.0,
        }

    sweep = [scored(float(t)) for t in grid]
    best = max(sweep, key=lambda s: (s["f1"], s["threshold"]))

    # Random-threshold control: thresholds drawn uniformly from the same grid.
    # The tuned point must beat their mean, or the tuning did nothing.
    rng = np.random.default_rng(cfg.cache.tune_seed)
    draws = rng.uniform(float(grid.min()), float(grid.max()), size=64)
    control_f1s = [scored(float(t))["f1"] for t in draws]
    control_mean = float(np.mean(control_f1s))
    control_max = float(np.max(control_f1s))

    record: dict[str, Any] = {
        "method": (
            "replay simulation: max cosine over the growing cache, scored against "
            "the workload's PAWS-derived should-hit labels on the tuning half"
        ),
        "threshold": float(best["threshold"]),
        "tune_stats": {k: float(v) for k, v in best.items()},
        "tuned_f1": float(best["f1"]),
        "control_random_mean_f1": control_mean,
        "control_random_max_f1": control_max,
        "control_n_draws": int(len(draws)),
        "tuned_beats_control": bool(best["f1"] > control_mean),
        "per_fraction": per_frac,
        "grid": {"start": 0.30, "stop": 0.999, "step": 0.005, "n": int(len(grid))},
        "sweep_file": "threshold_sweep.parquet",
    }
    # The sweep is a table, not a config value: keeping ~140 rows of it inside
    # tuning.json made the record unreadable and invited drift between the
    # chosen operating point and the curve it came from.
    write_parquet(pd.DataFrame(sweep), cfg.data.runs_dir / "threshold_sweep.parquet")
    write_json(record, cfg.data.runs_dir / "tuning.json")
    return record


def _replay_arrays(
    embedder: Embedder, rows: list[records.WorkloadRow]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings in workload order plus the should-hit and eligibility masks.

    should_hit is True for a paraphrase or an exact repeat, because an earlier
    equivalent request exists and the cache ought to serve it. It is False for
    a novel request and for a trap (PAWS label 0: lexically close, different
    meaning), both of which must miss.

    eligible is False for exact repeats: the exact-match short-circuit serves
    them before any embedding, so they never exercise the similarity threshold
    and would otherwise inflate its apparent recall.
    """
    texts = [r.request for r in rows]
    vecs = embedder.encode(texts, batch_size=64)
    seen: set[str] = set()
    should_hit = np.zeros(len(rows), dtype=bool)
    eligible = np.ones(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        key = normalize_exact(r.request)
        if key in seen:
            eligible[i] = False  # exact short-circuit, never reaches the threshold
        elif r.dup_type == "paraphrase":
            should_hit[i] = True
        seen.add(key)
    return vecs, should_hit, eligible


# ---------------------------------------------------------------- execution


def _outcome(config: str, frac: str, r: records.WorkloadRow, **fields: Any) -> records.DecisionRow:
    base: dict[str, Any] = {
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


def _outcome_row(o: records.DecisionRow) -> dict[str, Any]:
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


def _anchor_dependencies(rows: list[records.WorkloadRow]) -> dict[int, int | None]:
    """Map each row's idx to the idx of the row it must follow, or None for a
    novel row. Every paraphrase/trap/exact has exactly one anchor: the novel
    row of its group, which the workload builder guarantees appears earlier."""
    novel_by_group: dict[str, int] = {}
    for r in rows:
        if r.dup_type == "novel":
            novel_by_group[r.group_id] = r.idx
    deps: dict[int, int | None] = {}
    for r in rows:
        if r.dup_type == "novel":
            deps[r.idx] = None
        else:
            deps[r.idx] = novel_by_group.get(r.group_id)
    return deps


def _schedule(rows: list[records.WorkloadRow]) -> list[records.WorkloadRow]:
    """A dispatch order that keeps every anchor ahead of its followers but
    otherwise preserves workload order. Used only for diagnostics; the executor
    dispatches by dependency completion."""
    deps = _anchor_dependencies(rows)
    placed: set[int] = set()
    out: list[records.WorkloadRow] = []
    remaining = list(rows)
    while remaining:
        for r in list(remaining):
            dep = deps[r.idx]
            if dep is None or dep in placed:
                out.append(r)
                placed.add(r.idx)
                remaining.remove(r)
        if not out or len(placed) == len(rows):
            break
    return out


def execute_config(
    cfg: Config,
    config: str,
    frac: str,
    threshold: float,
    embedder: Embedder,
    run_dir: Path,
    limit: int | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    all_rows = load_workload(cfg, frac)
    _tune_rows, rows = split_tune_report(all_rows, cfg.cache.tune_fraction, cfg.cache.tune_seed)
    if limit:
        rows = rows[:limit]
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "cache_items.parquet"
    outcomes_path = run_dir / "outcomes.parquet"
    done: set[str] = set()
    existing = read_parquet(outcomes_path)
    if existing is not None and len(existing):
        done = {f"{r.config}|{r.frac}|{r.idx}" for r in existing.itertuples(index=False)}
    gw = Gateway(cfg)
    cheap = cfg.models.cheap
    expensive = cfg.models.expensive
    cache = SemanticCache.load(cache_path, threshold, embedder, exact=cfg.cache.exact_match)
    heuristic = HeuristicRouter(cfg.router) if config == "router_heuristic" else None
    cascade = CascadeRouter(cfg.router) if config == "router_cascade" else None
    n_workers = workers if workers is not None else cfg.gateway.concurrency
    lock = threading.Lock()
    pending = [r for r in rows if f"{config}|{frac}|{r.idx}" not in done]
    deps = _anchor_dependencies(pending)
    completed: set[int] = set()

    def process(r: records.WorkloadRow) -> None:
        out: records.DecisionRow | None = None
        last_error = ""
        for attempt in range(cfg.gateway.max_retries + 1):
            try:
                # The model call runs without the executor lock; the cache guards
                # its own reads/writes internally.
                out = _run_one(
                    cfg, gw, cache, heuristic, cascade, config, frac, r, cheap, expensive
                )
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                if "content-blocked" in str(exc) or attempt >= cfg.gateway.max_retries:
                    break
                time.sleep(2.0 * (attempt + 1))
        if out is None:
            out = _outcome(config, frac, r, error=last_error)
        with lock:
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
            completed.add(r.idx)

    def next_ready() -> records.WorkloadRow | None:
        with lock:
            for r in pending:
                if r.idx in completed:
                    continue
                dep = deps[r.idx]
                if dep is None or dep in completed:
                    pending.remove(r)
                    return r
        return None

    if n_workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            while True:
                r = next_ready()
                if r is None:
                    if all(p.idx in completed for p in pending) or not pending:
                        break
                    time.sleep(0.05)
                    continue
                futures.append(pool.submit(process, r))
            for f in futures:
                f.result()
    else:
        for r in pending:
            process(r)
    final = read_parquet(outcomes_path)
    assert final is not None
    return final.sort_values("idx").reset_index(drop=True)


def _run_one(
    cfg: Config,
    gw: Gateway,
    cache: SemanticCache,
    heuristic: HeuristicRouter | None,
    cascade: CascadeRouter | None,
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
