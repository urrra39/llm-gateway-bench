"""Metrics and validity gates, assembled into the one primary results file.

Validity gates are code, not prose:
- baseline must cost more than the cached and routed runs, or the measurement
  is broken;
- the false-hit rate is reported and never folded into the hit rate;
- the tuned threshold must beat a random (0.5) threshold on the tuning half.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lgb.config import Config
from lgb.judge import kappa_report
from lgb.run import CONFIGS, run_dir
from lgb.store import read_json, read_parquet, write_json


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        nan = float("nan")
        return {"p50": nan, "p95": nan, "p99": nan, "mean": nan}
    arr = np.asarray(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


def config_metrics(cfg: Config, config: str, frac: str) -> dict[str, Any]:
    out = read_parquet(run_dir(cfg, config, frac) / "outcomes.parquet")
    if out is None or not len(out):
        return {"config": config, "frac": frac, "present": False}
    judge = read_parquet(run_dir(cfg, config, frac) / "judge.parquet")
    rows = list(out.itertuples(index=False))
    errors = [r for r in rows if r.error]
    ok = [r for r in rows if not r.error]
    latencies = [float(r.latency_ms) for r in ok]
    # warm = every request except the first (first includes cold connection)
    warm = latencies[1:] if len(latencies) > 1 else []
    tokens_in = sum(int(r.tokens_in) for r in ok)
    tokens_out = sum(int(r.tokens_out) for r in ok)
    cost = float(sum(float(r.cost_usd) for r in ok))
    cheap_calls = sum(1 for r in ok if r.served_by == "cheap" and r.tokens_out > 0)
    exp_calls = sum(1 for r in ok if r.served_by == "expensive" and r.tokens_out > 0)
    exact_hits = sum(1 for r in ok if r.kind == "cache_exact")
    semantic_hits = sum(1 for r in ok if r.kind == "cache_semantic")
    misses = sum(1 for r in ok if r.kind in ("miss", "model_direct"))
    n = len(ok)

    metric = {
        "config": config,
        "frac": frac,
        "present": True,
        "n_rows": n,
        "n_errors": len(errors),
        "cost_usd": cost,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cheap_calls": cheap_calls,
        "expensive_calls": exp_calls,
        "cache_exact_hits": exact_hits,
        "cache_semantic_hits": semantic_hits,
        "cache_misses": misses,
        "hit_rate": round((exact_hits + semantic_hits) / n, 4) if n else 0.0,
        "first_latency_ms": latencies[0] if latencies else None,
        "latency": percentiles(warm) if warm else percentiles(latencies),
        "per_request_latency_note": "warm percentiles exclude the first request",
    }
    metric.update(_quality_block(judge, metric))
    return metric


def _quality_block(judge: pd.DataFrame | None, metric: dict[str, Any]) -> dict[str, Any]:
    """Equivalence vs baseline from judge rows. Rows identical to the baseline
    are auto-scored 2 (no judge call); everything else carries a judge score.
    False hits are semantic-cache hits the judge scored 0."""
    if judge is None or not len(judge):
        return {
            "judged_rows": 0,
            "quality_equiv": None,
            "false_hit_rate": None,
            "quality_by_dup": {},
        }
    rows = list(judge.itertuples(index=False))
    scores: list[float] = []
    false_hits: list[int] = []
    by_dup: dict[str, list[float]] = {}
    for r in rows:
        s = 2 if bool(r.identical) else r.judge1_score
        if s is None:
            continue
        scores.append(float(s))
        dt = str(r.dup_type)
        by_dup.setdefault(dt, []).append(float(s))
        if dt in ("paraphrase", "trap") and s == 0 and r.answer_text != r.baseline_text:
            false_hits.append(int(r.idx))
    semantic_hits = metric["cache_semantic_hits"]
    quality = {
        "judged_rows": len(scores),
        "quality_equiv": round(float(np.mean(scores)) / 2.0, 4) if scores else None,
        "mean_score": round(float(np.mean(scores)), 4) if scores else None,
        "false_hit_rate": (round(len(false_hits) / semantic_hits, 4) if semantic_hits else None),
        "false_hit_rows": false_hits,
        "quality_by_dup": {
            dt: round(float(np.mean(v)) / 2.0, 4) for dt, v in sorted(by_dup.items())
        },
    }
    return quality


def assemble(cfg: Config, run_name: str, note: str = "") -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for frac_cfg in cfg.workload.duplicate_fractions:
        frac = frac_cfg.name
        for config in CONFIGS:
            key = f"{frac}_{config}"
            metrics[key] = config_metrics(cfg, config, frac)

    baseline_cost = {
        f.name: metrics[f"{f.name}_baseline"].get("cost_usd")
        for f in cfg.workload.duplicate_fractions
    }
    gates = []
    for f in cfg.workload.duplicate_fractions:
        frac = f.name
        for config in ("cache", "router_cascade", "router_heuristic"):
            m = metrics[f"{frac}_{config}"]
            gates.append(
                {
                    "name": f"baseline_costs_more_{frac}_{config}",
                    "passed": m["present"] and m["cost_usd"] < baseline_cost[frac],
                    "observed": (
                        f"baseline {baseline_cost[frac]:.4f} vs {config} {m['cost_usd']:.4f}"
                    ),
                }
            )
        cm = metrics[f"{frac}_cache"]
        gates.append(
            {
                "name": f"false_hit_rate_reported_{frac}",
                "passed": cm["present"] and cm["false_hit_rate"] is not None,
                "observed": str(cm["false_hit_rate"]),
            }
        )
    tuning = read_json(cfg.data.runs_dir / "tuning.json") or {}
    gates.append(
        {
            "name": "tuned_threshold_beats_random_control",
            "passed": bool(tuning.get("tuned_beats_control")),
            "observed": (
                f"tuned {tuning.get('threshold')} f1 {tuning.get('tuned_f1')} "
                f"vs control {tuning.get('control_f1')}"
            ),
        }
    )
    kappa = kappa_report(cfg)
    payload = {
        "run_name": run_name,
        "note": note,
        "gates": gates,
        "all_gates_passed": all(g["passed"] for g in gates),
        "metrics": metrics,
        "tuning": tuning,
        "kappa": kappa,
        "environment": {
            "models": {
                "cheap": cfg.models.cheap,
                "expensive": cfg.models.expensive,
                "judge_primary": cfg.models.judge_primary,
                "judge_secondary": cfg.models.judge_secondary,
            },
            "prices": {
                "as_of": cfg.prices.as_of,
                "basis": cfg.prices.basis,
                "per_million_tokens": {
                    m: dict(p.model_dump()) for m, p in cfg.prices.per_million_tokens.items()
                },
            },
            "gateway_base_url": cfg.gateway.base_url,
            "concurrency": cfg.gateway.concurrency,
            "cache_threshold": cfg.cache.sim_threshold,
        },
    }
    # archive the previous primary results, keep exactly one current
    primary = cfg.data.primary_results
    if primary.exists():
        import shutil

        archive_dir = cfg.data.archive_dir
        archive_dir.mkdir(parents=True, exist_ok=True)
        old = read_json(primary) or {}
        old_name = old.get("run_name", "previous")
        dest = archive_dir / f"{old_name}.json"
        shutil.copy2(primary, dest)
        note_file = archive_dir / f"{old_name}.note"
        note_file.write_text(f"Superseded by {run_name}. Archived {_now()}.\n", encoding="utf-8")
    write_json(payload, primary)
    return payload


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat(timespec="minutes")
