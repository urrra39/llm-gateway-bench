"""Metrics and validity gates, assembled into the one primary results file.

Validity gates are code, not prose:
- baseline must cost more than the cached and routed runs, or the measurement
  is broken;
- the false-hit rate is reported and never folded into the hit rate;
- the tuned threshold must beat the mean of 64 random thresholds drawn from
  the tuning grid, or the tuning did nothing;
- human_label_coverage and judge_independence record the label-quality state
  honestly: both fail until a human verifies labels and the judges differ.
  A gate set that passes under zero human verification is not a gate set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lgb.cache import normalize_exact
from lgb.config import Config
from lgb.intervals import (
    BOOTSTRAP_B,
    INTERVAL_SEED,
    bootstrap_ci,
    bootstrap_percentiles,
    derive_seed,
    wilson,
)
from lgb.judge import kappa_report
from lgb.run import CONFIGS, run_dir
from lgb.store import read_json, read_parquet, write_json


def _has_error(value: Any) -> bool:
    """True when an outcome row carries a real error string.

    Parquet restores a null `error` as NaN, and `bool(nan)` is True, so a plain
    truthiness test silently reclassifies every successful row as an error and
    drops its cost and latency from the metrics. Missing means no error.
    """
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return bool(str(value).strip()) and str(value).strip().lower() != "nan"


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
    errors = [r for r in rows if _has_error(r.error)]
    ok = [r for r in rows if not _has_error(r.error)]
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

    cost_seed = derive_seed(INTERVAL_SEED, frac, config, "cost")
    cost_lo, cost_hi, cost_sha = bootstrap_ci(
        [float(r.cost_usd) for r in ok], cost_seed, stat=lambda v: float(np.sum(v))
    )
    lat_seed = derive_seed(INTERVAL_SEED, frac, config, "latency")
    lat_cis, lat_sha = bootstrap_percentiles(warm if warm else latencies, lat_seed)

    metric = {
        "config": config,
        "frac": frac,
        "present": True,
        "n_rows": n,
        "n_errors": len(errors),
        "cost_usd": cost,
        "cost_ci": [round(cost_lo, 4), round(cost_hi, 4)],
        "cost_seed": cost_seed,
        "cost_idx_sha": cost_sha,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cheap_calls": cheap_calls,
        "expensive_calls": exp_calls,
        "cache_exact_hits": exact_hits,
        "cache_semantic_hits": semantic_hits,
        "cache_misses": misses,
        "hit_rate": round((exact_hits + semantic_hits) / n, 4) if n else 0.0,
        "hit_rate_ci": [round(v, 4) for v in wilson(exact_hits + semantic_hits, n)],
        "first_latency_ms": latencies[0] if latencies else None,
        "latency": percentiles(warm) if warm else percentiles(latencies),
        "latency_ci": {
            "p50": [round(lat_cis[50.0][0], 1), round(lat_cis[50.0][1], 1)],
            "p95": [round(lat_cis[95.0][0], 1), round(lat_cis[95.0][1], 1)],
            "p99": [round(lat_cis[99.0][0], 1), round(lat_cis[99.0][1], 1)],
        },
        "latency_seed": lat_seed,
        "latency_idx_sha": lat_sha,
        "per_request_latency_note": "warm percentiles exclude the first request",
    }
    metric.update(_quality_block(judge, metric))
    return metric


def exact_only_metrics(cfg: Config, frac: str) -> dict[str, Any]:
    """Replay an exact-match-only cache over the baseline's stored rows.

    Deterministic arithmetic over committed parquet, no model calls: a request
    is served from store iff its normalized text matched a previous successful
    row verbatim; every other row inherits exactly what the baseline paid on
    that same row. Hit latency is the measured median cache_exact latency from
    the cache run on the same fraction, labelled replayed, not measured.

    Quality cannot be judged without a judge call: the replay serves the
    baseline run's anchor text, while the cache run judged different stored
    texts (temperature-0 repeats still differ). quality_equiv is therefore
    None, with the cache run's exact-hit verdicts reported as the closest
    observed proxy, explicitly not a measurement of this replay.
    """
    base = read_parquet(run_dir(cfg, "baseline", frac) / "outcomes.parquet")
    cache_out = read_parquet(run_dir(cfg, "cache", frac) / "outcomes.parquet")
    cache_judge = read_parquet(run_dir(cfg, "cache", frac) / "judge.parquet")
    if base is None or not len(base) or cache_out is None:
        return {"config": "exact_only", "frac": frac, "present": False}
    nominal_hit_ms = float(cache_out.loc[cache_out["kind"] == "cache_exact", "latency_ms"].median())
    rows = list(base.sort_values("idx").itertuples(index=False))
    stored: dict[str, bool] = {}
    n_errors = 0
    hits = 0
    costs: list[float] = []
    latencies: list[float] = []
    tokens_in = 0
    tokens_out = 0
    for r in rows:
        if _has_error(r.error) or not str(r.answer_text).strip():
            n_errors += 1
            continue
        key = normalize_exact(str(r.request))
        if key in stored:
            hits += 1
            costs.append(0.0)
            latencies.append(nominal_hit_ms)
        else:
            stored[key] = True
            costs.append(float(r.cost_usd))
            latencies.append(float(r.latency_ms))
            tokens_in += int(r.tokens_in)
            tokens_out += int(r.tokens_out)
    n = len(costs)
    warm = latencies[1:] if len(latencies) > 1 else []
    cost_seed = derive_seed(INTERVAL_SEED, frac, "exact_only", "cost")
    cost_lo, cost_hi, cost_sha = bootstrap_ci(costs, cost_seed, stat=lambda v: float(np.sum(v)))
    lat_seed = derive_seed(INTERVAL_SEED, frac, "exact_only", "latency")
    lat_cis, lat_sha = bootstrap_percentiles(warm if warm else latencies, lat_seed)
    proxy = _exact_proxy(cache_out, cache_judge)
    return {
        "config": "exact_only",
        "frac": frac,
        "present": True,
        "replay": True,
        "replay_note": (
            "hits and misses replayed from baseline rows in idx order; miss "
            "cost/latency inherited from the baseline row, hit latency is the "
            f"measured median cache_exact latency ({nominal_hit_ms:.2f} ms): "
            "replayed, not measured; fresh upstream latency variance not captured"
        ),
        "n_rows": n,
        "n_errors": n_errors,
        "cost_usd": round(float(np.sum(costs)), 4),
        "cost_ci": [round(cost_lo, 4), round(cost_hi, 4)],
        "cost_seed": cost_seed,
        "cost_idx_sha": cost_sha,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cheap_calls": 0,
        "expensive_calls": n - hits,
        "cache_exact_hits": hits,
        "cache_semantic_hits": 0,
        "cache_misses": n - hits,
        "hit_rate": round(hits / n, 4) if n else 0.0,
        "hit_rate_ci": [round(v, 4) for v in wilson(hits, n)],
        "first_latency_ms": latencies[0] if latencies else None,
        "latency": percentiles(warm) if warm else percentiles(latencies),
        "latency_ci": {
            "p50": [round(lat_cis[50.0][0], 1), round(lat_cis[50.0][1], 1)],
            "p95": [round(lat_cis[95.0][0], 1), round(lat_cis[95.0][1], 1)],
            "p99": [round(lat_cis[99.0][0], 1), round(lat_cis[99.0][1], 1)],
        },
        "latency_seed": lat_seed,
        "latency_idx_sha": lat_sha,
        "per_request_latency_note": "warm percentiles exclude the first request",
        "judged_rows": 0,
        "quality_equiv": None,
        "quality_ci": None,
        "quality_note": (
            "unmeasured: judging the replay's stored texts would need new judge "
            "calls; the proxy below judged different stored texts"
        ),
        "mean_score": None,
        "false_hit_rate": None,
        "false_hit_rate_ci": None,
        "false_hit_rows": [],
        "exact_proxy": proxy,
        "quality_by_dup": {},
    }


def _exact_proxy(
    cache_out: pd.DataFrame | None, cache_judge: pd.DataFrame | None
) -> dict[str, Any]:
    """Closest observed proxy for exact-hit quality: the cache run's verdicts
    on its own exact-hit rows. Not a measurement of the replay (different
    stored texts); reported so the bound is explicit, not hidden."""
    if cache_out is None or cache_judge is None:
        return {"judged": 0, "score_2": 0, "score_1": 0, "score_0": 0, "unjudged": 0}
    exact_idx = set(cache_out.loc[cache_out["kind"] == "cache_exact", "idx"].tolist())
    counts = {"judged": 0, "score_2": 0, "score_1": 0, "score_0": 0, "unjudged": 0}
    for r in cache_judge.itertuples(index=False):
        if int(r.idx) not in exact_idx:
            continue
        try:
            f = float(r.judge1_score)
        except (TypeError, ValueError):
            counts["unjudged"] += 1
            continue
        if np.isnan(f):
            counts["unjudged"] += 1
            continue
        counts["judged"] += 1
        counts[f"score_{int(f)}"] += 1
    return counts


def _quality_block(judge: pd.DataFrame | None, metric: dict[str, Any]) -> dict[str, Any]:
    """Equivalence vs baseline from judge rows. Rows identical to the baseline
    are auto-scored 2 (no judge call); everything else carries a judge score.
    False hits are semantic-cache hits the judge scored 0."""
    if judge is None or not len(judge):
        return {
            "judged_rows": 0,
            "quality_equiv": None,
            "quality_ci": None,
            "quality_seed": None,
            "quality_idx_sha": None,
            "false_hit_rate": None,
            "false_hit_rate_ci": None,
            "quality_by_dup": {},
        }
    rows = list(judge.itertuples(index=False))
    scores: list[float] = []
    false_hits: list[int] = []
    by_dup: dict[str, list[float]] = {}
    for r in rows:
        if bool(r.identical):
            s: float = 2.0
        else:
            raw = r.judge1_score
            if raw is None:
                continue
            try:
                f = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isnan(f):
                continue
            s = f
        scores.append(s)
        dt = str(r.dup_type)
        by_dup.setdefault(dt, []).append(s)
        if dt in ("paraphrase", "trap") and s == 0 and r.answer_text != r.baseline_text:
            false_hits.append(int(r.idx))
    semantic_hits = metric["cache_semantic_hits"]
    q_seed = derive_seed(INTERVAL_SEED, str(metric["frac"]), str(metric["config"]), "quality")
    q_vals = [s / 2.0 for s in scores]
    q_lo, q_hi, q_sha = bootstrap_ci(q_vals, q_seed)
    quality = {
        "judged_rows": len(scores),
        "quality_equiv": round(float(np.mean(scores)) / 2.0, 4) if scores else None,
        "quality_ci": [round(q_lo, 4), round(q_hi, 4)] if scores else None,
        "quality_seed": q_seed,
        "quality_idx_sha": q_sha,
        "mean_score": round(float(np.mean(scores)), 4) if scores else None,
        "false_hit_rate": (round(len(false_hits) / semantic_hits, 4) if semantic_hits else None),
        "false_hit_rate_ci": (
            [round(v, 4) for v in wilson(len(false_hits), semantic_hits)] if semantic_hits else None
        ),
        "false_hit_rows": false_hits,
        "quality_by_dup": {
            dt: round(float(np.mean(v)) / 2.0, 4) for dt, v in sorted(by_dup.items())
        },
    }
    return quality


def human_label_coverage(cfg: Config) -> dict[str, Any]:
    """Coverage of the shipped human-validation CSV. Fails until a human
    fills at least half the highest-value rows; an empty human_label column
    is a template, not validation."""
    import csv

    path = cfg.data.runs_dir.parent / "human_validation.csv"
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return {"name": "human_label_coverage", "passed": False, "observed": "csv absent"}
    filled = sum(1 for r in rows if str(r.get("human_label", "")).strip())
    observed = f"{filled}/{len(rows)} = {filled / len(rows):.3f}" if rows else "csv empty"
    return {
        "name": "human_label_coverage",
        "passed": bool(rows) and filled / len(rows) >= 0.50,
        "observed": observed,
    }


def judge_independence_gate(cfg: Config, kappa: dict[str, Any]) -> dict[str, Any]:
    """Fails while both judges are the same model: kappa is self-consistency,
    not agreement between independent judges, so no correctness claim about
    the quality column can rest on it."""
    same = cfg.models.judge_primary == cfg.models.judge_secondary
    return {
        "name": "judge_independence",
        "passed": not same,
        "observed": (
            f"primary {cfg.models.judge_primary} == secondary {cfg.models.judge_secondary}; "
            f"kappa {kappa.get('kappa')} (n={kappa.get('n')}) is self-consistency"
            if same
            else f"primary {cfg.models.judge_primary} vs secondary {cfg.models.judge_secondary}"
        ),
    }


def cost_weighted_table(cfg: Config) -> dict[str, Any]:
    """Argmin threshold of L = r*fp + fn over the committed sweep grid.

    r is the false-hit cost in units of one miss. Ties keep the lowest
    threshold (pandas idxmin). A false hit serves a wrong answer; a miss only
    pays the model price, so r=1 is not the honest operating point.
    """
    sweep = read_parquet(cfg.data.runs_dir / "threshold_sweep.parquet")
    if sweep is None or not len(sweep):
        return {"ratios": [], "note": "sweep absent"}
    rows = []
    for r in (1, 3, 10, 30, 100):
        loss = r * sweep["fp"] + sweep["fn"]
        best = sweep.loc[loss.idxmin()]
        rows.append(
            {
                "cost_ratio": r,
                "threshold": round(float(best["threshold"]), 4),
                "fp": int(best["fp"]),
                "fn": int(best["fn"]),
                "false_hit_rate": round(float(best["false_hit_rate"]), 4),
                "recall": round(float(best["recall"]), 4),
            }
        )
    return {
        "loss": "L = cost_ratio * false_hits + misses, over the sweep grid",
        "ratios": rows,
    }


def cost_weighted_thresholds(cfg: Config) -> dict[str, Any]:
    """Argmin threshold under L = r*fp + fn for loss ratios r in {1,3,10,30,100}.

    Recomputed from the committed sweep grid (pandas idxmin: lowest threshold
    wins ties). At high ratios the argmin sits on the grid edge, which is the
    finding: the grid contains no acceptable point.
    """
    sweep = read_parquet(cfg.data.runs_dir / "threshold_sweep.parquet")
    if sweep is None or not len(sweep):
        return {"ratios": [], "note": "sweep absent"}
    rows = []
    for r in (1, 3, 10, 30, 100):
        loss = r * sweep["fp"] + sweep["fn"]
        best = sweep.loc[loss.idxmin()]
        rows.append(
            {
                "loss_ratio": r,
                "threshold": round(float(best["threshold"]), 4),
                "fp": int(best["fp"]),
                "fn": int(best["fn"]),
                "false_hit_rate": round(float(best["false_hit_rate"]), 4),
                "recall": round(float(best["recall"]), 4),
            }
        )
    return {
        "loss": "L = r*fp + fn over the committed sweep grid",
        "ratios": rows,
        "tie_rule": "lowest threshold wins ties (pandas idxmin)",
    }


def assemble(cfg: Config, run_name: str, note: str = "") -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for frac_cfg in cfg.workload.duplicate_fractions:
        frac = frac_cfg.name
        for config in CONFIGS:
            key = f"{frac}_{config}"
            metrics[key] = config_metrics(cfg, config, frac)
        metrics[f"{frac}_exact_only"] = exact_only_metrics(cfg, frac)

    baseline_cost = {
        f.name: metrics[f"{f.name}_baseline"].get("cost_usd")
        for f in cfg.workload.duplicate_fractions
    }
    gates = []
    for f in cfg.workload.duplicate_fractions:
        frac = f.name
        for config in ("cache", "router_cascade", "router_heuristic"):
            m = metrics[f"{frac}_{config}"]
            if not (m["present"] and baseline_cost[frac] is not None):
                gates.append(
                    {
                        "name": f"baseline_costs_more_{frac}_{config}",
                        "passed": False,
                        "observed": "run not present (resume incomplete)",
                    }
                )
                continue
            gates.append(
                {
                    "name": f"baseline_costs_more_{frac}_{config}",
                    "passed": m["cost_usd"] < baseline_cost[frac],
                    "observed": (
                        f"baseline {baseline_cost[frac]:.4f} vs {config} {m['cost_usd']:.4f}"
                    ),
                }
            )
        cm = metrics[f"{frac}_cache"]
        gates.append(
            {
                "name": f"false_hit_rate_reported_{frac}",
                "passed": bool(cm.get("present")) and cm.get("false_hit_rate") is not None,
                "observed": str(cm.get("false_hit_rate")),
            }
        )
    tuning = read_json(cfg.data.runs_dir / "tuning.json") or {}
    # Strictly greater: an equal F1 means the tuning picked a point no better
    # than a random threshold, which is a published failure, not a pass.
    # The control is the mean F1 of 64 thresholds drawn uniformly from the
    # tuning grid; see run.tune_threshold.
    tuned_f1 = tuning.get("tuned_f1")
    control_f1 = tuning.get("control_random_mean_f1", tuning.get("control_f1"))
    tuned_ok = isinstance(tuned_f1, (int, float)) and isinstance(control_f1, (int, float))
    gates.append(
        {
            "name": "tuned_threshold_beats_random_control",
            "passed": bool(tuned_ok and float(tuned_f1) > float(control_f1)),  # type: ignore[arg-type]
            "observed": (
                f"tuned {tuning.get('threshold')} f1 {tuned_f1} "
                f"vs control_mean {control_f1} max {tuning.get('control_random_max_f1')}"
            ),
        }
    )
    kappa = kappa_report(cfg)
    gates.append(human_label_coverage(cfg))
    gates.append(judge_independence_gate(cfg, kappa))
    cost_weighted = cost_weighted_thresholds(cfg)
    payload = {
        "run_name": run_name,
        "note": note,
        "gates": gates,
        "all_gates_passed": all(g["passed"] for g in gates),
        "metrics": metrics,
        "tuning": tuning,
        "cost_weighted_thresholds": cost_weighted,
        "kappa": kappa,
        "interval_methods": {
            "wilson": "Wilson score interval, z=1.96, for binomial rates",
            "bootstrap": (
                f"percentile bootstrap, B={BOOTSTRAP_B}, numpy default_rng streams "
                f"from recorded per-interval seeds (master {INTERVAL_SEED}); "
                "the sha of each resample index matrix is stored beside its interval"
            ),
            "newcombe": "Newcombe score interval (method 10) for differences",
            "paired_bootstrap": "lockstep resampling over shared row idx",
        },
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
