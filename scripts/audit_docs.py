"""Re-derive every documented number from results.json and fail on drift.

Checks:
- results.json exists and every expected config x fraction is present.
- Validity gates pass or their failure is published (all_gates_passed is a bool;
  individual gate rows are recomputed where cheap to recompute).
- Numbers repeated across README, docs/CEILING.md and docs/DECISIONS.md agree.
- docs/OPEN_DEFECTS.md matches the DEFECTS source list below, so the two
  cannot drift.

Usage: python scripts/audit_docs.py [--check-only]
With --check-only, exit non-zero on drift without rewriting OPEN_DEFECTS.md.
Otherwise rewrite docs/OPEN_DEFECTS.md from DEFECTS first, then check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results.json"
README = REPO / "README.md"
CEILING = REPO / "docs" / "CEILING.md"
DECISIONS = REPO / "docs" / "DECISIONS.md"
OPEN_DEFECTS = REPO / "docs" / "OPEN_DEFECTS.md"

EXPECTED_CONFIGS = ("baseline", "cache", "router_cascade", "router_heuristic")
EXPECTED_FRACS = ("low", "high")

# Source list for docs/OPEN_DEFECTS.md. Edit here; the audit regenerates the
# file so prose and list cannot drift.
DEFECTS: list[dict[str, str]] = [
    {
        "id": "D1",
        "status": "open",
        "title": "Single working model; tiers are short/long recipes, not vendors",
        "detail": (
            "Probed 2026-09-17T07:43Z: of claude-opus-4-8, claude-opus-5, "
            "deepseek-v4-flash, gpt-5.6-sol, gpt-6-astra and glm-5.3, only "
            "deepseek-v4-flash answers. Both tiers are recipes of that model: "
            "short (one sentence, 512 tokens) and long (three sentences, 4096 "
            "tokens). Router savings are token savings from shorter answers, "
            "not a vendor price ratio; tables and figures name recipes."
        ),
    },
    {
        "id": "D2",
        "status": "open",
        "title": "Second judge is the same model as the first",
        "detail": (
            "Judge primary and secondary are both deepseek-v4-flash, so the "
            "reported kappa is judge self-consistency at temperature 0, not "
            "inter-family agreement. No human has verified any label."
        ),
    },
    {
        "id": "D3",
        "status": "open",
        "title": "Pilot history used a dead model pair",
        "detail": (
            "Runs before 2026-09-17 used glm-5.3 as the cheap tier. Those rows "
            "are superseded and archived; only runs with the current operating "
            "points reproduce from a clean clone."
        ),
    },
    {
        "id": "D4",
        "status": "open",
        "title": "Cascade escalation worsens the latency tail",
        "detail": (
            "An escalation pays two sequential model calls. On the low fraction "
            "cascade p99 is 37085.7 ms against a 25987.8 ms baseline. Fix, not "
            "implemented: skip escalation once the cheap call has consumed most "
            "of the latency budget, or issue speculative parallel calls."
        ),
    },
    {
        "id": "D5",
        "status": "open",
        "title": "Cascade model_ms double-counts the cheap call on escalation",
        "detail": (
            "On escalated rows model_ms adds the cheap-call time twice, so it "
            "can exceed latency_ms (12 low-fraction and 5 high-fraction "
            "escalated rows, worst excess 12650.3 ms and 11833.5 ms). Exact "
            "fix: record cheap_ms and expensive_ms as two timing fields per "
            "escalated row instead of accumulating into one model_ms. The "
            "stored rows carry no per-call breakdown, so past rows cannot be "
            "repaired by recomputation. Blocks: nothing published (headline "
            "tables use latency_ms; the tail decomposition uses latency_ms "
            "for tail rows); blocks any future claim that decomposes cascade "
            "latency from model_ms."
        ),
    },
]


def load_results() -> dict[str, object]:
    if not RESULTS.exists():
        raise SystemExit(f"missing {RESULTS}; run `make all` first")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("results.json did not load as a mapping")
    return data


def check_completeness(data: dict[str, object]) -> list[str]:
    errors: list[str] = []
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return ["results.json has no metrics mapping"]
    for frac in EXPECTED_FRACS:
        for config in EXPECTED_CONFIGS:
            key = f"{frac}_{config}"
            entry = metrics.get(key)
            if not isinstance(entry, dict) or not entry.get("present"):
                errors.append(f"missing or incomplete run: {key}")
    return errors


def check_gates(data: dict[str, object]) -> list[str]:
    errors: list[str] = []
    gates = data.get("gates")
    if not isinstance(gates, list):
        return ["results.json has no gates list"]
    by_name = {g.get("name"): g for g in gates if isinstance(g, dict)}
    for name in ("human_label_coverage", "judge_independence"):
        gate = by_name.get(name)
        if gate is None:
            errors.append(f"results.json lacks the {name} gate")
        elif gate.get("passed") is not False:
            errors.append(f"{name} gate should fail")
    if len(gates) != 11:
        errors.append(f"results.json has {len(gates)} gates, want 11")
    metrics = data.get("metrics", {})
    assert isinstance(metrics, dict)
    for frac in EXPECTED_FRACS:
        baseline = metrics.get(f"{frac}_baseline", {})
        assert isinstance(baseline, dict)
        base_cost = baseline.get("cost_usd")
        for config in ("cache", "router_cascade", "router_heuristic"):
            entry = metrics.get(f"{frac}_{config}", {})
            assert isinstance(entry, dict)
            if entry.get("present") and isinstance(base_cost, (int, float)):
                cost = entry.get("cost_usd")
                if isinstance(cost, (int, float)) and not cost < float(base_cost):
                    errors.append(
                        f"gate broken: {frac}_{config} costs {cost} vs baseline {base_cost}"
                    )
    return errors


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+\.\d+", text))


#: README headline tables: section header per fraction, then one markdown row
#: per config with a recipe column. The audit parses these rows and compares
#: every number against results.json, so prose and artifact cannot drift.
TABLE_HEADER = "| config | recipe | cost USD | p50 ms | p95 ms | p99 ms | hit rate |"


def _parse_interval(text: str) -> tuple[float, float] | None:
    """Parse 'lo-hi' from a parenthetical like '(95% boot 0.81-1.04)'."""
    match = re.search(r"(\d+\.\d+)-(\d+\.\d+)", text)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))


def _parse_k_of_n(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+) of (\d+)", text)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def check_headline_tables(data: dict[str, object]) -> list[str]:
    errors: list[str] = []
    readme = README.read_text(encoding="utf-8")
    metrics = data.get("metrics")
    assert isinstance(metrics, dict)
    attempted = {"low": 245, "high": 249}
    table_configs = (*EXPECTED_CONFIGS, "exact_only")
    for frac in EXPECTED_FRACS:
        boundary = r"(?=^\w+ duplicate fraction|^# |\Z)"
        section = re.search(
            rf"^{frac} duplicate fraction \(report half: ([^)]+)\):$(.*?)" + boundary,
            readme,
            re.MULTILINE | re.DOTALL,
        )
        if not section:
            errors.append(f"README missing '{frac} duplicate fraction (report half: ...)' header")
            continue
        header = section.group(1)
        if str(attempted[frac]) not in header:
            errors.append(f"README {frac} header does not state {attempted[frac]} attempted rows")
        short = {
            "baseline": "baseline",
            "cache": "cache",
            "router_cascade": "cascade",
            "router_heuristic": "heuristic",
            "exact_only": "exact_only",
        }
        for config in table_configs:
            match = re.search(rf"successful:[^:]*\b{short[config]} (\d+)", header)
            if not match:
                errors.append(f"README {frac} header does not state {config} successful rows")
                continue
            entry = metrics.get(f"{frac}_{config}")
            if not isinstance(entry, dict):
                errors.append(f"no metrics for {frac}_{config}")
                continue
            if int(match.group(1)) != entry.get("n_rows"):
                errors.append(
                    f"README {frac} header {config} n={match.group(1)} "
                    f"!= results.json {entry.get('n_rows')}"
                )
        rows = re.findall(r"^\| (\w+) \| ([^|]+) \| (.+)$", section.group(2), re.MULTILINE)
        seen: set[str] = set()
        for config, _recipe, rest in rows:
            if config not in table_configs:
                continue
            seen.add(config)
            entry = metrics.get(f"{frac}_{config}")
            if not isinstance(entry, dict):
                errors.append(f"no metrics for {frac}_{config}")
                continue
            cells = [c.strip() for c in rest.split("|")]
            # cells: cost, p50, p95, p99, hit rate, false-hit rate, quality equiv
            try:
                cost, p50, p95, p99, hit = (float(cells[i].split()[0]) for i in range(5))
            except (ValueError, IndexError):
                errors.append(f"README {frac}/{config} row does not parse as numbers")
                continue
            checks = [
                ("cost_usd", cost, 0.0001),
                ("latency.p50", p50, 0.06),
                ("latency.p95", p95, 0.06),
                ("latency.p99", p99, 0.06),
                ("hit_rate", hit, 0.0001),
            ]
            lat = entry.get("latency")
            assert isinstance(lat, dict)
            actual = {
                "cost_usd": entry.get("cost_usd"),
                "latency.p50": lat.get("p50"),
                "latency.p95": lat.get("p95"),
                "latency.p99": lat.get("p99"),
                "hit_rate": entry.get("hit_rate"),
            }
            for name, documented, tol in checks:
                value = actual[name]
                if not isinstance(value, (int, float)) or abs(float(value) - documented) > tol:
                    errors.append(
                        f"README {frac}/{config} {name}={documented} != results.json {value}"
                    )
            errors.extend(_check_cell_intervals(frac, config, cells, entry))
            total = entry.get("n_rows", 0) + entry.get("n_errors", 0)
            if total != attempted[frac]:
                errors.append(
                    f"results.json {frac}_{config} attempted rows {total} != {attempted[frac]}"
                )
        missing = set(table_configs) - seen
        if missing:
            errors.append(f"README {frac} table missing rows: {sorted(missing)}")
    return errors


def _check_cell_intervals(
    frac: str, config: str, cells: list[str], entry: dict[str, Any]
) -> list[str]:
    """Denominators and 95% intervals in one table row, against results.json."""
    from lgb.intervals import wilson

    errors: list[str] = []
    tag = f"README {frac}/{config}"
    # cost + latency cells carry '(95% boot lo-hi)'.
    for cell, key in (
        (cells[0], "cost_ci"),
        (cells[1], "latency_ci.p50"),
        (cells[2], "latency_ci.p95"),
        (cells[3], "latency_ci.p99"),
    ):
        shown = _parse_interval(cell)
        if shown is None:
            errors.append(f"{tag} cell {cell.split()[0]!r} lacks a 95% interval")
            continue
        node: Any = entry
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, list) or any(
            abs(a - b) > 0.061 for a, b in zip(shown, node, strict=True)
        ):
            errors.append(f"{tag} {key} {shown} != results.json {node}")
    # hit-rate cell carries '(k of n; 95% Wilson lo-hi)'.
    pair = _parse_k_of_n(cells[4])
    shown_hit = _parse_interval(cells[4])
    if pair is None or shown_hit is None:
        errors.append(f"{tag} hit-rate cell lacks 'k of n' and a Wilson interval")
    else:
        k, n = pair
        exp_hits = entry.get("cache_exact_hits", 0) + entry.get("cache_semantic_hits", 0)
        if k != exp_hits or n != entry.get("n_rows"):
            errors.append(f"{tag} hit counts {k} of {n} != results.json")
        if any(abs(a - b) > 0.0001 for a, b in zip(shown_hit, wilson(k, n), strict=True)):
            errors.append(f"{tag} hit Wilson {shown_hit} != recomputed {wilson(k, n)}")
        stored = entry.get("hit_rate_ci")
        if (
            not isinstance(stored, list)
            or any(abs(a - b) > 0.0001 for a, b in zip(shown_hit, stored, strict=True))
            or any(abs(a - b) > 0.0001 for a, b in zip(stored, wilson(k, n), strict=True))
        ):
            errors.append(f"{tag} hit interval {shown_hit} != results.json {stored}")
    # false-hit cell: value + '(k of n; ...)', 'null', 'n/a (...)' or exact-only proxy.
    fhr_cell = cells[5] if len(cells) > 5 else ""
    actual_fhr = entry.get("false_hit_rate")
    if fhr_cell.startswith("null"):
        if actual_fhr is not None:
            errors.append(f"{tag} false-hit null != {actual_fhr}")
    elif fhr_cell.startswith("n/a"):
        pair = _parse_k_of_n(fhr_cell)
        proxy = entry.get("exact_proxy", {})
        if pair is None or pair != (proxy.get("score_0", -1), proxy.get("judged", -2)):
            errors.append(f"{tag} exact proxy {pair} != results.json {proxy}")
    else:
        pair = _parse_k_of_n(fhr_cell)
        shown_fhr = _parse_interval(fhr_cell)
        try:
            documented_fhr = float(fhr_cell.split()[0])
        except ValueError:
            errors.append(f"{tag} false-hit cell does not parse")
            return errors
        if actual_fhr is None or abs(float(actual_fhr) - documented_fhr) > 0.0001:
            errors.append(f"{tag} false-hit {documented_fhr} != {actual_fhr}")
        if pair is None or shown_fhr is None:
            errors.append(f"{tag} false-hit cell lacks 'k of n' and a Wilson interval")
        else:
            k, n = pair
            if k != len(entry.get("false_hit_rows", [])) or n != entry.get("cache_semantic_hits"):
                errors.append(f"{tag} false-hit counts {k} of {n} != results.json")
            if any(abs(a - b) > 0.0001 for a, b in zip(shown_fhr, wilson(k, n), strict=True)):
                errors.append(f"{tag} false-hit Wilson {shown_fhr} != recomputed")
    # quality cell: value + '(n=...; ...)', 'null ...' or 'unmeasured ...'.
    q_cell = cells[6] if len(cells) > 6 else ""
    actual_q = entry.get("quality_equiv")
    if q_cell.startswith("null"):
        if actual_q is not None:
            errors.append(f"{tag} quality null != {actual_q}")
    elif q_cell.startswith("unmeasured"):
        if actual_q is not None or entry.get("quality_note") is None:
            errors.append(f"{tag} unmeasured quality needs null equiv plus a note")
    else:
        try:
            documented_q = float(q_cell.split()[0])
        except ValueError:
            errors.append(f"{tag} quality cell does not parse")
            return errors
        if actual_q is None or abs(float(actual_q) - documented_q) > 0.0001:
            errors.append(f"{tag} quality {documented_q} != {actual_q}")
        match = re.search(r"n=(\d+)", q_cell)
        shown_q = _parse_interval(q_cell)
        if match is None or shown_q is None:
            errors.append(f"{tag} quality cell lacks a denominator and interval")
        elif int(match.group(1)) != entry.get("judged_rows"):
            errors.append(f"{tag} quality n != results.json judged_rows")
    return errors


def check_figures() -> list[str]:
    errors: list[str] = []
    readme = README.read_text(encoding="utf-8")
    for doc in (README, CEILING, DECISIONS):
        text = doc.read_text(encoding="utf-8")
        if re.search(r"!\[[^\]]*\]\(https?://|<img[^>]+src=[\"']https?://", text):
            errors.append(f"{doc.name} embeds an http(s) image source")
    referenced = set(re.findall(r"docs/figures/([\w\-.]+)", readme))
    on_disk = {p.name for p in (REPO / "docs" / "figures").glob("*") if p.is_file()}
    for name in sorted(referenced):
        if name not in on_disk:
            errors.append(f"README references docs/figures/{name} which is not on disk")
        if not name.endswith(".svg"):
            errors.append(f"docs/figures/{name} is referenced as a figure but is not a plot")
    for name in sorted(on_disk):
        if name not in referenced:
            errors.append(f"docs/figures/{name} is on disk but unreferenced (orphaned)")
    return errors


#: Resolved inside check_ratios (not at import) so tests can point README at
#: a doctored copy; a module-level tuple would freeze the original path.
def _ratio_docs() -> tuple[Path, ...]:
    return (README, CEILING, DECISIONS, OPEN_DEFECTS)


def _known_pairs(data: dict[str, object]) -> set[tuple[int, int]]:
    """Every (count, denominator) pair the docs may state, re-derived."""
    pairs: set[tuple[int, int]] = set()
    metrics = data.get("metrics")
    assert isinstance(metrics, dict)
    for _key, entry in metrics.items():
        if not isinstance(entry, dict) or not entry.get("present"):
            continue
        n = int(entry.get("n_rows", 0))
        exact = int(entry.get("cache_exact_hits", 0))
        sem = int(entry.get("cache_semantic_hits", 0))
        pairs.add((exact + sem, n))
        pairs.add((exact, exact + sem if exact + sem else 1))
        pairs.add((exact, n + int(entry.get("n_errors", 0))))
        pairs.add((n, n + int(entry.get("n_errors", 0))))
        pairs.add((len(entry.get("false_hit_rows", [])), sem))
        pairs.add((int(entry.get("judged_rows", 0)), n))
        proxy = entry.get("exact_proxy", {})
        if isinstance(proxy, dict) and proxy.get("judged"):
            pairs.add((int(proxy.get("score_0", 0)), int(proxy.get("judged", 1))))
            pairs.add((int(proxy.get("judged", 0)), n))
    tuning = data.get("tuning", {})
    assert isinstance(tuning, dict)
    stats = tuning.get("tune_stats", {})
    if isinstance(stats, dict):
        tp, fp, fn, tn = (int(stats.get(k, 0)) for k in ("tp", "fp", "fn", "tn"))
        pairs.add((tp, tp + fp + fn + tn))
        pairs.add((fp, tp + fp))
    cw = data.get("cost_weighted_thresholds", {})
    if isinstance(cw, dict):
        for row in cw.get("ratios", []):
            if isinstance(row, dict):
                pairs.add((int(row.get("fp", 0)), int(row.get("fp", 0)) + int(row.get("tp", 0))))
    kappa = data.get("kappa", {})
    if isinstance(kappa, dict) and kappa.get("n"):
        pairs.add((int(kappa["n"]), int(kappa["n"])))
    pairs.add((30, 60))  # human gate: 30 of the 60 shipped rows
    pairs.add((0, 60))
    return pairs


def _known_percentages(data: dict[str, object]) -> list[float]:
    """Every percentage the docs may state, re-derived from artifacts."""
    values: list[float] = []
    metrics = data.get("metrics")
    assert isinstance(metrics, dict)
    for _key, entry in metrics.items():
        if not isinstance(entry, dict) or not entry.get("present"):
            continue
        for rate_key in ("hit_rate", "false_hit_rate", "quality_equiv"):
            rate = entry.get(rate_key)
            if isinstance(rate, (int, float)):
                values.append(float(rate) * 100.0)
        exact = int(entry.get("cache_exact_hits", 0))
        sem = int(entry.get("cache_semantic_hits", 0))
        attempted = int(entry.get("n_rows", 0)) + int(entry.get("n_errors", 0))
        if attempted:
            values.append(exact / attempted * 100.0)
        if exact + sem:
            values.append(exact / (exact + sem) * 100.0)
        base = metrics.get(f"{entry.get('frac')}_baseline", {})
        if isinstance(base, dict) and isinstance(base.get("cost_usd"), (int, float)):
            cost, baseline_cost = float(entry["cost_usd"]), float(base["cost_usd"])
            if entry.get("config") != "baseline" and baseline_cost:
                values.append((baseline_cost - cost) / baseline_cost * 100.0)
    sweep = _sweep_frame()
    if sweep is not None:
        values.extend(float(v) * 100.0 for v in sweep["false_hit_rate"].tolist())
        values.extend(float(v) * 100.0 for v in sweep["recall"].tolist())
    values.extend(_workload_fractions())
    values.append(50.0)  # human gate threshold: 30 of 60 rows
    values.append(2.0)  # shippability bar for false hits; the sweep-minimum
    # guard below keeps it meaningful (it must stay below every sweep value)
    return values


def _workload_fractions() -> list[float]:
    meta = REPO / "data" / "runs" / "workloads" / "meta.json"
    try:
        fractions = json.loads(meta.read_text(encoding="utf-8")).get("fractions", [])
    except (OSError, ValueError):
        return []
    out: list[float] = []
    for frac in fractions:
        if isinstance(frac, dict):
            for key in ("exact", "paraphrase", "trap", "novel"):
                if isinstance(frac.get(key), (int, float)):
                    out.append(float(frac[key]) * 100.0)
    return out


def _sweep_frame() -> Any:
    import pandas as pd

    path = REPO / "data" / "runs" / "threshold_sweep.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def check_ratios(data: dict[str, object]) -> list[str]:
    """Every 'X of Y' names a re-derived pair; every 'N%' a re-derived value.

    A bare percentage whose denominator has to be guessed is the defect class
    that shipped 10%-of-hits for 24-of-245-requests once; the check verifies
    the arithmetic relationship, not just that each number exists somewhere.
    """
    errors: list[str] = []
    pairs = _known_pairs(data)
    percentages = _known_percentages(data)
    sweep = _sweep_frame()
    sweep_min_fhr = float(sweep["false_hit_rate"].min()) * 100.0 if sweep is not None else None
    for doc in _ratio_docs():
        flat = re.sub(r"\s+", " ", doc.read_text(encoding="utf-8"))
        for match in re.finditer(r"(\d+) of (\d+)", flat):
            if (int(match.group(1)), int(match.group(2))) not in pairs:
                errors.append(f"{doc.name} states {match.group(0)} with no matching counts")
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            tag = f"{doc.name}:{lineno}"
            for match in re.finditer(r"(\d+(?:\.\d+)?)%", line):
                stated = float(match.group(1))
                decimals = len(match.group(1).split(".")[1]) if "." in match.group(1) else 0
                tol = 0.5 * 10**-decimals + 0.02
                if not any(abs(stated - v) <= tol for v in percentages):
                    errors.append(f"{tag} states {stated}% with no matching value")
            for match in re.finditer(r"(\d+)\s*[-\u2013\u2014]\s*(\d+)\s+", line):
                after = line[match.end() :]
                if re.match(
                    r"(successful|rows|requests|hits|errors|calls|verdicts|labels)\b", after
                ):
                    errors.append(f"{tag} states a count as a range: {match.group(0).strip()}")
    if sweep_min_fhr is not None and sweep_min_fhr <= 2.0:
        errors.append("sweep minimum false-hit rate reaches 2%; the no-ship claim moved")
    return errors


def check_cross_document_agreement(data: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for path in (README, CEILING, DECISIONS):
        if not path.exists():
            errors.append(f"missing {path.relative_to(REPO)}")
    if errors:
        return errors
    readme = README.read_text(encoding="utf-8")
    run_name = str(data.get("run_name", ""))
    if run_name and run_name not in readme:
        errors.append(f"README does not mention current run {run_name!r}")
    tuning = data.get("tuning", {})
    assert isinstance(tuning, dict)
    threshold = tuning.get("threshold")
    if threshold is not None and str(threshold) not in readme:
        errors.append(f"README does not contain tuned threshold {threshold}")
    kappa = data.get("kappa", {})
    assert isinstance(kappa, dict)
    kval = kappa.get("kappa")
    nval = kappa.get("n")
    if kval is not None and str(kval) not in readme:
        errors.append(f"README does not contain judge self-consistency {kval}")
    if nval is not None and f"n={nval}" not in readme and str(nval) not in readme:
        errors.append(f"README does not contain self-consistency denominator {nval}")
    if "self-consistency" not in readme:
        errors.append("README never qualifies kappa as judge self-consistency")
    return errors


#: Comparative claims the README may state. Each fragment must appear verbatim;
#: a "not separated" fragment must carry that label in the same sentence, and a
#: "separated" fragment must carry a 95% interval. Any other comparative
#: sentence in the README fails the audit, so new comparisons must be
#: registered here with their verdict. comparison_id points at the
#: results.json comparisons block the numbers come from.
COMPARISONS: list[dict[str, str]] = [
    {
        "id": "s-cache-cost-low",
        "fragment": "a 33% saving with non-overlapping 95% cost intervals",
        "verdict": "separated",
        "comparison_id": "cost-cache-baseline-low",
    },
    {
        "id": "s-cache-cost-high",
        "fragment": "a 59% saving, also separated",
        "verdict": "separated",
        "comparison_id": "cost-cache-baseline-high",
    },
    {
        "id": "s-heur-casc-cost-low",
        "fragment": "the heuristic costs less than the cascade (paired 95%",
        "verdict": "separated",
        "comparison_id": "cost-heur-casc-low",
    },
    {
        "id": "n-heur-casc-cost-high",
        "fragment": "on high duplicates the pair is not separated",
        "verdict": "not-separated",
        "comparison_id": "cost-heur-casc-high",
    },
    {
        "id": "s-casc-heur-qual-high",
        "fragment": "the cascade scores higher quality (paired 95%",
        "verdict": "separated",
        "comparison_id": "qual-heur-casc-high",
    },
    {
        "id": "n-casc-heur-qual-low",
        "fragment": "on low duplicates quality is not separated",
        "verdict": "not-separated",
        "comparison_id": "qual-heur-casc-low",
    },
    {
        "id": "n-fhr-high",
        "fragment": "False-hit rates are not separated on either fraction",
        "verdict": "not-separated",
        "comparison_id": "fhr-casc-heur-high",
    },
    {
        "id": "s-p99-heur-low",
        "fragment": "heuristic p99 reads lower than baseline on low duplicates",
        "verdict": "separated",
        "comparison_id": "p99-heur-base-low",
    },
    {
        "id": "s-p99-heur-high",
        "fragment": "on high duplicates (10517.9 ms",
        "verdict": "separated",
        "comparison_id": "p99-heur-base-high",
    },
    {
        "id": "s-heur-saving-low",
        "fragment": "a 64% saving on low, separated",
        "verdict": "separated",
        "comparison_id": "cost-heur-base-low",
    },
    {
        "id": "s-heur-saving-high",
        "fragment": "76% on high, separated",
        "verdict": "separated",
        "comparison_id": "cost-heur-base-high",
    },
]

COMPARATIVE_WORDS = re.compile(
    r"\b(beats?|higher|lower|safer|worse|better|cheaper|costlier|improves?|"
    r"reduces?|cuts?|exceeds?|outperforms?|saves?|gaps?|differences?|"
    r"separates?|separated|wins|loses|more than|less than|greater|smaller|"
    r"faster|slower|halves|improving|survives?)\b",
    re.IGNORECASE,
)
INTERVAL_MARKERS = ("95%", "not separated", "unquantified", "not resolved", "cannot separate")


def check_comparisons(data: dict[str, object]) -> list[str]:
    errors: list[str] = []
    readme = README.read_text(encoding="utf-8")
    flat_doc = re.sub(r"\s+", " ", readme)
    sentences = re.split(r"(?<=[.!?])\s+", flat_doc)
    for claim in COMPARISONS:
        fragment = claim["fragment"]
        if fragment not in flat_doc:
            errors.append(f"registered comparison missing from README: {fragment[:60]!r}")
            continue
        holding = [s for s in sentences if fragment in s]
        marker = "not separated" if claim["verdict"] == "not-separated" else "95%"
        if not any(marker in s for s in holding):
            errors.append(f"comparison {fragment[:60]!r} lacks its {claim['verdict']} marker")
    for sentence in sentences:
        flat = " ".join(sentence.split())
        if not COMPARATIVE_WORDS.search(flat) or len(flat) > 600:
            continue
        if flat.rstrip().endswith("?"):
            continue  # interrogatives ask; they do not claim
        if any(claim["fragment"] in flat for claim in COMPARISONS):
            continue
        if not any(marker in flat for marker in INTERVAL_MARKERS):
            errors.append(f"unregistered comparison without interval: {flat[:100]!r}")
    _verify_separated(data, errors)
    return errors


def _stored_comparisons(data: dict[str, object]) -> dict[str, dict[str, Any]]:
    comparisons = data.get("comparisons", [])
    assert isinstance(comparisons, list)
    out = {}
    for block in comparisons:
        if isinstance(block, dict) and isinstance(block.get("id"), str):
            out[block["id"]] = block
    return out


def _verify_separated(data: dict[str, object], errors: list[str]) -> None:
    """Every registry verdict re-derived from parquet via the same helpers
    the pipeline used: recomputed exclusion-of-zero must agree with the stored
    verdict, and the README sentence must print the stored numbers."""
    from lgb.config import Config
    from lgb.intervals import (
        INTERVAL_SEED,
        derive_seed,
        newcombe_diff,
        paired_bootstrap_diff,
    )
    from lgb.metrics import (
        _false_hit_count,
        _judge_scores,
        _paired_latencies,
        _paired_series,
        exact_only_metrics,
    )
    from lgb.run import run_dir
    from lgb.store import read_parquet

    cfg = Config.load(REPO / "config" / "bench.yaml")
    readme = README.read_text(encoding="utf-8")
    stored = _stored_comparisons(data)
    # exact_only replay, fresh from parquet, must match the stored entries.
    metrics = data.get("metrics")
    assert isinstance(metrics, dict)
    for frac in EXPECTED_FRACS:
        fresh = exact_only_metrics(cfg, frac)
        entry = metrics.get(f"{frac}_exact_only")
        if not isinstance(entry, dict):
            errors.append(f"no metrics for {frac}_exact_only")
            continue
        for key in ("cost_usd", "hit_rate", "cache_exact_hits", "n_rows", "n_errors"):
            if fresh.get(key) != entry.get(key):
                errors.append(f"exact_only {frac} {key} != stored")
    for claim in COMPARISONS:
        block = stored.get(claim["comparison_id"])
        if block is None:
            errors.append(f"no stored comparison for {claim['id']}")
            continue
        frac, a, b = block["frac"], block["a"], block["b"]
        if block.get("method") == "newcombe":
            sem: dict[str, int] = {}
            for name in (a, b):
                frame = read_parquet(run_dir(cfg, name, frac) / "outcomes.parquet")
                ok = frame[frame["error"].isna()] if frame is not None else None
                sem[name] = int((ok["kind"] == "cache_semantic").sum()) if ok is not None else 0
            k_a = _false_hit_count(read_parquet(run_dir(cfg, a, frac) / "judge.parquet"))
            k_b = _false_hit_count(read_parquet(run_dir(cfg, b, frac) / "judge.parquet"))
            _diff, lo, hi = newcombe_diff(k_a, sem[a], k_b, sem[b])
            tol = 1e-4
        else:
            seed = derive_seed(INTERVAL_SEED, frac, f"{a}-vs-{b}", block["quantity"])
            if seed != block.get("seed"):
                errors.append(f"comparison {block['id']} seed moved")
            if block["quantity"] == "cost":
                out_a = read_parquet(run_dir(cfg, a, frac) / "outcomes.parquet")
                out_b = read_parquet(run_dir(cfg, b, frac) / "outcomes.parquet")
                left = _paired_series(out_a, "cost_usd")
                right = _paired_series(out_b, "cost_usd")
            elif block["quantity"] == "quality":
                left = _judge_scores(read_parquet(run_dir(cfg, a, frac) / "judge.parquet"))
                right = _judge_scores(read_parquet(run_dir(cfg, b, frac) / "judge.parquet"))
            else:
                left = _paired_latencies(cfg, frac, a)
                right = _paired_latencies(cfg, frac, b)
            if block["quantity"] == "p99":
                from lgb.metrics import _paired_percentile_diff

                _diff, lo, hi, _sha, _n = _paired_percentile_diff(left, right, seed)
            else:
                _diff, lo, hi, _sha, _n = paired_bootstrap_diff(left, right, seed)
            tol = 0.06 if block["quantity"] == "p99" else 1e-6
        for label, value in (("diff", _diff), ("lo", lo), ("hi", hi)):
            if abs(float(value) - float(block[label])) > tol:
                errors.append(f"comparison {block['id']} {label} recomputed {value} != stored")
        separates = lo > 0 or hi < 0
        if separates != (claim["verdict"] == "separated"):
            errors.append(f"comparison {block['id']} verdict flipped on recompute")
        flat = re.sub(r"\s+", " ", readme)
        holding = [s for s in re.split(r"(?<=[.!?])\s+", flat) if claim["fragment"] in s]
        shown = [float(v) for s in holding for v in re.findall(r"-?\d+\.\d+", s)]
        for label in ("lo", "hi"):
            if not any(abs(v - float(block[label])) <= tol for v in shown):
                errors.append(f"README omits {block['id']} {label}={block[label]}")


def check_gate_table(data: dict[str, object]) -> list[str]:
    """The front-page gate table restates results.json gates exactly."""
    import csv

    errors: list[str] = []
    gates = data.get("gates")
    assert isinstance(gates, list)
    passed = sum(1 for g in gates if g.get("passed"))
    total = len(gates)
    readme = README.read_text(encoding="utf-8")
    line = f"{passed}/{total} PASS, {total - passed} FAIL"
    if line not in readme:
        errors.append(f"README gate line {line!r} not found")
    path = REPO / "data" / "human_validation.csv"
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        rows = []
    filled = sum(1 for r in rows if str(r.get("human_label", "")).strip())
    for gate in gates:
        if gate.get("name") == "human_label_coverage":
            if gate.get("passed") is not False:
                errors.append("human_label_coverage gate should fail")
            if f"{filled}/{len(rows)}" not in str(gate.get("observed", "")):
                errors.append("human_label_coverage observed != recomputed coverage")
        if gate.get("name") == "judge_independence" and gate.get("passed") is not False:
            errors.append("judge_independence gate should fail")
    if data.get("all_gates_passed") is not False:
        errors.append("all_gates_passed should be false")
    return errors


def check_cost_weighted_table(data: dict[str, object]) -> list[str]:
    """The README cost-weighted table matches a fresh argmin over the sweep."""
    errors: list[str] = []
    sweep = _sweep_frame()
    if sweep is None:
        return ["threshold sweep parquet absent"]
    readme = README.read_text(encoding="utf-8")
    stored = data.get("cost_weighted_thresholds", {})
    assert isinstance(stored, dict)
    rows = re.findall(
        r"^\| (\d+) \| ([\d.]+) \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \|",
        readme,
        re.MULTILINE,
    )
    want = {str(r["loss_ratio"]): r for r in stored.get("ratios", []) if isinstance(r, dict)}
    seen: set[str] = set()
    for ratio, thr, fp, fn, fhr, recall in rows:
        if ratio not in want:
            continue
        seen.add(ratio)
        loss = int(ratio) * sweep["fp"] + sweep["fn"]
        best = sweep.loc[loss.idxmin()]
        for label, documented, actual in (
            ("threshold", float(thr), float(best["threshold"])),
            ("fp", float(fp), float(best["fp"])),
            ("fn", float(fn), float(best["fn"])),
        ):
            if abs(documented - actual) > 1e-9:
                errors.append(f"cost-weighted r={ratio} {label} != recomputed")
        exp = want[ratio]
        if abs(float(fhr) - float(exp.get("false_hit_rate", -1))) > 0.0001:
            errors.append(f"cost-weighted r={ratio} fhr != results.json")
        if abs(float(recall) - float(exp.get("recall", -1))) > 0.0001:
            errors.append(f"cost-weighted r={ratio} recall != results.json")
    if set(want) - seen:
        errors.append(f"cost-weighted rows missing from README: {sorted(set(want) - seen)}")
    return errors


def render_open_defects() -> str:
    lines = [
        "# Open defects",
        "",
        "Source list lives in scripts/audit_docs.py as DEFECTS; this file is",
        "generated by `python scripts/audit_docs.py` so the two cannot drift.",
        "",
    ]
    for d in DEFECTS:
        lines.append(f"## {d['id']}: {d['title']}")
        lines.append("")
        lines.append(f"Status: {d['status']}")
        lines.append("")
        lines.append(d["detail"])
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    data = load_results()
    errors = check_completeness(data)
    errors.extend(check_gates(data))
    errors.extend(check_cross_document_agreement(data))
    errors.extend(check_headline_tables(data))
    errors.extend(check_figures())
    errors.extend(check_ratios(data))
    errors.extend(check_comparisons(data))
    errors.extend(check_gate_table(data))
    errors.extend(check_cost_weighted_table(data))
    expected = render_open_defects()
    current = OPEN_DEFECTS.read_text(encoding="utf-8") if OPEN_DEFECTS.exists() else ""
    if current != expected:
        if args.check_only:
            errors.append("docs/OPEN_DEFECTS.md drifts from scripts/audit_docs.py DEFECTS")
        else:
            OPEN_DEFECTS.parent.mkdir(parents=True, exist_ok=True)
            OPEN_DEFECTS.write_text(expected, encoding="utf-8")
            current = expected
            if current != expected:  # pragma: no cover
                errors.append("failed to regenerate OPEN_DEFECTS.md")
    if errors:
        for e in errors:
            print(f"audit FAIL: {e}", file=sys.stderr)
        return 1
    print("audit OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
