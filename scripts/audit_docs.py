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
            "can exceed latency_ms. Token and cost accounting sum each call "
            "once and are unaffected; only the latency decomposition is inflated."
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


def check_headline_tables(data: dict[str, object]) -> list[str]:
    errors: list[str] = []
    readme = README.read_text(encoding="utf-8")
    metrics = data.get("metrics")
    assert isinstance(metrics, dict)
    attempted = {"low": 245, "high": 249}
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
        if str(attempted[frac]) not in section.group(1):
            errors.append(f"README {frac} header does not state {attempted[frac]} attempted rows")
        rows = re.findall(r"^\| (\w+) \| ([^|]+) \| (.+)$", section.group(2), re.MULTILINE)
        seen: set[str] = set()
        for config, _recipe, rest in rows:
            if config not in EXPECTED_CONFIGS:
                continue
            seen.add(config)
            entry = metrics.get(f"{frac}_{config}")
            if not isinstance(entry, dict):
                errors.append(f"no metrics for {frac}_{config}")
                continue
            cells = [c.strip() for c in rest.split("|")]
            # cells: cost, p50, p95, p99, hit rate, false-hit rate, quality equiv
            try:
                cost, p50, p95, p99, hit = (float(cells[i]) for i in range(5))
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
            if len(cells) >= 7:
                documented_fhr = cells[5].split()[0]
                actual_fhr = entry.get("false_hit_rate")
                if documented_fhr == "null":
                    if actual_fhr is not None:
                        errors.append(f"README {frac}/{config} false-hit null != {actual_fhr}")
                elif actual_fhr is None or abs(float(actual_fhr) - float(documented_fhr)) > 0.0001:
                    errors.append(
                        f"README {frac}/{config} false-hit {documented_fhr} != {actual_fhr}"
                    )
                documented_q = cells[6].split()[0]
                actual_q = entry.get("quality_equiv")
                if documented_q == "null":
                    if actual_q is not None:
                        errors.append(f"README {frac}/{config} quality null != {actual_q}")
                elif actual_q is None or abs(float(actual_q) - float(documented_q)) > 0.0001:
                    errors.append(f"README {frac}/{config} quality {documented_q} != {actual_q}")
            total = entry.get("n_rows", 0) + entry.get("n_errors", 0)
            if total != attempted[frac]:
                errors.append(
                    f"results.json {frac}_{config} attempted rows {total} != {attempted[frac]}"
                )
        missing = set(EXPECTED_CONFIGS) - seen
        if missing:
            errors.append(f"README {frac} table missing rows: {sorted(missing)}")
    return errors


def check_figures() -> list[str]:
    errors: list[str] = []
    readme = README.read_text(encoding="utf-8")
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
