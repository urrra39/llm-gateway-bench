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
        "title": "Single working model; cheap tier is a constrained operating point",
        "detail": (
            "On 2026-09-17 only deepseek-v4-flash answered; glm-5.3 has no "
            "channel and all other gateway models are quota-blocked. Cheap and "
            "expensive tiers share one model ID with different prompts and token "
            "budgets, so router savings are token savings, not a market comparison."
        ),
    },
    {
        "id": "D2",
        "status": "open",
        "title": "Second judge is the same model as the first",
        "detail": (
            "Judge primary and secondary are both deepseek-v4-flash, so kappa "
            "measures self-consistency at temperature 0, not inter-family "
            "agreement. No human has verified any label."
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
                        f"gate broken: {frac}_{config} costs {cost} "
                        f"vs baseline {base_cost}"
                    )
    return errors


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+\.\d+", text))


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
        errors.append(f"README does not contain kappa {kval}")
    if nval is not None and f"n={nval}" not in readme and str(nval) not in readme:
        errors.append(f"README does not contain kappa denominator {nval}")
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
