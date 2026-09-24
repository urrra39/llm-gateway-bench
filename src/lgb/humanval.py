"""Human validation of judge verdicts.

Ships a CSV of the highest-value rows with an empty human_label column,
ordered so that the rows most likely to change a published conclusion come
first. When a human fills it, `lgb human-agreement --csv path` reports
judge-versus-human agreement, expected agreement and Cohen's kappa against a
stated denominator. Nothing here fills a human label. There is no default, no
inference from the judge, and no imputation: an unfilled row is dropped from
every agreement denominator and counted as unlabelled.

The scoring scale is the judge's own: 2 correct, 1 partial, 0 incorrect. A
human may write either the digit or the word. docs/HUMAN_LABELING.md defines
each value operationally; this module only reads them.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lgb.store import read_parquet

#: Priority tiers, lowest number first, as written into the CSV's `priority`
#: column. A row's tier is the strongest claim it could overturn.
#:
#: 1. A semantic hit the judge scored 0. This is a false hit: the cache served
#:    a different question's answer and the judge called it wrong. The false-hit
#:    rate is the number the no-ship verdict rests on, so a human disagreeing
#:    here moves the headline.
#: 2. A row the two judge passes scored differently. Judge self-consistency is
#:    the only label-quality number this repository has; these rows are where
#:    it is weakest.
#: 3. A cascade escalation. Two sequential model calls produced this answer and
#:    the escalation rule is a published design choice.
#: 4. Everything else, present so the file has a tail to sample from.
PRIORITY_LABELS: dict[int, str] = {
    1: "semantic hit judged incorrect (false hit)",
    2: "judge passes disagree",
    3: "cascade escalation",
    4: "other judged row",
}

#: Written to the CSV so a labeller reads the scale without the docs open, and
#: so `human_agreement` can reject a value that is not on it.
VALID_LABELS: tuple[str, ...] = ("2", "1", "0", "correct", "partial", "incorrect")

COLUMNS: tuple[str, ...] = (
    "priority",
    "priority_reason",
    "config",
    "frac",
    "idx",
    "kind",
    "served_by",
    "similarity",
    "hit_source_idx",
    "dup_type",
    "request",
    "answer_text",
    "baseline_text",
    "judge1_score",
    "judge2_score",
    "human_label",
)


def _judged_rows(cfg: Any) -> pd.DataFrame:
    """Every judged row with its outcome columns merged in.

    judge.parquet carries no `kind`, so whether a row was a semantic hit, an
    exact hit or a miss lives in outcomes.parquet and has to be joined on
    (config, frac, idx). Without that join `priority` cannot see a false hit,
    which is how the shipped file came to be sixty exact hits.
    """
    frames: list[pd.DataFrame] = []
    for run in sorted(cfg.data.runs_dir.glob("*")):
        judge = read_parquet(run / "judge.parquet")
        outcomes = read_parquet(run / "outcomes.parquet")
        if judge is None or outcomes is None:
            continue
        keys = ["config", "frac", "idx"]
        extra = outcomes[[*keys, "kind", "served_by", "similarity", "hit_source_idx"]]
        frames.append(judge.merge(extra, on=keys, how="left", validate="one_to_one"))
    if not frames:
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _score(v: object) -> int | None:
    """A judge score as an int, or None when the row was never judged.

    Scores arrive from parquet as floats with NaN for unjudged, so the old
    `isinstance(v, int)` test in the sort key was false for every row.
    """
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else int(f)


def _priority(row: Any) -> int:
    j1 = _score(row.judge1_score)
    j2 = _score(row.judge2_score)
    if row.kind == "cache_semantic" and j1 == 0:
        return 1
    if j1 is not None and j2 is not None and j1 != j2:
        return 2
    if row.config == "router_cascade" and row.served_by == "expensive":
        return 3
    return 4


def _cell(v: object) -> str:
    """A CSV cell: empty for missing, plain int for a score, else str.

    Writing "2.0" here was the second half of the priority defect — the sort
    key wanted ints and the exporter wrote floats — and it also made every
    label comparison in `human_agreement` a string mismatch.
    """
    if v is None:
        return ""
    score = _score(v)
    if score is not None and isinstance(v, float):
        return str(score)
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v)


def export_human_csv(cfg: Any, out_path: Path, limit: int = 60) -> Path:
    """Write the top `limit` judged rows, highest priority first, unlabelled.

    Ordering within a tier is (config, frac, idx), so the file is a pure
    function of the run store and regenerating it produces byte-identical
    output. human_label is written empty for every row without exception.
    """
    judged = _judged_rows(cfg)
    if not len(judged):
        pd.DataFrame(columns=list(COLUMNS)).to_csv(out_path, index=False)
        return out_path

    records: list[dict[str, Any]] = []
    for r in judged.itertuples(index=False):
        priority = _priority(r)
        records.append(
            {
                "priority": priority,
                "priority_reason": PRIORITY_LABELS[priority],
                "config": str(r.config),
                "frac": str(r.frac),
                "idx": int(r.idx),
                "kind": _cell(r.kind),
                "served_by": _cell(r.served_by),
                "similarity": _cell(r.similarity),
                "hit_source_idx": _cell(r.hit_source_idx),
                "dup_type": str(r.dup_type),
                "request": str(r.request),
                "answer_text": str(r.answer_text),
                "baseline_text": str(r.baseline_text),
                "judge1_score": _cell(r.judge1_score),
                "judge2_score": _cell(r.judge2_score),
                "human_label": "",
            }
        )
    records.sort(key=lambda r: (r["priority"], r["config"], r["frac"], r["idx"]))
    pd.DataFrame(records[:limit], columns=list(COLUMNS)).to_csv(out_path, index=False)
    return out_path


def priority_counts(cfg: Any) -> dict[int, int]:
    """How many judged rows sit in each tier, for docs and for the audit."""
    judged = _judged_rows(cfg)
    counts = dict.fromkeys(PRIORITY_LABELS, 0)
    for r in judged.itertuples(index=False):
        counts[_priority(r)] += 1
    return counts


def human_agreement(csv_path: Path) -> dict[str, Any]:
    """Judge-versus-human agreement from a partially filled CSV.

    Every unfilled human_label and every unjudged machine score is dropped
    from the pair list, so `n` is the number of rows that carry both and is
    the denominator of observed agreement, expected agreement and kappa. A
    file with no labels reports n=0 and None for each statistic rather than
    inventing a value.
    """
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    human = [str(r.get("human_label", "")).strip().lower() for r in rows]
    invalid = [(r, h) for r, h in zip(rows, human, strict=True) if h and h not in VALID_LABELS]
    if invalid:
        raise ValueError(
            f"{len(invalid)} rows have unrecognised human labels; "
            f"allowed values are {', '.join(VALID_LABELS)} or empty"
        )
    labelled = sum(1 for h in human if h)
    report: dict[str, Any] = {
        "path": str(csv_path),
        "n_rows": len(rows),
        "n_labelled": labelled,
        "n_unlabelled": len(rows) - labelled,
    }
    for col in ("judge1_score", "judge2_score"):
        machine_scores: list[str] = []
        human_scores: list[str] = []
        for r, h in zip(rows, human, strict=True):
            machine = _norm(str(r.get(col, "")))
            label = _norm(h)
            if not label or not machine:
                continue
            machine_scores.append(machine)
            human_scores.append(label)
        report[f"{col}_vs_human"] = _agreement(machine_scores, human_scores)
    return report


def _norm(v: str) -> str:
    """A score cell as a category name, or "" when the cell holds no score.

    Accepts the float spellings parquet produces ("2.0") as well as the ints
    and words a human writes, and maps pandas' "nan" to missing rather than
    letting it become a fourth category that silently deflates agreement.
    """
    text = v.strip().lower()
    if text in ("", "nan", "none", "na"):
        return ""
    mapping = {
        "2": "correct",
        "1": "partial",
        "0": "incorrect",
        "2.0": "correct",
        "1.0": "partial",
        "0.0": "incorrect",
        "correct": "correct",
        "incorrect": "incorrect",
        "partial": "partial",
    }
    return mapping.get(text, text)


def _agreement(machine: list[str], human: list[str]) -> dict[str, Any]:
    """Observed agreement, chance agreement and kappa over `n` paired rows.

    Expected agreement is the chance-agreement term of Cohen's kappa, the sum
    over categories of each rater's marginal share multiplied together. It is
    reported because kappa alone is unreadable: on a set where one category
    dominates, a high observed agreement and a near-zero kappa are the same
    measurement, and only the expected term says so.
    """
    if not machine:
        return {
            "n": 0,
            "observed": None,
            "expected": None,
            "kappa": None,
            "note": "no row carries both a human label and a judge score",
        }
    ma = np.asarray(machine)
    ha = np.asarray(human)
    observed = float((ma == ha).mean())
    cats = sorted(set(machine) | set(human))
    pa = {c: float((ma == c).mean()) for c in cats}
    pb = {c: float((ha == c).mean()) for c in cats}
    expected = sum(pa[c] * pb[c] for c in cats)
    kappa = (observed - expected) / (1 - expected) if expected < 1.0 else math.nan
    report: dict[str, Any] = {
        "n": len(machine),
        "observed": round(observed, 4),
        "expected": round(float(expected), 4),
        "kappa": round(float(kappa), 4),
    }
    if math.isnan(kappa):
        report["note"] = (
            "kappa undefined: both raters used a single category, so chance "
            "agreement is 1.0 and the denominator is zero"
        )
    return report


def format_report(report: dict[str, Any]) -> str:
    """The agreement report as lines a human reads, not a dict repr."""
    lines = [
        f"file: {report['path']}",
        f"rows: {report['n_rows']}  labelled: {report['n_labelled']}"
        f"  unlabelled: {report['n_unlabelled']}",
    ]
    for col in ("judge1_score", "judge2_score"):
        block = report.get(f"{col}_vs_human")
        if not isinstance(block, dict):
            continue
        lines.append(f"{col} vs human:")
        lines.append(f"  n (rows with both)   {block['n']}")
        if block["n"] == 0:
            lines.append(f"  {block.get('note', 'nothing to compare')}")
            continue
        lines.append(f"  observed agreement   {block['observed']}")
        lines.append(f"  expected agreement   {block['expected']}")
        kappa = block["kappa"]
        lines.append(f"  Cohen's kappa        {kappa}")
        if "note" in block:
            lines.append(f"  {block['note']}")
    if report["n_labelled"] == 0:
        lines.append("no human labels present; agreement is unmeasured, not zero")
    return "\n".join(lines)
