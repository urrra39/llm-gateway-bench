"""Human validation of judge verdicts.

Ships a CSV of the highest-value rows (false hits, judge disagreements,
escalations) with an empty human_label column. When a human fills it,
`lgb human-agreement --csv path` reports judge-versus-human agreement and
Cohen's kappa. Nothing fills human labels except a human.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lgb.store import read_parquet


def export_human_csv(cfg: Any, out_path: Path, limit: int = 60) -> Path:
    rows: list[dict[str, Any]] = []
    for run in sorted((cfg.data.runs_dir).glob("*")):
        judge = read_parquet(run / "judge.parquet")
        if judge is None:
            continue
        for r in judge.itertuples(index=False):
            rows.append(
                {
                    "config": str(r.config),
                    "frac": str(r.frac),
                    "idx": int(r.idx),
                    "request": str(r.request),
                    "dup_type": str(r.dup_type),
                    "answer_text": str(r.answer_text),
                    "baseline_text": str(r.baseline_text),
                    "judge1_score": _cell(r.judge1_score),
                    "judge2_score": _cell(r.judge2_score),
                    "human_label": "",
                }
            )
    frame = pd.DataFrame(rows)
    if not len(frame):
        frame.to_csv(out_path, index=False)
        return out_path

    def priority(r: dict[str, Any]) -> tuple[int, int, str]:
        j1 = r["judge1_score"]
        j2 = r["judge2_score"]
        false_hit = 0 if (isinstance(j1, int) and j1 == 0) else 1
        disagreement = 0 if (isinstance(j1, int) and isinstance(j2, int) and j1 != j2) else 1
        return (false_hit, disagreement, r["dup_type"])

    records: list[dict[str, Any]] = [dict(x) for x in rows]
    records.sort(key=priority)
    chosen = records[:limit]
    out = pd.DataFrame(chosen)
    out.to_csv(out_path, index=False)
    return out_path


def _cell(v: object) -> str:
    return "" if v is None else str(v)


def human_agreement(csv_path: Path) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    human = [str(r.get("human_label", "")).strip().lower() for r in rows]
    invalid = [
        r
        for r, h in zip(rows, human, strict=True)
        if h not in ("", "correct", "incorrect", "2", "1", "0")
    ]
    if invalid:
        raise ValueError(f"{len(invalid)} rows have unrecognised human labels")
    report: dict[str, Any] = {"path": str(csv_path), "n_rows": len(rows)}
    for col in ("judge1_score", "judge2_score"):
        machine_scores: list[str] = []
        human_scores: list[str] = []
        for r, h in zip(rows, human, strict=True):
            machine = str(r.get(col, "")).strip()
            if not h or not machine:
                continue
            machine_scores.append(_norm(machine))
            human_scores.append(_norm(h))
        report[f"{col}_vs_human"] = _agreement(machine_scores, human_scores)
    return report


def _norm(v: str) -> str:
    mapping = {
        "2": "correct",
        "1": "partial",
        "0": "incorrect",
        "correct": "correct",
        "incorrect": "incorrect",
        "partial": "partial",
    }
    return mapping.get(v, v)


def _agreement(machine: list[str], human: list[str]) -> dict[str, Any]:
    if not machine:
        return {"n": 0, "observed": None, "kappa": None}
    ma = np.asarray(machine)
    ha = np.asarray(human)
    observed = float((ma == ha).mean())
    cats = sorted(set(machine) | set(human))
    pa = {c: float((ma == c).mean()) for c in cats}
    pb = {c: float((ha == c).mean()) for c in cats}
    expected = sum(pa[c] * pb[c] for c in cats)
    kappa = (observed - expected) / (1 - expected) if expected < 1.0 else math.nan
    return {"n": len(machine), "observed": round(observed, 4), "kappa": round(float(kappa), 4)}
