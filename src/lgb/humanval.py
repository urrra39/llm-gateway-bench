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

import numpy as np
import pandas as pd

from lgb.store import read_parquet


def export_human_csv(cfg, out_path: Path, limit: int = 60) -> Path:
    rows: list[dict] = []
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

    def priority(r: dict) -> tuple[int, int, str]:
        j1 = r["judge1_score"]
        j2 = r["judge2_score"]
        false_hit = 0 if (isinstance(j1, int) and j1 == 0) else 1
        disagreement = 0 if (isinstance(j1, int) and isinstance(j2, int) and j1 != j2) else 1
        return (false_hit, disagreement, r["dup_type"])

    records = [dict(x) for x in rows]
    records.sort(key=priority)
    chosen = records[:limit]
    out = pd.DataFrame(chosen)
    out.to_csv(out_path, index=False)
    return out_path


def _cell(v: object) -> str:
    return "" if v is None else str(v)


def human_agreement(csv_path: Path) -> dict:
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
    report: dict = {"path": str(csv_path), "n_rows": len(rows)}
    for col in ("judge1_score", "judge2_score"):
        a, b = [], []
        for r, h in zip(rows, human, strict=True):
            machine = str(r.get(col, "")).strip()
            if not h or not machine:
                continue
            a.append(_norm(machine))
            b.append(_norm(h))
        report[f"{col}_vs_human"] = _agreement(a, b)
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


def _agreement(a: list[str], b: list[str]) -> dict:
    if not a:
        return {"n": 0, "observed": None, "kappa": None}
    a = np.asarray(a)
    b = np.asarray(b)
    observed = float((a == b).mean())
    cats = sorted(set(a) | set(b))
    pa = {c: float((a == c).mean()) for c in cats}
    pb = {c: float((b == c).mean()) for c in cats}
    expected = sum(pa[c] * pb[c] for c in cats)
    kappa = (observed - expected) / (1 - expected) if expected < 1.0 else math.nan
    return {"n": len(a), "observed": round(observed, 4), "kappa": round(float(kappa), 4)}
