"""Quality judging.

Every gateway response is compared against the baseline (expensive-model)
response for the same request. A response byte-identical to the baseline is
equivalent by construction (score 2, no judge call). Everything else is scored
by the primary judge model on a frozen rubric at temperature 0; a seeded subset
gets a second judge for inter-judge agreement (Cohen's kappa with its
denominator). Verdicts persist per row. Human labels are never filled by code.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lgb.chat import Gateway
from lgb.config import Config
from lgb.store import append_rows, read_parquet


def _load_run(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    if frame is None:
        raise FileNotFoundError(path)
    return frame


def judge_run(
    cfg: Config,
    config: str,
    frac: str,
    baseline: pd.DataFrame,
    outcomes: pd.DataFrame,
    run_dir: Path,
) -> pd.DataFrame:
    judge_path = run_dir / "judge.parquet"
    existing = read_parquet(judge_path)
    done: set[str] = set()
    if existing is not None and len(existing):
        done = {f"{r.config}|{r.frac}|{r.idx}" for r in existing.itertuples(index=False)}
    baseline_by_idx = {int(r.idx): r for r in baseline.itertuples(index=False)}
    judge_cfg = cfg.judge
    gw = Gateway(cfg)
    primary = cfg.models.judge_primary
    secondary = cfg.models.judge_secondary

    sample: list[int] = []
    if judge_cfg.second_judge_sample > 0:
        rng = np.random.default_rng(judge_cfg.sample_seed)
        candidate_idx = [
            int(r.idx)
            for r in outcomes.itertuples(index=False)
            if f"{config}|{frac}|{r.idx}" not in done
        ]
        k = min(judge_cfg.second_judge_sample, len(candidate_idx))
        sample = (
            sorted(int(x) for x in rng.choice(candidate_idx, size=k, replace=False))
            if candidate_idx
            else []
        )

    def prompt(request: str, a: str, b: str) -> str:
        return (
            f"{judge_cfg.rubric}\n\nREQUEST: {request}\n\nANSWER_A (reference): "
            f"{a}\n\nANSWER_B (candidate): {b}\n\nSCORE:"
        )

    def parse(text: str) -> int | None:
        for tok in text.strip().split():
            if tok in ("0", "1", "2"):
                return int(tok)
        return None

    pending: list[dict[str, Any]] = []
    for r in outcomes.itertuples(index=False):
        key = f"{config}|{frac}|{r.idx}"
        if key in done:
            continue
        ref = baseline_by_idx.get(int(r.idx))
        answer_b = str(r.answer_text)
        if ref is None or not str(ref.answer_text).strip():
            continue  # baseline missing; nothing to compare against
        answer_a = str(ref.answer_text)
        identical = answer_a == answer_b
        row: dict[str, Any] = {
            "config": config,
            "frac": frac,
            "idx": int(r.idx),
            "request": str(r.request),
            "dup_type": str(r.dup_type),
            "answer_text": answer_b,
            "baseline_text": answer_a,
            "identical": identical,
            "judge1_score": None,
            "judge2_score": None,
            "judge1_raw": None,
            "judge2_raw": None,
            "sampled_for_judge2": int(r.idx) in sample,
            "human_label": "",
            "note": "",
        }
        if identical:
            row["judge1_score"] = 2
            row["note"] = "identical to baseline, no judge call"
        else:
            text1 = gw.chat(
                primary, prompt(str(r.request), answer_a, answer_b), max_tokens=judge_cfg.max_tokens
            ).text
            row["judge1_raw"] = text1
            row["judge1_score"] = parse(text1)
            if row["sampled_for_judge2"]:
                text2 = gw.chat(
                    secondary,
                    prompt(str(r.request), answer_a, answer_b),
                    max_tokens=judge_cfg.max_tokens,
                ).text
                row["judge2_raw"] = text2
                row["judge2_score"] = parse(text2)
        pending.append(row)
        if len(pending) >= 20:
            append_rows(pd.DataFrame(pending), judge_path)
            pending.clear()
    if pending:
        append_rows(pd.DataFrame(pending), judge_path)
    final = read_parquet(judge_path)
    assert final is not None
    return final


def kappa_report(cfg: Config) -> dict[str, Any]:
    """Judge1 vs judge2 agreement over rows both scored, from committed files."""
    pairs: list[tuple[int, int]] = []
    for run in cfg.data.runs_dir.glob("*"):
        frame = read_parquet(run / "judge.parquet")
        if frame is None:
            continue
        for r in frame.itertuples(index=False):
            a, b = r.judge1_score, r.judge2_score
            if a is None or b is None:
                continue
            pairs.append((int(a), int(b)))
    if not pairs:
        return {"kappa": None, "n": 0, "observed": None, "note": "no double-scored rows"}
    a = np.asarray([p[0] for p in pairs])
    b = np.asarray([p[1] for p in pairs])
    observed = float((a == b).mean())
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    pa = {c: float((a == c).mean()) for c in cats}
    pb = {c: float((b == c).mean()) for c in cats}
    expected = sum(pa[c] * pb[c] for c in cats)
    kappa = (observed - expected) / (1 - expected) if expected < 1.0 else math.nan
    return {
        "kappa": round(float(kappa), 4),
        "n": len(pairs),
        "observed": round(observed, 4),
        "expected": round(float(expected), 4),
    }
