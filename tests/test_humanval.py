"""human-agreement runs on a partially filled file and reports kappa with n.

The fixture is fully synthetic: six invented rows exercising agreement,
disagreement and skipped empty labels. No real human labels exist anywhere.
"""

from __future__ import annotations

import csv
from pathlib import Path

from lgb.humanval import human_agreement

COLUMNS = [
    "config",
    "frac",
    "idx",
    "request",
    "dup_type",
    "answer_text",
    "baseline_text",
    "judge1_score",
    "judge2_score",
    "human_label",
]


def _row(idx: int, j1: str, human: str) -> dict[str, str]:
    return {
        "config": "cache",
        "frac": "low",
        "idx": str(idx),
        "request": f"synthetic request {idx}",
        "dup_type": "novel",
        "answer_text": "synthetic answer",
        "baseline_text": "synthetic baseline",
        "judge1_score": j1,
        "judge2_score": j1,
        "human_label": human,
    }


def test_partial_file_reports_kappa_with_denominator(tmp_path: Path) -> None:
    path = tmp_path / "partial.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(
            [
                _row(0, "2", "correct"),
                _row(1, "2", "correct"),
                _row(2, "0", "incorrect"),
                _row(3, "2", "incorrect"),
                _row(4, "2", ""),
                _row(5, "0", ""),
            ]
        )
    report = human_agreement(path)
    assert report["n_rows"] == 6
    assert report["judge1_score_vs_human"]["n"] == 4
    assert report["judge1_score_vs_human"]["observed"] == 0.75
    assert report["judge1_score_vs_human"]["kappa"] is not None


def test_empty_labels_report_nothing() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerow(_row(0, "2", ""))
        report = human_agreement(path)
        assert report["judge1_score_vs_human"] == {"n": 0, "observed": None, "kappa": None}
