"""The human gate: export ordering, label arithmetic, and no fabricated labels.

Every fixture here is synthetic. `tests/fixtures/human_validation_partial.csv`
holds seven invented rows written in the exact spellings the real pipeline
produces — judge scores as "2.0"/"1.0"/"0.0" with "nan" for unjudged — because
those spellings are what the agreement code used to mishandle silently. No
real human label exists anywhere in this repository and no test writes one.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lgb.config import Config
from lgb.humanval import (
    PRIORITY_LABELS,
    export_human_csv,
    format_report,
    human_agreement,
    priority_counts,
)

FIXTURE = Path("tests/fixtures/human_validation_partial.csv")
SHIPPED = Path("data/human_validation.csv")

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
    assert report["n_labelled"] == 4
    assert report["n_unlabelled"] == 2
    assert report["judge1_score_vs_human"]["n"] == 4
    assert report["judge1_score_vs_human"]["observed"] == 0.75
    assert report["judge1_score_vs_human"]["expected"] is not None
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
        block = report["judge1_score_vs_human"]
        assert block["n"] == 0
        assert block["observed"] is None and block["expected"] is None
        assert block["kappa"] is None
        assert "no row carries both" in block["note"]


def test_committed_fixture_reports_the_arithmetic_by_hand() -> None:
    """The whole statistic, recomputed here without the module.

    judge1 pairs (machine, human): incorrect/incorrect, incorrect/correct,
    correct/correct, partial/partial, correct/correct. Observed 4/5 = 0.8.
    Marginals 0.4/0.4/0.2 against 0.2/0.6/0.2 give expected 0.36, so kappa is
    0.44/0.64 = 0.6875. The two unlabelled rows and the unjudged row are out
    of the denominator, not counted as agreement.
    """
    report = human_agreement(FIXTURE)
    assert report["n_rows"] == 7
    assert report["n_labelled"] == 5
    assert report["n_unlabelled"] == 2

    first = report["judge1_score_vs_human"]
    assert first == {"n": 5, "observed": 0.8, "expected": 0.36, "kappa": 0.6875}

    second = report["judge2_score_vs_human"]
    assert second == {"n": 4, "observed": 0.5, "expected": 0.375, "kappa": 0.2}


def test_float_and_nan_spellings_are_not_a_fourth_category() -> None:
    """ "2.0" must mean correct and "nan" must mean missing.

    Read literally, "2.0" matched no mapping entry and passed through, so it
    could never equal a human "correct"; "nan" became its own category and
    deflated the marginals. Both would have silently reported a near-zero
    kappa on the real file.
    """
    report = human_agreement(FIXTURE)
    assert report["judge1_score_vs_human"]["observed"] == 0.8
    assert report["judge2_score_vs_human"]["n"] == 4


def test_the_cli_prints_kappa_observed_expected_and_n(capsys: pytest.CaptureFixture[str]) -> None:
    from lgb.__main__ import main

    assert main(["human-agreement", "--csv", str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert "rows: 7  labelled: 5  unlabelled: 2" in out
    assert "n (rows with both)   5" in out
    assert "observed agreement   0.8" in out
    assert "expected agreement   0.36" in out
    assert "Cohen's kappa        0.6875" in out


def test_an_unrecognised_label_is_refused(tmp_path: Path) -> None:
    rows = FIXTURE.read_text(encoding="utf-8").replace(",incorrect\n", ",mostly fine\n", 1)
    path = tmp_path / "bad.csv"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognised human labels"):
        human_agreement(path)


def test_undefined_kappa_is_reported_not_hidden(tmp_path: Path) -> None:
    """One category on both sides means chance agreement is 1 and kappa is not
    a number. The report says so instead of printing nan without comment."""
    path = tmp_path / "single.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows([_row(0, "2", "correct"), _row(1, "2", "2")])
    block = human_agreement(path)["judge1_score_vs_human"]
    assert block["n"] == 2 and block["observed"] == 1.0 and block["expected"] == 1.0
    assert "kappa undefined" in block["note"]
    assert format_report(human_agreement(path)).count("kappa undefined") == 2


def test_the_shipped_file_carries_no_human_label() -> None:
    """The one unrecoverable offence, checked as a test.

    If any process ever writes a label into data/human_validation.csv, this
    fails and the human_label_coverage gate's 0/60 becomes a lie.
    """
    with SHIPPED.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 60
    assert {str(r["human_label"]).strip() for r in rows} == {""}


def test_the_shipped_file_leads_with_false_hits_then_disagreements() -> None:
    """The ordering the README promises, checked against the file it ships.

    The defect was a file of sixty exact hits scored 2 or 1 — the least
    informative rows the store holds — under a README sentence claiming false
    hits came first.
    """
    with SHIPPED.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    tiers = [int(r["priority"]) for r in rows]
    assert tiers == sorted(tiers)
    assert tiers.count(1) == 34
    assert tiers.count(2) == 19
    assert tiers.count(3) == 7
    for row in rows:
        assert row["priority_reason"] == PRIORITY_LABELS[int(row["priority"])]
    for row in rows[:34]:
        assert row["kind"] == "cache_semantic"
        assert row["judge1_score"] == "0"


def test_export_is_deterministic_and_never_writes_a_label(tmp_path: Path) -> None:
    """Regenerating the file reproduces it byte for byte, still unlabelled."""
    cfg = Config.load(Path("config/bench.yaml"))
    out = tmp_path / "regenerated.csv"
    export_human_csv(cfg, out, limit=60)
    assert out.read_bytes() == SHIPPED.read_bytes()
    again = tmp_path / "again.csv"
    export_human_csv(cfg, again, limit=60)
    assert again.read_bytes() == out.read_bytes()
    with out.open(newline="", encoding="utf-8") as fh:
        assert {str(r["human_label"]) for r in csv.DictReader(fh)} == {""}


def test_priority_counts_match_the_documented_tiers() -> None:
    """docs/HUMAN_LABELING.md states these four counts; they come from here."""
    cfg = Config.load(Path("config/bench.yaml"))
    assert priority_counts(cfg) == {1: 34, 2: 19, 3: 26, 4: 1370}
