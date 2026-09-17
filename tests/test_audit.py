"""The docs audit catches number drift between README and results.json."""

from __future__ import annotations

from pathlib import Path

import pytest
import scripts.audit_docs as audit


def test_headline_tables_agree_with_results() -> None:
    data = audit.load_results()
    assert audit.check_headline_tables(data) == []


def test_figures_referenced_exist_and_none_orphaned() -> None:
    assert audit.check_figures() == []


def test_open_defects_regenerates_identically() -> None:
    assert audit.render_open_defects() == Path("docs/OPEN_DEFECTS.md").read_text(encoding="utf-8")


def test_audit_flags_a_doctored_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    doctored = readme.replace(
        "| baseline | long only | 0.9226 |", "| baseline | long only | 0.1000 |", 1
    )
    target = tmp_path / "README.md"
    target.write_text(doctored, encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors = audit.check_headline_tables(audit.load_results())
    assert any("cost_usd" in e for e in errors)
