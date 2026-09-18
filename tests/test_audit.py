"""The docs audit catches number drift between README and results.json."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_audit() -> ModuleType:
    path = Path("scripts/audit_docs.py")
    spec = importlib.util.spec_from_file_location("audit_docs_standalone", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_headline_tables_agree_with_results() -> None:
    audit = _load_audit()
    assert audit.check_headline_tables(audit.load_results()) == []


def test_figures_referenced_exist_and_none_orphaned() -> None:
    audit = _load_audit()
    assert audit.check_figures() == []


def test_open_defects_regenerates_identically() -> None:
    audit = _load_audit()
    rendered: str = audit.render_open_defects()
    assert rendered == Path("docs/OPEN_DEFECTS.md").read_text(encoding="utf-8")


def test_audit_flags_a_doctored_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    doctored = readme.replace(
        "| baseline | long only | 0.9226 (95% boot",
        "| baseline | long only | 0.1000 (95% boot",
        1,
    )
    assert doctored != readme
    target = tmp_path / "README.md"
    target.write_text(doctored, encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_headline_tables(audit.load_results())
    assert any("cost_usd" in e for e in errors)


def test_audit_flags_a_doctored_ratio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One wrong digit in '24 of 69' must fail the arithmetic check."""
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    doctored = readme.replace("24 of\n69 hits (34.8%)", "24 of\n70 hits (34.8%)", 1)
    assert doctored != readme
    target = tmp_path / "README.md"
    target.write_text(doctored, encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_ratios(audit.load_results())
    assert any("24 of 70" in e for e in errors)


def test_audit_flags_a_doctored_percentage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    doctored = readme.replace("0.0889 (4 of 45", "0.0999 (4 of 45", 1)
    assert doctored != readme
    target = tmp_path / "README.md"
    target.write_text(doctored, encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_headline_tables(audit.load_results())
    assert any("false-hit" in e for e in errors)


def test_audit_flags_a_count_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text(readme + "\n237-238 successful rows.\n", encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_ratios(audit.load_results())
    assert any("range" in e for e in errors)


def test_audit_flags_an_unregistered_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text(readme + "\nThe cache beats every baseline everywhere.\n", encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_comparisons(audit.load_results())
    assert any("unregistered comparison" in e for e in errors)


def test_audit_flags_an_external_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text(readme + "\n![](https://example.com/plot.png)\n", encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_figures()
    assert any("http" in e for e in errors)
