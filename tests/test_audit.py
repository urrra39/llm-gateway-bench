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
        "| baseline | long only | 0.9226 |", "| baseline | long only | 0.1000 |", 1
    )
    target = tmp_path / "README.md"
    target.write_text(doctored, encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_headline_tables(audit.load_results())
    assert any("cost_usd" in e for e in errors)
