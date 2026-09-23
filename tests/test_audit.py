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


def test_audit_flags_a_sized_image_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped defect: an image host URL with a display-width query."""
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text(
        readme + "\n![](https://sspark.example.ai/i/lYwyABsnZmcYkfsx?width=1024)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_figures()
    assert any("external image URL" in e for e in errors)


def test_audit_flags_a_bare_url_used_as_a_figure_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2's damage class: a URL in the code span where a filename belongs."""
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text(
        readme + "\n`https://example.com/Iz3YIk6YJvK8ujGF` plots the trade curve.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_figures()
    assert any("bare external URL as a name" in e for e in errors)


def test_audit_allows_localhost_in_a_code_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gateway's own loopback URL is a real address, not a figure name."""
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text(readme + "\nUpstream is `http://127.0.0.1:8787/v1` locally.\n", "utf-8")
    monkeypatch.setattr(audit, "README", target)
    assert audit.check_figures() == []


def test_audit_flags_a_dangling_figure_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text(readme + "\n![](docs/figures/not_generated.svg)\n", encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_figures()
    assert any("not_generated.svg" in e and "not on disk" in e for e in errors)


def test_audit_flags_an_orphaned_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A committed plot no document mentions is dead weight, and was.

    Runs against an isolated tree: the guard scans every tracked document, so
    orphanhood is only observable when the whole document set is controlled.
    """
    audit = _load_audit()
    figures = tmp_path / "docs" / "figures"
    figures.mkdir(parents=True)
    (figures / "referenced.svg").write_text("<svg/>", encoding="utf-8")
    (figures / "orphan.svg").write_text("<svg/>", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("![](docs/figures/referenced.svg)\n", encoding="utf-8")
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setattr(audit, "README", readme)
    for name in ("CEILING", "DECISIONS", "OPEN_DEFECTS", "HUMAN_LABELING"):
        monkeypatch.setattr(audit, name, tmp_path / "docs" / f"{name}.md")
    errors: list[str] = audit.check_figures()
    assert errors == ["docs/figures/orphan.svg is on disk but unreferenced (orphaned)"]


def test_audit_scans_documents_beyond_the_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An external image smuggled into any tracked document must fail too."""
    audit = _load_audit()
    target = tmp_path / "HUMAN_LABELING.md"
    target.write_text("![](https://example.com/labels.png)\n", encoding="utf-8")
    monkeypatch.setattr(audit, "HUMAN_LABELING", target)
    errors: list[str] = audit.check_figures()
    assert any("HUMAN_LABELING.md" in e and "http" in e for e in errors)
