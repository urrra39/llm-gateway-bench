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
    doctored = readme.replace("24 of 69 hits (34.8%)", "24 of 70 hits (34.8%)", 1)
    assert doctored != readme
    target = tmp_path / "README.md"
    target.write_text(doctored, encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_ratios(audit.load_results())
    assert any("24 of 70" in e for e in errors)


def test_audit_flags_a_doctored_percentage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    # The gate table states the same rate earlier in the file, so target the
    # headline-table cell by its interval suffix.
    doctored = readme.replace("0.0889 (4 of 45; 95% Wilson", "0.0999 (4 of 45; 95% Wilson", 1)
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


def test_every_gate_has_a_registered_bound() -> None:
    """The falsifiability check passes on the shipped results.json."""
    audit = _load_audit()
    assert audit.check_gate_bounds(audit.load_results()) == []


def test_audit_rejects_a_gate_with_no_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The defect class: a gate asserting only that a number exists."""
    audit = _load_audit()
    data = audit.load_results()
    gates = list(data["gates"])
    gates.append({"name": "false_hit_rate_reported_low", "passed": True, "observed": "0.0889"})
    data["gates"] = gates
    errors: list[str] = audit.check_gate_bounds(data)
    assert any("no registered bound" in e for e in errors)


def test_audit_rejects_a_gate_verdict_the_bar_contradicts() -> None:
    """Flipping the stored verdict must not survive recomputation."""
    audit = _load_audit()
    data = audit.load_results()
    gates = [dict(g) for g in data["gates"]]
    for gate in gates:
        if gate["name"] == "false_hit_rate_within_bound_low_cache":
            gate["passed"] = True
    data["gates"] = gates
    errors: list[str] = audit.check_gate_bounds(data)
    assert any("false_hit_rate_within_bound_low_cache says True" in e for e in errors)


def test_audit_requires_the_readme_to_restate_every_gate_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing gate cannot be dropped from the front page."""
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    dropped = readme.replace(
        "| false_hit_rate_within_bound_high_cache | FAIL | "
        "0.0941 (8 of 85 semantic hits) vs bar 0.0500 |\n",
        "",
        1,
    )
    assert dropped != readme
    target = tmp_path / "README.md"
    target.write_text(dropped, encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_gate_bounds(audit.load_results())
    assert any(
        "false_hit_rate_within_bound_high_cache" in e and "lacks the row" in e for e in errors
    )


def test_exact_hit_counts_agree_with_parquet() -> None:
    """Both exact-hit counts, recomputed from outcomes.parquet, as shipped."""
    audit = _load_audit()
    assert audit.check_exact_hit_counts(audit.load_results()) == []


def test_audit_rejects_conflating_the_two_exact_hit_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """24 exact hits and 26 exact hits are different quantities.

    Stating the cache run's count against the replay's denominator is the
    defect: it reads as one number explaining the other.
    """
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text(readme + "\nThe replay serves 24 of 237 rows.\n", encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_exact_hit_counts(audit.load_results())
    assert any("24 of 237" in e and "attaches" in e for e in errors)


def test_audit_requires_the_exact_hit_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the two counts differ, the README must write the arithmetic."""
    audit = _load_audit()
    readme = Path("README.md").read_text(encoding="utf-8")
    doctored = readme.replace("26 is 24 plus", "26 is 25 plus", 1)
    assert doctored != readme
    target = tmp_path / "README.md"
    target.write_text(doctored, encoding="utf-8")
    monkeypatch.setattr(audit, "README", target)
    errors: list[str] = audit.check_exact_hit_counts(audit.load_results())
    assert any("does not reconcile" in e for e in errors)


def test_audit_human_gate_follows_the_csv_row_count() -> None:
    """The human-gate denominator is the CSV's real row count, not a literal.

    Reads data/human_validation.csv independently and requires the audit's
    derived pairs to carry (ceil(bar * n), n) and (filled, n) for that exact n.
    A change to the file's length without the audit re-deriving fails here,
    which is what stops a hand-written (30, 60) from surviving a resized file.
    """
    import csv as _csv
    from math import ceil

    from lgb.metrics import HUMAN_LABEL_COVERAGE_BAR

    audit = _load_audit()
    with Path("data/human_validation.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    total = len(rows)
    filled = sum(1 for r in rows if str(r.get("human_label", "")).strip())
    assert audit._human_validation_counts() == (filled, total)
    pairs = audit._known_pairs(audit.load_results())
    assert (ceil(HUMAN_LABEL_COVERAGE_BAR * total), total) in pairs
    assert (filled, total) in pairs


def test_known_pairs_track_a_resized_human_validation_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Growing the CSV moves the human-gate pair the audit accepts.

    With the counts helper returning a different (filled, total), the known
    pairs carry the new (needed, total), so a reversion to a literal (30, 60)
    would fail: neither derived pair below would be present.
    """
    from math import ceil

    from lgb.metrics import HUMAN_LABEL_COVERAGE_BAR

    audit = _load_audit()
    monkeypatch.setattr(audit, "_human_validation_counts", lambda: (7, 83))
    pairs = audit._known_pairs(audit.load_results())
    assert (ceil(HUMAN_LABEL_COVERAGE_BAR * 83), 83) in pairs
    assert (7, 83) in pairs
