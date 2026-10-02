"""Benchmark engine accounting, with a stubbed upstream."""

from __future__ import annotations

from typing import Any

from lgb.chat import ChatResult
from lgb.config import Config
from lgb.run import _chat_nonempty


class EmptyThenAnswer:
    def __init__(self) -> None:
        self.budgets: list[int | None] = []

    def chat(self, _model: str, _text: str, **kw: Any) -> ChatResult:
        self.budgets.append(kw.get("max_tokens"))
        text = "" if len(self.budgets) == 1 else "answer"
        return ChatResult(text, 10, 512 if text == "" else 40, {}, latency_s=1.0)


def test_empty_retry_bills_both_calls() -> None:
    cfg = Config.load("config/bench.yaml")
    gw = EmptyThenAnswer()
    text, tokens_in, tokens_out, latency = _chat_nonempty(
        gw,  # type: ignore[arg-type]
        "m",
        "q",
        cfg,
        tier="cheap",
    )
    assert text == "answer"
    assert gw.budgets == [512, 1024]
    # the empty first call consumed its whole budget and is paid for
    assert (tokens_in, tokens_out, latency) == (20, 552, 2.0)


def test_executor_surfaces_a_persist_failure_instead_of_hanging(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A follower waits on its anchor; if persisting the anchor raised, the
    dispatcher used to spin forever because the anchor never completed."""
    import threading

    import pytest

    import lgb.run as run_mod
    from lgb.records import WorkloadRow

    rows = [
        WorkloadRow(0, "a", "novel", "g0"),
        WorkloadRow(1, "b", "novel", "g1"),
        WorkloadRow(2, "a2", "paraphrase", "g0"),
    ]
    monkeypatch.setattr(run_mod, "load_workload", lambda _cfg, _frac: rows)
    monkeypatch.setattr(run_mod, "split_tune_report", lambda r, _f, _s: ([], r))
    monkeypatch.setattr(run_mod, "Gateway", lambda _cfg: EmptyThenAnswer())

    def boom(*_a: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(run_mod, "append_rows", boom)
    cfg = Config.load("config/bench.yaml")
    errors: list[BaseException] = []

    def go() -> None:
        try:
            run_mod.execute_config(cfg, "baseline", "low", 0.9, None, tmp_path, workers=2)  # type: ignore[arg-type]
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=20)
    assert not t.is_alive(), "execute_config hung"
    with pytest.raises(OSError):
        raise errors[0]
