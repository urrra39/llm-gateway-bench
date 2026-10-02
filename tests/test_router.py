"""Router decisions with hand-computed expectations."""

from __future__ import annotations

from lgb.config import Config
from lgb.router import CascadeRouter, HeuristicRouter


def _spec() -> Config:
    return Config.load("config/bench.yaml")


def test_heuristic_routes_long_request_to_expensive() -> None:
    router = HeuristicRouter(_spec().router)
    long_req = "x" * 500
    assert router.decide(long_req).route == "hard"


def test_heuristic_routes_reasoning_cue_to_expensive() -> None:
    router = HeuristicRouter(_spec().router)
    dec = router.decide("Explain why the sky is blue because of scattering")
    assert dec.route == "hard"


def test_heuristic_routes_short_plain_request_to_cheap() -> None:
    router = HeuristicRouter(_spec().router)
    dec = router.decide("The cat sat on the mat.")
    assert dec.route == "easy"


def test_cascade_escalates_on_uncertainty() -> None:
    router = CascadeRouter(_spec().router)
    dec = router.decide("anything", "I am not sure about this answer")
    assert dec.escalate is True
    assert dec.route == "hard"


def test_cascade_escalates_on_empty_answer() -> None:
    router = CascadeRouter(_spec().router)
    dec = router.decide("anything", "   ")
    assert dec.escalate is True


def test_cascade_keeps_confident_cheap_answer() -> None:
    router = CascadeRouter(_spec().router)
    dec = router.decide("anything", "The cat sat on the mat.")
    assert dec.escalate is False
    assert dec.route == "easy"


def test_cascade_reads_escalation_words_from_config() -> None:
    spec = _spec().router.model_copy(update={"cascade_escalation_words": ("maybe",)})
    router = CascadeRouter(spec)
    assert router.decide("anything", "Maybe it is blue.").escalate is True
    assert router.decide("anything", "I am not sure.").escalate is False


def test_cascade_escalates_on_curly_apostrophe() -> None:
    router = CascadeRouter(_spec().router)
    assert router.decide("anything", "I don\u2019t know.").escalate is True
