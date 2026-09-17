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
