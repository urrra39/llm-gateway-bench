"""Two routing strategies, compared rather than assumed.

- cheap_heuristic: request length + presence of reasoning cues decides easy/hard.
- cascade: call the cheap model first; escalate to the expensive model when the
  cheap answer signals low confidence. Costs the cheap call either way.
"""

from __future__ import annotations

from dataclasses import dataclass

from lgb.config import RouterSpec

UNCERTAINTY_HINTS = (
    "not sure",
    "uncertain",
    "cannot",
    "can't",
    "i don't know",
    "i don't know",
    "unable",
    "sorry",
)


@dataclass(frozen=True)
class RouteDecision:
    route: str  # easy -> cheap model; hard -> expensive model
    reason: str
    cheap_answer: str | None = None  # cascade only
    escalate: bool = False


class HeuristicRouter:
    def __init__(self, spec: RouterSpec) -> None:
        self.spec = spec

    def decide(self, request: str) -> RouteDecision:
        cues = self.spec.heuristic.reasoning_cues
        is_long = len(request) > self.spec.heuristic.max_chars
        low = request.casefold()
        has_cue = any(cue in low for cue in cues)
        if is_long or has_cue:
            return RouteDecision("hard", f"long={is_long} cue={has_cue}")
        return RouteDecision("easy", "short and no reasoning cue")


class CascadeRouter:
    """Cheap-first with escalation. The decision is only final after the cheap
    model answers: an explicit uncertainty hint, an empty answer, or a refusal
    escalates to the expensive model."""

    def __init__(self, spec: RouterSpec) -> None:
        self.spec = spec

    def decide(self, _request: str, cheap_answer: str) -> RouteDecision:
        low = cheap_answer.casefold()
        escalate = any(w in low for w in UNCERTAINTY_HINTS) or not cheap_answer.strip()
        if escalate:
            return RouteDecision(
                "hard", "cheap answer uncertain or empty", cheap_answer=cheap_answer, escalate=True
            )
        return RouteDecision("easy", "cheap answer confident", cheap_answer=cheap_answer)


def route_for(config: str, spec: RouterSpec) -> HeuristicRouter | CascadeRouter | None:
    if config == "router_heuristic":
        return HeuristicRouter(spec)
    if config == "router_cascade":
        return CascadeRouter(spec)
    return None
