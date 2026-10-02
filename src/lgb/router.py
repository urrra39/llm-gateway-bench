"""Two routing strategies, compared rather than assumed.

- cheap_heuristic: request length + presence of reasoning cues decides easy/hard.
- cascade: call the cheap model first; escalate to the expensive model when the
  cheap answer signals low confidence. Costs the cheap call either way.
"""

from __future__ import annotations

from dataclasses import dataclass

from lgb.config import RouterSpec


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
        # Substring, not word, matching: "how" also fires inside "show". The
        # committed runs were routed this way, so changing it would make them
        # irreproducible; see docs/OPEN_DEFECTS.md D8.
        cues = self.spec.heuristic.reasoning_cues
        is_long = len(request) > self.spec.heuristic.max_chars
        low = request.casefold()
        has_cue = any(cue in low for cue in cues)
        if is_long or has_cue:
            return RouteDecision("hard", f"long={is_long} cue={has_cue}")
        return RouteDecision("easy", "short and no reasoning cue")


class CascadeRouter:
    """Cheap-first with escalation. The decision is only final after the cheap
    model answers: an empty answer, or one containing any of the configured
    `cascade_escalation_words`, escalates to the expensive model."""

    def __init__(self, spec: RouterSpec) -> None:
        self.words = tuple(w.casefold() for w in spec.cascade_escalation_words)

    def decide(self, _request: str, cheap_answer: str) -> RouteDecision:
        low = cheap_answer.casefold()
        if not cheap_answer.strip() or any(w in low for w in self.words):
            return RouteDecision(
                "hard", "cheap answer uncertain or empty", cheap_answer=cheap_answer, escalate=True
            )
        return RouteDecision("easy", "cheap answer confident", cheap_answer=cheap_answer)
