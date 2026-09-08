"""Row schemas shared across the pipeline.

A workload row is one request occurrence with its known duplicate structure.
A decision row records what the gateway did with it: which model served it,
whether the cache produced the answer, at what similarity, and the timings and
tokens. Judge rows hold the rubric score(s) against the baseline answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DupType = Literal["novel", "exact", "paraphrase", "trap"]
DecisionKind = Literal[
    "model_direct",  # no cache in front, model answered
    "cache_exact",  # exact-match short-circuit hit
    "cache_semantic",  # semantic hit above threshold
    "miss",  # cache miss, model answered
]
ServedBy = Literal["cheap", "expensive"]


@dataclass(frozen=True)
class WorkloadRow:
    idx: int
    request: str
    dup_type: DupType
    group_id: str  # same underlying question shares a group id


@dataclass(frozen=True)
class DecisionRow:
    config: str  # baseline | cache | router
    frac: str  # low | high
    idx: int
    request: str
    dup_type: DupType
    group_id: str
    kind: DecisionKind
    served_by: ServedBy
    similarity: float | None  # semantic-hit similarity; None otherwise
    hit_source_idx: int | None  # row idx whose stored answer was returned
    latency_ms: float
    embed_ms: float
    lookup_ms: float
    model_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    answer_text: str
    baseline_row: int | None = None  # for router: baseline idx already answered
    error: str | None = None


@dataclass
class JudgeRow:
    config: str
    frac: str
    idx: int
    request: str
    dup_type: DupType
    answer_text: str
    baseline_text: str
    identical: bool
    judge1_score: int | None = None  # 0/1/2
    judge2_score: int | None = None
    judge1_raw: str | None = None
    judge2_raw: str | None = None
    human_label: str = ""  # empty until a human fills it
    sampled_for_judge2: bool = False
    note: str = ""


def decision_columns() -> list[str]:
    return list(DecisionRow.__dataclass_fields__)
