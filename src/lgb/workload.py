"""Workload construction from the committed PAWS sample.

Ground truth for "should this request have been a cache hit" exists because
PAWS is a labelled paraphrase corpus: label 1 says two sentences are
paraphrases (a later request SHOULD hit the earlier one), label 0 says two
sentences are near-duplicates but NOT paraphrases (a later request SHOULD NOT
hit; these are the false-hit traps).

Duplicate structure per occurrence list:
- novel: a request with no earlier equivalent.
- paraphrase: sentence2 of a label-1 pair whose sentence1 appeared earlier.
- trap: sentence2 of a label-0 pair whose sentence1 appeared earlier.
- exact: a verbatim repeat of an earlier request.

Occurrences are ordered so every anchor precedes its paraphrase/trap, and the
duplicate fractions are input parameters recorded per workload. The anchor
block comes first, so the cache is fully warm before any paraphrase fires;
that is an optimistic choice for paraphrase hits and is stated as such. The
reported numbers are only as meaningful as this construction.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lgb.config import Config, DuplicateFractionSpec
from lgb.records import DupType, WorkloadRow


@dataclass(frozen=True)
class Pair:
    anchor: str
    follow: str
    label: int  # 1 paraphrase, 0 trap
    source: str


def load_pairs(path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pairs.append(
                Pair(
                    anchor=str(row["sentence1"]),
                    follow=str(row["sentence2"]),
                    label=int(row["label"]),
                    source=str(row["id"]),
                )
            )
    return pairs


def _fill_requests(template: str, texts: list[str]) -> list[str]:
    return [template.format(sentence=t) for t in texts]


def build_workload(
    cfg: Config, frac: DuplicateFractionSpec, pairs: list[Pair], rng: np.random.Generator
) -> list[WorkloadRow]:
    n = cfg.workload.n_per_fraction
    p = round(frac.paraphrase * n)
    t = round(frac.trap * n)
    e = round(frac.exact * n)
    novel_occ = n - p - t - e
    if novel_occ < p + t:
        raise ValueError("workload too small for its duplicate structure")
    template = cfg.workload.request_template

    paraphrase_pairs = [x for x in pairs if x.label == 1]
    trap_pairs = [x for x in pairs if x.label == 0]
    rng.shuffle(paraphrase_pairs)
    rng.shuffle(trap_pairs)
    p_anchors = paraphrase_pairs[:p]
    t_anchors = trap_pairs[:t]

    anchor_pairs = p_anchors + t_anchors
    anchor_texts = [x.anchor for x in anchor_pairs]
    used = set(anchor_texts)
    pool = [x.anchor for x in pairs if x.anchor not in used]
    rng.shuffle(pool)
    extra_needed = novel_occ - len(anchor_pairs)
    extra_texts: list[str] = []
    i = 0
    while len(extra_texts) < extra_needed:
        text = pool[i % len(pool)]
        if text not in used:
            extra_texts.append(text)
            used.add(text)
        i += 1

    # anchor rows: first novel occurrence of every underlying request
    novel_rows: list[WorkloadRow] = []
    all_texts_so_far: list[str] = []
    for i, text in enumerate(anchor_texts + extra_texts):
        group = f"g{i:05d}"
        req = template.format(sentence=text)
        novel_rows.append(WorkloadRow(idx=i, request=req, dup_type="novel", group_id=group))
        all_texts_so_far.append(req)

    # follower events: paraphrase / trap / exact, in random order
    events: list[tuple[str, DupType, str]] = []  # (request, type, group)
    for pr in p_anchors:
        events.append((template.format(sentence=pr.follow), "paraphrase", ""))
    for tr in t_anchors:
        events.append((template.format(sentence=tr.follow), "trap", ""))
    # map both the anchor and the follow text of every paired row to its group
    anchor_group: dict[str, str] = {}
    for i, x in enumerate(anchor_pairs):
        anchor_group[template.format(sentence=x.anchor)] = f"g{i:05d}"
        anchor_group[template.format(sentence=x.follow)] = f"g{i:05d}"
    events = [_resolve(req, dtype, anchor_group) for req, dtype, _g in events]

    for _ in range(e):
        target_text = all_texts_so_far[rng.integers(0, len(all_texts_so_far))]
        events.append((target_text, "exact", _group_of(novel_rows, target_text)))
    rng.shuffle(events)

    tail_rows: list[WorkloadRow] = []
    for req, dtype, group in events:
        tail_rows.append(WorkloadRow(idx=0, request=req, dup_type=dtype, group_id=group))
        all_texts_so_far.append(req)

    ordered = _linear_extension(novel_rows, tail_rows, rng)
    rows = [
        WorkloadRow(idx=i, request=r.request, dup_type=r.dup_type, group_id=r.group_id)
        for i, r in enumerate(ordered)
    ]
    if len(rows) != n:
        raise RuntimeError(f"workload {frac.name}: built {len(rows)} rows, wanted {n}")
    return rows


def _linear_extension(
    novel_rows: list[WorkloadRow], tail_rows: list[WorkloadRow], rng: np.random.Generator
) -> list[WorkloadRow]:
    """A random order in which every anchor precedes its paraphrase/trap and
    every exact repeat follows the request it duplicates, so a short prefix of
    the workload already contains cache hits and misses mixed."""
    remaining = list(novel_rows) + list(tail_rows)
    placed_groups: set[str] = set()
    placed_texts: set[str] = set()
    out: list[WorkloadRow] = []
    for _ in range(len(remaining)):
        candidates = [
            r
            for r in remaining
            if (r.dup_type == "novel")
            or (r.dup_type in ("paraphrase", "trap") and r.group_id in placed_groups)
            or (r.dup_type == "exact" and r.request in placed_texts)
        ]
        if not candidates:  # pragma: no cover - invariant guarantees one exists
            raise RuntimeError("linear extension stalled")
        pick = candidates[int(rng.integers(0, len(candidates)))]
        remaining.remove(pick)
        placed_groups.add(pick.group_id)
        placed_texts.add(pick.request)
        out.append(pick)
    return out


def _resolve(req: str, dtype: DupType, anchor_group: dict[str, str]) -> tuple[str, DupType, str]:
    group = anchor_group.get(req)
    if group is None:
        raise RuntimeError(f"cannot find anchor group for {req[:60]!r}")
    return (req, dtype, group)


def _group_of(rows: list[WorkloadRow], text: str) -> str:
    for r in rows:
        if r.request == text:
            return r.group_id
    raise RuntimeError("exact-repeat target not found")


def split_tune_report(
    rows: list[WorkloadRow], frac: float, seed: int
) -> tuple[list[WorkloadRow], list[WorkloadRow]]:
    """Split whole groups into a tuning half and a reporting half so no group
    straddles the boundary (an anchor and its paraphrase stay together)."""
    groups: dict[str, list[WorkloadRow]] = {}
    for r in rows:
        groups.setdefault(r.group_id, []).append(r)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    order = list(range(len(ids)))
    rng.shuffle(order)
    cut = int(len(ids) * frac)
    tune_ids = {ids[i] for i in order[:cut]}
    tune = [r for g in tune_ids for r in groups[g]]
    report = [r for g, grp in groups.items() if g not in tune_ids for r in grp]
    tune.sort(key=lambda r: r.idx)
    report.sort(key=lambda r: r.idx)
    return tune, report
