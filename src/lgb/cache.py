"""The semantic cache under test: exact-match short-circuit, then embedding
cosine similarity above a threshold. Persisted to disk so a run resumes with
its history.

A "should-hit" label exists in the workload for tuning (a paraphrase of an
earlier request should hit; a near-duplicate non-paraphrase — a trap — should
not). The threshold is chosen on a held-out half of the workload, never on the
rows that are reported.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lgb.embed import Embedder
from lgb.store import read_parquet, write_parquet


def normalize_exact(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


@dataclass
class CacheItem:
    idx: int
    request: str
    group_id: str
    answer_text: str
    model: str


@dataclass
class LookupResult:
    kind: str  # cache_exact | cache_semantic | miss
    similarity: float
    hit_idx: int | None


@dataclass
class SemanticCache:
    threshold: float
    embedder: Embedder | None = None
    exact: bool = True
    items: list[CacheItem] = field(default_factory=list)
    _vectors: np.ndarray | None = None
    #: normalized request -> idx of the item stored for it; kept in step
    #: with items so an exact lookup is O(1) instead of a rebuild per request.
    _exact: dict[str, int] = field(default_factory=dict)
    last_embed_ms: float = 0.0
    last_lookup_ms: float = 0.0
    #: Guards items/_vectors so concurrent lookups and adds (the benchmark runs
    #: several model calls in parallel) stay consistent. An RLock because the
    #: public methods are called from code that may already hold it.
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def add(self, item: CacheItem) -> None:
        with self._lock:
            self.items.append(item)
            self._exact[normalize_exact(item.request)] = item.idx
            if self._vectors is not None and self.embedder is not None:
                vec = self.embedder.encode([item.request])[0]
                self._vectors = np.vstack([self._vectors, vec])
            else:
                self._vectors = None

    def __len__(self) -> int:
        with self._lock:
            return len(self.items)

    def get(self, idx: int) -> CacheItem:
        """The stored item with this idx (the one a lookup's hit_idx names)."""
        with self._lock:
            return next(i for i in self.items if i.idx == idx)

    def _stored_vectors(self) -> np.ndarray:
        if self._vectors is None or self._vectors.shape[0] != len(self.items):
            assert self.embedder is not None, "semantic lookup needs an embedder"
            vecs = self.embedder.encode([i.request for i in self.items])
            self._vectors = vecs
        return self._vectors

    def lookup(self, request: str) -> LookupResult:
        import time as _time

        with self._lock:
            start = _time.perf_counter()
            if self.exact and not self.items:
                return LookupResult("miss", 0.0, None)
            if self.exact:
                hit = self._exact.get(normalize_exact(request))
                if hit is not None:
                    self.last_embed_ms = 0.0
                    self.last_lookup_ms = (_time.perf_counter() - start) * 1000.0
                    return LookupResult("cache_exact", 1.0, hit)
            if self.embedder is None or not self.items:
                return LookupResult("miss", 0.0, None)
            stored = self._stored_vectors()
            e_start = _time.perf_counter()
            q = self.embedder.encode([request])[0]
            embed_ms = (_time.perf_counter() - e_start) * 1000.0
            sims = stored @ q
            best = int(np.argmax(sims))
            self.last_embed_ms = embed_ms
            self.last_lookup_ms = (_time.perf_counter() - start) * 1000.0
            if float(sims[best]) >= self.threshold:
                return LookupResult("cache_semantic", float(sims[best]), self.items[best].idx)
            return LookupResult("miss", float(sims[best]), None)

    def save(self, path: Path) -> None:
        with self._lock:
            if not self.items:
                write_parquet(pd.DataFrame(), path)
                return
            frame = pd.DataFrame(
                [
                    {
                        "idx": i.idx,
                        "request": i.request,
                        "group_id": i.group_id,
                        "answer_text": i.answer_text,
                        "model": i.model,
                    }
                    for i in self.items
                ]
            )
            write_parquet(frame, path)

    @classmethod
    def load(
        cls, path: Path, threshold: float, embedder: Embedder | None, exact: bool = True
    ) -> SemanticCache:
        cache = cls(threshold=threshold, embedder=embedder, exact=exact)
        table = read_parquet(path)
        if table is not None and len(table):
            for r in table.itertuples(index=False):
                cache.add(
                    CacheItem(
                        idx=int(r.idx),
                        request=str(r.request),
                        group_id=str(r.group_id),
                        answer_text=str(r.answer_text),
                        model=str(r.model),
                    )
                )
        return cache


def simulate_threshold(
    vectors: np.ndarray,
    should_hit: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Replay a workload against a growing cache and score the hit decisions.

    The live cache does not compare a request against one stored request; it
    takes the maximum cosine similarity over everything stored so far. That
    maximum rises as the cache fills, so a threshold chosen on isolated pairs
    understates the false-hit rate badly: with hundreds of stored requests an
    unrelated request finds some spurious near-neighbour. Tuning therefore has
    to score the same statistic the cache uses.

    vectors:    (n, d) row embeddings in workload order, L2-normalised.
    should_hit: (n,) bool ground truth; True only where an earlier equivalent
                request exists (paraphrase / exact repeat).
    eligible:   (n,) bool, False for rows the exact-match short-circuit serves
                before any embedding, which the semantic threshold never sees.
    """
    tp = fp = fn = tn = 0
    for i in range(len(vectors)):
        if not eligible[i]:
            continue
        best = 0.0 if i == 0 else float(np.max(vectors[:i] @ vectors[i]))
        hit = best >= threshold
        if should_hit[i] and hit:
            tp += 1
        elif should_hit[i] and not hit:
            fn += 1
        elif not should_hit[i] and hit:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denom = precision + recall
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / denom if denom else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "false_hit_rate": fp / (tp + fp) if tp + fp else 0.0,
    }
