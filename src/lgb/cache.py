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
    last_embed_ms: float = 0.0
    last_lookup_ms: float = 0.0

    # -- exact index ------------------------------------------------
    def _exact_index(self) -> dict[str, int]:
        return {normalize_exact(i.request): i.idx for i in self.items}

    def add(self, item: CacheItem) -> None:
        self.items.append(item)
        if self._vectors is not None and self.embedder is not None:
            vec = self.embedder.encode([item.request])[0]
            self._vectors = np.vstack([self._vectors, vec])
        else:
            self._vectors = None

    def __len__(self) -> int:
        return len(self.items)

    def _stored_vectors(self) -> np.ndarray:
        if self._vectors is None or self._vectors.shape[0] != len(self.items):
            assert self.embedder is not None, "semantic lookup needs an embedder"
            vecs = self.embedder.encode([i.request for i in self.items])
            self._vectors = vecs
        return self._vectors

    def lookup(self, request: str) -> LookupResult:
        import time as _time

        start = _time.perf_counter()
        if self.exact and not self.items:
            return LookupResult("miss", 0.0, None)
        if self.exact:
            hit = self._exact_index().get(normalize_exact(request))
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
            cache.items = [
                CacheItem(
                    idx=int(r.idx),
                    request=str(r.request),
                    group_id=str(r.group_id),
                    answer_text=str(r.answer_text),
                    model=str(r.model),
                )
                for r in table.itertuples(index=False)
            ]
            cache._vectors = None
        return cache


def similarity_of(embedder: Embedder, a: str, b: str) -> float:
    va, vb = embedder.encode([a, b])
    return float(va @ vb)


def evaluate_threshold(
    positives: np.ndarray, negatives: np.ndarray, threshold: float
) -> dict[str, float]:
    """Precision/recall for 'should this pair hit' at a threshold.

    positives: cosine similarities of paraphrase (should-hit) pairs.
    negatives: cosine similarities of trap and unrelated (should-not-hit) pairs.
    """
    tp = int(np.sum(positives >= threshold))
    fn = len(positives) - tp
    fp = int(np.sum(negatives >= threshold))
    tn = len(negatives) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def choose_threshold(
    positives: np.ndarray,
    negatives: np.ndarray,
    candidates: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Pick the threshold that maximises F1 over the labelled pairs.

    False hits (positives below threshold / negatives above it) both cost, but
    a false hit is the expensive failure, so the objective weights precision
    and recall equally (F1) and the caller reports the chosen operating point.
    """
    if candidates is None:
        all_sims = np.concatenate([positives, negatives])
        candidates = np.unique(np.round(all_sims, 4)) if len(all_sims) else np.array([0.5])
    best_f1 = -1.0
    best_thr = float(np.max(candidates))
    best_stats: dict[str, float] = {}
    for thr in np.sort(candidates)[::-1]:
        stats = evaluate_threshold(positives, negatives, float(thr))
        denom = stats["precision"] + stats["recall"]
        f1 = 2 * stats["precision"] * stats["recall"] / denom if denom else 0.0
        if f1 > best_f1:  # ties keep the higher threshold (fewer false hits)
            best_f1 = f1
            best_thr = float(thr)
            best_stats = stats
    return best_thr, best_stats
