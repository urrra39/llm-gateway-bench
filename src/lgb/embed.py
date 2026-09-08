"""Embeddings from a small CPU sentence-transformer, loaded lazily.

Choice recorded: local all-MiniLM-L6-v2 (about 90 MB, runs on CPU) rather than
an embedding API, because the environment has no GPU and no embedding API key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_name: str, models_dir: Path) -> None:
        self.model_name = model_name
        self.models_dir = models_dir
        self._model = None  # type: ignore[assignment]
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(self.model_name, cache_folder=str(self.models_dir))
        self._loaded = True

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        assert self._model is not None
        vecs = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def close(self) -> None:  # pragma: no cover - lifecycle helper
        self._model = None
        self._loaded = False


def cosine_sim_matrix(query: np.ndarray, stored: np.ndarray) -> np.ndarray:
    """query (m, d) x stored (n, d) normalized cosine similarity."""
    return query @ stored.T
