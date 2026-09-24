"""Embedding providers.

Two providers ship:

* ``hashing`` (default) - a deterministic, dependency-light embedder built on
  scikit-learn's HashingVectorizer over word + char n-grams, L2-normalised. No
  model download, no API key, runs anywhere, and is perfectly reproducible,
  which makes CI and the evaluation harness deterministic.
* ``sentence-transformers`` - real neural embeddings (default
  ``BAAI/bge-small-en-v1.5``), used in the graded runs reported in the README.
  Install ``requirements-ml.txt`` to enable it.

Both expose the same interface so the rest of the system never branches on
which one is active, and vectors are stored in the DB with the dimension they
were produced at (a provider switch triggers a re-embed - see
``ingestion/pipeline.py::reembed_all``).
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from core.config import settings
from core.logging_conf import get_logger

log = get_logger(__name__)


def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class BaseEmbedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.dim}"


class HashingEmbedder(BaseEmbedder):
    """Deterministic lexical-semantic embedding, zero downloads."""

    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        word_dim = max(dim // 2, 1)
        self._word = HashingVectorizer(
            n_features=word_dim,
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            stop_words="english",
            alternate_sign=False,
            norm=None,
        )
        self._char = HashingVectorizer(
            n_features=dim - word_dim,
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            alternate_sign=False,
            norm=None,
        )

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        safe = [t if t.strip() else " " for t in texts]
        word = self._word.transform(safe).toarray()
        char = self._char.transform(safe).toarray()
        mat = np.hstack([np.sqrt(word), 0.5 * np.sqrt(char)]).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class SentenceTransformerEmbedder(BaseEmbedder):
    """Neural embeddings via sentence-transformers (optional dependency)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._is_e5 = "e5" in model_name.lower()
        self._is_bge = "bge" in model_name.lower()

    def _prefix(self, text: str, is_query: bool) -> str:
        if self._is_e5:
            return ("query: " if is_query else "passage: ") + text
        if self._is_bge and is_query:
            return "Represent this sentence for searching relevant passages: " + text
        return text

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prepped = [self._prefix(t, is_query) for t in texts]
        vecs = self._model.encode(
            prepped, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.model_name}:{self.dim}"


@lru_cache(maxsize=4)
def get_embedder(provider: str | None = None, model: str | None = None) -> BaseEmbedder:
    provider = (provider or settings.embedding_provider).lower()
    model = model or settings.embedding_model
    if provider in {"sentence-transformers", "st", "neural"}:
        try:
            return SentenceTransformerEmbedder(model)
        except Exception as exc:  # pragma: no cover - optional dep missing
            log.warning(
                "sentence-transformers unavailable (%s); falling back to hashing embedder.", exc
            )
    return HashingEmbedder(dim=settings.embedding_dim)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
