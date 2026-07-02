"""
embedders.py — контракт Embedder + реализации.

Embedder.encode(texts) -> matrix. Приводит любой эмбеддер (TF-IDF,
sentence-transformers, ...) к общему интерфейсу, чтобы probe/synth
работали поверх любого, не зная внутренностей.

Шаг 9.2: TfidfEmbedder (есть, лёгкий) + SentenceTransformerEmbedder
(реалистичный, optional dependency).
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Контракт эмбеддера: fit на корпусе, encode произвольных текстов."""
    def fit(self, texts: list[str]) -> "Embedder": ...
    def encode(self, texts: list[str]) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


class TfidfEmbedder:
    """Лёгкий TF-IDF эмбеддер (sklearn). Без тяжёлых моделей."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 1)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(ngram_range=ngram_range)
        self._fitted = False
        self._dim_val = 0

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self._vec.fit(texts)
        self._fitted = True
        self._dim_val = len(self._vec.vocabulary_)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder not fitted; call .fit() first")
        return self._vec.transform(texts).toarray().astype(float)

    @property
    def dim(self) -> int:
        return self._dim_val


class SentenceTransformerEmbedder:
    """
    Dense-эмбеддер через sentence-transformers (all-MiniLM-L6-v2 по умолчанию).
    Optional: pip install 'retrieval-fairness[models]'.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder requires sentence-transformers: "
                "pip install 'retrieval-fairness[models]'"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._dim_val = self._model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        # dense-эмбеддеры не требуют fit на корпусе; no-op
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(self._model.encode(texts, show_progress_bar=False), dtype=float)

    @property
    def dim(self) -> int:
        return self._dim_val


def get_embedder(name: str, **kw) -> Embedder:
    """Реестр эмбеддеров: 'tfidf' | 'minilm' | 'sbert'."""
    name = name.lower()
    if name == "tfidf":
        return TfidfEmbedder(**kw)
    if name in ("minilm", "sbert"):
        model = "sentence-transformers/all-MiniLM-L6-v2" if name == "minilm" else kw.pop("model_name", None)
        return SentenceTransformerEmbedder(model_name=model) if name == "minilm" else SentenceTransformerEmbedder(**kw)
    raise ValueError(f"unknown embedder: {name}")
