"""
stores.py — реализации VectorStore.

InMemoryVectorStore — лёгкий стор на косинусном сходстве через numpy.
Нужен для разработки/тестов/демо: работает без внешних БД и тяжёлых
зависимостей. В проде заменяется адаптером под FAISS/Qdrant/Pinecone/pgvector
по тому же контракту VectorStore.
"""

from __future__ import annotations
from typing import Iterator

import numpy as np

from retrieval_fairness.types import Chunk, Hit


def _cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Косинусное сходство query (1D) к каждой строке matrix (2D)."""
    if matrix.shape[0] == 0:
        return np.array([])
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-12:
        q_norm = 1e-12
    m_norm = np.linalg.norm(matrix, axis=1)
    m_norm = np.where(m_norm < 1e-12, 1e-12, m_norm)
    return (matrix @ query) / (m_norm * q_norm + 1e-12)


class InMemoryVectorStore:
    """
    Векторный стор в памяти: косинусный поиск по numpy-матрице.

    chunks: list[Chunk] — корпус; Chunk.vector обязан быть заполнен.
    """

    def __init__(self, chunks: list[Chunk]):
        assert all(c.vector is not None for c in chunks), "InMemoryVectorStore требует Chunk.vector"
        self._chunks = list(chunks)
        self._ids = [c.id for c in self._chunks]
        self._matrix = np.array([c.vector for c in self._chunks], dtype=float)

    def search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        sims = _cosine_matrix(np.array(query_vec, dtype=float), self._matrix)
        if sims.size == 0:
            return []
        k = min(top_k, sims.size)
        # top-k индексов по убыванию сходства
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [
            Hit(chunk_id=self._ids[i], score=float(sims[i]), rank=r + 1)
            for r, i in enumerate(idx)
        ]

    def list_chunks(self) -> Iterator[Chunk]:
        yield from self._chunks

    @property
    def size(self) -> int:
        return len(self._chunks)
