"""
adapters/inmemory.py — in-memory векторный стор (косинус, numpy).

Лёгкий стор для разработки/тестов/демо: работает без внешних БД.
В проде заменяется FAISS/Qdrant/pgvector адаптерами по тому же контракту.
"""

from __future__ import annotations
from typing import Iterator

import numpy as np

from retrieval_fairness.types import Chunk, Hit
from retrieval_fairness.adapters.base import BaseVectorStoreAdapter


def _cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.array([])
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-12:
        q_norm = 1e-12
    m_norm = np.linalg.norm(matrix, axis=1)
    m_norm = np.where(m_norm < 1e-12, 1e-12, m_norm)
    return (matrix @ query) / (m_norm * q_norm + 1e-12)


class InMemoryVectorStore(BaseVectorStoreAdapter):
    """
    Векторный стор в памяти: косинусный поиск по numpy-матрице.
    chunks: list[Chunk] — корпус; Chunk.vector обязан быть заполнен.
    """

    def __init__(self, chunks: list[Chunk]):
        super().__init__()
        assert all(c.vector is not None for c in chunks), "InMemoryVectorStore требует Chunk.vector"
        self._chunks = list(chunks)
        self._ids = [c.id for c in self._chunks]
        self._matrix = np.array([c.vector for c in self._chunks], dtype=float)

    def _search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        sims = _cosine_matrix(np.array(query_vec, dtype=float), self._matrix)
        if sims.size == 0:
            return []
        k = min(top_k, sims.size)
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [
            Hit(chunk_id=self._ids[i], score=float(sims[i]), rank=r + 1)
            for r, i in enumerate(idx)
        ]

    def _list_chunk_ids(self) -> Iterator[str]:
        yield from self._ids

    def list_chunks(self) -> Iterator[Chunk]:
        yield from self._chunks

    @property
    def size(self) -> int:
        return len(self._chunks)
