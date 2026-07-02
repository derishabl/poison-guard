"""
adapters/qdrant.py — Qdrant-адаптер.

Работает поверх Qdrant (local или cloud) через qdrant-client.
Зависимость: pip install 'retrieval-fairness[qdrant]'

Тесты skip'аются без QDRANT_TEST_URL.
"""

from __future__ import annotations
from typing import Iterator

from retrieval_fairness.types import Hit
from retrieval_fairness.adapters.base import BaseVectorStoreAdapter


class QdrantAdapter(BaseVectorStoreAdapter):
    """
    Qdrant-адаптер.

    url: Qdrant endpoint (http://localhost:6333).
    collection: имя коллекции.
    api_key: для Qdrant Cloud (опционально).
    vector_name: имя вектора в коллекции (для named-vectors; None = default).
    """

    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        vector_name: str | None = None,
    ):
        super().__init__()
        try:
            from qdrant_client import QdrantClient  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "QdrantAdapter requires qdrant-client: pip install 'retrieval-fairness[qdrant]'"
            ) from e
        self._url = url
        self._collection = collection
        self._api_key = api_key
        self._vector_name = vector_name

    def _client(self):
        from qdrant_client import QdrantClient
        return QdrantClient(url=self._url, api_key=self._api_key)

    def _search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        from qdrant_client.models import SearchParams, Query
        with self._client() as client:
            # client.search() deprecated в qdrant-client >=1.10 (убран в 2.x).
            # query_points() — современный API. Совместимость: пробуем новый,
            # при AttributeError падаем на старый (старые версии qdrant-client).
            query = Query(nearest=query_vec if self._vector_name is None
                          else {self._vector_name: query_vec})
            try:
                res = client.query_points(
                    collection_name=self._collection,
                    query=query,
                    limit=top_k,
                    search_params=SearchParams(hnsw_ef=128, exact=False),
                ).points
            except AttributeError:
                # старый qdrant-client (<1.10): нет query_points -> search()
                res = client.search(
                    collection_name=self._collection,
                    query_vector=query_vec if self._vector_name is None else (self._vector_name, query_vec),
                    limit=top_k,
                    search_params=SearchParams(hnsw_ef=128, exact=False),
                )
        out = []
        for rank, point in enumerate(res, start=1):
            out.append(Hit(chunk_id=str(point.id), score=float(point.score), rank=rank))
        return out

    def _list_chunk_ids(self) -> Iterator[str]:
        # scroll API — пагинация по всему корпусу
        with self._client() as client:
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=self._collection,
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for p in points:
                    yield str(p.id)
                if offset is None:
                    break

    @property
    def size(self) -> int:
        with self._client() as client:
            info = client.get_collection(self._collection)
            return info.points_count or 0
