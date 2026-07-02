"""
probe.py — прогон workload'а по векторному стору и сбор retrieval-frequency.

Прогоняет запросы через store.search(top_k), собирает top-k id на каждый
запрос и переводит в retrieval-frequency, из которой метрики считаются
отдельно (metrics.build_report).
"""

from __future__ import annotations
from dataclasses import dataclass, field

from retrieval_fairness.types import Query, VectorStore
from retrieval_fairness.metrics import retrieval_frequencies, build_report, FairnessReport


@dataclass
class ProbeResult:
    """Результат прогона: частоты + сводный отчёт."""
    freqs: dict[str, int]
    hits_per_query: list[list[str]] = field(default_factory=list)
    report: FairnessReport | None = None


def probe(
    store: VectorStore,
    queries: list[Query],
    top_k: int = 10,
    corpus_ids: list[str] | None = None,
) -> ProbeResult:
    """
    Прогнать запросы по стору, собрать retrieval-frequency и отчёт.

    corpus_ids: список id корпуса. Если None — берётся из store.list_chunks().
    Нужен, чтобы включить чанки, которые ни разу не нашлись (dark matter).
    """
    if corpus_ids is None:
        corpus_ids = [c.id for c in store.list_chunks()]

    hits_per_query: list[list[str]] = []
    for q in queries:
        hits = store.search(q.vector, top_k)
        hits_per_query.append([h.chunk_id for h in hits])

    freqs = retrieval_frequencies(hits_per_query, corpus_ids)
    report = build_report(freqs, n_queries=len(queries), top_k=top_k)
    return ProbeResult(freqs=freqs, hits_per_query=hits_per_query, report=report)
