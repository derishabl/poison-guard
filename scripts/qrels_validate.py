"""
qrels_validate.py — валидация dark matter через qrels (BEIR NQ, Шаг 9.7).

Отвечает на вопрос из docs/case_study_nq.md:
  «Сколько dark-matter чанков на самом деле релевантны каким-то запросам
  по qrels?» Если много — «потерянное золото» (корпус содержит релевантное,
  но ретривер его не находит). Если мало — dark matter действительно шум.

Дополнительно считает recall@k по qrels — сколько релевантных пар
(запрос, чанк) реально попало в top-k.

Использование:
  python scripts/qrels_validate.py --probe cases/nq_tfidf_sample.json \
      --qrels data/nq/qrels.json --queries data/nq/queries.jsonl

Не часть ядра exposure — инструмент интерпретации для кейса.
"""

from __future__ import annotations
import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate dark matter against qrels")
    ap.add_argument("--probe", required=True, help="probe JSON (save_probe / case_run --out)")
    ap.add_argument("--qrels", required=True, help="qrels.json: {query_id: {doc_id: score}}")
    ap.add_argument("--queries", required=True, help="queries.jsonl (порядок = hits_per_query)")
    ap.add_argument("--json", help="сохранить результат в JSON")
    args = ap.parse_args()

    with open(args.probe, encoding="utf-8") as f:
        probe = json.load(f)
    with open(args.qrels, encoding="utf-8") as f:
        qrels: dict[str, dict[str, int]] = json.load(f)
    query_ids = []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                query_ids.append(str(json.loads(line)["id"]))

    freqs: dict[str, int] = probe["freqs"]
    hits_per_query: list[list[str]] = probe["hits_per_query"]
    if len(query_ids) != len(hits_per_query):
        print(f"ERROR: queries ({len(query_ids)}) != hits_per_query ({len(hits_per_query)}); "
              "порядок/файл не соответствует прогону")
        return 2

    corpus_ids = set(freqs)
    dark = {cid for cid, v in freqs.items() if v == 0}

    # какие чанки корпуса релевантны хоть одному запросу workload'а (по qrels)
    relevant: set[str] = set()
    for qid in query_ids:
        for did in qrels.get(qid, {}):
            if did in corpus_ids:
                relevant.add(did)

    dark_relevant = dark & relevant  # «потерянное золото»

    # recall@k по qrels: релевантные пары, попавшие в top-k
    pairs_total = 0
    pairs_hit = 0
    for qid, hits in zip(query_ids, hits_per_query):
        rel_docs = [d for d in qrels.get(qid, {}) if d in corpus_ids]
        pairs_total += len(rel_docs)
        hit_set = set(hits)
        pairs_hit += sum(1 for d in rel_docs if d in hit_set)

    result = {
        "n_chunks": len(corpus_ids),
        "n_queries": len(query_ids),
        "dark_matter": len(dark),
        "relevant_in_corpus": len(relevant),
        "dark_and_relevant": len(dark_relevant),
        "dark_relevant_pct_of_dark": round(len(dark_relevant) / len(dark), 4) if dark else 0.0,
        "dark_relevant_pct_of_relevant": round(len(dark_relevant) / len(relevant), 4) if relevant else 0.0,
        "qrels_pairs_total": pairs_total,
        "qrels_pairs_in_topk": pairs_hit,
        "recall_at_k": round(pairs_hit / pairs_total, 4) if pairs_total else 0.0,
        "dark_relevant_ids": sorted(dark_relevant),
    }

    print("=" * 64)
    print("QRELS VALIDATE — dark matter vs релевантность")
    print("=" * 64)
    print(f"  Корпус: {result['n_chunks']} чанков, запросов: {result['n_queries']}")
    print(f"  Dark matter:                {result['dark_matter']}")
    print(f"  Релевантных (qrels) в корпусе: {result['relevant_in_corpus']}")
    print("-" * 64)
    print(f"  «Потерянное золото» (dark И релевантные): {result['dark_and_relevant']}")
    print(f"    = {result['dark_relevant_pct_of_dark']*100:5.1f}% dark matter")
    print(f"    = {result['dark_relevant_pct_of_relevant']*100:5.1f}% всех релевантных чанков")
    print("-" * 64)
    print(f"  Recall@k по qrels: {result['recall_at_k']*100:5.1f}% "
          f"({result['qrels_pairs_in_topk']}/{result['qrels_pairs_total']} пар)")
    print("=" * 64)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"saved -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
