"""
scripts/download_nq.py — загрузка BEIR NQ для кейса (Шаг 9).

BEIR-формат: готовые queries/corpus/qrels, без парсинга сырого NQ.
Скачивает в data/nq/. data/ в .gitignore (большой).

Использование:
  python scripts/download_nq.py --out data/nq --sample-corpus 5000 --sample-queries 500
  (без --sample-* — полный корпус NQ ~260k чанков)

Зависимости: beir, datasets (HuggingFace).
"""

from __future__ import annotations
import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser(description="Download BEIR NQ for retrieval-fairness case study")
    ap.add_argument("--out", default="data/nq")
    ap.add_argument("--sample-corpus", type=int, default=0, help="0 = full corpus; else sample N chunks")
    ap.add_argument("--sample-queries", type=int, default=0, help="0 = all dev queries; else sample N")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading BEIR NQ -> {args.out}")

    try:
        from beir.datasets.data_loader import GenericDataLoader
        from beir.configs import dataset_settings
        from beir.util import export_dataset
    except ImportError:
        print("ERROR: install beir: pip install beir", flush=True)
        return 2

    # BEIR качает и распаковывает NQ
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
    out_dir = os.path.dirname(args.out)
    data_path = export_dataset(url, out_dir, "nq")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # sample
    if args.sample_corpus > 0:
        cids = list(corpus.keys())[: args.sample_corpus]
        corpus = {k: corpus[k] for k in cids}
    if args.sample_queries > 0:
        qids = list(queries.keys())[: args.sample_queries]
        queries = {k: queries[k] for k in qids}
        qrels = {k: v for k, v in qrels.items() if k in set(qids)}

    # запись в JSONL (наш формат)
    with open(os.path.join(args.out, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for cid, doc in corpus.items():
            text = doc.get("text", "")
            title = doc.get("title", "")
            full = f"{title}\n{text}" if title else text
            f.write(json.dumps({"id": cid, "text": full}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "queries.jsonl"), "w", encoding="utf-8") as f:
        for qid, q in queries.items():
            f.write(json.dumps({"id": qid, "text": q}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "qrels.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in qrels.items()}, f)

    print(f"  corpus:  {len(corpus)} chunks")
    print(f"  queries: {len(queries)}")
    print(f"  qrels:   {len(qrels)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
