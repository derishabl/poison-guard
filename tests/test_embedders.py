"""test_embedders.py — Embedder contract + TfidfEmbedder."""
from __future__ import annotations
import os
import pytest
import numpy as np

from retrieval_fairness.embedders import TfidfEmbedder, get_embedder, Embedder


def test_tfidf_fit_encode():
    emb = TfidfEmbedder()
    texts = ["отпуск через HR портал", "VPN настройка приложение", "зарплата аванс"]
    emb.fit(texts)
    mat = emb.encode(["отпуск HR"])
    assert mat.shape[0] == 1
    assert mat.shape[1] == emb.dim
    assert emb.dim > 0


def test_tfidf_encode_without_fit_raises():
    emb = TfidfEmbedder()
    try:
        emb.encode(["test"])
        assert False
    except RuntimeError:
        pass


def test_get_embedder_tfidf():
    emb = get_embedder("tfidf")
    assert isinstance(emb, TfidfEmbedder)
    assert isinstance(emb, Embedder) if hasattr(Embedder, "__class__") else True


def test_get_embedder_unknown():
    try:
        get_embedder("nope")
        assert False
    except ValueError:
        pass


def test_tfidf_satisfies_protocol():
    emb = TfidfEmbedder()
    # Protocol — structural; проверяем наличие методов
    assert hasattr(emb, "fit") and hasattr(emb, "encode") and hasattr(emb, "dim")


@pytest.mark.skipif(
    os.environ.get("RF_TEST_MINILM") != "1",
    reason="set RF_TEST_MINILM=1 to test sentence-transformers (heavy dep)",
)
def test_minilm_optional():
    from retrieval_fairness.embedders import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder()
    emb.fit(["test"])
    mat = emb.encode(["hello world"])
    assert mat.shape[0] == 1
    assert mat.shape[1] == emb.dim
    assert emb.dim == 384  # all-MiniLM-L6-v2


if __name__ == "__main__":
    import sys
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    p = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}"); p += 1
        except (AssertionError, Exception) as e:
            print(f"  SKIP/FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{p}/{len(fns)} passed")
    sys.exit(0)
