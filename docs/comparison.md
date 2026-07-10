# retrieval-fairness vs existing tools — honest comparison

Goal: clearly show what each tool does, and where retrieval-fairness is
stronger or weaker. No exaggeration. Reviewed July 2026.

## Positioning

retrieval-fairness — **exposure-bias audit for vector search / RAG**:
shows what share of the corpus queries never retrieve, measures
concentration (Gini), hub capture, and provides a CI gate / regression
diff for embedder/chunking migrations. The metrics are honestly borrowed
(Gini/Lorenz — economics; retrievability — T-Retrievability); the
novelty is RAG-specific packaging.

## Comparison

| Tool | Focus | Exposure bias? | CI gate? | Regression diff? | Dashboard? | Vector store | Maturity |
|---|---|---|---|---|---|---|---|
| **retrieval-fairness** | corpus exposure audit (RAG) | ✅ core | ✅ | ✅ | ✅ | adapters (FAISS/Qdrant/pgvector/InMemory) | early |
| **retrieval-observatory (retobs)** | pipeline reliability (RAG) | ❌ (recall/NDCG, not exposure) | ✅ | ✅ (golden sets + pytest) | ✅ | 8 adapters (pgvector/Qdrant/BM25/...) | **v0.4.1, active PyPI** |
| T-Retrievability | retrievability metric | ✅ (the metric) | ❌ | ❌ | ❌ | TREC run-files only | paper + code (CIKM 2025) |
| **vector-guardrails** | semantic-change detection | ❌ (overlap/churn) | ✅ (exit codes) | ✅ | ❌ | snapshots (any store) | early v0.1.0, not on PyPI |
| **ragcheck** | retrieval quality harness (qrels) | ❌ (recall/nDCG, not exposure) | partial (deterministic) | ✅ (regression diffs) | ❌ | offline runs | active GitHub |
| semantic-coverage | knowledge gaps (RAG) | partial | ❌ | ❌ | ✅ (web app) | Pinecone/Chroma connectors | small |

## retobs is the closest competitor

`retrieval-observatory` (retobs, v0.4.1 on PyPI) is a mature, actively
developed retrieval-reliability platform with CI gates, golden-run
regression detection, a pytest plugin, BEIR benchmarks (NFCorpus /
SciFact / FiQA with real NDCG numbers), per-stage attribution, a
dashboard, production tracing, and 8 adapters including pgvector and
Qdrant. It overlaps our CI-gate, regression-diff, dashboard, adapter,
and BEIR-case work directly.

**But retobs audits a different object — the pipeline, not the corpus.**
It answers "which stage drops recall and why does a query fail?" via
per-operator replay-graded attribution. It does **not** compute
coverage, dark matter, Gini, hub capture, or Lorenz — i.e. it has no
notion of per-chunk exposure or corpus health. Conversely, fairness of
the corpus is our entire core, not theirs.

| Audit question | retobs | retrieval-fairness |
|---|---|---|
| "Which operator drops recall?" | ✅ native | ❌ |
| "What share of the corpus is NEVER retrieved?" | ❌ | ✅ |
| "Who are the hub chunks?" | ❌ | ✅ |
| "How concentrated is exposure (Gini)?" | ❌ | ✅ |
| "Which relevant chunks are dark matter ('lost gold')?" | ❌ | ✅ (qrels validation) |
| "Pipeline migration regression?" | ✅ (golden runs) | ✅ (per-query/chunk diff) |
| "Corpus migration regression?" | ❌ | ✅ (chunk-level deltas) |

## ragcheck — the closest tool in *spirit* (offline, deterministic, diffs)

`ragcheck` (JSLEEKR/ragcheck) is an offline retrieval-quality harness:
recall@k / precision@k / MRR / nDCG (binary + graded), chunking
diagnostics, and **deterministic regression diffs** with no LLM-as-judge.
It shares our philosophy (offline, reproducible, CI-friendly) but audits
a different axis: **qrels-based quality** ("did the relevant docs land
in top-k?"), not per-chunk exposure. It has no retrieval-frequency
notion — no coverage, dark matter, Gini, or hub leaderboard. Our `qrels`
command overlaps its recall@k, but as a cross-check against dark matter
("lost gold"), not as the primary metric. Complementary, worth watching.

## ⚠️ Marketing note: "code coverage for RAG" is used by semantic-coverage

`semantic-coverage` ships the literal tagline **"The Code Coverage tool
for RAG Knowledge Bases."** Its mechanism is different (UMAP + HDBSCAN
blind-spot detection of query clusters vs document clusters, shipped as a
FastAPI + React web app — no retrieval-frequency, no Gini/dark matter, no
CI gate, not a pip library), so the tools are complementary in function.
retrieval-fairness nonetheless leads with "exposure audit" / "dark
matter" / "antihub inventory" (the latter grounded in the hubness
literature, Radovanović et al. 2010) and keeps "code coverage" as a
secondary explainer only.

## What is unique to retrieval-fairness

1. **Corpus-level exposure.** coverage / dark-matter / Gini / Lorenz /
   hub-leaderboard are not implemented by retobs, vector-guardrails,
   ragcheck, semantic-coverage, or rag-sentinel. T-Retrievability has
   the metric the academic way but ships no product (TREC run-files, no
   store adapter, no gate, no dashboard).
2. **Qrels "lost gold" cross-check** — validating dark-matter chunks
   against qrels to show "lost relevant material" — is a packaging
   insight nobody else has shipped. First-class CLI: `retrieval-fairness qrels`.
3. **`coverage of reachable ceiling`** — coverage reported not only as
   "share of corpus" but also as "share of what the workload can
   physically reach" (n_queries × top_k). On full NQ: coverage 11.72% =
   88.27% of the reachable ceiling. None of retobs, ragcheck,
   vector-guardrails, or T-Retrievability-as-a-product reports coverage
   relative to a workload ceiling.
4. **Regression-diff as corpus migration audit** — per-chunk deltas,
   rescued/newly-dark-matter chunks. retobs regression is per-query
   golden runs; vector-guardrails is overlap/churn on snapshots.

## Where we are weaker (honest)

1. **The metric engine is borrowed** — anyone can assemble an analog
   from T-Retrievability + a dashboard. Defensibility is execution and
   the corpus-exposure focus, not a patent.
2. **retobs is far more mature on the pipeline axis** — per-stage
   attribution, production tracing, LangChain/LlamaIndex integration.
3. **No production tracing** — offline audit only.
4. **Synthetic queries are crude** — deterministic TF-IDF, no LLM
   paraphrase. Mitigation: `synth` is honestly framed as an **antihub
   self-query audit** ("can a chunk be found even by a query aimed
   directly at it?"), which practitioner guidance recommends running on
   every corpus update. Crude as a workload simulator, precise as an
   invisibility detector.
5. **No Pinecone adapter yet.**