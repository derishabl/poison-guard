# retrieval-fairness vs existing tools — honest comparison

Goal: clearly show where we are stronger and where we are not. No
exaggeration. Reviewed July 2026.

## Positioning

retrieval-fairness — **exposure-bias audit for vector search / RAG**:
shows what share of the corpus queries never retrieve, measures
concentration (Gini), hub capture, and provides a CI gate / regression
diff for embedder/chunking migrations. Metrics are honestly borrowed
(Gini/Lorenz — economics; retrievability — T-Retrievability); the
novelty is RAG-specific packaging.

## Comparison

| Tool | Focus | Exposure bias? | CI gate? | Regression diff? | Dashboard? | Vector store | Maturity |
|---|---|---|---|---|---|---|---|
| **retrieval-fairness** | corpus exposure audit (RAG) | ✅ core | ✅ | ✅ | ✅ | adapters (FAISS/Qdrant/pgvector/InMemory) | early |
| **retrieval-observatory (retobs)** | pipeline reliability (RAG) | ❌ (recall/NDCG, not exposure) | ✅ | ✅ (golden sets + pytest) | ✅ | 8 adapters (pgvector/Qdrant/BM25/...) | **v0.4.1, active PyPI** |
| T-Retrievability | retrievability metric | ✅ (the metric) | ❌ | ❌ | ❌ | TREC run-files only | paper + code (CIKM 2025) |
| **vector-guardrails** | semantic-change detection | ❌ (overlap/churn) | ✅ (exit codes) | ✅ | ❌ | snapshots (any store) | early v0.1.0, not on PyPI |
| semantic-coverage | knowledge gaps (RAG) | partial | ❌ | ❌ | ❌ | vector store | small |
| rag-sentinel | governance (RAG) | ❌ (freshness/PII) | ❌ | partial | partial | vector store | small |
| RankAudit | generic ranking fairness | ✅ (general) | ❌ | ❌ | ❌ | black-box ranker | small |
| Drift-Adapter | embedding migration | ❌ | ❌ | ❌ | ❌ | research | paper |

## The key 2026 finding: retobs is the closest competitor

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

The two tools are **complementary, not competing**: retobs profiles and
tunes a retrieval pipeline (recall / NDCG optimization); retrieval-fairness
audits the corpus (exposure uniformity). The PyPI names (`retrieval-observatory`
vs `retrieval-fairness`) reinforce that the audience reads them as
*different axes of retrieval health*.

## Why we still have a defensible lane

1. **Corpus-level exposure is our monopoly.** coverage / dark-matter /
   Gini / Lorenz / hub-leaderboard are not implemented by retobs,
   vector-guardrails, semantic-coverage, or rag-sentinel. T-Retrievability
   has the metric but ships no product (TREC run-files, conda envs, no
   store adapter, no gate, no dashboard).
2. **Qrels "lost gold" cross-check** — validating dark-matter chunks
   against qrels to show "lost relevant material" (905 relevant dark-
   matter chunks on the full NQ run) — is a packaging insight nobody
   else has shipped. Worth promoting as a feature, not a footnote.
3. **Regression-diff as corpus migration audit** — per-chunk deltas,
   rescued/newly-dark-matter chunks. retobs regression is per-query
   golden runs; vector-guardrails is overlap/churn on snapshots. We
   show *which chunks became dark matter*, which is the corpus-migration
   pain point.
4. **PyPI name `retrieval-fairness` is free** (verified); retobs occupies
   `retrieval-observatory`. No naming collision.

## Where we are weaker (honest)

1. **The metric engine is borrowed** — anyone can assemble an analog
   from T-Retrievability + a dashboard in a week. Defensibility =
   execution + the corpus-exposure lane, not a patent.
2. **retobs is far more mature on the pipeline axis** — per-stage
   attribution, production tracing, LangChain/LlamaIndex integration.
   We should not compete there; it would shadow our corpus lane.
3. **No production tracing** — we are offline audit only. retobs does
   live trace observability; we do not. Stay offline-purposeful.
4. **Synthetic queries are crude** — deterministic TF-IDF, no LLM
   paraphrase. retobs has `generate_testset` + LLM-judge labels.
5. **No Pinecone adapter yet** (SaaS, manual check); retobs notes
   Pinecone / Weaviate as unsupported too.

## Differentiators to sharpen (next focus)

- **Embed qrels "lost gold" as a first-class feature** — it is the
  bridge between exposure (corpus) and relevance (quality) and no
  competitor does it.
- **Lead with corpus exposure, not pipeline tuning.** Compared to
  retobs we win on "code coverage for your index", not "which stage
  broke recall".
- **Publish to PyPI** under `retrieval-fairness` (free name) before
  the corpus-exposure lane is noticed and occupied — timing is the
  remaining moat while the niche tightens.

## What not to do (to avoid diluting the lane)

- Do not chase per-stage pipeline attribution — retobs is already strong
  there; we would be a weaker copy.
- Do not do freshness/PII/governance — that is rag-sentinel.
- Do not do generic ranking fairness — that is RankAudit.
- Do not do attack-cost/poisoning — a crowded separate area (see archive/).