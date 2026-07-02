# retrieval-fairness

**Code coverage для retrieval.** Показывает, какую долю векторного
корпуса реально достают запросы, а какую — никогда не находят (dark
matter); меряет концентрацию exposure (Gini), захват хабами, и
regression-diff при смене эмбеддера/чанкинга.

> Статус: ранняя разработка. Это **packaging-новизна** (метрики Gini /
> retrievability честно заимствованы из IR-fairness / T-Retrievability),
> не virgin-изобретение. См. `RETRIEVAL_FAIRNESS_PLAN.md`.

## Установка

```bash
pip install -e .            # или pip install retrieval-fairness
```

## Быстрый старт

```bash
retrieval-fairness demo --top-k 5          # демо на синтетическом корпусе
retrieval-fairness demo-diff --top-k 5     # regression diff при миграции эмбеддера
```

## Использование

### Прогон по реальным запросам

```bash
# corpus.jsonl: {"id": "...", "text": "...", "vector": [...]}
# queries.jsonl: {"id": "...", "vector": [...]}
retrieval-fairness probe --corpus corpus.jsonl --queries queries.jsonl \
    --top-k 10 --json report.json --html dashboard.html
```

### Без query-логов: синтетические запросы из корпуса

```bash
retrieval-fairness synth --corpus corpus.jsonl --top-k 10 --html dashboard.html
```

### Regression diff (смена эмбеддера/чанкинга)

```bash
retrieval-fairness diff --baseline before.json --candidate after.json
```

### CI-гейт

```bash
retrieval-fairness gate --baseline v1.json --candidate new.json --strict \
    --max-coverage-drop 0.05 --max-dark-matter-rise 0.05
# exit 1 в strict-режиме, если coverage упал > 5 п.п. -> CI блокирует деплой
```

## Метрики

| Метрика | Что показывает |
|---|---|
| Coverage % | доля корпуса, что находится хотя бы раз |
| Dark matter % | доля, что НИ РАЗУ не нашлась |
| Gini | концентрация exposure (0 = равномерно, 1 = всё в одном) |
| Hub capture top5/10 | доля exposure в top-N хабах |
| Lorenz curve | визуализация неравенства |
| Per-query overlap | стабильность выдачи при миграции |

## Как это устроено

- `retrieval_fairness/types.py` — контракт `VectorStore` (Protocol).
  Любой стор (FAISS, Qdrant, Pinecone, pgvector) приводится к нему
  адаптером.
- `stores.py` — `InMemoryVectorStore` для дев/тестов/демо.
- `metrics.py` — coverage, gini, lorenz, hub_capture, FairnessReport.
- `probe.py` — прогон workload → retrieval-frequency → отчёт.
- `diff.py` — regression diff между двумя прогонами.
- `gate.py` — CI-гейт с настраиваемыми правилами.
- `synth.py` — синтетические запросы из корпуса.
- `dashboard.py` — автономный HTML-отчёт (Lorenz, histogram, PCA-карта).

## Тесты

```bash
pytest tests/ -q    # 36 тестов
```

## Лицензия

MIT
