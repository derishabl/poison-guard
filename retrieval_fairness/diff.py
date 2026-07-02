"""
diff.py — regression diff между двумя прогонами probe.

Сравнивает baseline и candidate (например, до и после смены эмбеддера):
- per-chunk exposure delta (на сколько изменилась частота попадания в top-k)
- chunks, ставшие dark matter (были найдены -> не нашлись)
- chunks, вышедшие из dark matter
- per-query top-k overlap (насколько изменилась выдача по запросам)
- сводная дельта метрик (coverage, gini, hub-capture)

Вход: два ProbeResult. Выход: DiffReport + текстовый/JSON вывод.
"""

from __future__ import annotations
from dataclasses import dataclass, field


def per_chunk_delta(
    baseline: dict[str, int], candidate: dict[str, int]
) -> dict[str, int]:
    """Delta retrieval-frequency по каждому чанку: candidate - baseline."""
    keys = set(baseline) | set(candidate)
    return {k: candidate.get(k, 0) - baseline.get(k, 0) for k in keys}


def newly_dark_matter(
    baseline: dict[str, int], candidate: dict[str, int]
) -> list[str]:
    """Чанки, которые находились в baseline, но стали dark matter в candidate."""
    return [k for k in baseline if baseline[k] > 0 and candidate.get(k, 0) == 0]


def rescued_from_dark_matter(
    baseline: dict[str, int], candidate: dict[str, int]
) -> list[str]:
    """Чанки, которые были dark matter в baseline, а теперь находятся."""
    return [k for k in candidate if baseline.get(k, 0) == 0 and candidate[k] > 0]


def per_query_overlap(
    baseline_hits: list[list[str]], candidate_hits: list[list[str]]
) -> list[float]:
    """
    Per-query Jaccard overlap top-k между baseline и candidate.
    1.0 = выдача не изменилась; 0.0 = полностью разная.
    Список выровнен по индексу запроса (предполагается одинаковый порядок
    и одинаковое число запросов).

    Раньше использовался zip(), который при разном числе запросов молча
    обрезает по короткому — это давал мусорный mean overlap. Теперь
    разное число запросов = явная ошибка (тихо неверный результат хуже
    падения).
    """
    if len(baseline_hits) != len(candidate_hits):
        raise ValueError(
            f"per_query_overlap: число запросов отличается "
            f"(baseline={len(baseline_hits)}, candidate={len(candidate_hits)}). "
            f"Regression-diff осмыслен только для одинакового workload'а; "
            f"сравнение разных наборов запросов даст невалидный overlap."
        )
    out = []
    for b, c in zip(baseline_hits, candidate_hits):
        sb, sc = set(b), set(c)
        if not sb and not sc:
            out.append(1.0)
            continue
        union = sb | sc
        out.append(len(sb & sc) / len(union) if union else 0.0)
    return out


@dataclass
class DiffReport:
    # сводные дельты метрик
    coverage_delta: float
    gini_delta: float
    hub_capture_top5_delta: float
    dark_matter_delta: float
    # per-chunk
    chunk_deltas: dict[str, int] = field(default_factory=dict)
    new_dark_matter: list[str] = field(default_factory=list)
    rescued: list[str] = field(default_factory=list)
    # per-query
    per_query_overlap: list[float] = field(default_factory=list)
    mean_query_overlap: float = 0.0
    # worst-affected чанки (по убыванию |delta|)
    worst_losses: list[tuple[str, int]] = field(default_factory=list)
    worst_gains: list[tuple[str, int]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 64,
            "REGRESSION DIFF (baseline -> candidate)",
            "=" * 64,
            f"  Coverage delta:     {self.coverage_delta*100:+.2f}%   (отрицательно = хуже)",
            f"  Dark matter delta:  {self.dark_matter_delta*100:+.2f}%   (положительно = больше тёмной материи)",
            f"  Gini delta:         {self.gini_delta:+.3f}   (рост = больше концентрация)",
            f"  Hub-capture top5:   {self.hub_capture_top5_delta*100:+.2f}%",
            "-" * 64,
            f"  Mean per-query overlap: {self.mean_query_overlap:.3f}  (1=выдача та же, 0=полностью иная)",
            f"  Новых dark-matter:  {len(self.new_dark_matter)} чанков",
            f"  Спасённых из dark:  {len(self.rescued)} чанков",
            "-" * 64,
            "  Худшие потери (chunk: delta freq):",
        ]
        for cid, d in self.worst_losses[:10]:
            lines.append(f"    {cid:30} {d:+d}")
        lines.append("  Наибольшие улучшения (chunk: delta freq):")
        for cid, d in self.worst_gains[:10]:
            lines.append(f"    {cid:30} {d:+d}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "coverage_delta": round(self.coverage_delta, 4),
            "dark_matter_delta": round(self.dark_matter_delta, 4),
            "gini_delta": round(self.gini_delta, 4),
            "hub_capture_top5_delta": round(self.hub_capture_top5_delta, 4),
            "mean_query_overlap": round(self.mean_query_overlap, 4),
            "new_dark_matter": self.new_dark_matter,
            "rescued": self.rescued,
            "worst_losses": self.worst_losses[:20],
            "worst_gains": self.worst_gains[:20],
        }


def diff_reports(baseline, candidate) -> DiffReport:
    """
    Сравнить два ProbeResult.
    baseline, candidate: ProbeResult (поля freqs, report, hits_per_query).
    """
    b_rep, c_rep = baseline.report, candidate.report
    deltas = per_chunk_delta(baseline.freqs, candidate.freqs)
    new_dm = newly_dark_matter(baseline.freqs, candidate.freqs)
    rescued = rescued_from_dark_matter(baseline.freqs, candidate.freqs)
    overlaps = per_query_overlap(baseline.hits_per_query, candidate.hits_per_query)
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # worst losses/gains по |delta|, исключая 0
    losses = sorted([(k, v) for k, v in deltas.items() if v < 0], key=lambda kv: kv[1])
    gains = sorted([(k, v) for k, v in deltas.items() if v > 0], key=lambda kv: -kv[1])

    return DiffReport(
        coverage_delta=c_rep.coverage_pct - b_rep.coverage_pct,
        gini_delta=c_rep.gini - b_rep.gini,
        hub_capture_top5_delta=c_rep.hub_capture_top5 - b_rep.hub_capture_top5,
        dark_matter_delta=c_rep.dark_matter_pct - b_rep.dark_matter_pct,
        chunk_deltas=deltas,
        new_dark_matter=new_dm,
        rescued=rescued,
        per_query_overlap=overlaps,
        mean_query_overlap=mean_overlap,
        worst_losses=losses,
        worst_gains=gains,
    )
