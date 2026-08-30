"""Retrieval and answer metrics.

Retrieval is measured separately from generation on purpose. When quality drops
you need to know *which half* broke, and a single end-to-end score cannot tell
you. In practice retrieval is the culprit far more often than the LLM.

    Hit@k      did we retrieve at least one correct source in the top k?
               The blunt one. If Hit@k is low, nothing downstream can save you.
    Recall@k   what fraction of ALL correct sources did we get? Matters for
               multi-hop questions that need two documents.
    MRR        1/rank of the first correct result, averaged. Rewards putting
               the right chunk FIRST, not merely somewhere in the list --
               which matters because of the lost-in-the-middle effect.
    nDCG@k     rank-discounted gain. The most sensitive of the four: it
               distinguishes rank 1 from rank 2, where Hit@k cannot.

Answer-side:
    Keyword coverage  cheap, deterministic proxy for correctness.
    Refusal accuracy  did it refuse when it should have, and only then?
    Citation validity did every cited number point at a real source?
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """1.0 if any expected source appears in the top k."""
    if not expected:
        return 0.0
    return 1.0 if set(retrieved[:k]) & set(expected) else 0.0


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of expected sources present in the top k."""
    if not expected:
        return 0.0
    return len(set(retrieved[:k]) & set(expected)) / len(set(expected))


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not retrieved[:k]:
        return 0.0
    return len(set(retrieved[:k]) & set(expected)) / len(retrieved[:k])


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """1/rank of the first correct source; 0 if none found."""
    for i, src in enumerate(retrieved, start=1):
        if src in expected:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Binary-relevance nDCG.

    DCG discounts each hit by log2(rank+1), so a correct result at rank 1 is
    worth ~1.0 and at rank 5 only ~0.39. IDCG is the score of a perfect
    ranking, which normalises the result into [0, 1] so questions with
    different numbers of correct sources stay comparable.
    """
    if not expected:
        return 0.0
    exp = set(expected)
    dcg = sum(1.0 / math.log2(i + 1) for i, src in enumerate(retrieved[:k], start=1) if src in exp)
    ideal_hits = min(len(exp), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def keyword_coverage(answer: str, must_include: list[str]) -> float:
    """Fraction of required keywords present, case-insensitively."""
    if not must_include:
        return 1.0
    low = answer.lower()
    return sum(1 for kw in must_include if kw.lower() in low) / len(must_include)


@dataclass
class RetrievalMetrics:
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    n: int = 0

    def as_row(self) -> dict[str, float]:
        return {
            "Hit@1": self.hit_at_1, "Hit@3": self.hit_at_3, "Hit@5": self.hit_at_5,
            "Recall@5": self.recall_at_5, "MRR": self.mrr, "nDCG@5": self.ndcg_at_5,
        }


@dataclass
class AnswerMetrics:
    keyword_coverage: float = 0.0
    refusal_accuracy: float = 0.0
    false_refusals: int = 0      # refused a question it could have answered
    hallucinated_answers: int = 0  # answered a question it should have refused
    invalid_citations: int = 0
    uncited_answers: int = 0
    n: int = 0


@dataclass
class RunMetrics:
    label: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    answer: AnswerMetrics = field(default_factory=AnswerMetrics)
    latencies_ms: list[float] = field(default_factory=list)
    total_cost_usd: float = 0.0

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies_ms, 95)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round(pct / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


def aggregate_retrieval(rows: list[tuple[list[str], list[str]]]) -> RetrievalMetrics:
    """rows = [(retrieved_sources_in_rank_order, expected_sources), ...]"""
    rows = [(r, e) for r, e in rows if e]
    if not rows:
        return RetrievalMetrics()
    n = len(rows)
    return RetrievalMetrics(
        hit_at_1=sum(hit_at_k(r, e, 1) for r, e in rows) / n,
        hit_at_3=sum(hit_at_k(r, e, 3) for r, e in rows) / n,
        hit_at_5=sum(hit_at_k(r, e, 5) for r, e in rows) / n,
        recall_at_5=sum(recall_at_k(r, e, 5) for r, e in rows) / n,
        mrr=sum(reciprocal_rank(r, e) for r, e in rows) / n,
        ndcg_at_5=sum(ndcg_at_k(r, e, 5) for r, e in rows) / n,
        n=n,
    )
