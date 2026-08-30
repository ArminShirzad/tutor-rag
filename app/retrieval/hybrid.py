"""Stage 5 -- Hybrid search: fusing dense (vector) and sparse (BM25) results.

The problem: BM25 scores are unbounded (0 to ~30, corpus-dependent) while
cosine scores live in [-1, 1]. You cannot add them. `0.7*cosine + 0.3*bm25` is
the answer people give in interviews and it is fragile -- the weights need
retuning for every corpus, and one outlier BM25 score swamps the blend.

Two principled fixes are implemented here.

1. Reciprocal Rank Fusion (RRF) -- the default, and what you should name first.
   Throw the scores away and keep only the *ranks*:

       RRF(d) = SUM_r  1 / (k + rank_r(d))          k = 60 by convention

   Scale-free, needs no tuning, no normalisation, and it is robust when one
   retriever returns garbage. A document ranked #1 by both retrievers beats one
   ranked #1 by a single retriever, which is exactly the behaviour you want.
   The constant k damps the influence of top ranks: with k=60, the gap between
   rank 1 and rank 2 is small, so a near-miss is not punished harshly.

2. Weighted score fusion -- min-max normalise each ranker into [0,1], then
   blend with alpha. More tunable, more brittle. Included for comparison
   because `scripts/`-level evaluation should decide, not folklore.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    """One retrieved chunk plus the provenance of *why* it was retrieved.

    Keeping per-retriever scores on the object is what makes the system
    debuggable: when an answer is wrong you can see instantly whether retrieval
    failed or generation did.
    """

    index: int
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


def reciprocal_rank_fusion(
    vector_hits: list[tuple[int, float]],
    bm25_hits: list[tuple[int, float]],
    k: int = 60,
    top_n: int = 20,
) -> list[Hit]:
    fused: dict[int, Hit] = {}

    for rank, (idx, score) in enumerate(vector_hits, start=1):
        hit = fused.setdefault(idx, Hit(index=idx, score=0.0))
        hit.score += 1.0 / (k + rank)
        hit.vector_score = score
        hit.vector_rank = rank

    for rank, (idx, score) in enumerate(bm25_hits, start=1):
        hit = fused.setdefault(idx, Hit(index=idx, score=0.0))
        hit.score += 1.0 / (k + rank)
        hit.bm25_score = score
        hit.bm25_rank = rank

    ranked = sorted(fused.values(), key=lambda h: h.score, reverse=True)
    return ranked[:top_n]


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def weighted_fusion(
    vector_hits: list[tuple[int, float]],
    bm25_hits: list[tuple[int, float]],
    alpha: float = 0.5,
    top_n: int = 20,
) -> list[Hit]:
    """alpha=1.0 is pure vector, alpha=0.0 is pure BM25."""
    v_idx = np.array([i for i, _ in vector_hits], dtype=np.int64)
    v_sc = _minmax(np.array([s for _, s in vector_hits], dtype="float32"))
    b_idx = np.array([i for i, _ in bm25_hits], dtype=np.int64)
    b_sc = _minmax(np.array([s for _, s in bm25_hits], dtype="float32"))

    fused: dict[int, Hit] = {}
    for rank, (idx, norm, raw) in enumerate(zip(v_idx, v_sc, [s for _, s in vector_hits]), start=1):
        hit = fused.setdefault(int(idx), Hit(index=int(idx), score=0.0))
        hit.score += alpha * float(norm)
        hit.vector_score, hit.vector_rank = float(raw), rank
    for rank, (idx, norm, raw) in enumerate(zip(b_idx, b_sc, [s for _, s in bm25_hits]), start=1):
        hit = fused.setdefault(int(idx), Hit(index=int(idx), score=0.0))
        hit.score += (1 - alpha) * float(norm)
        hit.bm25_score, hit.bm25_rank = float(raw), rank

    return sorted(fused.values(), key=lambda h: h.score, reverse=True)[:top_n]
