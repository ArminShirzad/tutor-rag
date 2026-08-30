"""The retrieval pipeline -- stages 3-6 wired together.

    query
      |
      v
    embed query ------------+
      |                     |
      v                     v
    vector search (k=20)   BM25 search (k=20)
      |                     |
      +----> RRF fusion <---+
                |
                v
        cross-encoder rerank (20 -> 5)
                |
                v
        threshold: below min_score -> refuse
                |
                v
        top-5 chunks + provenance
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, settings as default_settings
from app.index.bm25 import BM25
from app.index.embeddings import Embedder, build_embedder
from app.index.vector_store import VectorStore
from app.ingest.chunking import Chunk
from app.retrieval.hybrid import Hit, reciprocal_rank_fusion, weighted_fusion
from app.retrieval.rerank import Reranker, build_reranker


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "citation": self.chunk.citation,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
        }


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    mode: str = "hybrid"
    reranked: bool = False
    below_threshold: bool = False   # nothing cleared min_score -> we should refuse


class Retriever:
    """Owns the index and answers retrieval queries.

    Built once and reused: loading the embedding model takes ~2s and the
    reranker another ~2s, which is why they are instance state and not
    per-request work.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        config: Settings | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = config or default_settings
        # BM25 indexes the same text we embed, so both retrievers see the
        # heading context added during chunking.
        self.bm25 = BM25([c.embed_text for c in store.chunks])

    # ---------------- construction helpers ----------------
    @classmethod
    def from_index(cls, index_dir: Path | None = None, config: Settings | None = None) -> "Retriever":
        cfg = config or default_settings
        index_dir = Path(index_dir or cfg.index_dir)
        store = VectorStore.load(index_dir)
        embedder = build_embedder(
            provider=cfg.embedding.provider,
            model=cfg.embedding.local_model if cfg.embedding.provider == "local" else cfg.embedding.gemini_model,
            api_key=cfg.gemini_api_key,
            batch_size=cfg.embedding.batch_size,
        )
        if embedder.dim != store.dim:
            raise ValueError(
                f"Index was built with a {store.dim}-d embedder ({store.embedder_name}) but the "
                f"current embedder is {embedder.dim}-d ({embedder.name}). Re-run: python -m app.cli ingest"
            )
        reranker = build_reranker(cfg.retrieval.reranker_model) if cfg.retrieval.use_reranker else None
        return cls(store, embedder, reranker, cfg)

    # ---------------- the pipeline ----------------
    def retrieve(
        self,
        query: str,
        mode: str | None = None,
        candidate_k: int | None = None,
        final_k: int | None = None,
        use_reranker: bool | None = None,
        alpha: float | None = None,
    ) -> RetrievalResult:
        rc = self.settings.retrieval
        mode = mode or rc.mode
        candidate_k = candidate_k or rc.candidate_k
        final_k = final_k or rc.final_k
        do_rerank = rc.use_reranker if use_reranker is None else use_reranker
        do_rerank = do_rerank and self.reranker is not None

        timings: dict[str, float] = {}

        # -- dense retrieval
        vector_hits: list[tuple[int, float]] = []
        if mode in {"vector", "hybrid"}:
            t0 = time.perf_counter()
            qvec = self.embedder.encode_one(query, is_query=True)
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            vector_hits = self.store.search(qvec, k=candidate_k)
            timings["vector_search_ms"] = (time.perf_counter() - t0) * 1000

        # -- sparse retrieval
        bm25_hits: list[tuple[int, float]] = []
        if mode in {"bm25", "hybrid"}:
            t0 = time.perf_counter()
            bm25_hits = self.bm25.search(query, k=candidate_k)
            timings["bm25_search_ms"] = (time.perf_counter() - t0) * 1000

        # -- fusion
        if mode == "hybrid":
            t0 = time.perf_counter()
            if alpha is None:
                hits = reciprocal_rank_fusion(vector_hits, bm25_hits, k=rc.rrf_k, top_n=candidate_k)
            else:
                hits = weighted_fusion(vector_hits, bm25_hits, alpha=alpha, top_n=candidate_k)
            timings["fusion_ms"] = (time.perf_counter() - t0) * 1000
        elif mode == "vector":
            hits = [Hit(index=i, score=s, vector_score=s, vector_rank=r)
                    for r, (i, s) in enumerate(vector_hits, start=1)]
        else:
            hits = [Hit(index=i, score=s, bm25_score=s, bm25_rank=r)
                    for r, (i, s) in enumerate(bm25_hits, start=1)]

        # -- reranking
        below_threshold = False
        if do_rerank and hits:
            t0 = time.perf_counter()
            docs = [self.store.chunks[h.index].embed_text for h in hits]
            scores = self.reranker.score(query, docs)
            for hit, s in zip(hits, scores):
                hit.rerank_score = s
            hits.sort(key=lambda h: h.rerank_score, reverse=True)
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

            # Refusal gate. If even the best chunk is weakly relevant, the
            # corpus probably does not contain the answer -- say so instead of
            # handing the LLM irrelevant context and inviting a hallucination.
            if hits and hits[0].rerank_score < rc.min_score:
                below_threshold = True
            else:
                # Prune weak chunks *individually*. Padding the prompt out to
                # final_k with barely-relevant text is actively harmful: it
                # invites the model to use it, and every extra chunk costs
                # tokens and latency. Better five good chunks than five slots
                # filled. Always keep at least one so we never hand the
                # generator an empty context after clearing the gate.
                kept = [h for h in hits if h.rerank_score >= rc.min_score]
                hits = kept or hits[:1]

        top = hits[:final_k]
        chunks = [
            RetrievedChunk(
                chunk=self.store.chunks[h.index],
                score=h.final_score,
                vector_score=h.vector_score,
                bm25_score=h.bm25_score,
                rerank_score=h.rerank_score,
                vector_rank=h.vector_rank,
                bm25_rank=h.bm25_rank,
            )
            for h in top
        ]
        timings["total_ms"] = sum(v for k, v in timings.items() if k.endswith("_ms"))

        return RetrievalResult(
            query=query,
            chunks=chunks,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
            mode=mode,
            reranked=do_rerank,
            below_threshold=below_threshold,
        )
