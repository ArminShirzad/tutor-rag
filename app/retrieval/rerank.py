"""Stage 6 -- Reranking, the highest-value stage per line of code.

Bi-encoder vs cross-encoder -- know this cold, it is the standard question:

  BI-ENCODER (the embedding model, stage 3)
      encode(query) -> vector          |  encode(doc) -> vector
      similarity = cosine(a, b)
    The two texts never meet. Document vectors are computed *once at index
    time*, so query time is one matrix multiply over the whole corpus.
    Fast, scalable, approximate.

  CROSS-ENCODER (the reranker, this file)
      score = model("[CLS] query [SEP] document [SEP]")
    Query and document go through the transformer *together*, so every query
    token can attend to every document token. It can tell that "the model
    overfits" and "training loss keeps dropping while validation loss rises"
    are the same idea. Far more accurate -- and impossible to precompute,
    because the score depends on the pair.

Cost is the whole story: a cross-encoder needs one forward pass **per
(query, document) pair**. Over 50k chunks that is 50k forward passes per query
-- seconds to minutes. Over the 20 candidates the bi-encoder nominated, it is
~50ms.

Hence the two-stage architecture, which is the shape of every serious retrieval
system:

    cheap + wide (recall)        ->      expensive + narrow (precision)
    bi-encoder/BM25, top-20              cross-encoder, keep top-5

Measured on this project's eval set, reranking is worth roughly +20 points of
Hit@3 -- more than any other single change. See `EVALUATION.md`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.index.bm25 import tokenize


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document. Higher is better."""


class CrossEncoderReranker(Reranker):
    """sentence-transformers cross-encoder, ms-marco trained.

    Output is an unbounded logit, typically ~ -11 (irrelevant) to ~ +11
    (highly relevant). It is *not* a probability -- do not threshold it as one.
    We use the raw logit for a refusal threshold because it is monotonic in
    relevance, which is all a threshold needs.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu", max_length=512)
        self.name = f"cross-encoder:{model_name.split('/')[-1]}"

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        return [float(s) for s in self.model.predict(pairs, show_progress_bar=False)]


class LexicalOverlapReranker(Reranker):
    """Dependency-free fallback.

    Scores by weighted token overlap (Jaccard-ish, with a coverage bonus for
    matching *all* query terms). Much weaker than a cross-encoder, but it keeps
    the two-stage architecture intact when no model can be downloaded, and it
    is genuinely better than nothing on keyword-heavy queries.

    Scores are mapped into the same rough range as the cross-encoder so a
    single `min_score` threshold works for both.
    """

    name = "lexical-overlap"

    def score(self, query: str, documents: list[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(documents)
        out: list[float] = []
        for doc in documents:
            d = set(tokenize(doc))
            if not d:
                out.append(-11.0)
                continue
            overlap = len(q & d)
            coverage = overlap / len(q)              # how much of the query is present
            precision = overlap / max(len(d), 1)     # how focused the doc is
            raw = 0.8 * coverage + 0.2 * min(precision * 10, 1.0)
            out.append(raw * 22.0 - 11.0)            # map [0,1] -> [-11, +11]
        return out


def build_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    try:
        return CrossEncoderReranker(model_name)
    except Exception as exc:
        print(f"[rerank] cross-encoder unavailable ({exc}) -> lexical fallback")
        return LexicalOverlapReranker()
