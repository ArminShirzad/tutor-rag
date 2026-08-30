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

import os
from abc import ABC, abstractmethod

from app.index.bm25 import tokenize
from app.resilience import RateLimiter, call_with_retry


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


class LLMReranker(Reranker):
    """Rerank with the LLM itself (a RankGPT-style listwise reranker).

    Why this exists: a cross-encoder needs torch plus a 90MB model, which does
    not fit in a 512MB free-tier container. This trades that local dependency
    for one extra API call.

    The important design choice is **listwise, not pointwise**. Scoring each
    passage in its own request would be 20 API calls per query -- slow and
    expensive. Sending all 20 passages in one request and asking for an array
    of scores is a single call, and it is also *more* accurate, because the
    model can compare passages against each other rather than judging each in
    isolation.

    Trade-offs against a cross-encoder, which is the honest comparison:
      + no local model, tiny memory footprint, no cold-start model load
      + often better on nuanced relevance, since it is a far larger model
      - one extra network round-trip (~1-2s vs ~300ms)
      - costs tokens per query
      - non-deterministic; temperature 0 reduces but does not remove this
    """

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash", max_chars: int = 700):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_chars = max_chars
        self.name = f"llm-reranker:{model}"
        # Reranking shares the generation quota, so it needs the same limiter.
        # Note this means one query costs *two* generation calls -- the rerank
        # and the answer -- which halves effective throughput on a tight quota.
        self.limiter = RateLimiter(int(os.environ.get("LLM_RPM", 5)))

    _SCHEMA = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "relevance": {"type": "integer"},
                    },
                    "required": ["index", "relevance"],
                },
            }
        },
        "required": ["scores"],
    }

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        from google.genai import types

        listing = "\n\n".join(
            f"[{i}] {doc[: self.max_chars]}" for i, doc in enumerate(documents)
        )
        prompt = (
            f"Rate how well each passage answers the question.\n\n"
            f"QUESTION: {query}\n\nPASSAGES:\n{listing}\n\n"
            "Score each passage 0-10: 10 = directly and completely answers the "
            "question, 5 = related topic but does not answer it, 0 = unrelated. "
            "Be strict: most passages in a corpus are irrelevant to any given "
            "question. Return a score for every passage index."
        )

        try:
            resp = call_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=self._SCHEMA,
                    ),
                ),
                limiter=self.limiter,
                label="rerank",
            )
            import json

            data = json.loads(resp.text)
            # Default to "irrelevant" for anything the model failed to score,
            # so a truncated response degrades toward refusal rather than
            # toward confidently surfacing unscored passages.
            out = [-11.0] * len(documents)
            for item in data.get("scores", []):
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(documents):
                    rel = max(0, min(10, int(item.get("relevance", 0))))
                    # map 0..10 onto the cross-encoder's rough -11..+11 range so
                    # a single MIN_SCORE threshold works for either reranker
                    out[idx] = rel * 2.2 - 11.0
            return out
        except Exception as exc:
            print(f"[rerank] LLM reranker failed ({exc}) -> lexical fallback for this query")
            return LexicalOverlapReranker().score(query, documents)


def build_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                   provider: str = "auto", api_key: str | None = None,
                   llm_model: str = "gemini-3.6-flash") -> Reranker:
    """Factory with graceful degradation: cross-encoder -> LLM -> lexical.

    `provider="llm"` is what the slim deployment profile uses, because torch
    does not fit in a 512MB container.
    """
    provider = (provider or "auto").lower()

    if provider in {"auto", "cross-encoder"}:
        try:
            return CrossEncoderReranker(model_name)
        except Exception as exc:
            print(f"[rerank] cross-encoder unavailable ({exc})")

    if provider in {"auto", "llm"} and api_key:
        try:
            return LLMReranker(api_key, llm_model)
        except Exception as exc:
            print(f"[rerank] LLM reranker unavailable ({exc})")

    print("[rerank] using lexical fallback")
    return LexicalOverlapReranker()
