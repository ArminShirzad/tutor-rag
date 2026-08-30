"""Lexical retrieval -- BM25, implemented from scratch.

Why keep a keyword index when we already have embeddings?

Because embeddings are bad at exactly the things users type most often:
  - rare proper nouns and product names ("Adam", "ReLU", "pgvector")
  - identifiers, error codes, version numbers ("HTTP 429", "v2.1")
  - acronyms the model never saw during training

A bi-encoder maps "ReLU" to a fuzzy region of "activation-function-ish" space
and will happily return sigmoid. BM25 matches the literal token and cannot
drift. Vector search gives recall on *meaning*; BM25 gives precision on
*wording*. Hybrid search takes both, which is why it beats either alone on
essentially every public benchmark.

The formula:

    score(D, Q) = SUM_q  IDF(q) * ( f(q,D) * (k1 + 1) )
                            / ( f(q,D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(q) = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )

Reading the knobs, which is what an interviewer actually wants:
  k1 (~1.5) -- term-frequency saturation. The 10th occurrence of a word adds
               far less than the 2nd. Without it, keyword spam wins.
  b  (0.75) -- length normalisation. b=1 fully penalises long documents,
               b=0 ignores length entirely.
  IDF       -- rare terms are worth more than common ones. A term appearing in
               every document carries no signal and scores ~0.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")

# Stopwords carry no discriminative signal; dropping them speeds scoring and
# stops "the" from contributing noise. Kept deliberately small -- an aggressive
# list breaks queries like "how to be a good learner".
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "as", "from",
    "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "it", "its", "we", "you", "they", "he", "she", "i",
    "do", "does", "did", "can", "could", "will", "would", "should",
    "have", "has", "had", "not", "no", "so", "such", "up", "out", "about",
}


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


class BM25:
    """Okapi BM25 over an in-memory inverted index."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = [tokenize(doc) for doc in corpus]
        self.n_docs = len(self.doc_tokens)
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype="float32")
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        # inverted index: term -> {doc_index: term_frequency}
        self.index: dict[str, dict[int, int]] = {}
        for i, toks in enumerate(self.doc_tokens):
            for term, tf in Counter(toks).items():
                self.index.setdefault(term, {})[i] = tf

        # precompute IDF once -- it depends only on the corpus, not the query
        self.idf: dict[str, float] = {}
        for term, postings in self.index.items():
            n_q = len(postings)
            self.idf[term] = math.log(1 + (self.n_docs - n_q + 0.5) / (n_q + 0.5))

    def get_scores(self, query: str) -> np.ndarray:
        """Score every document against the query. Returns shape (n_docs,)."""
        scores = np.zeros(self.n_docs, dtype="float32")
        if self.n_docs == 0:
            return scores

        # Denominator's length term is query-independent, so hoist it out.
        len_norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))

        for term in tokenize(query):
            postings = self.index.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            docs = np.fromiter(postings.keys(), dtype=np.int64, count=len(postings))
            freqs = np.fromiter(postings.values(), dtype="float32", count=len(postings))
            scores[docs] += idf * (freqs * (self.k1 + 1)) / (freqs + len_norm[docs])
        return scores

    def search(self, query: str, k: int = 10,
               allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-k (doc_index, score), highest first, zero-scores dropped.

        `allowed` is the same pre-filter mask the vector store takes. Both
        retrievers must apply it -- filtering only the dense side would leak
        restricted content through the keyword side, which is exactly the kind
        of gap that makes hybrid search a security surface and not just a
        quality one.
        """
        scores = self.get_scores(query)
        if not len(scores):
            return []
        if allowed is not None:
            if allowed.shape[0] != scores.shape[0]:
                raise ValueError("access mask length does not match the index")
            scores = np.where(allowed, scores, 0.0)
        k = min(k, len(scores))
        # argpartition is O(n) vs O(n log n) for a full sort -- matters at scale
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
