"""Stage 3 -- Embeddings.

An embedding maps text to a vector such that *semantically similar text lands
close together*. That is what lets "how do I stop my model memorising the
training data?" retrieve a chunk titled "Regularisation and Dropout" which
shares almost no keywords with the question.

Three providers behind one interface, chosen at runtime:
  local   -- sentence-transformers/all-MiniLM-L6-v2. 384-d, ~90MB, CPU, free,
             offline. The right default for a demo and for most production
             workloads under ~1M chunks.
  gemini  -- Google text-embedding-004, 768-d. Better on nuanced queries;
             needs a key and a network round-trip.
  hashing -- dependency-free fallback so the system still runs anywhere. It is
             a bag-of-character-ngrams projection: genuinely worse, but it keeps
             the demo alive on a machine with no model download.

Interview point -- the asymmetry that trips people up: the query and the
document must be embedded by the *same* model into the *same* space. Mixing
models, or changing model without re-indexing, silently destroys retrieval
while every test still "passes".
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalise rows to unit length.

    Why: once vectors are unit-length, the dot product *is* cosine similarity.
    That turns the whole search into one matrix multiply, which is why this
    scales to hundreds of thousands of chunks in pure numpy.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return mat / norms


class Embedder(ABC):
    dim: int
    name: str

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Return an (n, dim) float32 array of unit-normalised vectors."""

    def encode_one(self, text: str, is_query: bool = True) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]


class LocalEmbedder(Embedder):
    """sentence-transformers bi-encoder running locally on CPU."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        # renamed in sentence-transformers 6.x; support both
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.name = f"local:{model_name.split('/')[-1]}"
        self.batch_size = batch_size

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")


class GeminiEmbedder(Embedder):
    """Google text-embedding-004 via the google-genai SDK.

    Note the task_type switch: Google trains these models asymmetrically, so a
    query and a document are embedded with different prefixes. Using the wrong
    one costs real recall -- a classic silent bug.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-004", batch_size: int = 64):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = 768
        self.name = f"gemini:{model}"
        self.batch_size = batch_size

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        from google.genai import types

        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task),
            )
            out.extend(e.values for e in resp.embeddings)
        return l2_normalize(np.asarray(out, dtype="float32"))


class HashingEmbedder(Embedder):
    """Zero-dependency fallback: hashed character 3-grams + sublinear tf.

    This is not a semantic model -- it is closer to a fuzzy keyword index. It
    exists so `python -m app.cli` works on a fresh machine before any model has
    been downloaded, and so the test suite never needs the network.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing:{dim}d"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        t = f" {text.lower()} "
        for n in (3, 4):
            for i in range(len(t) - n + 1):
                gram = t[i : i + n]
                h = int.from_bytes(hashlib.md5(gram.encode()).digest()[:4], "little")
                v[h % self.dim] += 1.0
        # sublinear scaling stops long documents from dominating
        return np.sign(v) * np.log1p(np.abs(v))

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return l2_normalize(np.vstack([self._vec(t) for t in texts]))


def build_embedder(provider: str = "local", model: str = "", api_key: str | None = None,
                   batch_size: int = 64) -> Embedder:
    """Factory with graceful degradation -- never hard-fails the pipeline."""
    provider = (provider or "local").lower()

    if provider == "gemini":
        if not api_key:
            print("[embeddings] GEMINI_API_KEY missing -> falling back to local model")
        else:
            try:
                return GeminiEmbedder(api_key, model or "text-embedding-004", batch_size)
            except Exception as exc:
                print(f"[embeddings] Gemini unavailable ({exc}) -> falling back to local model")

    if provider in {"local", "gemini"}:
        try:
            return LocalEmbedder(model or "sentence-transformers/all-MiniLM-L6-v2", batch_size)
        except Exception as exc:
            print(f"[embeddings] local model unavailable ({exc}) -> falling back to hashing")

    return HashingEmbedder()
