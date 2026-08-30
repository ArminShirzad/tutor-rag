"""Stage 4 -- The vector store.

Deliberately implemented in ~150 lines of numpy rather than pulled in from a
library, because the point of this project is to show the mechanism.

A "vector database" does three things:
  1. store vectors + their metadata
  2. find the nearest neighbours of a query vector
  3. persist and reload without re-embedding

With unit-normalised vectors, step 2 is a single matrix multiply:

    scores = matrix @ query      # (n, d) @ (d,) -> (n,)

That is exact brute-force search: O(n*d). On 100k chunks of 384 dims it is a
~40M-FLOP operation -- a few milliseconds. **Exact search is the correct choice
until you can prove otherwise**; that is the answer to "why not FAISS?".

When you outgrow it (roughly >1M vectors, or when p99 latency matters more than
perfect recall), you move to an *approximate* index -- HNSW (a navigable
small-world graph, what pgvector, Qdrant and Weaviate use) or IVF-PQ. Those
trade a few percent of recall for 10-100x speed. The API here matches what
pgvector would give you, so swapping is a store change, not a rewrite:

    -- the pgvector equivalent of `search()`
    SELECT chunk_id, 1 - (embedding <=> %(q)s) AS score
    FROM chunks
    ORDER BY embedding <=> %(q)s
    LIMIT 20;
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.ingest.chunking import Chunk


class VectorStore:
    def __init__(self, dim: int, embedder_name: str = ""):
        self.dim = dim
        self.embedder_name = embedder_name
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray = np.zeros((0, dim), dtype="float32")

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"dimension mismatch: store is {self.dim}-d, got {vectors.shape[1]}-d. "
                "Re-index after changing the embedding model."
            )
        self.chunks.extend(chunks)
        self.matrix = np.vstack([self.matrix, vectors.astype("float32")])

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Exact cosine nearest-neighbour search. Returns (index, score)."""
        if len(self.chunks) == 0:
            return []
        scores = self.matrix @ query_vec.astype("float32")
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    # ---------------- persistence ----------------
    def save(self, index_dir: Path) -> None:
        """Vectors as .npy (compact binary), metadata as .jsonl (greppable).

        Embeddings are the expensive part of ingestion -- persisting them turns
        a 30-second rebuild into a 200ms load.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.matrix)
        with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        (index_dir / "meta.json").write_text(
            json.dumps(
                {"dim": self.dim, "count": len(self.chunks), "embedder": self.embedder_name},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        store = cls(dim=meta["dim"], embedder_name=meta.get("embedder", ""))
        store.matrix = np.load(index_dir / "vectors.npy")
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as fh:
            store.chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
        if len(store.chunks) != store.matrix.shape[0]:
            raise ValueError("corrupt index: chunk/vector count mismatch -- re-run ingestion")
        return store

    @staticmethod
    def exists(index_dir: Path) -> bool:
        index_dir = Path(index_dir)
        return all((index_dir / f).exists() for f in ("vectors.npy", "chunks.jsonl", "meta.json"))
