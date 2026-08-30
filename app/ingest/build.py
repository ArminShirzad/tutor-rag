"""Index building, callable from both the CLI and the API.

Extracted so a deployment can build its own index at startup. That matters for
container hosts with an ephemeral filesystem: baking the index into the image
requires an API key at *build* time, which is awkward with secret managers that
only inject at runtime. Building on first boot costs a few seconds once and
removes that whole class of problem.
"""
from __future__ import annotations

import time
from pathlib import Path

from app.config import Settings, settings as default_settings
from app.index.embeddings import build_embedder
from app.index.vector_store import VectorStore
from app.ingest.chunking import chunk_corpus
from app.ingest.loaders import load_corpus


def build_index(
    corpus_dir: Path | None = None,
    index_dir: Path | None = None,
    config: Settings | None = None,
    strategy: str | None = None,
    size: int | None = None,
    overlap: int | None = None,
    contextual: bool = True,
    verbose: bool = True,
) -> VectorStore:
    """Load -> chunk -> embed -> persist. Returns the populated store."""
    cfg = config or default_settings
    corpus_dir = Path(corpus_dir or cfg.corpus_dir)
    index_dir = Path(index_dir or cfg.index_dir)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    t0 = time.perf_counter()

    docs = load_corpus(corpus_dir)
    if not docs:
        raise FileNotFoundError(
            f"No supported documents in {corpus_dir} (.md .txt .pdf .docx)"
        )
    log(f"[index] {len(docs)} documents, {sum(len(d.text) for d in docs):,} chars")

    chunks = chunk_corpus(
        docs,
        strategy=strategy or cfg.chunking.strategy,
        size=size or cfg.chunking.chunk_size,
        overlap=overlap if overlap is not None else cfg.chunking.chunk_overlap,
        min_chars=cfg.chunking.min_chunk_chars,
        contextual=contextual,
    )
    if not chunks:
        raise ValueError("Chunking produced no chunks -- check min_chunk_chars")
    log(f"[index] {len(chunks)} chunks")

    embedder = build_embedder(
        provider=cfg.embedding.provider,
        model=(cfg.embedding.local_model if cfg.embedding.provider == "local"
               else cfg.embedding.gemini_model),
        api_key=cfg.gemini_api_key,
        batch_size=cfg.embedding.batch_size,
    )
    vectors = embedder.encode([c.embed_text for c in chunks], is_query=False)
    log(f"[index] embedded with {embedder.name} -> {vectors.shape}")

    store = VectorStore(dim=embedder.dim, embedder_name=embedder.name)
    store.add(chunks, vectors)
    store.save(index_dir)
    log(f"[index] saved to {index_dir} in {time.perf_counter() - t0:.1f}s")
    return store


def ensure_index(config: Settings | None = None) -> bool:
    """Build the index only if it is missing. Returns True if it built one.

    Also rebuilds when the persisted index was made by a *different* embedder:
    querying a 384-d index with a 768-d model is a silent correctness bug, and
    it is exactly what happens when a deployment flips EMBED_PROVIDER without
    clearing its volume.
    """
    cfg = config or default_settings
    index_dir = Path(cfg.index_dir)

    if not VectorStore.exists(index_dir):
        print(f"[index] no index at {index_dir} -- building")
        build_index(config=cfg)
        return True

    import json

    try:
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        expected = "gemini" if cfg.embedding.provider == "gemini" else "local"
        stored = str(meta.get("embedder", ""))
        if not stored.startswith(expected):
            print(f"[index] embedder changed ({stored} -> {expected}) -- rebuilding")
            build_index(config=cfg)
            return True
    except Exception as exc:
        print(f"[index] could not verify index metadata ({exc}) -- rebuilding")
        build_index(config=cfg)
        return True

    return False
