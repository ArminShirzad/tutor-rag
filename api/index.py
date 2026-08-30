"""Vercel serverless entrypoint.

Vercel's Python runtime looks for an ASGI app named `app` in files under api/.
Re-exporting the same FastAPI application means the serverless deployment and
the long-running server run identical code.

Two constraints this file works around, both specific to serverless:

1. **Read-only filesystem.** Only /tmp is writable, so the index cannot be
   written next to the code. INDEX_DIR points at /tmp; the index is rebuilt on
   a cold start (~2s for this corpus, one batched embedding call) and reused
   for the lifetime of the warm instance.

2. **No local models.** The 250MB bundle limit rules out torch, so the
   deployment uses hosted embeddings and the listwise LLM reranker. Same
   pipeline shape, different providers -- which is exactly what the provider
   adapters exist for.
"""
import os

os.environ.setdefault("INDEX_DIR", "/tmp/tutor-rag-index")
os.environ.setdefault("EMBED_PROVIDER", "gemini")
os.environ.setdefault("RERANK_PROVIDER", "llm")
os.environ.setdefault("LLM_PROVIDER", "auto")
# Reranking spends a whole LLM call here, so nominate fewer candidates than the
# local profile's 20.
os.environ.setdefault("CANDIDATE_K", "10")

from app.api import app  # noqa: E402

__all__ = ["app"]
