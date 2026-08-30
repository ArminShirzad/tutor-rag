"""REST API (FastAPI).

    uvicorn app.api:app --reload

Endpoints
    GET  /                 the web UI
    GET  /health           liveness + index stats
    POST /ask              question -> grounded answer with citations
    POST /search           retrieval only (no LLM) -- for debugging retrieval
    GET  /stats            index statistics

Design notes worth saying out loud in an interview:

* The engine is built ONCE at startup, in a lifespan handler, not per request.
  Loading the embedding model and cross-encoder takes ~4s; doing that per
  request would make every call unusable.
* Every response carries a `diagnostics` block -- timings per stage, token
  counts, cost, and the chunks actually retrieved. This is the difference
  between an LLM feature you can operate and one you can only pray to. It is
  also the raw material for the observability the JD asks about.
* The heavy work is synchronous and CPU-bound, so endpoints are `def`, not
  `async def`. FastAPI then runs them in a threadpool instead of blocking the
  event loop -- declaring them `async` would serialise every request behind
  the reranker.
"""
from __future__ import annotations

import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.engine import RAGEngine
from app.index.vector_store import VectorStore
from app.ingest.build import ensure_index

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Container hosts have an ephemeral filesystem, so the index may not
        # exist on boot. Building it here costs a few seconds once and means a
        # deployment needs no build step and no pre-baked artefact.
        ensure_index(settings)
        print("[api] loading index and models ...")
        engine = RAGEngine.from_index(settings.index_dir, settings)

        # HR mode swaps the prompt pack for Persian and enables the
        # sensitive-topic router. The retrieval pipeline is unchanged -- only
        # the wording, language and escalation rules are per-product.
        if os.environ.get("HR_MODE", "").lower() == "true":
            from app.generation.prompts_hr import (
                REFUSAL_FA,
                SYSTEM_PROMPT_FA,
                build_user_prompt_fa,
            )
            from app.safety import SensitiveTopicRouter

            engine.system_prompt = SYSTEM_PROMPT_FA
            engine.prompt_builder = build_user_prompt_fa
            engine.refusal = REFUSAL_FA
            engine.router = SensitiveTopicRouter(embedder=engine.retriever.embedder)
            STATE["hr_mode"] = True
            print("[api] HR mode: Persian prompts + sensitive-topic routing enabled")

        STATE["engine"] = engine
        print(f"[api] ready: {len(engine.retriever.store)} chunks, "
              f"embedder={engine.retriever.embedder.name}, "
              f"reranker={engine.retriever.reranker.name if engine.retriever.reranker else None}, "
              f"llm={engine.llm.name}")
    except Exception as exc:
        # Never crash-loop the container on a startup failure: serve /health as
        # degraded so the platform reports something actionable instead of an
        # opaque restart loop.
        print(f"[api] STARTUP FAILED: {exc}")
        STATE["engine"] = None
        STATE["error"] = str(exc)
    yield
    STATE.clear()


app = FastAPI(
    title="tutor-rag",
    description="A production-shaped RAG system: hybrid retrieval, cross-encoder "
                "reranking, grounded generation with citations, and an eval harness.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    mode: str | None = Field(None, description="vector | bm25 | hybrid")
    k: int | None = Field(None, ge=1, le=20)
    rerank: bool | None = None
    # DEMO ONLY. In production the principal comes from the authenticated
    # session, never from the request body -- a client that can name its own
    # role can read anything. It is a request field here purely so the access
    # control is visible without a login flow.
    role: str | None = Field(None, description="employee | manager | hr (demo only)")
    employee_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    mode: str | None = None
    k: int | None = Field(None, ge=1, le=20)
    rerank: bool | None = None


def _engine() -> RAGEngine:
    engine = STATE.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="Index not built. Run: python -m app.cli ingest")
    return engine


@app.get("/health")
def health() -> dict:
    engine = STATE.get("engine")
    if engine is None:
        return {"status": "degraded", "reason": STATE.get("error", "no index")}
    return {
        "status": "ok",
        "hr_mode": bool(STATE.get("hr_mode")),
        "chunks": len(engine.retriever.store),
        "embedder": engine.retriever.embedder.name,
        "reranker": engine.retriever.reranker.name if engine.retriever.reranker else None,
        "llm": engine.llm.name,
        "retrieval_mode": settings.retrieval.mode,
    }


@app.get("/stats")
def stats() -> dict:
    engine = _engine()
    store = engine.retriever.store
    by_source: dict[str, int] = {}
    for c in store.chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    lens = [len(c.text) for c in store.chunks]
    return {
        "chunks": len(store.chunks),
        "documents": len(by_source),
        "dimension": store.dim,
        "embedder": store.embedder_name,
        "vector_memory_mb": round(store.matrix.nbytes / 1e6, 3),
        "chunk_chars": {"min": min(lens), "mean": sum(lens) // len(lens), "max": max(lens)},
        "by_source": by_source,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    engine = _engine()
    principal = None
    if req.role:
        from app.access import Principal

        principal = Principal(
            employee_id=req.employee_id or "demo-user",
            role=req.role,
            name=req.employee_id or req.role,
        )
    answer = engine.answer(
        req.question, mode=req.mode, final_k=req.k, use_reranker=req.rerank,
        principal=principal,
    )
    out = answer.to_dict()
    if principal is not None:
        mask = engine.retriever.access_mask(principal)
        out["diagnostics"]["viewer"] = principal.describe()
        out["diagnostics"]["visible_chunks"] = f"{int(mask.sum())}/{len(mask)}"
    return out


@app.post("/search")
def search(req: SearchRequest) -> dict:
    """Retrieval without generation. The first thing to call when an answer
    looks wrong -- it isolates retrieval failures from generation failures."""
    engine = _engine()
    res = engine.retriever.retrieve(req.query, mode=req.mode, final_k=req.k,
                                    use_reranker=req.rerank)
    return {
        "query": res.query,
        "mode": res.mode,
        "reranked": res.reranked,
        "below_threshold": res.below_threshold,
        "timings_ms": res.timings_ms,
        "results": [c.to_dict() for c in res.chunks],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    ui = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    if ui.exists():
        return ui.read_text(encoding="utf-8")
    return "<h1>tutor-rag</h1><p>UI not found. API docs at <a href='/docs'>/docs</a></p>"
