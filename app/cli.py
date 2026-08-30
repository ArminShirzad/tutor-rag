"""Command line interface.

    python -m app.cli ingest                 build the index from data/corpus
    python -m app.cli ask "your question"    ask one question
    python -m app.cli search "query"         retrieval only, no LLM (debugging)
    python -m app.cli repl                   interactive session
    python -m app.cli stats                  index statistics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.config import settings
from app.engine import RAGEngine
from app.index.embeddings import build_embedder
from app.index.vector_store import VectorStore
from app.ingest.chunking import chunk_corpus
from app.ingest.loaders import load_corpus
from app.retrieval.pipeline import Retriever

# ANSI colours; disabled automatically when piping to a file.
_TTY = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


BOLD = lambda s: c(s, "1")
DIM = lambda s: c(s, "2")
GREEN = lambda s: c(s, "32")
YELLOW = lambda s: c(s, "33")
CYAN = lambda s: c(s, "36")
RED = lambda s: c(s, "31")


def cmd_ingest(args: argparse.Namespace) -> int:
    corpus_dir = Path(args.corpus or settings.corpus_dir)
    index_dir = Path(args.index or settings.index_dir)

    print(BOLD("\n=== Ingestion ===\n"))
    t0 = time.perf_counter()

    print(f"1. Loading documents from {corpus_dir} ...")
    docs = load_corpus(corpus_dir)
    if not docs:
        print(RED(f"   No supported documents found in {corpus_dir}"))
        print(DIM("   Supported: .md .txt .pdf .docx"))
        return 1
    total_chars = sum(len(d.text) for d in docs)
    print(f"   {GREEN(str(len(docs)))} documents, {total_chars:,} characters")
    for d in docs:
        print(DIM(f"     - {d.source} ({len(d.text):,} chars) -> {d.title}"))

    strategy = args.strategy or settings.chunking.strategy
    print(f"\n2. Chunking (strategy={CYAN(strategy)}, size={args.size}, overlap={args.overlap}) ...")
    chunks = chunk_corpus(
        docs,
        strategy=strategy,
        size=args.size,
        overlap=args.overlap,
        min_chars=settings.chunking.min_chunk_chars,
        contextual=not args.no_contextual,
    )
    if not chunks:
        print(RED("   Chunking produced nothing -- check min_chunk_chars"))
        return 1
    lens = [len(ch.text) for ch in chunks]
    print(f"   {GREEN(str(len(chunks)))} chunks "
          f"(min {min(lens)}, mean {sum(lens)//len(lens)}, max {max(lens)} chars)")

    print(f"\n3. Embedding (provider={CYAN(settings.embedding.provider)}) ...")
    embedder = build_embedder(
        provider=settings.embedding.provider,
        model=(settings.embedding.local_model if settings.embedding.provider == "local"
               else settings.embedding.gemini_model),
        api_key=settings.gemini_api_key,
        batch_size=settings.embedding.batch_size,
    )
    t_embed = time.perf_counter()
    vectors = embedder.encode([ch.embed_text for ch in chunks], is_query=False)
    embed_s = time.perf_counter() - t_embed
    print(f"   {GREEN(embedder.name)} -> {vectors.shape[0]} x {vectors.shape[1]} "
          f"in {embed_s:.1f}s ({len(chunks)/max(embed_s,1e-6):.0f} chunks/s)")

    print(f"\n4. Building index ...")
    store = VectorStore(dim=embedder.dim, embedder_name=embedder.name)
    store.add(chunks, vectors)
    store.save(index_dir)
    size_mb = sum(f.stat().st_size for f in index_dir.glob("*")) / 1e6
    print(f"   saved to {index_dir} ({size_mb:.2f} MB)")

    print(GREEN(f"\nDone in {time.perf_counter()-t0:.1f}s. "
                f"{len(docs)} docs -> {len(chunks)} chunks -> {embedder.dim}-d vectors.\n"))
    return 0


def _print_answer(ans, show_chunks: bool = False) -> None:
    print()
    if ans.answered:
        print(BOLD(GREEN("ANSWER")))
    else:
        print(BOLD(YELLOW("NO ANSWER IN CORPUS")))
    print(ans.answer)

    if ans.citations:
        print(BOLD("\nSOURCES"))
        for cit in ans.citations:
            print(f"  [{cit.number}] {CYAN(cit.citation)}  {DIM(f'score={cit.score:.3f}')}")
            if show_chunks:
                print(DIM("      " + cit.text[:300].replace("\n", " ") + "..."))

    if ans.warnings:
        for w in ans.warnings:
            print(YELLOW(f"\n! {w}"))

    d = ans.to_dict()["diagnostics"]
    rt = d["retrieval_timings_ms"]
    parts = [f"{k.replace('_ms','')}={v}ms" for k, v in rt.items() if k != "total_ms"]
    print(DIM(f"\n  mode={d['retrieval_mode']} reranked={d['reranked']} "
              f"confidence={ans.confidence} model={d['model']}"))
    print(DIM(f"  retrieval: {' '.join(parts)}"))
    print(DIM(f"  total={d['timings_ms'].get('total_ms')}ms  "
              f"tokens={d['input_tokens']}in/{d['output_tokens']}out  "
              f"cost=${d['cost_usd']:.6f}"))
    print()


def _load_engine() -> RAGEngine | None:
    if not VectorStore.exists(settings.index_dir):
        print(RED(f"No index at {settings.index_dir}. Run: python -m app.cli ingest"))
        return None
    return RAGEngine.from_index(settings.index_dir)


def cmd_ask(args: argparse.Namespace) -> int:
    engine = _load_engine()
    if engine is None:
        return 1
    ans = engine.answer(
        args.question,
        mode=args.mode,
        final_k=args.k,
        use_reranker=not args.no_rerank,
    )
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_answer(ans, show_chunks=args.show_chunks)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Retrieval only. The first thing to run when an answer looks wrong --
    it separates 'retrieval failed' from 'generation failed'."""
    if not VectorStore.exists(settings.index_dir):
        print(RED(f"No index at {settings.index_dir}. Run: python -m app.cli ingest"))
        return 1
    retriever = Retriever.from_index(settings.index_dir)
    res = retriever.retrieve(args.query, mode=args.mode, final_k=args.k,
                             use_reranker=not args.no_rerank)
    print(BOLD(f"\n{len(res.chunks)} results  ") + DIM(f"mode={res.mode} reranked={res.reranked}"))
    print(DIM(f"timings: {res.timings_ms}\n"))
    for i, rc in enumerate(res.chunks, start=1):
        print(f"{BOLD(f'[{i}]')} {CYAN(rc.chunk.citation)}")
        bits = [f"final={rc.score:.4f}"]
        if rc.vector_rank:
            bits.append(f"vec#{rc.vector_rank}({rc.vector_score:.3f})")
        if rc.bm25_rank:
            bits.append(f"bm25#{rc.bm25_rank}({rc.bm25_score:.2f})")
        if rc.rerank_score is not None:
            bits.append(f"rerank={rc.rerank_score:.3f}")
        print(DIM("    " + "  ".join(bits)))
        print("    " + rc.chunk.text[:260].replace("\n", " ") + "...\n")
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    engine = _load_engine()
    if engine is None:
        return 1
    print(BOLD("\nStudy assistant. Ask a question, or /quit to exit.\n"))
    print(DIM("  /mode vector|bm25|hybrid   /rerank on|off   /k N   /chunks\n"))
    mode, k, rerank, show = None, None, True, False
    while True:
        try:
            q = input(BOLD("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in {"/quit", "/exit", "/q"}:
            return 0
        if q.startswith("/mode "):
            mode = q.split()[1]; print(DIM(f"  mode={mode}")); continue
        if q.startswith("/rerank "):
            rerank = q.split()[1] == "on"; print(DIM(f"  rerank={rerank}")); continue
        if q.startswith("/k "):
            k = int(q.split()[1]); print(DIM(f"  k={k}")); continue
        if q == "/chunks":
            show = not show; print(DIM(f"  show_chunks={show}")); continue
        _print_answer(engine.answer(q, mode=mode, final_k=k, use_reranker=rerank), show_chunks=show)


def cmd_stats(args: argparse.Namespace) -> int:
    if not VectorStore.exists(settings.index_dir):
        print(RED(f"No index at {settings.index_dir}. Run: python -m app.cli ingest"))
        return 1
    store = VectorStore.load(settings.index_dir)
    lens = [len(c.text) for c in store.chunks]
    by_source: dict[str, int] = {}
    for ch in store.chunks:
        by_source[ch.source] = by_source.get(ch.source, 0) + 1
    print(BOLD("\n=== Index ==="))
    print(f"  chunks     {len(store.chunks)}")
    print(f"  dimension  {store.dim}")
    print(f"  embedder   {store.embedder_name}")
    print(f"  chunk size min={min(lens)} mean={sum(lens)//len(lens)} max={max(lens)}")
    print(f"  memory     {store.matrix.nbytes/1e6:.2f} MB of vectors")
    print(BOLD("\n=== Documents ==="))
    for src, n in sorted(by_source.items()):
        print(f"  {n:4d}  {src}")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="app.cli", description="tutor-rag command line")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="build the index")
    pi.add_argument("--corpus"); pi.add_argument("--index")
    pi.add_argument("--strategy", choices=["fixed", "sentence", "recursive", "structural"])
    pi.add_argument("--size", type=int, default=settings.chunking.chunk_size)
    pi.add_argument("--overlap", type=int, default=settings.chunking.chunk_overlap)
    pi.add_argument("--no-contextual", action="store_true",
                    help="do not prefix chunks with title/section before embedding")
    pi.set_defaults(func=cmd_ingest)

    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("--mode", choices=["vector", "bm25", "hybrid"])
    pa.add_argument("-k", type=int); pa.add_argument("--no-rerank", action="store_true")
    pa.add_argument("--json", action="store_true"); pa.add_argument("--show-chunks", action="store_true")
    pa.set_defaults(func=cmd_ask)

    ps = sub.add_parser("search", help="retrieval only, no generation")
    ps.add_argument("query")
    ps.add_argument("--mode", choices=["vector", "bm25", "hybrid"])
    ps.add_argument("-k", type=int); ps.add_argument("--no-rerank", action="store_true")
    ps.set_defaults(func=cmd_search)

    pr = sub.add_parser("repl", help="interactive session"); pr.set_defaults(func=cmd_repl)
    pst = sub.add_parser("stats", help="index statistics"); pst.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
