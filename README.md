# tutor-rag

A grounded question-answering system over course material. Built to be *operated*,
not just demoed: hybrid retrieval, cross-encoder reranking, citation-verified
answers, a calibrated refusal path, and an evaluation harness that measures what
every component is actually worth.

Every retrieval component here is implemented from primitives — BM25, reciprocal
rank fusion, the vector store, the two-stage retrieve-then-rerank pipeline — rather
than imported from a framework. The point was to understand the mechanism, then
measure it.

```
question
   │
   ├─► embed query ──────► vector search (cosine, top-20) ──┐
   │                                                        ├─► RRF fusion
   └─► tokenize ─────────► BM25 search (top-20) ────────────┘        │
                                                                     ▼
                                                    cross-encoder rerank (20 → 5)
                                                                     │
                                            score < threshold? ──► refuse, 0 tokens
                                                                     │
                                                       grounded prompt + [n] citations
                                                                     │
                                                       LLM (JSON schema-constrained)
                                                                     │
                                                    verify every citation resolves
                                                                     │
                                                                  answer
```

## Results

Measured on a 30-question golden set (`app/evaluation/dataset.py`) covering
paraphrases, acronym lookups, multi-hop questions, and questions that are
deliberately *not* answerable from the corpus.

```bash
python -m app.evaluation.run_eval retrieval
```

| Configuration | Hit@1 | Hit@3 | Recall@5 | MRR | nDCG@5 | p50 latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.846 | 0.962 | 0.981 | 0.913 | 0.921 | **0 ms** |
| Vector only | 0.846 | 0.923 | 0.923 | 0.885 | 0.895 | 4 ms |
| Hybrid (RRF) | 0.808 | 0.923 | 0.923 | 0.865 | 0.880 | 5 ms |
| Vector + rerank | 0.846 | 0.923 | 0.923 | 0.885 | 0.895 | 315 ms |
| **Hybrid + rerank** | **0.923** | **1.000** | **1.000** | **0.962** | **0.972** | 334 ms |

Three findings worth more than the top-line number:

**Hybrid fusion alone made things slightly worse** (nDCG 0.880 vs 0.895 for pure
vector). RRF widens the candidate pool but blurs the top of the ranking, because a
document both retrievers rank *moderately* can outrank one that a single retriever
ranks *first*. Fusion only pays off once something precise re-sorts the pool — which
is exactly what the next row shows.

**Reranking is where hybrid becomes worth it.** Hybrid+rerank reaches Hit@3 and
Recall@5 of 1.000: the wide pool contains every correct document, and the
cross-encoder puts it on top. Vector+rerank cannot match it, because you cannot
rerank a document you never retrieved. Recall is the ceiling; precision is what
reranking buys.

**BM25 alone is a genuinely strong baseline** on a small, jargon-dense corpus — it
beat pure vector search here, at literally zero milliseconds. Always measure the
boring baseline before claiming the sophisticated system earned its cost.

The cost of that quality is honest: reranking takes retrieval from ~5 ms to ~334 ms.
On this corpus that is the right trade. On a latency-critical path it might not be,
which is why `USE_RERANKER` is a config flag and the ablation is reproducible.

## Hallucination control

Three independent layers, because prompting alone is not a control:

1. **A calibrated refusal gate.** Out-of-corpus questions are rejected *before*
   generation, so they cost zero tokens. The threshold is measured, not guessed —
   `scripts/calibrate_threshold.py` scores known-answerable against known-unanswerable
   questions:

   | | reranker score range |
   |---|---|
   | in-corpus | **+0.85 … +6.24** |
   | out-of-corpus | **−11.11 … −10.89** |

   A ~12-point margin means the threshold sits in open space, not on a cliff edge.
   If those ranges ever overlap, the fix is better retrieval — not a different number.

2. **Weak-context pruning.** Chunks below the threshold are dropped individually
   rather than padding the prompt out to `final_k`. Marginal context is not free:
   it invites the model to use it, and it costs tokens on every call.

3. **Citation verification.** After generation, every `[n]` the model emitted is
   checked against the sources actually supplied. Invented citations are stripped and
   surfaced as a warning; an answer with no citations is flagged and downgraded to
   low confidence. A model that cites `[7]` when it was given five sources has
   hallucinated its provenance, and that is detectable without a second LLM.

## Reliability

Hosted LLM APIs rate-limit hard — the Gemini free tier allows 5 requests per
minute. Every hosted call (generation, embedding, reranking) goes through
`app/resilience.py`, which pairs a minimum-interval limiter with retry-and-backoff.

Retries honour the `retryDelay` the API returns rather than guessing, and
non-transient errors (bad key, unknown model) are re-raised immediately —
retrying those converts a clear error into a timeout.

This was not defensive over-engineering. The first end-to-end eval reported 42%
keyword coverage with perfect retrieval, and every failure turned out to be a
429 rather than a bad answer. A system that cannot distinguish *"the model
answered badly"* from *"the model was never called"* cannot be evaluated at all.
See [EVALUATION.md](EVALUATION.md).

## Deployment profiles

| | Full | Slim |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2`, local CPU | Gemini `gemini-embedding-001` |
| Reranker | `ms-marco-MiniLM` cross-encoder, local | Listwise LLM reranker |
| Memory | ~800 MB (torch + 2 models) | ~150 MB |
| Per-query cost | 1 LLM call | 2 LLM calls (rerank + answer) |
| Requirements | `requirements.txt` | `requirements-slim.txt` |

Free-tier container hosts give 512 MB, and torch alone is ~250 MB resident
before any model loads. The slim profile moves both models to hosted APIs while
keeping the pipeline shape identical — still hybrid retrieval, still RRF, still
two-stage retrieve-then-rerank.

The **LLM reranker** is listwise rather than pointwise: one request scores all
candidates, instead of one request per candidate. That is ~20x cheaper, and
also more accurate, because the model compares passages against each other
rather than judging each in isolation.

```bash
# slim profile locally
EMBED_PROVIDER=gemini RERANK_PROVIDER=llm uvicorn app.api:app
```

`render.yaml` deploys the slim profile to Render's free tier. The API builds its
index on startup if one is missing, and rebuilds when the embedder changed —
querying a 384-d index with a 768-d model is a silent correctness bug.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli ingest
python -m app.cli ask "why does my validation loss increase while training loss drops"
```

No API key is required. With no key the system uses local embeddings
(`all-MiniLM-L6-v2`, CPU) and an **extractive** answerer that returns retrieved
sentences verbatim with citations — it cannot hallucinate by construction, and it
serves as the faithfulness floor in evaluation. Add a free
[Gemini key](https://aistudio.google.com/apikey) to `.env` for fluent generation:

```bash
GEMINI_API_KEY=your_key_here
```

Web UI and API:

```bash
uvicorn app.api:app --reload    # http://localhost:8000
```

The UI exposes the retrieval mode, the rerank toggle, and a retrieval-only switch,
so the effect of each stage is visible live rather than described.

## CLI

| Command | Purpose |
|---|---|
| `python -m app.cli ingest` | Build the index; prints per-stage document/chunk/vector counts |
| `python -m app.cli ask "..."` | Grounded answer with citations, timings, and cost |
| `python -m app.cli search "..."` | **Retrieval only.** The first thing to run when an answer looks wrong — it separates a retrieval failure from a generation failure |
| `python -m app.cli repl` | Interactive session with `/mode`, `/rerank`, `/k` switches |
| `python -m app.cli stats` | Index statistics |

`ingest` accepts `--strategy fixed|sentence|recursive|structural`, `--size`,
`--overlap`, and `--no-contextual` so chunking choices can be A/B tested rather
than argued about.

## API

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Question → grounded answer, citations, and full diagnostics |
| `POST /search` | Retrieval only, with per-retriever ranks and scores |
| `GET /health` | Liveness, chunk count, active models |
| `GET /stats` | Index statistics |
| `GET /docs` | OpenAPI |

Every `/ask` response carries a `diagnostics` block: per-stage timings, token counts,
cost in USD, and the chunks actually retrieved with their vector rank, BM25 rank and
rerank score. That block is the difference between an LLM feature you can operate and
one you can only pray to.

## Design decisions

**Contextual chunk embedding.** The text that gets *embedded* is prefixed with its
document title and section heading; the text shown to the LLM stays clean. A chunk
reading "It runs in O(n log n)" is unretrievable on its own — prefixed with
"Sorting Algorithms > Merge Sort" it becomes findable. Usually worth more recall than
swapping embedding models.

**Exact search, deliberately.** The vector store is a numpy matrix multiply — exact
brute-force k-NN, no ANN index. At this scale it is ~1 ms and perfectly accurate.
Approximate indexes (HNSW, IVF-PQ) trade recall for speed you do not need yet. The
interface mirrors what pgvector would give you, so the migration is a store swap, not
a rewrite:

```sql
SELECT chunk_id, 1 - (embedding <=> %(q)s) AS score
FROM chunks ORDER BY embedding <=> %(q)s LIMIT 20;
```

**Reciprocal Rank Fusion over weighted score blending.** BM25 scores are unbounded and
corpus-dependent; cosine scores are bounded. Adding them requires normalisation
constants that need retuning per corpus. RRF discards scores and uses only ranks:
`Σ 1/(k + rank)`, k=60. Scale-free, no tuning, robust when one retriever returns
garbage. Weighted fusion is implemented too, for comparison.

**Schema-constrained generation.** Answers come back as JSON with `answer`,
`answered`, `citations`, `confidence` and `missing_information` via Gemini's native
JSON mode, so the decoder is constrained rather than politely asked. `answered: false`
becomes a first-class signal the product can render and alert on, instead of a string
match against a refusal sentence.

**Provider adapters everywhere.** Embeddings and generation each sit behind one
interface with a graceful-degradation factory: Gemini → local model → dependency-free
fallback. Switching providers is a config change, and the eval harness can A/B two
providers on the same golden set without touching the pipeline.

**Synchronous endpoints on purpose.** The reranker is CPU-bound. FastAPI runs `def`
endpoints in a threadpool; marking them `async def` would serialise every request
behind the event loop.

## Project layout

```
app/
  config.py              every tunable knob, with the reasoning for each default
  engine.py              orchestration: retrieve → ground → generate → verify
  api.py                 FastAPI service
  cli.py                 command line interface
  ingest/
    loaders.py           .md .txt .pdf .docx → Document
    chunking.py          4 strategies + contextual embedding
  index/
    embeddings.py        local / Gemini / hashing providers
    bm25.py              Okapi BM25 from scratch
    vector_store.py      numpy exact k-NN + persistence
  retrieval/
    hybrid.py            RRF and weighted fusion
    rerank.py            cross-encoder + lexical fallback
    pipeline.py          the two-stage pipeline
  generation/
    prompts.py           grounding, citation and refusal design
    llm.py               Gemini / extractive adapters
  evaluation/
    dataset.py           30-question golden set
    metrics.py           Hit@k, Recall@k, MRR, nDCG@k
    run_eval.py          the ablation harness
scripts/
  calibrate_threshold.py measure the refusal threshold
```

## What I would do next

- **pgvector** instead of the numpy store, once the corpus outgrows memory — keeps
  vectors in the same transactional database as application data, so metadata
  filtering is just SQL and there is no second system to keep in sync.
- **Query rewriting** for conversational follow-ups ("what about the second one?"
  currently retrieves nothing useful, because it carries no standalone meaning).
- **An LLM-as-judge faithfulness metric** to complement keyword coverage — checking
  every claim against its cited span, sampled rather than on every run, because it
  costs tokens and has its own variance.
- **Caching** — embedding queries and full answers, keyed by normalised query. Most
  student questions repeat, and cached answers are free and instant.
- **Streaming** responses, so time-to-first-token stops being time-to-full-answer.

## License

MIT
