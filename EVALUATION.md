# Evaluation

The point of this document: every claim about this system is reproducible from
the repository in under a minute.

```bash
python -m app.evaluation.run_eval retrieval   # ablation, no LLM, ~30s
python -m app.evaluation.run_eval answers     # end-to-end
python scripts/calibrate_threshold.py         # refusal threshold
```

## Why retrieval and generation are measured separately

A single end-to-end score tells you quality dropped. It does not tell you
*which half* broke, and the two halves have completely different fixes. In
practice retrieval is the culprit far more often than the LLM.

So the retrieval ablation runs with no LLM at all: it is fast, free and
deterministic, which means it can run on every commit. Generation quality is
measured separately, and costs tokens.

## The golden set

30 questions in `app/evaluation/dataset.py`. Small on purpose — a set you
actually run on every change beats a large one you ran once.

Each case labels the **source document**, not the answer text:

```python
EvalCase(
    question="how do I stop my model from memorising the training examples",
    expected_sources=["02-overfitting-regularization.md"],
    must_include=["overfit"],
    category="paraphrase",
)
```

Labelling sources makes retrieval measurable exactly, with no LLM in the loop
and no judgement calls. `must_include` is a cheap, stable proxy for answer
correctness that needs no second model.

Categories are chosen to exercise specific failure modes:

| Category | What it tests | Which retriever should win |
|---|---|---|
| `paraphrase` | Question and passage share almost no vocabulary | Vector |
| `keyword` | Acronyms and rare proper nouns (`AdamW`, `RoPE`, `HNSW`) | BM25 |
| `factual` | The answer is a single sentence | Chunking-sensitive |
| `multihop` | The answer needs two documents | Recall@5 |
| `debugging` | How a student actually phrases a problem | End-to-end |
| `refusal` | **Not answerable from the corpus at all** | Nothing — must refuse |

The refusal cases matter as much as the rest. A system that answers everything
scores perfectly on the positives and is dangerous in production.

## Metrics

| Metric | Definition | Why it is here |
|---|---|---|
| **Hit@k** | Did any correct source appear in the top k? | The blunt one. If Hit@k is low nothing downstream can help. |
| **Recall@k** | What fraction of *all* correct sources did we get? | Multi-hop questions need two documents; Hit@k cannot see that. |
| **MRR** | Mean of 1/rank of the first correct result | Rewards putting the right chunk **first**, not merely somewhere. Matters because of the lost-in-the-middle effect. |
| **nDCG@k** | Rank-discounted gain, normalised | The most sensitive: it distinguishes rank 1 from rank 2, where Hit@k is blind. |

## Results

| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | nDCG@5 | p50 |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.846 | 0.962 | 1.000 | 0.981 | 0.913 | 0.921 | 0 ms |
| Vector only | 0.846 | 0.923 | 0.923 | 0.923 | 0.885 | 0.895 | 4 ms |
| Hybrid (RRF) | 0.808 | 0.923 | 0.923 | 0.923 | 0.865 | 0.880 | 5 ms |
| Vector + rerank | 0.846 | 0.923 | 0.923 | 0.923 | 0.885 | 0.895 | 315 ms |
| **Hybrid + rerank** | **0.923** | **1.000** | **1.000** | **1.000** | **0.962** | **0.972** | 334 ms |

### Reading the table

**Hybrid fusion alone slightly hurt.** nDCG 0.880 vs 0.895 for pure vector.
This is the most interesting row. RRF widens the candidate pool but blurs the
top of the ranking: a document that *both* retrievers rank moderately can
outrank one that a *single* retriever ranks first. Fusion is only worth it once
something precise re-sorts the pool.

**Reranking is what makes hybrid pay off.** Hybrid+rerank reaches Hit@3 and
Recall@5 of 1.000 — the wide pool contains every correct document and the
cross-encoder lifts it to the top. Vector+rerank cannot match it, because you
cannot rerank a document you never retrieved. **Recall is the ceiling;
reranking buys precision underneath it.**

**BM25 alone is a strong baseline** and beat pure vector search here, at zero
milliseconds, on a small jargon-dense corpus. Measure the boring baseline before
concluding the sophisticated system earned its cost.

### The latency trade

Reranking takes retrieval from ~5 ms to ~334 ms — about 90% of the budget, for
20 cross-encoder forward passes. On this corpus that is clearly the right trade.
On a latency-critical path it might not be, which is why `USE_RERANKER` is a
config flag and this table is reproducible.

If latency became critical: shrink `CANDIDATE_K` below 20, use a smaller
cross-encoder, move it to GPU, or cache aggressively.

## Refusal threshold calibration

`scripts/calibrate_threshold.py` scores known-answerable questions against
known-unanswerable ones and reports the separation:

| | reranker score range |
|---|---|
| in-corpus (10 questions) | **+0.85 … +6.24** |
| out-of-corpus (8 questions) | **−11.11 … −10.89** |

A ~12-point margin means `MIN_SCORE = -5.0` sits in open space rather than on a
cliff edge. The script prints the recommended midpoint.

These are raw cross-encoder logits, **not probabilities**, and they are not
comparable across models. Re-run the calibration whenever the reranker or the
corpus changes.

If the two distributions ever overlap, no threshold separates them. The fix
then is better retrieval — chunking, embeddings, reranker — not a different
number. Tuning the threshold in that situation only hides the problem.

## A failure the evaluation caught in itself

Worth recording, because it is the most useful thing this harness has done so
far.

The first end-to-end run reported **42.3% keyword coverage** and **15 answers
with no citations** — while retrieval scored a perfect **Hit@3 of 1.000** on
the same run. Those two numbers cannot both describe a working system, and the
contradiction is what made the bug findable.

Every one of those 15 failures was an **HTTP 429**. The Gemini free tier allows
5 requests per minute; the eval fired 30 as fast as it could. The generation
step never ran. The answers were error strings.

Two things follow:

1. **A rate limit is indistinguishable from a quality regression** unless you
   look. `answer` was a string, `answered` was true, and the metric dutifully
   scored it as a bad answer. This is the failure mode where you spend an
   afternoon tuning prompts against a network error.

2. **The fix belongs in the client, not the eval.** `app/resilience.py` adds a
   minimum-interval limiter (so we mostly never hit the limit) and retry with
   backoff (for when we do anyway). Retries use the `retryDelay` the API itself
   returns rather than a guess, and **non-transient errors are re-raised
   immediately** — retrying a bad API key just converts a clear error into a
   timeout.

The eval harness now surfaces `llm_error` warnings in its per-case output
precisely so this cannot recur silently.

## What this evaluation does not cover

Stated plainly, because knowing the limits of your own measurement is the point:

- **Faithfulness is proxied, not verified.** Keyword coverage is cheap and
  deterministic but it cannot tell whether a claim is genuinely supported by its
  cited chunk. The proper version is LLM-as-judge over (claim, cited span)
  pairs, sampled rather than run every time, because it costs tokens and carries
  its own variance.
- **The corpus is small** (5 documents, 42 chunks). Rankings between
  configurations could change on a corpus three orders of magnitude larger —
  in particular, BM25's strong showing partly reflects a small, jargon-dense
  corpus.
- **30 questions is a small sample.** A difference of one or two questions moves
  a metric by ~3 points, so small gaps in this table are not significant. The
  hybrid+rerank gap is large enough to trust; the hybrid-vs-vector gap is at the
  edge.
- **No multi-turn evaluation.** Conversational follow-ups ("what about the
  second one?") are not tested, and are known to fail — they carry no standalone
  meaning, so retrieval has nothing to work with. Query rewriting is the fix.
