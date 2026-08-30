"""The evaluation harness.

    python -m app.evaluation.run_eval retrieval   # compare retrieval configs (no LLM, fast)
    python -m app.evaluation.run_eval answers     # end-to-end, needs the generator
    python -m app.evaluation.run_eval all

`retrieval` deliberately runs without an LLM: it is fast, free and
deterministic, so it can run on every commit. Ablations are the point -- the
table shows what each component is actually worth, so "we added a reranker" is
backed by a number instead of a vibe.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")

from app.config import settings
from app.engine import RAGEngine
from app.evaluation.dataset import GOLDEN_SET, EvalCase
from app.evaluation.metrics import (
    AnswerMetrics,
    RunMetrics,
    aggregate_retrieval,
    keyword_coverage,
)
from app.generation.prompts import REFUSAL_MESSAGE
from app.retrieval.pipeline import Retriever


def _unique_sources(chunks) -> list[str]:
    """Chunk ranks -> document ranks, preserving order and dropping duplicates.

    Several chunks often come from the same document; for source-level metrics
    only the best rank of each document counts.
    """
    seen: list[str] = []
    for c in chunks:
        if c.chunk.source not in seen:
            seen.append(c.chunk.source)
    return seen


# --------------------------------------------------------------------------
# Retrieval-only evaluation (no LLM)
# --------------------------------------------------------------------------
CONFIGS = [
    ("BM25 only",              dict(mode="bm25",   use_reranker=False)),
    ("Vector only",            dict(mode="vector", use_reranker=False)),
    ("Hybrid (RRF)",           dict(mode="hybrid", use_reranker=False)),
    ("Vector + rerank",        dict(mode="vector", use_reranker=True)),
    ("Hybrid + rerank",        dict(mode="hybrid", use_reranker=True)),
]


def eval_retrieval(retriever: Retriever, cases: list[EvalCase], config: dict,
                   label: str) -> RunMetrics:
    answerable = [c for c in cases if c.answerable and c.expected_sources]
    rows, latencies = [], []
    for case in answerable:
        t0 = time.perf_counter()
        res = retriever.retrieve(case.question, final_k=5, **config)
        latencies.append((time.perf_counter() - t0) * 1000)
        rows.append((_unique_sources(res.chunks), case.expected_sources))

    metrics = RunMetrics(label=label, retrieval=aggregate_retrieval(rows))
    metrics.latencies_ms = latencies
    return metrics


def print_retrieval_table(results: list[RunMetrics]) -> str:
    headers = ["Configuration", "Hit@1", "Hit@3", "Hit@5", "Recall@5", "MRR", "nDCG@5", "p50 ms"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in results:
        row = r.retrieval.as_row()
        lines.append(
            "| " + " | ".join([
                r.label,
                *[f"{row[h]:.3f}" for h in ["Hit@1", "Hit@3", "Hit@5", "Recall@5", "MRR", "nDCG@5"]],
                f"{r.p50_ms:.0f}",
            ]) + " |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# End-to-end evaluation (retrieval + generation)
# --------------------------------------------------------------------------
def eval_answers(engine: RAGEngine, cases: list[EvalCase], label: str) -> tuple[RunMetrics, list[dict]]:
    m = RunMetrics(label=label)
    am = AnswerMetrics()
    rows, details = [], []

    for case in cases:
        ans = engine.answer(case.question)
        m.latencies_ms.append(ans.timings_ms.get("total_ms", 0.0))
        if ans.llm:
            m.total_cost_usd += ans.llm.cost_usd

        refused = (not ans.answered) or REFUSAL_MESSAGE.lower() in ans.answer.lower()

        if case.answerable:
            rows.append((_unique_sources(ans.retrieval.chunks), case.expected_sources))
            cov = keyword_coverage(ans.answer, case.must_include)
            am.keyword_coverage += cov
            if refused:
                am.false_refusals += 1
            if not refused and not ans.citations:
                am.uncited_answers += 1
        else:
            # correct behaviour is to refuse
            if refused:
                am.refusal_accuracy += 1
            else:
                am.hallucinated_answers += 1
            cov = 1.0 if refused else 0.0

        am.invalid_citations += sum(1 for w in ans.warnings if "non-existent sources" in w)
        am.n += 1

        details.append({
            "question": case.question,
            "category": case.category,
            "answerable": case.answerable,
            "refused": refused,
            "keyword_coverage": round(cov, 3),
            "expected_sources": case.expected_sources,
            "retrieved_sources": _unique_sources(ans.retrieval.chunks) if ans.retrieval else [],
            "answer": ans.answer[:500],
            "warnings": ans.warnings,
            "latency_ms": ans.timings_ms.get("total_ms", 0.0),
        })

    n_answerable = sum(1 for c in cases if c.answerable)
    n_refusal = sum(1 for c in cases if not c.answerable)
    am.keyword_coverage = am.keyword_coverage / n_answerable if n_answerable else 0.0
    am.refusal_accuracy = am.refusal_accuracy / n_refusal if n_refusal else 0.0

    m.retrieval = aggregate_retrieval(rows)
    m.answer = am
    return m, details


def print_answer_report(m: RunMetrics) -> str:
    a = m.answer
    return "\n".join([
        f"| Metric | Value |",
        f"|---|---|",
        f"| Keyword coverage (answerable) | {a.keyword_coverage:.1%} |",
        f"| Refusal accuracy (unanswerable) | {a.refusal_accuracy:.1%} |",
        f"| False refusals | {a.false_refusals} |",
        f"| Hallucinated answers (should have refused) | {a.hallucinated_answers} |",
        f"| Answers with no citation | {a.uncited_answers} |",
        f"| Invalid citations emitted | {a.invalid_citations} |",
        f"| Retrieval Hit@3 | {m.retrieval.hit_at_3:.3f} |",
        f"| Retrieval MRR | {m.retrieval.mrr:.3f} |",
        f"| Latency p50 / p95 | {m.p50_ms:.0f} ms / {m.p95_ms:.0f} ms |",
        f"| Total cost for {a.n} questions | ${m.total_cost_usd:.6f} |",
    ])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_eval")
    p.add_argument("what", choices=["retrieval", "answers", "all"], default="all", nargs="?")
    p.add_argument("--out", help="write JSON results here")
    args = p.parse_args(argv)

    out: dict = {}

    if args.what in {"retrieval", "all"}:
        print("\n" + "=" * 78)
        print("RETRIEVAL ABLATION  --  what is each component actually worth?")
        print("=" * 78)
        retriever = Retriever.from_index()
        results = []
        for label, cfg in CONFIGS:
            print(f"  running {label} ...", flush=True)
            results.append(eval_retrieval(retriever, GOLDEN_SET, cfg, label))
        table = print_retrieval_table(results)
        print("\n" + table + "\n")
        best = max(results, key=lambda r: r.retrieval.ndcg_at_5)
        base = next(r for r in results if r.label == "Vector only")
        print(f"Best: {best.label} (nDCG@5 = {best.retrieval.ndcg_at_5:.3f})")
        delta = best.retrieval.ndcg_at_5 - base.retrieval.ndcg_at_5
        print(f"Improvement over plain vector search: {delta:+.3f} nDCG@5 "
              f"({delta / max(base.retrieval.ndcg_at_5, 1e-9):+.1%})\n")
        out["retrieval"] = {r.label: {**r.retrieval.as_row(), "p50_ms": r.p50_ms} for r in results}
        out["retrieval_table_markdown"] = table

    if args.what in {"answers", "all"}:
        print("=" * 78)
        print("END-TO-END ANSWER QUALITY")
        print("=" * 78)
        engine = RAGEngine.from_index()
        print(f"  generator: {engine.llm.name}")
        print(f"  running {len(GOLDEN_SET)} questions ...", flush=True)
        m, details = eval_answers(engine, GOLDEN_SET, engine.llm.name)
        report = print_answer_report(m)
        print("\n" + report + "\n")

        failures = [d for d in details
                    if (d["answerable"] and (d["refused"] or d["keyword_coverage"] < 0.5))
                    or (not d["answerable"] and not d["refused"])]
        if failures:
            print(f"{len(failures)} failing cases (inspect these -- they are the roadmap):\n")
            for d in failures[:12]:
                why = "false refusal" if d["answerable"] and d["refused"] else (
                    "hallucinated" if not d["answerable"] else
                    f"coverage {d['keyword_coverage']:.0%}")
                print(f"  [{d['category']:9}] {why:16} {d['question'][:60]}")
            print()
        out["answers"] = {"metrics": asdict(m.answer), "model": m.label,
                          "p50_ms": m.p50_ms, "cost_usd": m.total_cost_usd}
        out["answers_table_markdown"] = report
        out["details"] = details

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
