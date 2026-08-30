"""Calibrate the refusal threshold from the reranker's score distribution.

The `min_score` in config.py must not be a guess. Run this and read the gap.

    python scripts/calibrate_threshold.py

It scores questions we know are answerable from the corpus against questions we
know are not, and reports the separation. If the two distributions overlap, no
threshold can work and the answer is better retrieval, not a different number.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.pipeline import Retriever  # noqa: E402

IN_CORPUS = [
    "what is dropout and why does it work",
    "explain the self-attention formula",
    "what does AdamW change about weight decay",
    "why do we scale attention by the square root of d_k",
    "what is the difference between L1 and L2 regularization",
    "how does reciprocal rank fusion work",
    "what causes vanishing gradients",
    "when should I use layer normalization instead of batch normalization",
    "what is the difference between a bi-encoder and a cross-encoder",
    "why is cross entropy used for classification",
]

OUT_OF_CORPUS = [
    "what is the capital of France",
    "how do I bake sourdough bread",
    "what is the refund policy for this course",
    "who won the 2022 world cup",
    "how much does the premium subscription cost",
    "what time does the live session start tomorrow",
    "can I get a certificate after finishing",
    "what is the phone number for support",
]


def main() -> int:
    retriever = Retriever.from_index()
    if retriever.reranker is None:
        print("Reranker disabled -- calibration is meaningless. Set USE_RERANKER=true.")
        return 1

    def top_scores(questions: list[str]) -> list[tuple[str, float]]:
        out = []
        for q in questions:
            res = retriever.retrieve(q)
            out.append((q, res.chunks[0].rerank_score if res.chunks else float("-inf")))
        return out

    ins = top_scores(IN_CORPUS)
    outs = top_scores(OUT_OF_CORPUS)

    print("\nIN-CORPUS (should be answered)")
    for q, s in sorted(ins, key=lambda x: x[1]):
        print(f"  {s:8.3f}  {q}")
    print("\nOUT-OF-CORPUS (should be refused)")
    for q, s in sorted(outs, key=lambda x: x[1], reverse=True):
        print(f"  {s:8.3f}  {q}")

    lo_in = min(s for _, s in ins)
    hi_out = max(s for _, s in outs)
    print("\n" + "=" * 58)
    print(f"  lowest in-corpus score   {lo_in:8.3f}")
    print(f"  highest out-corpus score {hi_out:8.3f}")
    gap = lo_in - hi_out
    if gap > 0:
        print(f"  SEPARABLE. gap = {gap:.3f}")
        print(f"  recommended MIN_SCORE = {(lo_in + hi_out) / 2:.2f}  (midpoint of the margin)")
    else:
        print(f"  OVERLAP of {-gap:.3f} -- no threshold separates these.")
        print("  Fix retrieval (chunking, embeddings, reranker) rather than tuning the number.")
    print("=" * 58 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
