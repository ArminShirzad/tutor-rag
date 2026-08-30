"""The golden evaluation set.

This file is the most valuable artefact in the repository. Anyone can wire an
embedding model to an LLM; the engineering question is "did that change make it
better?", and you cannot answer it without a fixed set of questions with known
correct answers.

Design rules used here:

1. Label the SOURCE, not the answer text. `expected_sources` records which
   document must be retrieved. Retrieval quality is then measurable exactly,
   with no LLM in the loop and no judgement calls.
2. `must_include` holds keywords the answer must contain -- a cheap, stable
   proxy for correctness that needs no second model.
3. `answerable: False` cases are as important as the positive ones. A system
   that answers everything scores 100% on positives and is dangerous. These
   measure the refusal path.
4. Cover the failure modes deliberately: paraphrases (vector search should
   win), exact jargon and acronyms (BM25 should win), multi-hop questions
   spanning two documents, and questions whose answer sits in a single
   sentence (chunking-sensitive).

30 examples is small. That is fine and deliberate: a small set you actually run
on every change beats a large one you run once.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    question: str
    expected_sources: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    answerable: bool = True
    category: str = "general"
    note: str = ""


GOLDEN_SET: list[EvalCase] = [
    # ---- paraphrase: no keyword overlap, vector search should carry these ----
    EvalCase(
        question="how do I stop my model from memorising the training examples",
        expected_sources=["02-overfitting-regularization.md"],
        must_include=["overfit"],
        category="paraphrase",
        note="'memorising' never appears near 'overfitting' as a keyword pair",
    ),
    EvalCase(
        question="my network does well in training but badly on new data, what is wrong",
        expected_sources=["02-overfitting-regularization.md"],
        must_include=["overfit"],
        category="paraphrase",
    ),
    EvalCase(
        question="how can I make the model pay attention to different parts of a sentence at once",
        expected_sources=["04-transformers-attention.md"],
        must_include=["head"],
        category="paraphrase",
    ),
    EvalCase(
        question="what makes text with similar meaning end up close together numerically",
        expected_sources=["05-embeddings-vector-search.md"],
        must_include=["embedding"],
        category="paraphrase",
    ),

    # ---- exact jargon and acronyms: BM25 should carry these ----
    EvalCase(
        question="what is AdamW",
        expected_sources=["03-optimizers-training.md"],
        must_include=["weight decay"],
        category="keyword",
        note="acronym-heavy; embeddings blur AdamW into generic 'optimizer'",
    ),
    EvalCase(
        question="what is the dying ReLU problem",
        expected_sources=["01-neural-networks.md"],
        must_include=["negative", "zero"],
        category="keyword",
    ),
    EvalCase(
        question="what does HNSW stand for",
        expected_sources=["05-embeddings-vector-search.md"],
        must_include=["Hierarchical Navigable Small World"],
        category="keyword",
    ),
    EvalCase(
        question="what is RoPE",
        expected_sources=["04-transformers-attention.md"],
        must_include=["Rotary", "position"],
        category="keyword",
    ),
    EvalCase(
        question="what is BM25",
        expected_sources=["05-embeddings-vector-search.md"],
        must_include=["keyword", "frequency"],
        category="keyword",
    ),
    EvalCase(
        question="explain reciprocal rank fusion",
        expected_sources=["05-embeddings-vector-search.md"],
        must_include=["rank", "60"],
        category="keyword",
    ),

    # ---- factual lookups ----
    EvalCase(
        question="what is the formula for self attention",
        expected_sources=["04-transformers-attention.md"],
        must_include=["softmax", "sqrt"],
        category="factual",
    ),
    EvalCase(
        question="why do we divide by the square root of d_k in attention",
        expected_sources=["04-transformers-attention.md"],
        must_include=["variance", "softmax"],
        category="factual",
        note="the reason is one sentence -- punishes over-large chunks",
    ),
    EvalCase(
        question="what is the difference between L1 and L2 regularization",
        expected_sources=["02-overfitting-regularization.md"],
        must_include=["sparse", "zero"],
        category="factual",
    ),
    EvalCase(
        question="what learning rate should I start with for Adam",
        expected_sources=["03-optimizers-training.md"],
        must_include=["1e-3"],
        category="factual",
    ),
    EvalCase(
        question="what does gradient clipping do",
        expected_sources=["03-optimizers-training.md"],
        must_include=["norm", "threshold"],
        category="factual",
    ),
    EvalCase(
        question="which loss function should I use for classification",
        expected_sources=["01-neural-networks.md"],
        must_include=["cross-entropy"],
        category="factual",
    ),
    EvalCase(
        question="what is cosine similarity and what range does it have",
        expected_sources=["05-embeddings-vector-search.md"],
        must_include=["angle"],
        category="factual",
    ),
    EvalCase(
        question="how many characters is a token roughly",
        expected_sources=["04-transformers-attention.md"],
        must_include=["four"],
        category="factual",
    ),
    EvalCase(
        question="what temperature should I use for factual question answering",
        expected_sources=["04-transformers-attention.md"],
        must_include=["low"],
        category="factual",
    ),
    EvalCase(
        question="what is the difference between an epoch and a batch",
        expected_sources=["01-neural-networks.md"],
        must_include=["pass", "update"],
        category="factual",
    ),

    # ---- multi-hop: the answer needs two documents ----
    EvalCase(
        question="why do transformers use layer norm instead of batch norm",
        expected_sources=["03-optimizers-training.md", "04-transformers-attention.md"],
        must_include=["batch size"],
        category="multihop",
    ),
    EvalCase(
        question="how are dropout and early stopping different approaches to the same problem",
        expected_sources=["02-overfitting-regularization.md"],
        must_include=["overfit"],
        category="multihop",
    ),
    EvalCase(
        question="what is the difference between a bi-encoder and a cross-encoder and when do I use each",
        expected_sources=["05-embeddings-vector-search.md"],
        must_include=["rerank"],
        category="multihop",
    ),
    EvalCase(
        question="why does ReLU help with vanishing gradients",
        expected_sources=["01-neural-networks.md", "03-optimizers-training.md"],
        must_include=["gradient"],
        category="multihop",
    ),

    # ---- debugging questions: how a real student would phrase it ----
    EvalCase(
        question="my loss became NaN during training, what should I check",
        expected_sources=["03-optimizers-training.md"],
        must_include=["learning rate"],
        category="debugging",
    ),
    EvalCase(
        question="my validation scores are random and bad, what did I forget",
        expected_sources=["02-overfitting-regularization.md"],
        must_include=["eval"],
        category="debugging",
        note="answer is model.eval() -- tests retrieval of a specific gotcha",
    ),

    # ---- must be refused: not in the corpus at all ----
    EvalCase(
        question="what is the refund policy for this course",
        answerable=False, category="refusal",
    ),
    EvalCase(
        question="who is the instructor for this module",
        answerable=False, category="refusal",
    ),
    EvalCase(
        question="what is the capital of France",
        answerable=False, category="refusal",
        note="the LLM knows this -- tests that grounding beats parametric memory",
    ),
    EvalCase(
        question="how do I install PyTorch on Windows with CUDA 12",
        answerable=False, category="refusal",
        note="plausibly in-domain but absent -- the hardest refusal case",
    ),
]


def by_category() -> dict[str, list[EvalCase]]:
    out: dict[str, list[EvalCase]] = {}
    for case in GOLDEN_SET:
        out.setdefault(case.category, []).append(case)
    return out
