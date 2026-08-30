"""Tests that run offline in about a second.

Deliberately built on `HashingEmbedder` and `LexicalOverlapReranker` rather than
the real models: a test suite that needs a 90MB download and 4 seconds of model
loading is a test suite nobody runs. These cover the logic we wrote -- chunking
boundaries, BM25 maths, RRF fusion, persistence, citation verification -- not
the behaviour of somebody else's neural network.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.generation.llm import ExtractiveLLM
from app.index.bm25 import BM25, tokenize
from app.index.embeddings import HashingEmbedder, l2_normalize
from app.index.vector_store import VectorStore
from app.ingest.chunking import chunk_document, split_recursive
from app.ingest.loaders import Document
from app.retrieval.hybrid import reciprocal_rank_fusion, weighted_fusion
from app.retrieval.pipeline import Retriever
from app.retrieval.rerank import LexicalOverlapReranker

DOC = Document(
    doc_id="d1",
    source="ml.md",
    title="Machine Learning",
    text=(
        "# Machine Learning\n\n"
        "## Overfitting\n\n"
        "Overfitting happens when a model memorises the training data instead of "
        "learning the underlying pattern. The signature is training loss falling "
        "while validation loss rises.\n\n"
        "## Dropout\n\n"
        "Dropout randomly zeroes activations during training with probability p. "
        "It prevents neurons from co-adapting. Dropout is disabled at inference time.\n\n"
        "## Optimizers\n\n"
        "AdamW decouples weight decay from the gradient update. It is the standard "
        "choice for training transformer models.\n"
    ),
)


# ---------------------------------------------------------------- chunking
def test_chunking_respects_size_and_produces_chunks():
    chunks = chunk_document(DOC, strategy="recursive", size=200, overlap=30)
    assert chunks, "chunking produced nothing"
    # overlap means a chunk can exceed `size` by up to `overlap`
    assert all(len(c.text) <= 200 + 30 for c in chunks)


def test_structural_chunking_attaches_headings():
    chunks = chunk_document(DOC, strategy="structural", size=400, overlap=0)
    sections = {c.section for c in chunks}
    assert "Dropout" in sections
    assert "Optimizers" in sections


def test_contextual_embedding_prefixes_title_and_section():
    chunks = chunk_document(DOC, strategy="structural", size=400, overlap=0, contextual=True)
    dropout = next(c for c in chunks if c.section == "Dropout")
    # the embedded text carries the context...
    assert dropout.embed_text.startswith("Machine Learning > Dropout")
    # ...but the text shown to the LLM stays clean
    assert not dropout.text.startswith("Machine Learning >")


def test_contextual_can_be_disabled():
    chunks = chunk_document(DOC, strategy="structural", size=400, overlap=0, contextual=False)
    assert all(c.embed_text == c.text for c in chunks)


def test_recursive_split_covers_whole_text():
    text = "\n\n".join(f"Paragraph number {i} with some filler content." for i in range(20))
    spans = split_recursive(text, 150, 20)
    assert spans
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)


def test_citation_includes_section():
    chunks = chunk_document(DOC, strategy="structural", size=400, overlap=0)
    dropout = next(c for c in chunks if c.section == "Dropout")
    assert dropout.citation == "ml.md > Dropout"


# ---------------------------------------------------------------- bm25
def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the dropout layer")
    assert "dropout" in tokenize("the dropout layer")


def test_bm25_ranks_the_matching_document_first():
    corpus = [
        "dropout randomly zeroes activations during training",
        "adamw decouples weight decay from the gradient update",
        "cosine similarity measures the angle between two vectors",
    ]
    bm25 = BM25(corpus)
    top = bm25.search("what is dropout", k=3)
    assert top[0][0] == 0


def test_bm25_idf_penalises_ubiquitous_terms():
    # "model" appears everywhere -> near-zero IDF; "pgvector" is unique -> high
    corpus = [f"the model does thing {i}" for i in range(10)] + ["the model uses pgvector"]
    bm25 = BM25(corpus)
    assert bm25.idf["pgvector"] > bm25.idf["model"]


def test_bm25_returns_nothing_for_unknown_terms():
    bm25 = BM25(["dropout regularises networks"])
    assert bm25.search("kubernetes helm chart", k=5) == []


def test_bm25_handles_empty_corpus():
    assert BM25([]).search("anything", k=5) == []


# ---------------------------------------------------------------- fusion
def test_rrf_rewards_agreement_between_retrievers():
    # doc 1 is ranked #1 by both; doc 2 is #1 by one retriever only
    vector = [(1, 0.9), (2, 0.8)]
    bm25 = [(1, 12.0), (3, 9.0)]
    fused = reciprocal_rank_fusion(vector, bm25, k=60)
    assert fused[0].index == 1
    assert fused[0].score == pytest.approx(2 / 61)


def test_rrf_keeps_provenance():
    fused = reciprocal_rank_fusion([(1, 0.9)], [(1, 12.0)], k=60)
    hit = fused[0]
    assert hit.vector_rank == 1 and hit.bm25_rank == 1
    assert hit.vector_score == 0.9 and hit.bm25_score == 12.0


def test_rrf_survives_one_empty_retriever():
    fused = reciprocal_rank_fusion([(5, 0.7)], [], k=60)
    assert fused[0].index == 5


def test_weighted_fusion_alpha_selects_retriever():
    vector = [(1, 0.9), (2, 0.1)]
    bm25 = [(2, 20.0), (1, 1.0)]
    assert weighted_fusion(vector, bm25, alpha=1.0)[0].index == 1   # pure vector
    assert weighted_fusion(vector, bm25, alpha=0.0)[0].index == 2   # pure bm25


# ---------------------------------------------------------------- vectors
def test_l2_normalize_produces_unit_vectors():
    mat = np.array([[3.0, 4.0], [1.0, 0.0]], dtype="float32")
    assert np.allclose(np.linalg.norm(l2_normalize(mat), axis=1), 1.0)


def test_l2_normalize_survives_zero_vector():
    out = l2_normalize(np.zeros((1, 4), dtype="float32"))
    assert np.isfinite(out).all()


def test_vector_store_roundtrip(tmp_path):
    chunks = chunk_document(DOC, strategy="structural", size=400, overlap=0)
    emb = HashingEmbedder(dim=128)
    store = VectorStore(dim=emb.dim, embedder_name=emb.name)
    store.add(chunks, emb.encode([c.embed_text for c in chunks]))
    store.save(tmp_path)

    assert VectorStore.exists(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == len(store)
    assert loaded.dim == store.dim
    assert np.allclose(loaded.matrix, store.matrix)
    assert loaded.chunks[0].chunk_id == store.chunks[0].chunk_id


def test_vector_store_rejects_dimension_mismatch():
    store = VectorStore(dim=128)
    chunks = chunk_document(DOC, size=400)
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add(chunks, np.zeros((len(chunks), 64), dtype="float32"))


def test_vector_store_rejects_count_mismatch():
    store = VectorStore(dim=8)
    with pytest.raises(ValueError):
        store.add(chunk_document(DOC, size=400), np.zeros((1, 8), dtype="float32"))


def test_empty_store_search_returns_nothing():
    assert VectorStore(dim=8).search(np.zeros(8, dtype="float32"), k=5) == []


# ---------------------------------------------------------------- retrieval
@pytest.fixture
def retriever() -> Retriever:
    chunks = chunk_document(DOC, strategy="structural", size=400, overlap=0)
    emb = HashingEmbedder(dim=256)
    store = VectorStore(dim=emb.dim, embedder_name=emb.name)
    store.add(chunks, emb.encode([c.embed_text for c in chunks]))
    return Retriever(store, emb, LexicalOverlapReranker())


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_every_mode_finds_the_right_section(retriever, mode):
    res = retriever.retrieve("what does AdamW do to weight decay", mode=mode, final_k=3)
    assert res.chunks
    assert any("AdamW" in c.chunk.text for c in res.chunks)


def test_hybrid_records_both_retriever_ranks(retriever):
    res = retriever.retrieve("dropout", mode="hybrid", final_k=3, use_reranker=False)
    assert any(c.vector_rank is not None for c in res.chunks)
    assert any(c.bm25_rank is not None for c in res.chunks)


def test_reranking_populates_scores_and_sorts(retriever):
    res = retriever.retrieve("dropout at inference time", final_k=3, use_reranker=True)
    assert res.reranked
    scores = [c.rerank_score for c in res.chunks]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_timings_are_recorded(retriever):
    res = retriever.retrieve("dropout", mode="hybrid")
    assert res.timings_ms["total_ms"] > 0
    assert "bm25_search_ms" in res.timings_ms


# ---------------------------------------------------------------- generation
def test_extractive_llm_never_invents_content():
    llm = ExtractiveLLM()
    user = ("CONTEXT\n[1] Source: ml.md\nDropout randomly zeroes activations "
            "during training with probability p.\n\nQUESTION\nwhat is dropout\n")
    out = llm.generate("sys", user, schema={"type": "object"})
    assert out.structured["answered"] is True
    # every returned sentence must be present in the supplied context
    body = out.structured["answer"].replace(" [1]", "")
    assert body.strip(" .") in user


def test_extractive_llm_refuses_when_nothing_matches():
    llm = ExtractiveLLM()
    user = ("CONTEXT\n[1] Source: ml.md\nCosine similarity measures the angle "
            "between two vectors in a space.\n\nQUESTION\nkubernetes ingress "
            "controller configuration\n")
    out = llm.generate("sys", user, schema={"type": "object"})
    assert out.structured["answered"] is False
