"""Central configuration. Every tunable knob in the RAG pipeline lives here.

Interview note: being able to point at a single config surface and say
"these are the levers I tuned, and here is the eval that justified each value"
is the difference between 'I followed a tutorial' and 'I own this system'.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load .env if present. Real environment variables always win, so a container
# or CI can override the file without editing it.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # python-dotenv is optional
    pass
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = DATA_DIR / "index"


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class ChunkingConfig:
    # Why 800/120: with a ~512-token embedding model, 800 characters lands near
    # 180-200 tokens -- comfortably inside the model's window, so nothing is
    # silently truncated. Overlap prevents a fact from being split across a
    # boundary and becoming unretrievable from either side.
    strategy: str = field(default_factory=lambda: _env("CHUNK_STRATEGY", "recursive"))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))
    min_chunk_chars: int = 80


@dataclass
class EmbeddingConfig:
    # "local" = sentence-transformers on CPU (free, offline, no rate limit).
    # "gemini" = Google text-embedding-004 (better quality, needs a key).
    provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "local"))
    local_model: str = field(
        default_factory=lambda: _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_EMBED_MODEL", "gemini-embedding-001"))
    batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH", 64))


@dataclass
class RetrievalConfig:
    # Retrieve wide, rerank narrow. The bi-encoder is cheap but imprecise, so we
    # let it nominate `candidate_k` documents; the expensive cross-encoder then
    # sorts them and we keep only `final_k` for the prompt.
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 20))
    final_k: int = field(default_factory=lambda: _env_int("FINAL_K", 5))
    mode: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "hybrid"))  # vector|bm25|hybrid
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    use_reranker: bool = field(default_factory=lambda: _env("USE_RERANKER", "true").lower() == "true")
    reranker_model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    # auto -> cross-encoder if torch is available, else the LLM reranker, else
    # lexical. "llm" is what the slim deployment profile uses: a 512MB
    # container cannot hold torch plus a 90MB cross-encoder.
    reranker_provider: str = field(default_factory=lambda: _env("RERANK_PROVIDER", "auto"))
    # Below this reranker score we treat the corpus as having no answer and
    # refuse, rather than letting the LLM improvise. This is our main
    # hallucination lever.
    #
    # CALIBRATED, not guessed. Measuring the cross-encoder's top score over
    # in-corpus vs out-of-corpus questions (scripts/calibrate_threshold.py):
    #     in-corpus      +0.85 .. +6.24
    #     out-of-corpus -11.11 .. -10.89
    # A ~12-point gap means any threshold in between separates them perfectly,
    # so -5.0 sits in the middle of a wide margin rather than on a cliff edge.
    # Re-run the calibration whenever the reranker model or corpus changes --
    # these are raw logits and are not comparable across models.
    min_score: float = field(default_factory=lambda: _env_float("MIN_SCORE", -5.0))


@dataclass
class GenerationConfig:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))  # auto|gemini|extractive
    # flash-lite, not flash: the free tier caps the full flash models at 20
    # generation requests PER DAY, which is not enough to run the eval once,
    # let alone serve a demo. The lite tier has a far larger daily allowance and
    # is more than adequate for grounded extraction from retrieved context --
    # the hard reasoning in this system happens in retrieval, not generation.
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "gemini-3.1-flash-lite"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    max_output_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))


@dataclass
class Settings:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    corpus_dir: Path = field(default_factory=lambda: Path(_env("CORPUS_DIR", str(CORPUS_DIR))))
    # Overridable because serverless platforms mount a read-only filesystem
    # with only /tmp writable, so the index cannot live next to the code.
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(INDEX_DIR))))

    @property
    def gemini_api_key(self) -> str | None:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


settings = Settings()
