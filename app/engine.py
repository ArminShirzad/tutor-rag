"""The RAG engine -- the single entry point the API and CLI both call.

    question -> retrieve -> ground -> generate -> verify -> Answer

The `verify` step is what separates this from a tutorial RAG: after generation
we check that every citation the model emitted actually exists, and we strip
any it invented. A model that cites [7] when only 5 sources were provided has
hallucinated its provenance, and we would rather catch that than ship it.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, settings as default_settings
from app.generation.llm import LLM, LLMResponse, build_llm
from app.generation.prompts import (
    ANSWER_SCHEMA,
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.retrieval.pipeline import RetrievalResult, RetrievedChunk, Retriever


@dataclass
class Citation:
    number: int
    source: str
    citation: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "source": self.source,
            "citation": self.citation,
            "snippet": self.text[:400] + ("..." if len(self.text) > 400 else ""),
            "score": round(self.score, 4),
        }


@dataclass
class Answer:
    question: str
    answer: str
    answered: bool
    citations: list[Citation] = field(default_factory=list)
    confidence: str = "medium"
    missing_information: str = ""
    retrieval: RetrievalResult | None = None
    llm: LLMResponse | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "answered": self.answered,
            "confidence": self.confidence,
            "missing_information": self.missing_information,
            "citations": [c.to_dict() for c in self.citations],
            "warnings": self.warnings,
            "diagnostics": {
                "retrieval_mode": self.retrieval.mode if self.retrieval else None,
                "reranked": self.retrieval.reranked if self.retrieval else None,
                "chunks_considered": len(self.retrieval.chunks) if self.retrieval else 0,
                "below_threshold": self.retrieval.below_threshold if self.retrieval else False,
                "model": self.llm.model if self.llm else None,
                "input_tokens": self.llm.input_tokens if self.llm else 0,
                "output_tokens": self.llm.output_tokens if self.llm else 0,
                "cost_usd": round(self.llm.cost_usd, 8) if self.llm else 0.0,
                "timings_ms": self.timings_ms,
                "retrieval_timings_ms": self.retrieval.timings_ms if self.retrieval else {},
                "retrieved": [c.to_dict() for c in self.retrieval.chunks] if self.retrieval else [],
            },
        }


_CITE_RE = re.compile(r"\[(\d+)\]")


class RAGEngine:
    def __init__(self, retriever: Retriever, llm: LLM, config: Settings | None = None):
        self.retriever = retriever
        self.llm = llm
        self.settings = config or default_settings

    @classmethod
    def from_index(cls, index_dir: Path | None = None, config: Settings | None = None) -> "RAGEngine":
        cfg = config or default_settings
        retriever = Retriever.from_index(index_dir, cfg)
        llm = build_llm(cfg.generation.provider, cfg.generation.model, cfg.gemini_api_key)
        return cls(retriever, llm, cfg)

    def answer(
        self,
        question: str,
        mode: str | None = None,
        final_k: int | None = None,
        use_reranker: bool | None = None,
        structured: bool = True,
    ) -> Answer:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        # ---- 1. retrieve
        t0 = time.perf_counter()
        result = self.retriever.retrieve(
            question, mode=mode, final_k=final_k, use_reranker=use_reranker
        )
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- 2. refuse early if retrieval found nothing credible.
        # Cheaper and safer than asking the model to refuse: zero tokens spent,
        # and no chance it improvises from weak context.
        if not result.chunks or result.below_threshold:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return Answer(
                question=question,
                answer=REFUSAL_MESSAGE,
                answered=False,
                confidence="low",
                missing_information="No sufficiently relevant passage was found in the indexed course material.",
                retrieval=result,
                timings_ms=timings,
                warnings=["refused_pre_generation"],
            )

        # ---- 3. ground + generate
        user_prompt = build_user_prompt(question, result.chunks)
        t0 = time.perf_counter()
        response = self.llm.generate(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=self.settings.generation.temperature,
            max_tokens=self.settings.generation.max_output_tokens,
            schema=ANSWER_SCHEMA if structured else None,
        )
        timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- 4. parse
        data = response.structured or {}
        answer_text = (data.get("answer") or response.text or "").strip()
        answered = data.get("answered")
        if answered is None:
            answered = REFUSAL_MESSAGE.lower() not in answer_text.lower()
        confidence = data.get("confidence", "medium")
        missing = data.get("missing_information", "") or ""

        # ---- 5. verify citations against what we actually retrieved
        warnings: list[str] = []
        cited_numbers = {int(n) for n in _CITE_RE.findall(answer_text)}
        valid_range = set(range(1, len(result.chunks) + 1))

        invented = cited_numbers - valid_range
        if invented:
            warnings.append(f"model cited non-existent sources: {sorted(invented)}")
            # Strip them rather than showing a student a dead citation.
            answer_text = _CITE_RE.sub(
                lambda m: "" if int(m.group(1)) in invented else m.group(0), answer_text
            )
            answer_text = re.sub(r"\s+([.,])", r"\1", answer_text).strip()
            cited_numbers &= valid_range

        if answered and not cited_numbers:
            warnings.append("answer has no citations -- ungrounded claim risk")
            confidence = "low"

        citations = [
            Citation(
                number=i,
                source=rc.chunk.source,
                citation=rc.chunk.citation,
                text=rc.chunk.text,
                score=rc.score,
            )
            for i, rc in enumerate(result.chunks, start=1)
            if i in cited_numbers
        ]

        if response.degraded_from:
            warnings.append(
                f"degraded: {response.degraded_from} unavailable, answered extractively"
            )
        elif response.raw_error:
            warnings.append(f"llm_error: {response.raw_error}")

        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return Answer(
            question=question,
            answer=answer_text or REFUSAL_MESSAGE,
            answered=bool(answered),
            citations=citations,
            confidence=confidence,
            missing_information=missing,
            retrieval=result,
            llm=response,
            timings_ms=timings,
            warnings=warnings,
        )
