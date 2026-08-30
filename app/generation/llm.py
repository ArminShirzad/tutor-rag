"""Stage 7b -- The LLM adapter.

One interface, three backends, so the pipeline never hard-depends on a vendor:

  gemini     -- Google Gemini via google-genai. Free tier, supports native JSON
                mode and function calling.
  extractive -- NO LLM AT ALL. Returns the top reranked sentences verbatim with
                citations. It cannot hallucinate by construction, and it lets
                the whole system run and be evaluated with zero API keys.
  auto       -- gemini if a key is present, otherwise extractive.

Why the adapter matters in an interview: "we might switch models for cost or
latency" is a certainty, not a hypothetical. Vendor calls live behind one
interface so switching is a config change, and so `evaluation/` can A/B two
providers on the same eval set without touching the pipeline.
"""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.resilience import RateLimiter, call_with_retry


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    structured: dict | None = None
    raw_error: str | None = None

    @property
    def cost_usd(self) -> float:
        """Estimated cost for this request.

        Rates are per million tokens and configurable, because published
        pricing changes and a hardcoded constant silently goes stale. Defaults
        are Flash-tier list prices; override with PRICE_IN_PER_M /
        PRICE_OUT_PER_M to match your actual contract.

        Tracked per request because cost is one of the four qualities this
        system is judged on -- and because an unbounded context is the usual
        way a RAG bill explodes.
        """
        price_in = float(os.environ.get("PRICE_IN_PER_M", 0.10))
        price_out = float(os.environ.get("PRICE_OUT_PER_M", 0.40))
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


class LLM(ABC):
    name: str

    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.1,
                 max_tokens: int = 1024, schema: dict | None = None) -> LLMResponse:
        ...


class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str = "gemini-3.6-flash",
                 rpm: int | None = None, max_retries: int | None = None):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini:{model}"
        # LLM_RPM=0 disables spacing (use it on a paid tier where the limit is
        # high enough that the retry path alone is sufficient).
        self.limiter = RateLimiter(
            int(os.environ.get("LLM_RPM", 5)) if rpm is None else rpm
        )
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", 4)) if max_retries is None else max_retries

    def generate(self, system: str, user: str, temperature: float = 0.1,
                 max_tokens: int = 1024, schema: dict | None = None) -> LLMResponse:
        from google.genai import types

        cfg: dict = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if schema:
            # Native JSON mode: the decoder is constrained to the schema, so
            # the output is guaranteed parseable. Strictly better than asking
            # politely for JSON in the prompt and regex-ing the result.
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema

        t0 = time.perf_counter()
        try:
            resp = call_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(**cfg),
                ),
                limiter=self.limiter,
                max_retries=self.max_retries,
                label="llm",
            )
        except Exception as exc:
            return LLMResponse(
                text=f"[generation failed: {exc}]",
                model=self.name,
                latency_ms=(time.perf_counter() - t0) * 1000,
                raw_error=str(exc),
            )
        latency = (time.perf_counter() - t0) * 1000

        text = (resp.text or "").strip()
        usage = getattr(resp, "usage_metadata", None)
        structured = None
        if schema and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = _salvage_json(text)

        return LLMResponse(
            text=text,
            model=self.name,
            latency_ms=latency,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            structured=structured,
        )


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ExtractiveLLM(LLM):
    """No model. Selects the sentences from the context that best match the
    question and returns them verbatim with citations.

    This is not a toy: extractive QA is a legitimate production choice when
    hallucination is unacceptable and fluency is optional. Here it also serves
    as the *floor* in evaluation -- if a generative model cannot beat verbatim
    extraction on faithfulness, it is not earning its cost.
    """

    name = "extractive:no-llm"

    def generate(self, system: str, user: str, temperature: float = 0.1,
                 max_tokens: int = 1024, schema: dict | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        context_match = re.search(r"CONTEXT\n(.*?)\n\nQUESTION\n(.*?)\n", user, re.DOTALL)
        if not context_match:
            return LLMResponse(text="I could not find this in the course material.", model=self.name)
        context, question = context_match.group(1), context_match.group(2)

        blocks = re.split(r"\n\n(?=\[\d+\])", context)
        q_terms = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2}

        scored: list[tuple[float, str, int]] = []
        for block in blocks:
            num_match = re.match(r"\[(\d+)\]", block)
            if not num_match:
                continue
            num = int(num_match.group(1))
            body = re.sub(r"^\[\d+\] Source: .*?\n", "", block, flags=re.DOTALL)
            for sent in _SENT_SPLIT.split(body):
                sent = sent.strip()
                if len(sent) < 40:
                    continue
                terms = {w for w in re.findall(r"[a-z0-9]+", sent.lower()) if len(w) > 2}
                if not terms:
                    continue
                overlap = len(q_terms & terms) / max(len(q_terms), 1)
                scored.append((overlap, sent, num))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [s for s in scored[:3] if s[0] > 0.08]

        if not picked:
            answer = "I could not find this in the course material."
            structured = {"answer": answer, "answered": False, "citations": [],
                          "confidence": "low", "missing_information": "No relevant passage found."}
        else:
            # restore document order so the answer reads coherently
            picked.sort(key=lambda x: (x[2], -x[0]))
            answer = " ".join(f"{sent} [{num}]" for _, sent, num in picked)
            structured = {"answer": answer, "answered": True,
                          "citations": sorted({n for _, _, n in picked}),
                          "confidence": "medium", "missing_information": ""}

        return LLMResponse(
            text=json.dumps(structured) if schema else answer,
            model=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000,
            structured=structured if schema else None,
        )


def _salvage_json(text: str) -> dict | None:
    """Best-effort recovery when a model wraps JSON in prose or code fences."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return None


def build_llm(provider: str = "auto", model: str = "gemini-3.6-flash",
              api_key: str | None = None) -> LLM:
    provider = (provider or "auto").lower()
    if provider in {"auto", "gemini"} and api_key:
        try:
            return GeminiLLM(api_key, model)
        except Exception as exc:
            print(f"[llm] Gemini unavailable ({exc}) -> extractive mode")
    elif provider == "gemini" and not api_key:
        print("[llm] GEMINI_API_KEY not set -> extractive mode (no hallucination, no fluency)")
    return ExtractiveLLM()
