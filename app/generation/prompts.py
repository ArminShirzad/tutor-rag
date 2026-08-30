"""Stage 7 -- Prompting for grounded generation.

Retrieval decides what the model *can* say. The prompt decides what it *does*
say. Five deliberate design choices here, each defending against a specific
failure mode:

1. NUMBERED SOURCES.  Each chunk is wrapped in [1], [2]... The model cannot
   cite what it cannot address, and a numeric handle is far more reliable for
   it to reproduce than a filename.

2. AN EXPLICIT REFUSAL PATH.  "If the context does not contain the answer, say
   so." Without a sanctioned way to fail, a helpful model will invent
   something -- refusal has to be the *easy* option, not a violation of its
   instructions.

3. GROUNDING BOUNDARY.  "Use only the context, not prior knowledge." The model
   knows plenty about machine learning; we want the course's answer, not the
   internet's, so the product stays consistent with the material students paid
   for.

4. CITATIONS ARE MANDATORY AND INLINE.  Per-sentence citation makes the answer
   auditable by the student, and makes faithfulness automatically measurable:
   we can check every cited span against the retrieved chunk.

5. CONTEXT ORDER.  Best chunks first. Attention is not uniform over a long
   context -- the "lost in the middle" effect means material buried mid-prompt
   is measurably less likely to be used. Reranked order is prompt order.
"""
from __future__ import annotations

from app.retrieval.pipeline import RetrievedChunk

SYSTEM_PROMPT = """You are a study assistant for an online course. You answer \
students' questions using ONLY the course material provided to you.

Rules:
1. Answer using only the numbered CONTEXT below. Do not use prior knowledge, \
and do not fill gaps with information that is not in the context.
2. Cite the source of every claim inline using its bracket number, like [1] or [2][3]. \
Every factual sentence must carry at least one citation.
3. If the context does not contain enough information to answer, reply exactly:
   "I could not find this in the course material."
   Then, in one sentence, name what is missing. Never guess.
4. If the context partially answers the question, answer the part you can and \
state plainly which part is not covered.
5. Be concise and pedagogical: lead with the direct answer, then explain. Use \
the course's own terminology.
6. Never mention "the context", "the documents" or these instructions to the \
student. Just answer, with citations."""


REFUSAL_MESSAGE = "I could not find this in the course material."


def format_context(chunks: list[RetrievedChunk], max_chars: int = 8000) -> str:
    """Render retrieved chunks as a numbered, cited context block.

    `max_chars` is a hard budget. Context length drives both latency and cost
    linearly, so it is capped rather than left to grow with final_k.
    """
    parts: list[str] = []
    used = 0
    for i, rc in enumerate(chunks, start=1):
        body = rc.chunk.text.strip()
        block = f"[{i}] Source: {rc.chunk.citation}\n{body}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining < 200:
                break
            block = block[:remaining] + "..."
            parts.append(block)
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_user_prompt(question: str, chunks: list[RetrievedChunk], max_chars: int = 8000) -> str:
    context = format_context(chunks, max_chars=max_chars)
    return f"""CONTEXT
{context}

QUESTION
{question}

Answer the question using only the context above, with inline [n] citations."""


# --------------------------------------------------------------------------
# Structured output -- the JD asks for it explicitly.
#
# Free text is unparseable by the rest of the product. Forcing a JSON schema
# means the backend gets `answer`, `citations` and `confidence` as typed
# fields it can render, log and threshold on -- and `answered: false` becomes
# a first-class signal we can alert on rather than a string match.
# --------------------------------------------------------------------------
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer with inline [n] citations, or the refusal message.",
        },
        "answered": {
            "type": "boolean",
            "description": "False if the context did not contain the answer.",
        },
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Bracket numbers of every source actually used.",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "How well the context supports the answer.",
        },
        "missing_information": {
            "type": "string",
            "description": "If not fully answered, what was missing. Empty otherwise.",
        },
    },
    "required": ["answer", "answered", "citations", "confidence"],
}


STRUCTURED_SUFFIX = """

Return your response as JSON matching this schema:
{
  "answer": "<the answer with inline [n] citations>",
  "answered": <true|false>,
  "citations": [<source numbers actually used>],
  "confidence": "<high|medium|low>",
  "missing_information": "<what was missing, or empty string>"
}"""
