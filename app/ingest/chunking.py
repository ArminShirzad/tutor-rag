"""Stage 2 -- Chunking.

Why chunk at all? Two hard constraints:
  1. Embedding models have a fixed window (all-MiniLM-L6-v2 truncates at 256
     word-pieces). Embedding a whole document silently throws away the tail.
  2. An embedding is one vector -- an average of everything in the text. A
     10-page document averages into mush and matches nothing precisely. Small
     chunks keep the vector semantically sharp.

The trade-off, which is the real interview answer:
  chunks too small -> precise vectors, but the answer gets split in half and
                      the model sees a fragment without its context.
  chunks too large -> full context, but a diluted vector, worse retrieval, and
                      more tokens per query (higher latency and cost).

Four strategies are implemented so the choice can be *measured* rather than
argued about. `scripts/compare_chunking.py` runs them all against the eval set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from app.ingest.loaders import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str            # what the LLM reads
    embed_text: str      # what we embed -- may carry extra context (see below)
    source: str
    title: str
    section: str = ""
    index: int = 0
    start: int = 0
    end: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        loc = self.source
        if self.section:
            loc += f" > {self.section}"
        page = self.metadata.get("page")
        if page:
            loc += f" (p.{page})"
        return loc


# --------------------------------------------------------------------------
# Strategy 1: fixed-size character windows. The naive baseline.
# --------------------------------------------------------------------------
def split_fixed(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start, n = 0, len(text)
    step = max(1, size - overlap)
    while start < n:
        end = min(start + size, n)
        spans.append((start, end))
        if end == n:
            break
        start += step
    return spans


# --------------------------------------------------------------------------
# Strategy 2: sentence-aware packing. Never cuts mid-sentence.
# --------------------------------------------------------------------------
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n{2,}")


def split_sentences(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    # Build sentence spans, then greedily pack them up to `size`.
    bounds: list[tuple[int, int]] = []
    prev = 0
    for m in _SENT_RE.finditer(text):
        bounds.append((prev, m.start()))
        prev = m.end()
    bounds.append((prev, len(text)))
    bounds = [(a, b) for a, b in bounds if b > a]

    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int = 0
    for a, b in bounds:
        if cur_start is None:
            cur_start, cur_end = a, b
        elif b - cur_start <= size:
            cur_end = b
        else:
            spans.append((cur_start, cur_end))
            # step back `overlap` characters so consecutive chunks overlap
            back = max(cur_start, cur_end - overlap)
            cur_start, cur_end = (back if back < a else a), b
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return spans


# --------------------------------------------------------------------------
# Strategy 3: recursive -- the sane default. This is what LangChain's
# RecursiveCharacterTextSplitter does. Break on the most semantic separator
# available; fall back to finer ones only when a piece is still too big.
# --------------------------------------------------------------------------
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def _recursive(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size or not seps:
        return [text]
    sep, rest = seps[0], seps[1:]
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = part if not buf else buf + sep + part
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                out.append(buf)
                buf = ""
            if len(part) > size:
                out.extend(_recursive(part, size, rest))
            else:
                buf = part
    if buf:
        out.append(buf)
    return [p for p in out if p.strip()]


def split_recursive(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    pieces = _recursive(text, size, SEPARATORS)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        idx = text.find(piece, cursor)
        if idx == -1:
            idx = cursor
        start = max(0, idx - overlap) if spans else idx
        end = idx + len(piece)
        spans.append((start, end))
        cursor = end
    return spans


# --------------------------------------------------------------------------
# Strategy 4: structural / heading-aware. Best for course notes, docs and
# handbooks, because a heading tells you what a section is *about*.
# --------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def split_by_heading(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Returns (start, end, section_title) so the heading can be attached."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(a, b, "") for a, b in split_recursive(text, size, overlap)]

    sections: list[tuple[int, int, str]] = []
    if matches[0].start() > 0:
        sections.append((0, matches[0].start(), ""))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((start, end, m.group(2).strip()))

    out: list[tuple[int, int, str]] = []
    for start, end, heading in sections:
        body = text[start:end]
        if len(body) <= size:
            out.append((start, end, heading))
        else:
            # A long section is split further, but every piece keeps its heading.
            for a, b in split_recursive(body, size, overlap):
                out.append((start + a, start + b, heading))
    return out


STRATEGIES: dict[str, Callable] = {
    "fixed": split_fixed,
    "sentence": split_sentences,
    "recursive": split_recursive,
    "structural": split_by_heading,
}

_PAGE_RE = re.compile(r"\[page (\d+)\]")


def _page_at(text: str, pos: int) -> str | None:
    """Last `[page N]` marker at or before `pos` -- the page this chunk starts on."""
    last = None
    for m in _PAGE_RE.finditer(text, 0, max(pos, 1)):
        last = m.group(1)
    return last


def chunk_document(
    doc: Document,
    strategy: str = "recursive",
    size: int = 800,
    overlap: int = 120,
    min_chars: int = 80,
    contextual: bool = True,
) -> list[Chunk]:
    """Split one document into Chunks.

    `contextual=True` implements "contextual retrieval": the text we *embed* is
    prefixed with the document title and section heading, while the text we
    *show the LLM* stays clean. A chunk that says "It runs in O(n log n)" is
    meaningless on its own -- prefixed with "Sorting Algorithms > Merge Sort" it
    becomes retrievable by a query about merge sort complexity. This single
    trick is usually worth more recall than any amount of embedding-model
    shopping.
    """
    fn = STRATEGIES.get(strategy, split_recursive)
    raw = fn(doc.text, size, overlap)

    chunks: list[Chunk] = []
    idx = 0
    for item in raw:
        if len(item) == 3:
            start, end, section = item
        else:
            start, end = item
            section = ""

        body = doc.text[start:end].strip()
        if len(body) < min_chars:
            continue

        page = _page_at(doc.text, start)
        body_clean = _PAGE_RE.sub("", body).strip()
        if not body_clean:
            continue

        header_bits = [doc.title]
        if section and section != doc.title:
            header_bits.append(section)
        embed_text = (" > ".join(header_bits) + "\n" + body_clean) if contextual else body_clean

        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{idx}",
                doc_id=doc.doc_id,
                text=body_clean,
                embed_text=embed_text,
                source=doc.source,
                title=doc.title,
                section=section,
                index=idx,
                start=start,
                end=end,
                metadata={"page": page} if page else {},
            )
        )
        idx += 1
    return chunks


def chunk_corpus(docs: list[Document], **kwargs) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, **kwargs))
    return out
