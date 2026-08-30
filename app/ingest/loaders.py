"""Stage 1 -- Document processing.

Turn heterogeneous files (.md, .txt, .pdf, .docx) into a uniform `Document`.

The job interview version of this stage: "garbage in, garbage out". Most RAG
systems that fail in production fail here, not in the fancy retrieval math --
a PDF parsed with broken column order or headers/footers bleeding into every
page will poison every downstream stage.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SUPPORTED = {".md", ".markdown", ".txt", ".pdf", ".docx"}


@dataclass
class Document:
    """One source file after text extraction."""

    doc_id: str
    source: str          # human-readable path, shown in citations
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def _doc_id(path: Path, text: str) -> str:
    """Content-addressed id: re-ingesting an unchanged file yields the same id,
    which makes incremental re-indexing possible."""
    h = hashlib.sha256()
    h.update(str(path.name).encode("utf-8"))
    h.update(text[:4096].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def _clean(text: str) -> str:
    """Normalisation that is safe for every format."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")           # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse blank-line runs
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _title_from(path: Path, text: str) -> str:
    """Prefer a real markdown H1; fall back to a prettified filename."""
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    """Extract per page and tag page numbers so citations can point at a page."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required to ingest PDFs: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        body = page.extract_text() or ""
        if body.strip():
            # The marker survives chunking, so a chunk knows which page it came
            # from even after it is split.
            pages.append(f"[page {i}]\n{body}")
    return "\n\n".join(pages)


def load_docx(path: Path) -> str:
    """Minimal .docx reader -- a .docx is a zip with XML inside, so we avoid a
    dependency. Paragraphs become newlines, tabs are preserved."""
    import html

    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(xml)


LOADERS = {
    ".md": load_text,
    ".markdown": load_text,
    ".txt": load_text,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_file(path: Path) -> Document | None:
    suffix = path.suffix.lower()
    if suffix not in LOADERS:
        return None
    raw = LOADERS[suffix](path)
    text = _clean(raw)
    if not text:
        return None
    return Document(
        doc_id=_doc_id(path, text),
        source=path.name,
        title=_title_from(path, text),
        text=text,
        metadata={"format": suffix.lstrip("."), "chars": len(text)},
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Walk a directory and load every supported file, sorted for determinism."""
    docs: list[Document] = []
    for path in sorted(Path(corpus_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            doc = load_file(path)
            if doc:
                docs.append(doc)
    return docs
