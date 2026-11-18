"""PDF → pages → overlapping chunks that never lose their page number."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: int
    doc: str
    page: int          # 1-based
    heading: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def read_pdf(path: str | Path) -> list[str]:
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,&/()'-]{2,70}$")


def sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if para:
            out += [s.strip() for s in _SENT.split(para) if s.strip()]
    return out


def headings(text: str) -> list[str]:
    """Short title-like lines without ending punctuation are treated as section headings."""
    return [l.strip() for l in text.splitlines() if _HEADING.match(l.strip()) and not l.strip().endswith(".")
            and len(l.split()) <= 10]


def chunk_pages(pages: list[str], doc: str, size: int = 70, overlap: int = 15, start_id: int = 0) -> list[Chunk]:
    """Pack whole sentences up to `size` words; carry the last ~`overlap` words into the next chunk.
    Chunks stay within a page so every citation points at one page."""
    chunks, cid, heading = [], start_id, ""
    for pno, raw in enumerate(pages, 1):
        text = clean(raw)
        hs = headings(text)
        if hs:
            heading = hs[0]
        buf: list[str] = []
        count = 0
        for s in sentences(text):
            n = len(s.split())
            if buf and count + n > size:
                chunks.append(Chunk(cid, doc, pno, heading, " ".join(buf))); cid += 1
                keep, kept = [], 0
                for prev in reversed(buf):
                    if kept >= overlap:
                        break
                    keep.insert(0, prev); kept += len(prev.split())
                buf, count = keep, kept
            buf.append(s); count += n
        if buf:
            chunks.append(Chunk(cid, doc, pno, heading, " ".join(buf))); cid += 1
    return chunks
