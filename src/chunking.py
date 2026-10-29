"""Character-window chunking with overlap.

Window starts follow range(0, max(1, n - overlap), size - overlap), the rule described in
the upstream README (60k chars, 8k overlap). A span shorter than `overlap` is therefore
always fully inside at least one chunk. Credit agreements use 32k / 4k (see README).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    n_chunks: int
    start: int
    end: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}__chunk{self.index}"


def chunk_bounds(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= overlap:
        raise ValueError("chunk size must exceed overlap")
    starts = list(range(0, max(1, n - overlap), size - overlap))
    return [(s, min(s + size, n)) for s in starts]


def chunk_text(doc_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    bounds = chunk_bounds(len(text), size, overlap)
    return [Chunk(doc_id, i, len(bounds), s, e, text[s:e]) for i, (s, e) in enumerate(bounds)]


def safe_id(s: str) -> str:
    """Filesystem- and Hub-safe file stem for a chunk id."""
    return "".join(c if c.isalnum() or c in "-_.," else "_" for c in s)
