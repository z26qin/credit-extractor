"""Merge chunk outputs per document: union per clause type, deduplicated.

Dedup key is the whitespace-normalized quote; the first-seen original form is kept.
"""
from __future__ import annotations

from .parsing import norm_ws


def merge_chunks(chunk_outputs: list[dict[str, list[str]]],
                 clauses: list[str]) -> dict[str, list[str]]:
    merged = {c: [] for c in clauses}
    seen = {c: set() for c in clauses}
    for out in chunk_outputs:
        for c in clauses:
            for q in out.get(c, []):
                k = norm_ws(q)
                if k and k not in seen[c]:
                    seen[c].add(k)
                    merged[c].append(q)
    return merged
