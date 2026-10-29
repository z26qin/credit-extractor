"""Prompt strings and clause-name canon.

The CUAD prompt is copied verbatim from Model B's model card (Apache-2.0), which says the
model was trained on exactly this shape and anything else is out of distribution. Do not
edit these strings: tests pin them byte-for-byte.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CUAD_SYSTEM = ("You are a contract review system. Extract the requested clause types from the "
               "contract excerpt. Quote the contract text verbatim. A clause type not present "
               "in this excerpt gets an empty list. Reply with JSON only.")
CUAD_USER_PREFIX = ("Extract all 41 CUAD clause types from this contract excerpt as JSON "
                    "(keys: clause types, values: lists of verbatim quotes).\n\nCONTRACT EXCERPT:\n")


def load_cuad_clauses(path: Path | None = None) -> list[str]:
    """The 41 CUAD clause names in official order (data/category_descriptions.csv, CC BY 4.0)."""
    path = path or REPO_ROOT / "data" / "category_descriptions.csv"
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))[1:]
    clauses = [r[0].replace("Category: ", "").strip() for r in rows]
    if len(clauses) != 41:
        raise ValueError(f"expected 41 CUAD clause types, got {len(clauses)}")
    return clauses


def norm_key(s: str) -> str:
    """Key used to map model output keys back to canonical clause names."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def canon_map(clauses: list[str]) -> dict[str, str]:
    m = {norm_key(c): c for c in clauses}
    if len(m) != len(clauses):
        raise ValueError("clause names collide after normalization")
    return m


def messages(system: str, user_prefix: str, chunk_text: str) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": user_prefix + chunk_text}]


PROMPTS = {
    # name -> (system, user_prefix); credit prompts are added in Milestone 2
    "cuad_native": (CUAD_SYSTEM, CUAD_USER_PREFIX),
}
