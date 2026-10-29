"""Parse model output into {clause: [quotes]} and classify failures.

Generation is unconstrained (no JSON grammar), so JSON validity is a real, measurable
metric. Statuses:
  ok      - parsed to a JSON object
  loop    - hit the output-token cap or ended in a detected repetition loop
  invalid - finished normally but is not a JSON object
"""
from __future__ import annotations

import json
import re

from .prompts import norm_key

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def detect_repetition(text: str, min_period: int = 10, max_period: int = 1000,
                      min_repeats: int = 5, min_span: int = 200) -> bool:
    """True if the tail of `text` is one substring repeated >= min_repeats times.

    Catches the documented greedy failure (the same 1-2 sentences until the token cap).
    Legitimate JSON never repeats exactly: consecutive keys differ.
    """
    n = len(text)
    for p in range(min_period, min(max_period, n // min_repeats) + 1):
        span = p * min_repeats
        if span < min_span:
            continue
        unit = text[n - p:]
        if text[n - span:] == unit * min_repeats:
            return True
    return False


def parse_json_object(raw: str):
    """Return the dict the model produced, or None. Tolerates code fences / stray prose."""
    s = raw.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1)
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        pass
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            d = json.loads(s[i:j + 1])
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def classify(raw: str, finish_reason: str | None) -> tuple[str, dict | None]:
    d = parse_json_object(raw)
    if d is not None:
        return "ok", d
    if finish_reason in ("length", "repetition") or detect_repetition(raw):
        return "loop", None
    return "invalid", None


def norm_ws(s) -> str:
    return " ".join(str(s).split())


def canonicalize(d: dict, canon: dict[str, str]) -> tuple[dict[str, list[str]], int]:
    """Map output keys onto canonical clause names; keep non-empty string quotes.

    Returns ({clause: [quotes]} with every clause present, #dropped unknown keys).
    """
    out = {c: [] for c in canon.values()}
    dropped = 0
    for k, v in d.items():
        c = canon.get(norm_key(k))
        if c is None:
            dropped += 1
            continue
        if not isinstance(v, list):
            continue
        for q in v:
            if isinstance(q, str) and q.strip() and q not in out[c]:
                out[c].append(q)
    return out, dropped
