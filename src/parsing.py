"""Parse model output into {clause: [quotes]} and classify failures.

Generation is unconstrained (no JSON grammar), so JSON validity is a real, measurable
metric. Statuses:
  ok       - parsed to a JSON object
  repaired - parsed only after escaping double quotes inside strings
  loop     - hit the output-token cap or ended in a detected repetition loop
  invalid  - finished normally but is not a JSON object, even after repair
"""
from __future__ import annotations

import json
import re

from .prompts import norm_key

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)
# what can follow a string's real closing quote in {clause: [quotes]}: key, list end, next quote
_CLOSES_STRING = re.compile(r'\s*(?::\s*\[|\]|,\s*")')

OK_STATUSES = ("ok", "repaired")      # a usable decode; anything else goes to the ladder


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


def repair_quotes(s: str) -> str:
    """Escape double quotes inside strings, the model's copying error on quoted contract text.

    A quote ends a string only if a key colon, list end or next string follows. Ambiguous
    text such as `"Lender", "Agent"` inside a quote can still split it in two, which is why
    quotes from repaired decodes get their own verbatim rate.
    """
    out, in_string, i = [], False, 0
    while i < len(s):
        ch = s[i]
        if in_string and ch == "\\":
            out.append(s[i:i + 2])
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
            elif _CLOSES_STRING.match(s, i + 1):
                in_string = False
            else:
                out.append("\\")
        out.append(ch)
        i += 1
    return "".join(out)


def parse_output(raw: str):
    """Strict parse, then the same parse after `repair_quotes`. Returns a dict or None."""
    d = parse_json_object(raw)
    if d is None:
        d = parse_json_object(repair_quotes(raw))
    return d


def json_error(raw: str, context: int = 160) -> dict | None:
    """Where strict parsing fails (for diagnosis), or None if `raw` parses."""
    s = raw.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1)
    try:
        json.loads(s)
        return None
    except json.JSONDecodeError as e:
        return {"error": e.msg, "pos": e.pos,
                "context": s[max(0, e.pos - context):e.pos + context]}


def classify(raw: str, finish_reason: str | None) -> tuple[str, dict | None]:
    d = parse_json_object(raw)
    if d is not None:
        return "ok", d
    if finish_reason in ("length", "repetition") or detect_repetition(raw):
        return "loop", None
    d = parse_output(raw)
    if d is not None:
        return "repaired", d
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
