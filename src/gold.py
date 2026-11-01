"""Turn a labeled review sheet into a gold set, following docs/labeling_guide.md.

`yes` quotes, every `span:` and quotes moved by `belongs:` become gold passages, cut from the
agreement text. Page numbers and page footers are removed from that text first, so a passage
that crosses a page break still matches a model that left them out. Anything that cannot be
placed in the text comes back as a problem, and no gold file is written until none is left.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from .metrics import match_key
from .parsing import norm_ws
from .prompts import REPO_ROOT, clauses_for

PAGE_LINE = re.compile(r"^[ \t]*(?:\d{1,3}|LEGAL_US_E # [\d.]+)[ \t]*(?:\n|$)", re.M)
ANCHOR_WORDS = 5
MIN_SIMILARITY = 0.9       # word-level match needed to place a quote that is not verbatim
MAX_ANCHOR_SKIP = 12
MAX_ANCHOR_HITS = 20
Gold = dict[str, dict[str, list[dict]]]    # doc -> clause -> [{text, main, origin}]


def clean_source(text: str) -> str:
    return PAGE_LINE.sub("", text)


class Source:
    """Agreement text with a match key that maps back to the original characters."""

    def __init__(self, text: str):
        self.text = clean_source(text)
        chars, offsets, gap = [], [], False
        for i, ch in enumerate(self.text):
            if ch.isspace():
                if not gap:
                    chars.append(" ")
                    offsets.append(i)
                gap = True
                continue
            gap = False
            for c in match_key(ch):
                chars.append(c)
                offsets.append(i)
        self.key, self.offsets = "".join(chars), offsets

    def cut(self, start: int, end: int) -> str:
        return norm_ws(self.text[self.offsets[start]:self.offsets[end - 1] + 1])

    def span(self, span: str) -> str | None:
        """Text of a span, where ' ... ' joins its opening and closing words."""
        head, _, tail = (match_key(x) for x in span.partition(" ... "))
        start = self.key.find(head)
        if start < 0 or not head:
            return None
        if not tail:
            return self.cut(start, start + len(head))
        end = self.key.find(tail, start + len(head))
        return self.cut(start, end + len(tail)) if end >= 0 else None

    def place(self, quote: str) -> str | None:
        """Text of a quote, allowing small copying changes. None if it is not there."""
        key = match_key(quote)
        start = self.key.find(key)
        if start >= 0 and key:
            return self.cut(start, start + len(key))
        words = key.split()
        if len(words) < 2 * ANCHOR_WORDS:
            return None
        best, best_score = None, MIN_SIMILARITY
        reach = min(MAX_ANCHOR_SKIP, len(words) - 2 * ANCHOR_WORDS + 1)
        for skip in range(reach):                       # added lead-in words move the opening
            head = " ".join(words[skip:skip + ANCHOR_WORDS])
            hits = [m.start() for m in re.finditer(re.escape(head), self.key)]
            for a in hits[:MAX_ANCHOR_HITS]:
                for back in range(reach):               # and added closing words move the end
                    tail = " ".join(words[-ANCHOR_WORDS - back:len(words) - back])
                    b = self.key.find(tail, a + len(head))
                    if b < 0 or b + len(tail) - a > 1.3 * len(key):
                        continue
                    end = b + len(tail)
                    score = SequenceMatcher(None, self.key[a:end].split(), words,
                                            autojunk=False).ratio()
                    if score >= best_score:
                        best, best_score = (a, end), score
                    break
        return self.cut(*best) if best else None


def parse_note(note: str) -> tuple[list[str], list[tuple[str, bool]]]:
    """Split a note into its tags and its (span, is_main) pairs."""
    head, sep, tail = note.partition("span: ")
    tags = [t.strip() for t in head.split(";") if t.strip()]
    spans = [(s.removeprefix("[main] "), s.startswith("[main] "))
             for s in tail.split(" || ")] if sep else []
    return tags, spans


def build_gold(rows: list[dict], sources: dict[str, Source],
               clauses: list[str]) -> tuple[Gold, list[str]]:
    """Gold passages from labeled rows, plus every row that needs another look."""
    gold: Gold = {d: {c: [] for c in clauses} for d in sources}
    seen: dict[tuple[str, str, str], dict] = {}
    problems: list[str] = []

    def add(doc: str, clause: str, text: str, main: bool, origin: str) -> None:
        k = (doc, clause, match_key(text))
        if k in seen:
            seen[k]["main"] = seen[k]["main"] or main
            return
        seen[k] = {"text": text, "main": main, "origin": origin}
        gold[doc][clause].append(seen[k])

    for r in rows:
        if not r["correct"]:
            continue
        doc, clause, correct = r["doc_id"], r["clause"], r["correct"]
        where = f"{doc} | {clause} | #{r['quote_no']}"
        tags, spans = parse_note(r["note"])
        src = sources[doc]
        if correct in ("partial", "missed") and not spans:
            problems.append(f"{where}: {correct} needs a span")
        if correct == "yes" or any(t.startswith("belongs: ") for t in tags):
            text = src.place(r["quote"])
            if text is None:
                problems.append(f"{where}: quote not placed in the source")
        if correct == "yes" and text:
            add(doc, clause, text, "main" in tags, "quote")
        for t in tags:
            if t.startswith("belongs: "):
                target = t[len("belongs: "):]
                if target not in clauses:
                    problems.append(f"{where}: unknown clause {target!r}")
                elif text:
                    add(doc, target, text, False, "belongs")
        for span, main in spans:
            text = src.span(span)
            if text is None:
                problems.append(f"{where}: span not found: {span[:70]}")
            else:
                add(doc, clause, text, main, "span")
    for doc, by_clause in gold.items():
        for clause, passages in by_clause.items():
            if passages and not any(p["main"] for p in passages):
                problems.append(f"{doc} | {clause}: no main passage")
    return gold, problems


def read_sources(text_dir: Path, docs: set[str]) -> dict[str, Source]:
    return {d: Source((text_dir / f"{d}.txt").read_text(encoding="utf-8")) for d in sorted(docs)}


def read_sheet(path: Path, condition: str | None = None) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if condition is None or r["condition"] == condition]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the gold set from a labeled review sheet.")
    ap.add_argument("--sheet", type=Path, default=REPO_ROOT / "reports" / "review_sheet.csv")
    ap.add_argument("--text-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "gold_m3.json")
    ap.add_argument("--condition", default="c", help="labeled condition in the sheet")
    ap.add_argument("--labeled-by", required=True)
    a = ap.parse_args(argv)
    rows = read_sheet(a.sheet, a.condition)
    sources = read_sources(a.text_dir, {r["doc_id"] for r in rows})
    gold, problems = build_gold(rows, sources, clauses_for("credit_schema"))
    if problems:
        print(f"{len(problems)} rows need review, no gold file written:")
        print("\n".join(problems))
        return 1
    a.out.write_text(json.dumps({
        "sheet_sha256": hashlib.sha256(a.sheet.read_bytes()).hexdigest(),
        "labeled_by": a.labeled_by, "documents": gold}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    n = sum(len(p) for by_clause in gold.values() for p in by_clause.values())
    print(f"{n} gold passages over {len(gold)} documents -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
