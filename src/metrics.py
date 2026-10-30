"""Evaluation metrics, reimplemented from the definitions in Model B's model card / README.

Gold-based (CUAD reproduction, later the credit test set):
  detection P/R/F1 over (document, clause type) decisions: TP = gold and pred both
  non-empty; FP = pred only; FN = gold only. Bootstrap 95% CI over documents.
  overlap = mean over TPs of the best token-F1 between any predicted and gold quote.
  strict (ContractEval-style): TP only if every gold span is a casefolded substring of
  some predicted quote.
Label-free (Milestone 3):
  verbatim rate = share of quotes found in the source text after whitespace
  normalization (optionally casefolded, which is what the upstream 97.2% used).
"""
from __future__ import annotations

import random
from collections import Counter

from .parsing import norm_ws

Doc = dict[str, list[str]]          # clause -> quotes
_CURLY_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def tok_f1(a: str, b: str) -> float:
    wa, wb = a.lower().split(), b.lower().split()
    if not wa or not wb:
        return 0.0
    inter = sum((Counter(wa) & Counter(wb)).values())
    if inter == 0:
        return 0.0
    p, r = inter / len(wa), inter / len(wb)
    return 2 * p * r / (p + r)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def per_doc_counts(gold: dict[str, Doc], pred: dict[str, Doc], clauses: list[str]) -> dict:
    """{doc: {tp, fp, fn, ov_sum, ov_n, strict_tp, strict_fp, strict_fn}}."""
    out = {}
    for d in gold:
        c = dict(tp=0, fp=0, fn=0, ov_sum=0.0, ov_n=0, strict_tp=0, strict_fp=0, strict_fn=0)
        for cl in clauses:
            g = {norm_ws(q) for q in gold[d].get(cl, []) if norm_ws(q)}
            p = {norm_ws(q) for q in pred.get(d, {}).get(cl, []) if norm_ws(q)}
            if g and p:
                c["tp"] += 1
                c["ov_sum"] += max(tok_f1(pq, gq) for pq in p for gq in g)
                c["ov_n"] += 1
            elif p:
                c["fp"] += 1
            elif g:
                c["fn"] += 1
            if g:
                pf = [x.casefold() for x in p]
                if p and all(any(gq.casefold() in x for x in pf) for gq in g):
                    c["strict_tp"] += 1
                else:
                    c["strict_fn"] += 1
            elif p:
                c["strict_fp"] += 1
        out[d] = c
    return out


def _agg(counts: dict, docs: list[str], prefix: str = "") -> tuple[float, float, float]:
    tp = sum(counts[d][prefix + "tp"] for d in docs)
    fp = sum(counts[d][prefix + "fp"] for d in docs)
    fn = sum(counts[d][prefix + "fn"] for d in docs)
    return _prf(tp, fp, fn)


def bootstrap_ci(counts: dict, prefix: str = "", n: int = 1000,
                 seed: int = 42) -> tuple[float, float]:
    docs = list(counts)
    rng = random.Random(seed)
    fs = sorted(_agg(counts, rng.choices(docs, k=len(docs)), prefix)[2] for _ in range(n))
    # same indices as upstream for n=1000: fs[24], fs[974]
    return fs[max(0, int(0.025 * n) - 1)], fs[int(0.975 * n) - 1]


def detection_report(gold: dict[str, Doc], pred: dict[str, Doc], clauses: list[str],
                     n_boot: int = 1000) -> dict:
    counts = per_doc_counts(gold, pred, clauses)
    docs = list(counts)
    p, r, f = _agg(counts, docs)
    sp, sr, sf = _agg(counts, docs, "strict_")
    ov_n = sum(c["ov_n"] for c in counts.values())
    lo, hi = bootstrap_ci(counts, n=n_boot)
    slo, shi = bootstrap_ci(counts, "strict_", n=n_boot)
    return {
        "documents": len(docs), "decisions": len(docs) * len(clauses),
        "detection_precision": round(p, 4), "detection_recall": round(r, 4),
        "detection_f1": round(f, 4), "detection_f1_ci95": [round(lo, 4), round(hi, 4)],
        "overlap_tokenF1_on_TP": round(sum(c["ov_sum"] for c in counts.values()) / ov_n, 4)
        if ov_n else None,
        "strict_f1": round(sf, 4), "strict_f1_ci95": [round(slo, 4), round(shi, 4)],
        "strict_precision": round(sp, 4), "strict_recall": round(sr, 4),
    }


def verbatim_flags(quotes: list[str], source: str, casefold: bool = False,
                   fold_quotes: bool = False) -> list[bool]:
    """fold_quotes treats curly and straight quote marks as the same character."""
    def key(s: str) -> str:
        s = norm_ws(s)
        if fold_quotes:
            s = s.translate(_CURLY_QUOTES)
        return s.casefold() if casefold else s
    src = key(source)
    return [bool(k) and k in src for k in map(key, quotes)]


def verbatim_rate(quotes: list[str], source: str, casefold: bool = False) -> float | None:
    flags = verbatim_flags(quotes, source, casefold)
    return sum(flags) / len(flags) if flags else None
