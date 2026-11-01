"""Score models against a gold set from src/gold.py.

Every model goes through the same rule, so results stay comparable, including models whose
quotes nobody labeled. The report gives the guide's Found and Strict F1, overall and per clause
type, the upstream metrics, verbatim rates, and a paired comparison of each model with the
first one named.

    python -m src.score --gold reports/gold_m3.json --text-dir <dir> \
        --sheet "Model A=c" --sheet "Model B=b" --out reports/score_m3.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gold import clean_source, read_sheet
from .metrics import (_agg, bootstrap_ci, detection_report, guide_counts, match_key, paired_diff,
                      right_quote, verbatim_flags)
from .prompts import REPO_ROOT, clauses_for

Preds = dict[str, dict[str, list[str]]]


def load_gold(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["documents"]


def preds_from_sheet(path: Path, condition: str) -> Preds:
    preds: Preds = {}
    for r in read_sheet(path, condition):
        if r["found"] == "True":
            preds.setdefault(r["doc_id"], {}).setdefault(r["clause"], []).append(r["quote"])
    return preds


def preds_from_run(run_dir: Path, clauses: list[str]) -> Preds:
    from .baselines import merged_quotes
    return {d: {c: [q for q, _, _ in qs] for c, qs in by_clause.items()}
            for d, by_clause in merged_quotes(Path(run_dir), clauses).items()}


def _f1(counts: dict, prefix: str) -> float:
    return round(_agg(counts, list(counts), prefix)[2], 4)


def verbatim(preds: Preds, sources: dict[str, str]) -> dict:
    """Share of quotes in the document, as is and with page numbers and footers removed."""
    pairs = [(q, d) for d, by_clause in preds.items() for qs in by_clause.values() for q in qs]
    if not pairs:
        return {"quotes": 0, "rate": None, "rate_no_page_lines": None}

    def rate(texts: dict[str, str]) -> float:
        hits = sum(verbatim_flags([q], texts[d], casefold=True, fold_quotes=True)[0]
                   for q, d in pairs)
        return round(hits / len(pairs), 4)
    return {"quotes": len(pairs), "rate": rate(sources),
            "rate_no_page_lines": rate({d: clean_source(t) for d, t in sources.items()})}


def score(gold: dict, preds: Preds, clauses: list[str], sources: dict[str, str],
          n_boot: int = 1000) -> dict:
    counts = guide_counts(gold, preds, clauses)
    texts = {d: {c: [p["text"] for p in by_clause.get(c, [])] for c in clauses}
             for d, by_clause in gold.items()}
    upstream = detection_report(texts, preds, clauses, n_boot=n_boot)
    return {
        "counts": counts,
        "found_f1": _f1(counts, "found_"), "found_ci95": bootstrap_ci(counts, "found_", n=n_boot),
        "strict_f1": _f1(counts, "main_"), "strict_ci95": bootstrap_ci(counts, "main_", n=n_boot),
        "upstream_detection_f1": upstream["detection_f1"],
        "upstream_strict_f1": upstream["strict_f1"],
        "per_clause": {c: {k: _f1(one, f"{k}_") for k in ("found", "main")}
                       for c in clauses for one in [guide_counts(gold, preds, [c])]},
        "verbatim": verbatim(preds, sources),
    }


def rule_check(gold: dict, rows: list[dict]) -> dict:
    """How the automatic right-quote rule compares with the labels on a labeled sheet."""
    out = {"right_kept": 0, "right_lost": 0, "wrong_passed": 0, "wrong_rejected": 0}
    for r in rows:
        if r["found"] != "True":
            continue
        passages = [match_key(p["text"]) for p in gold[r["doc_id"]].get(r["clause"], [])]
        auto = right_quote(match_key(r["quote"]), passages)
        if r["correct"] in ("yes", "partial"):
            out["right_kept" if auto else "right_lost"] += 1
        else:
            out["wrong_passed" if auto else "wrong_rejected"] += 1
    return out


def report(gold: dict, results: dict[str, dict], clauses: list[str], labeled_by: str,
           check: dict | None = None, n_boot: int = 1000) -> str:
    names = list(results)
    present = {c: sum(bool(by_clause.get(c)) for by_clause in gold.values()) for c in clauses}
    L = [f"# Scores on {len(gold)} agreements", "",
         f"Gold set labeled by: {labeled_by}. {sum(present.values())} clause decisions have gold "
         f"text, out of {len(gold) * len(clauses)}.", "",
         "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def ci(pair) -> str:
        return f"{pair[0]:.3f}-{pair[1]:.3f}"
    for label, fn in (
            ("Found F1", lambda r: f"{r['found_f1']:.3f} ({ci(r['found_ci95'])})"),
            ("Strict F1", lambda r: f"{r['strict_f1']:.3f} ({ci(r['strict_ci95'])})"),
            ("upstream detection F1", lambda r: f"{r['upstream_detection_f1']:.3f}"),
            ("upstream strict F1", lambda r: f"{r['upstream_strict_f1']:.3f}"),
            ("quotes", lambda r: r["verbatim"]["quotes"]),
            ("verbatim, quote marks and case ignored", lambda r: r["verbatim"]["rate"]),
            ("same, page numbers removed", lambda r: r["verbatim"]["rate_no_page_lines"])):
        L.append(f"| {label} | " + " | ".join(str(fn(results[n])) for n in names) + " |")
    L += ["", "Brackets are 95% bootstrap intervals over agreements.", "",
          "| clause | agreements with it | " + " | ".join(f"{n}: Found | {n}: Strict" for n in names)
          + " |", "|---|---|" + "---|---|" * len(names)]
    for c in clauses:
        cells = [f"{results[n]['per_clause'][c]['found']:.3f} | "
                 f"{results[n]['per_clause'][c]['main']:.3f}" for n in names]
        L.append(f"| {c} | {present[c]} | " + " | ".join(cells) + " |")
    if len(names) > 1:
        L += ["", f"Paired difference from {names[0]}, same agreement resamples:", "",
              "| model | Found F1 difference (95% CI) | Strict F1 difference (95% CI) |",
              "|---|---|---|"]
        for n in names[1:]:
            pairs = [paired_diff(results[n]["counts"], results[names[0]]["counts"], p, n=n_boot)
                     for p in ("found_", "main_")]
            L.append(f"| {n} | " + " | ".join(f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})"
                                             for d, lo, hi in pairs) + " |")
    if check:
        L += ["", "Automatic right-quote rule against the labels: "
              f"{check['right_kept']} right quotes kept, {check['right_lost']} lost, "
              f"{check['wrong_passed']} wrong quotes passed, {check['wrong_rejected']} rejected."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score models against a gold set.")
    ap.add_argument("--gold", type=Path, default=REPO_ROOT / "reports" / "gold_m3.json")
    ap.add_argument("--text-dir", type=Path, required=True)
    ap.add_argument("--review-sheet", type=Path, default=REPO_ROOT / "reports" / "review_sheet.csv")
    ap.add_argument("--sheet", action="append", default=[], metavar="NAME=CONDITION",
                    help="predictions from the review sheet")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=RUN_DIR",
                    help="predictions from a run directory")
    ap.add_argument("--check-labels", metavar="CONDITION", help="check the rule on this condition")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    clauses = clauses_for("credit_schema")
    raw = json.loads(a.gold.read_text(encoding="utf-8"))
    gold = raw["documents"]
    sources = {d: (a.text_dir / f"{d}.txt").read_text(encoding="utf-8") for d in gold}
    preds = {}
    for spec in a.sheet:
        name, _, cond = spec.partition("=")
        preds[name] = preds_from_sheet(a.review_sheet, cond)
    for spec in a.run:
        name, _, run_dir = spec.partition("=")
        preds[name] = preds_from_run(Path(run_dir), clauses)
    if not preds:
        ap.error("name at least one model with --sheet or --run")
    results = {n: score(gold, p, clauses, sources) for n, p in preds.items()}
    check = rule_check(gold, read_sheet(a.review_sheet, a.check_labels)) if a.check_labels else None
    md = report(gold, results, clauses, raw.get("labeled_by", "unknown"), check)
    if a.out:
        a.out.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
