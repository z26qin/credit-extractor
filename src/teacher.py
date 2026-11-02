"""Teacher pre-labeling (configs/teacher.yaml): the trial on the 5 M1 agreements, scored
against their gold set, and the full run on the collected agreements.

    python -m src.teacher --text-dir <M1 text dir> [--out work/teacher_trial]
    python -m src.teacher --text-dir <dir> --manifest reports/collect_full_manifest.csv \
        --split reports/collect_full_split.json --out work/prelabel_full \
        --report reports/prelabel_full.md --max-usd <approved cap>

DeepSeek labels each chunk of the agreement text, with page numbers removed, and code keeps only
quotes it can check: the JSON parses, the key is a clause type, and the quote is in the chunk
after whitespace normalization, with curly and straight quote marks distinct. A chunk with a
failed quote is asked again, up to `retries` times at `retry_temperature`, and the passing quotes
of all its answers are kept. A chunk is left out of training if no answer parses, or if a clause
type had failed quotes and no passing one.

Every answer's tokens and cost are recorded. No request starts unless the spend so far plus the
request's worst case stays under `max_usd`, and a probe on a few chunks stops the run first if the
whole trial would not fit. Finished chunks are skipped on a rerun, and their cost still counts.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import yaml

from .chunking import Chunk, chunk_text, safe_id
from .edgar import read_manifest
from .gold import clean_source
from .parsing import norm_ws, parse_output
from .prompts import (CUAD_SYSTEM, REPO_ROOT, canon_map, credit_clauses, credit_user_prefix,
                      messages, norm_key)
from .score import load_gold, preds_from_sheet, report, score

EXCERPT = "CONTRACT EXCERPT:\n"
CHARS_PER_TOKEN = 3          # below the real ~4, so the worst case is an overestimate


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or REPO_ROOT / "configs" / "teacher.yaml").read_text("utf-8"))


def teacher_prefix(rules: str) -> str:
    """The student's user turn with the labeling rules before the excerpt. Students never see it."""
    prefix = credit_user_prefix()
    assert prefix.endswith(EXCERPT)
    return prefix[:-len(EXCERPT)] + "RULES:\n" + rules.strip() + "\n\n" + EXCERPT


def cost_usd(usage: dict, price: dict) -> float:
    hit = usage.get("prompt_cache_hit_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    return (hit * price["input_cache_hit"] + miss * price["input"]
            + usage.get("completion_tokens", 0) * price["output"]) / 1e6


def worst_case(msgs: list[dict], cfg: dict) -> float:
    tokens_in = sum(len(m["content"]) for m in msgs) / CHARS_PER_TOKEN
    price = cfg["price_per_million"]
    return (tokens_in * price["input"] + cfg["max_tokens"] * price["output"]) / 1e6


class Budget:
    """Spend so far plus the worst case of every request in flight never passes the cap."""

    def __init__(self, cap: float, spent: float = 0.0):
        self.cap, self.spent, self.held = cap, spent, 0.0
        self._lock = threading.Lock()

    def hold(self, worst: float) -> bool:
        with self._lock:
            if self.spent + self.held + worst > self.cap:
                return False
            self.held += worst
            return True

    def settle(self, worst: float, actual: float) -> None:
        with self._lock:
            self.held -= worst
            self.spent += actual


def chat(client: httpx.Client, cfg: dict, msgs: list[dict], temperature: float) -> dict:
    """One chat completion, retrying rate limits, server errors and dropped connections."""
    body = {"model": cfg["model"], "messages": msgs, "temperature": temperature,
            "max_tokens": cfg["max_tokens"], **cfg.get("request", {})}
    for attempt in range(cfg["http_tries"]):
        try:
            r = client.post("/chat/completions", json=body)
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            transient = (isinstance(e, httpx.TransportError) or e.response.status_code == 429
                         or e.response.status_code >= 500)
            if not transient or attempt == cfg["http_tries"] - 1:
                raise
            time.sleep(cfg["backoff_s"] * 2 ** attempt)
    raise AssertionError("unreachable")


def check(content: str, chunk_norm: str, canon: dict[str, str]) -> tuple[str, dict, list]:
    """(status, {clause: [passing quotes]}, [[clause or ?key, quote] that failed])."""
    d = parse_output(content)
    if not isinstance(d, dict):
        return "invalid", {}, []
    passing, failed = {}, []
    for k, v in d.items():
        clause = canon.get(norm_key(k))
        for q in v if isinstance(v, list) else [v]:
            if not (isinstance(q, str) and q.strip()):
                continue
            if clause is not None and norm_ws(q) in chunk_norm:
                passing.setdefault(clause, []).append(q)
            else:
                failed.append([clause or f"?{k}", q])
    return "ok", passing, failed


def label_chunk(client: httpx.Client, cfg: dict, chunk: Chunk, budget: Budget, prefix: str,
                canon: dict[str, str]) -> dict | None:
    """One chunk through the rejection flow. None if the budget stopped it before any answer."""
    msgs = messages(CUAD_SYSTEM, prefix, chunk.text)
    worst = worst_case(msgs, cfg)
    norm = norm_ws(chunk.text)
    kept: dict[str, dict[str, str]] = {}
    failed_clauses, attempts, stopped = set(), [], False
    for i in range(1 + cfg["retries"]):
        if not budget.hold(worst):
            stopped = True
            break
        temperature = 0.0 if i == 0 else cfg["retry_temperature"]
        start = time.time()
        try:
            resp = chat(client, cfg, msgs, temperature)
        except Exception:
            budget.settle(worst, 0.0)
            raise
        usage = resp.get("usage") or {}
        cost = cost_usd(usage, cfg["price_per_million"])
        budget.settle(worst, cost)
        choice = resp["choices"][0]
        content = choice["message"].get("content") or ""
        status, passing, failed = check(content, norm, canon)
        for clause, qs in passing.items():
            for q in qs:
                kept.setdefault(clause, {}).setdefault(norm_ws(q), q)
        failed_clauses |= {c for c, _ in failed if c in canon.values()}
        attempts.append({"temperature": temperature, "status": status,
                         "finish_reason": choice.get("finish_reason"), "usage": usage,
                         "cost_usd": round(cost, 6), "seconds": round(time.time() - start, 1),
                         "passing": {c: len(qs) for c, qs in passing.items()},
                         "failed": failed, "text": content})
        if status == "ok" and not failed:
            break
    if not attempts:
        return None
    lost = sorted(failed_clauses - set(kept))
    parsed = any(a["status"] == "ok" for a in attempts)
    reason = None if parsed and not lost else (
        "no answer parsed" if not parsed else "failed quotes and no passing one: " + ", ".join(lost))
    return {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "index": chunk.index,
            "attempts": attempts, "kept": {c: list(m.values()) for c, m in kept.items()},
            "left_out": reason is not None, "reason": reason, "budget_stopped": stopped}


def load_records(out: Path) -> list[dict]:
    return [json.loads(p.read_text("utf-8")) for p in sorted((out / "records").glob("*.json"))]


def _save(out: Path, rec: dict) -> None:
    path = out / "records" / f"{safe_id(rec['chunk_id'])}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def chunks_for(docs: dict[str, str], cfg: dict) -> list[Chunk]:
    return [c for d, t in docs.items()
            for c in chunk_text(d, clean_source(t), cfg["chunk_chars"], cfg["overlap"])]


def chunk_cost(rec: dict) -> float:
    return sum(a["cost_usd"] for a in rec["attempts"])


def run(cfg: dict, docs: dict[str, str], out: Path, transport: httpx.BaseTransport | None = None,
        api_key: str | None = None) -> dict:
    """Label every chunk not yet recorded. Returns what was spent and what is still unlabeled."""
    api_key = api_key or os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise SystemExit(f"set {cfg['api_key_env']} in the environment")
    (out / "records").mkdir(parents=True, exist_ok=True)
    chunks = chunks_for(docs, cfg)
    done = {r["chunk_id"] for r in load_records(out)}
    budget = Budget(cfg["max_usd"], sum(map(chunk_cost, load_records(out))))
    todo = [c for c in chunks if c.chunk_id not in done]
    canon, prefix = canon_map(credit_clauses()), teacher_prefix(cfg["rules"])
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=cfg["base_url"], headers=headers, timeout=cfg["timeout_s"],
                      transport=transport) as client:
        def one(c: Chunk) -> None:
            rec = label_chunk(client, cfg, c, budget, prefix, canon)
            if rec is not None:
                _save(out, rec)

        # spread over the agreements: the first chunks are mostly contents pages
        n_probe = max(0, min(cfg["probe_chunks"] - len(done), len(todo)))
        probe = todo[::max(1, len(todo) // n_probe)][:n_probe] if n_probe else []
        for c in probe:
            one(c)
        recs = load_records(out)
        if not recs:
            raise SystemExit(f"no chunk could be labeled under the ${cfg['max_usd']:.2f} cap")
        per_chunk = sum(map(chunk_cost, recs)) / len(recs)
        projected = per_chunk * len(chunks)
        print(f"{len(recs)} chunks labeled, ${budget.spent:.4f} so far; projected for all "
              f"{len(chunks)} chunks: ${projected:.2f} (cap ${cfg['max_usd']:.2f})", flush=True)
        if projected > cfg["max_usd"]:
            raise SystemExit(f"stopped: the projected ${projected:.2f} is over the approved "
                             f"${cfg['max_usd']:.2f}; ask before raising max_usd")
        with ThreadPoolExecutor(cfg["workers"]) as pool:
            list(pool.map(one, [c for c in todo if c not in probe]))
    labeled = {r["chunk_id"] for r in load_records(out)}
    return {"chunks": len(chunks), "spent_usd": round(budget.spent, 6),
            "not_labeled": [c.chunk_id for c in chunks if c.chunk_id not in labeled]}


# ------------------------------------------------------------------------------ scoring

def preds_from_records(records: list[dict], kept_only: bool = False) -> dict:
    """{doc: {clause: [quotes]}}, the union over chunks, deduplicated after normalization."""
    out: dict = {}
    for r in records:
        if kept_only and r["left_out"]:
            continue
        doc = out.setdefault(r["doc_id"], {})
        for clause, qs in r["kept"].items():
            for q in qs:
                doc.setdefault(clause, {}).setdefault(norm_ws(q), q)
    return {d: {c: list(m.values()) for c, m in by.items()} for d, by in out.items()}


def stats(records: list[dict], clauses: list[str]) -> dict:
    attempts = [a for r in records for a in r["attempts"]]
    usage = [a["usage"] for a in attempts]
    per_clause = {c: {"passing": 0, "failed": 0} for c in clauses}
    unknown = 0
    for a in attempts:
        for c, n in a["passing"].items():
            per_clause[c]["passing"] += n
        for c, _ in a["failed"]:
            if c in per_clause:
                per_clause[c]["failed"] += 1
            else:
                unknown += 1
    return {
        "chunks": len(records), "requests": len(attempts),
        "retried": sum(len(r["attempts"]) > 1 for r in records),
        "left_out": sum(r["left_out"] for r in records),
        "budget_stopped": sum(r["budget_stopped"] for r in records),
        "invalid_answers": sum(a["status"] != "ok" for a in attempts),
        "cut_off_answers": sum(a["finish_reason"] == "length" for a in attempts),
        "tokens_in": sum(u.get("prompt_tokens", 0) for u in usage),
        "tokens_in_cached": sum(u.get("prompt_cache_hit_tokens", 0) for u in usage),
        "tokens_out": sum(u.get("completion_tokens", 0) for u in usage),
        "cost_usd": round(sum(a["cost_usd"] for a in attempts), 4),
        "seconds_per_request": round(sum(a["seconds"] for a in attempts) / max(1, len(attempts)), 1),
        "per_clause": per_clause, "unknown_key_quotes": unknown,
    }


def how_labeled(cfg: dict, what: str) -> str:
    return (f"`{cfg['model']}` at `{cfg['base_url']}` labeled {what} in "
            f"{cfg['chunk_chars']:,}-character chunks with {cfg['overlap']:,} overlap, page numbers "
            f"removed, thinking off, at temperature 0, with up to {cfg['retries']} more answers at "
            f"{cfg['retry_temperature']} for a chunk with a failed quote.")


def run_lines(cfg: dict, s: dict, n_chunks: int) -> list[str]:
    """The Spend and Checks sections, shared by the trial and the full run."""
    L = ["## Spend", "",
         "| chunks labeled | requests | input tokens | of them cached | output tokens | cost at peak price |",
         "|---|---|---|---|---|---|",
         f"| {s['chunks']} of {n_chunks} | {s['requests']} | {s['tokens_in']:,} | "
         f"{s['tokens_in_cached']:,} | {s['tokens_out']:,} | ${s['cost_usd']:.4f} |", "",
         f"Cap ${cfg['max_usd']:.2f}. Off-peak hours bill half the peak price. "
         f"{s['seconds_per_request']} seconds per request on average.", "",
         "## Checks", "",
         f"- Chunks retried: {s['retried']}. Left out of training: {s['left_out']}. "
         f"Stopped by the budget: {s['budget_stopped']}.",
         f"- Answers that did not parse: {s['invalid_answers']}. Answers cut off at "
         f"{cfg['max_tokens']} tokens: {s['cut_off_answers']}. Quotes under a key that is not a "
         f"clause type: {s['unknown_key_quotes']}.", "",
         "| clause | quotes passing | quotes failing the check | share failing |", "|---|---|---|---|"]
    for c, n in s["per_clause"].items():
        total = n["passing"] + n["failed"]
        L.append(f"| {c} | {n['passing']} | {n['failed']} | "
                 f"{n['failed'] / total:.2f} |" if total else f"| {c} | 0 | 0 | - |")
    return L + ["", "Passing and failing are counted per answer, so a chunk asked again counts "
                "twice.", ""]


def write_report(cfg: dict, records: list[dict], docs: dict[str, str], gold_path: Path,
                 sheet: Path, n_chunks: int) -> tuple[str, bool]:
    """The trial on the 5 M1 agreements, scored against their gold set."""
    clauses = credit_clauses()
    raw = json.loads(gold_path.read_text("utf-8"))
    gold = load_gold(gold_path)
    preds = {"Model A": preds_from_sheet(sheet, "c"), "Teacher": preds_from_records(records),
             "Teacher, kept chunks only": preds_from_records(records, kept_only=True)}
    results = {n: score(gold, p, clauses, docs) for n, p in preds.items()}
    t, g = results["Teacher"], cfg["gate"]
    passed = t["found_f1"] >= g["found_f1"] and (t["right_share"] or 0) >= g["right_share"]
    L = ["# Teacher trial", "", how_labeled(cfg, "the 5 M1 agreements"), "",
         f"**Gate: {'pass' if passed else 'FAIL'}.** Found F1 {t['found_f1']:.3f} against "
         f"{g['found_f1']:.2f} needed, and share of quotes right {t['right_share']} against "
         f"{g['right_share']:.2f} needed.", ""]
    L += run_lines(cfg, stats(records, clauses), n_chunks)
    L += ["## Scores", "",
          "\"Teacher\" is every passing quote. \"Kept chunks only\" leaves out the chunks that "
          "would not enter training. Five agreements are few, and they shaped the labeling "
          "rules, so read this as a direction.", "",
          report(gold, results, clauses, raw.get("labeled_by", "unknown")).split("\n", 2)[2]]
    return "\n".join(L), passed


def prelabel_report(cfg: dict, records: list[dict], parts: dict[str, list[str]],
                    n_chunks: int) -> str:
    """The full run: spend, checks, and what the labels hold, per split. No gold set exists."""
    clauses = credit_clauses()
    split_of = {d: name for name, ids in parts.items() for d in ids}
    n_docs = len(split_of)
    L = ["# Teacher pre-labeling", "",
         how_labeled(cfg, f"{n_docs} agreements ({', '.join(f'{len(v)} {k}' for k, v in parts.items())})"),
         "", "These are pre-labels. Test labels come from your review, and no pre-label enters "
         "them unreviewed.", ""]
    L += run_lines(cfg, stats(records, clauses), n_chunks)
    L += ["## What the labels hold", "", "| | " + " | ".join(parts) + " |",
          "|---|" + "---|" * len(parts)]
    rows = {"agreements": {}, "chunks": {}, "chunks with a quote": {}, "chunks left out": {},
            "quotes kept": {}}
    for name in parts:
        recs = [r for r in records if split_of.get(r["doc_id"]) == name]
        rows["agreements"][name] = len(parts[name])
        rows["chunks"][name] = len(recs)
        rows["chunks with a quote"][name] = sum(bool(r["kept"]) and not r["left_out"] for r in recs)
        rows["chunks left out"][name] = sum(r["left_out"] for r in recs)
        rows["quotes kept"][name] = sum(len(q) for r in recs for q in r["kept"].values())
    L += [f"| {k} | " + " | ".join(str(v[n]) for n in parts) + " |" for k, v in rows.items()]
    preds = preds_from_records(records)
    L += ["", "Agreements with at least one quote for each clause type, after merging chunks:", "",
          "| clause | " + " | ".join(parts) + " |", "|---|" + "---|" * len(parts)]
    for c in clauses:
        L.append(f"| {c} | " + " | ".join(
            f"{sum(bool(preds.get(d, {}).get(c)) for d in parts[n])} of {len(parts[n])}"
            for n in parts) + " |")
    return "\n".join(L) + "\n"


def load_docs(text_dir: Path, manifest: Path) -> dict[str, str]:
    """Agreement text, checked against its manifest."""
    docs = {}
    for doc_id, row in read_manifest(manifest).items():
        text = (text_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        if len(text) != int(row["char_count"]):
            raise ValueError(f"{doc_id}: {len(text)} chars, manifest says {row['char_count']}")
        docs[doc_id] = text
    return docs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Teacher pre-labeling: the M1 trial by default, or "
                                             "the full run with --split.")
    ap.add_argument("--text-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "reports" / "m1_manifest.csv")
    ap.add_argument("--split", type=Path, help="train/test ids: a full run, with no gold scoring")
    ap.add_argument("--out", type=Path, default=Path("work/teacher_trial"))
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "reports" / "teacher_trial.md")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--max-usd", type=float, help="the cap the user approved for this run")
    ap.add_argument("--probe-chunks", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--score-only", action="store_true", help="report on what is recorded")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    for key in ("max_usd", "probe_chunks", "workers"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    docs = load_docs(a.text_dir, a.manifest)
    if not a.score_only:
        print(json.dumps(run(cfg, docs, a.out)), flush=True)
    n_chunks = len(chunks_for(docs, cfg))
    if a.split:
        md, passed = prelabel_report(cfg, load_records(a.out),
                                     json.loads(a.split.read_text("utf-8")), n_chunks), True
    else:
        md, passed = write_report(cfg, load_records(a.out), docs,
                                  REPO_ROOT / "reports" / "gold_m3.json",
                                  REPO_ROOT / "reports" / "review_sheet.csv", n_chunks)
    a.report.write_text(md, encoding="utf-8")
    print(md)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
