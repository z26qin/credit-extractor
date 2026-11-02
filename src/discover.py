"""Data scale-up: find about 150 more credit agreements on EDGAR (configs/collect_full.yaml).

    python -m src.discover --out work/collect_full

1. Full-text search lists exhibits that use the credit-agreement phrases, one search per form.
2. Hits are kept if they are EX-10 or EX-4 exhibits whose name and description do not mark an
   amendment or side document, from an issuer that is not a fund and not in M1.
3. Issuers are visited in the order of a hash of their CIK, so the picks do not lean on size or
   date. For each, up to `max_docs_per_issuer` exhibits are tried, newest first: size from the
   filing index, then the text, which must be a full credit agreement dated 2023 or later and of
   `text_chars` length. The first that passes is the issuer's pick.
4. Each pick is classified from its defined terms (`classify`) and kept while its archetype has
   room in `quotas`. The run stops when every quota is full.
5. Within each archetype, the picks with the lowest CIK hashes go to test (`test_share`).

Every exhibit looked at is logged in `decisions.jsonl` with the reason it was kept or dropped, so
a rerun skips it. Requests go through the M1 client: at most 5 per second, with the SEC
User-Agent from the SEC_USER_AGENT secret.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .edgar import (ARCHIVES, EdgarClient, agreement_date, html_to_text, read_manifest, title_of,
                    write_manifest)
from .prompts import REPO_ROOT

SEARCH = "https://efts.sec.gov/LATEST/search-index"
PAGE = 100
ARCHETYPES = ("investment_grade", "secured_mid_cap", "leveraged")

SECURED = ("Collateral Documents", "Security Documents", "Security Agreement",
           "Collateral Agreement", "Guarantee and Collateral Agreement",
           "Pledge and Security Agreement", "Collateral Agent")
LEVERAGED = ("Available Amount", "Cumulative Credit", "Available Basket Amount", "Builder Basket",
             "Cumulative Amount", "Incremental Equivalent Debt",
             "Incremental Equivalent Indebtedness", "Excess Cash Flow", "Effective Yield",
             "All-In Yield", "Term B Loans", "Ratio Debt")
RATING = ("Debt Rating", "Debt Ratings", "Index Debt", "Public Debt Rating", "Rating Level",
          "Ratings", "Investment Grade Rating")
AMENDMENT = re.compile(r"AMENDMENT\s+(?:NO\b|TO\b)|\b(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|"
                       r"SEVENTH|EIGHTH|NINTH|TENTH)\s+(?:\w+\s+)?AMENDMENT\b|\bWAIVER\b|\bJOINDER\b")
COVER_CHARS = 1500           # an amendment says so on its cover, before any contents page
# fund, BDC and net-asset-value vehicles borrow against investments, unlike operating companies;
# SIC codes miss many of them. Every agreement mentions the Investment Company Act, so not that.
FUND_TERMS = ("Net Asset Value", "Portfolio Investment", "Portfolio Investments",
              "Capital Commitment", "Capital Commitments", "Uncalled Capital Commitments")
FUND_PHRASES = re.compile(r"business development company|regulated investment company", re.I)


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or REPO_ROOT / "configs" / "collect_full.yaml").read_text())


def cik_hash(cik: int | str) -> str:
    return hashlib.sha256(str(int(cik)).encode()).hexdigest()


# ------------------------------------------------------------------------------ search

def search_url(q: str, form: str, start: str, end: str, offset: int) -> str:
    params = {"q": q, "dateRange": "custom", "category": "custom", "startdt": start,
              "enddt": end, "forms": form}
    if offset:
        params["from"] = offset
    return SEARCH + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def search(client: EdgarClient, cfg: dict) -> list[dict]:
    """Every hit of every query and form, one row per exhibit, deduplicated."""
    hits = {}
    for query in cfg["queries"]:
        for form in query["forms"]:
            offset, total = 0, None
            while total is None or offset < min(total, 10_000):
                d = client.get_json(search_url(query["q"], form, cfg["date_from"],
                                               cfg["date_to"], offset))
                total = d["hits"]["total"]["value"]
                page = d["hits"]["hits"]
                for h in page:
                    s = h["_source"]
                    adsh, _, name = h["_id"].partition(":")
                    hits.setdefault(h["_id"], {
                        "key": h["_id"], "accession": adsh, "document": name,
                        "cik": int(s["ciks"][0]), "issuer": (s.get("display_names") or [""])[0],
                        "form": s.get("form", ""), "file_type": s.get("file_type") or "",
                        "description": s.get("file_description") or "",
                        "file_date": s.get("file_date", ""), "sics": s.get("sics") or []})
                if not page:
                    break
                offset += PAGE
    return list(hits.values())


def keep_hit(h: dict, cfg: dict, skip_ciks: set[int]) -> str | None:
    """Why a hit is dropped before any download, or None to keep it."""
    if h["cik"] in skip_ciks:
        return "M1 issuer"
    if re.search(cfg["exclude_issuer"], h["issuer"], re.I):
        return "a fund by name"
    if not re.match(cfg["exhibit_types"], h["file_type"], re.I):
        return f"not an agreement exhibit ({h['file_type']})"
    if re.search(cfg["exclude_names"], f"{h['document']} {h['description']}".lower()):
        return "name or description marks an amendment or side document"
    sics = [str(s) for s in h["sics"]]
    if any(s.startswith(tuple(cfg["exclude_sic_prefix"])) and s not in cfg["keep_sic"]
           for s in sics):
        return f"fund or investment office (SIC {', '.join(sics)})"
    return None


# ---------------------------------------------------------------------- text checks

def defined(text: str, terms) -> list[str]:
    """Terms the agreement defines, as “Term” with curly or straight quotes."""
    return [t for t in terms if re.search(rf"[“\"]{re.escape(t)}[”\"]", text)]


def classify(text: str) -> tuple[str, dict]:
    """Archetype from defined terms. Unsecured is investment grade; secured with two or more
    leveraged-loan terms is leveraged; any other secured agreement is secured mid-cap."""
    signals = {"secured": defined(text, SECURED), "leveraged": defined(text, LEVERAGED),
               "rating": defined(text, RATING)}
    if not signals["secured"]:
        return "investment_grade", signals
    return ("leveraged" if len(signals["leveraged"]) >= 2 else "secured_mid_cap"), signals


def check_text(text: str, cfg: dict) -> str | None:
    """Why the text is not a full credit agreement in range, or None if it is."""
    lo, hi = cfg["text_chars"]
    if not lo <= len(text) <= hi:
        return f"{len(text):,} characters, outside {lo:,} to {hi:,}"
    title = title_of(text)
    if "CREDIT" not in title and "LOAN AGREEMENT" not in title:
        return f"title is not a credit agreement: {title[:80]!r}"
    if AMENDMENT.search(text[:COVER_CHARS].upper()):
        return "an amendment"
    if FUND_PHRASES.search(text) or defined(text, FUND_TERMS):
        return "a fund or BDC facility"
    date = agreement_date(text)
    if not date or date < cfg["date_from"]:
        return f"agreement dated {date or 'unknown'}, before {cfg['date_from']}"
    return None


def html_size(client: EdgarClient, hit: dict) -> int | None:
    idx = client.get_json(f"{ARCHIVES}/{hit['cik']}/{hit['accession'].replace('-', '')}/index.json")
    for item in idx["directory"]["item"]:
        if item["name"] == hit["document"]:
            return int(item.get("size") or 0)
    return None


# ------------------------------------------------------------------------------ picks

def doc_id(hit: dict) -> str:
    """Issuer word plus CIK, unique because there is one agreement per issuer."""
    name = re.sub(r"\(.*", "", hit["issuer"]).strip().lower()
    word = re.sub(r"[^a-z0-9]+", "-", name).strip("-").split("-")[0] or "issuer"
    return f"{word}-{hit['cik']}"


def load_decisions(out: Path) -> list[dict]:
    path = out / "decisions.jsonl"
    return [json.loads(l) for l in path.read_text("utf-8").splitlines()] if path.exists() else []


def _log(out: Path, row: dict) -> None:
    with open(out / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick(client: EdgarClient, cfg: dict, hits: list[dict], out: Path,
         skip_ciks: set[int]) -> list[dict]:
    """Visit issuers in hash order and fill the quotas. Returns the manifest rows."""
    for sub in ("raw", "text"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    seen = {d["key"]: d for d in load_decisions(out)}
    by_cik: dict[int, list[dict]] = {}
    for h in hits:
        if keep_hit(h, cfg, skip_ciks) is None:
            by_cik.setdefault(h["cik"], []).append(h)
    rows = []
    for d in seen.values():           # earlier picks stay unless a newer filter drops them
        if d["status"] != "picked":
            continue
        if d["cik"] in by_cik:
            rows.append(d["row"])
            continue
        for sub, ext in (("raw", "htm"), ("text", "txt")):
            (out / sub / f"{d['row']['id']}.{ext}").unlink(missing_ok=True)
        print(f"   removed  {d['issuer'][:40]:40}  a newer filter drops it", flush=True)
    have = {a: sum(r["archetype"] == a for r in rows) for a in ARCHETYPES}
    done_ciks = {r["cik"] for r in rows}
    for cik in sorted(by_cik, key=cik_hash):
        if all(have[a] >= cfg["quotas"][a] for a in ARCHETYPES):
            break
        if cik in done_ciks:
            continue
        newest = sorted(by_cik[cik], key=lambda h: h["file_date"], reverse=True)
        for h in newest[:cfg["max_docs_per_issuer"]]:
            decision = seen.get(h["key"])
            if decision is not None and decision["status"] == "quota full" and \
                    have[decision["reason"]] < cfg["quotas"][decision["reason"]]:
                decision = None                     # its archetype has room again
            if decision is None:
                decision = seen[h["key"]] = examine(client, cfg, h, out, have)
                _log(out, decision)
                print(f"{decision['status']:>10}  {h['issuer'][:40]:40}  {decision['reason']}",
                      flush=True)
                if decision["status"] == "picked":
                    rows.append(decision["row"])
                    have[decision["row"]["archetype"]] += 1
            if decision["status"] in ("picked", "quota full"):
                break
    write_manifest(out / "manifest.csv", rows)
    return rows


def examine(client: EdgarClient, cfg: dict, h: dict, out: Path, have: dict) -> dict:
    """Size check, download, text checks and archetype for one exhibit."""
    base = {"key": h["key"], "cik": h["cik"], "issuer": h["issuer"], "file_date": h["file_date"]}
    lo, hi = cfg["html_bytes"]
    size = html_size(client, h)
    if size is None or not lo <= size <= hi:
        return dict(base, status="dropped", reason=f"HTML size {size}, outside {lo:,} to {hi:,}")
    body = client.get(f"{ARCHIVES}/{h['cik']}/{h['accession'].replace('-', '')}/{h['document']}")
    text = html_to_text(body.decode("utf-8", errors="replace"))
    why = check_text(text, cfg)
    if why:
        return dict(base, status="dropped", reason=why)
    archetype, signals = classify(text)
    if have[archetype] >= cfg["quotas"][archetype]:
        return dict(base, status="quota full", reason=archetype, signals=signals)
    pid = doc_id(h)
    (out / "raw" / f"{pid}.htm").write_bytes(body)
    (out / "text" / f"{pid}.txt").write_text(text, encoding="utf-8")
    row = {"id": pid, "issuer": h["issuer"], "cik": h["cik"], "form": h["form"],
           "accession": h["accession"], "document": h["document"],
           "url": f"{ARCHIVES}/{h['cik']}/{h['accession'].replace('-', '')}/{h['document']}",
           "filing_date": h["file_date"], "agreement_date": agreement_date(text) or "",
           "title": title_of(text), "archetype": archetype,
           "sha256": hashlib.sha256(body).hexdigest(), "raw_bytes": len(body),
           "char_count": len(text),
           "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return dict(base, status="picked", reason=archetype, signals=signals, row=row)


def split(rows: list[dict], test_share: float) -> dict[str, list[str]]:
    """Test takes the lowest CIK hashes within each archetype, so the mix carries over."""
    out = {"train": [], "test": []}
    for a in ARCHETYPES:
        ids = sorted((r for r in rows if r["archetype"] == a), key=lambda r: cik_hash(r["cik"]))
        n_test = round(len(ids) * test_share)
        out["test"] += [r["id"] for r in ids[:n_test]]
        out["train"] += [r["id"] for r in ids[n_test:]]
    return {k: sorted(v) for k, v in out.items()}


def mix_report(rows: list[dict], decisions: list[dict], parts: dict, cfg: dict) -> str:
    decisions = list({d["key"]: d for d in decisions}.values())      # a re-examined exhibit counts once
    sig = {d["row"]["id"]: d["signals"] for d in decisions if d["status"] == "picked"}
    reasons: dict[str, int] = {}
    for d in decisions:
        if d["status"] != "picked":
            key = d["status"] if d["status"] == "quota full" else re.sub(r"[\d,]+", "N", d["reason"])[:70]
            reasons[key] = reasons.get(key, 0) + 1
    test = set(parts["test"])
    L = ["# Collection mix for review", "",
         f"{len(rows)} agreements picked, one per issuer, from {len(decisions)} exhibits examined. "
         "The archetype comes from defined terms (`src/discover.py`, `classify`) and needs your "
         "review: unsecured is investment grade, and secured with two or more leveraged-loan "
         "terms is leveraged.", "",
         "| archetype | quota | picked | train | test |", "|---|---|---|---|---|"]
    for a in ARCHETYPES:
        ids = [r["id"] for r in rows if r["archetype"] == a]
        L.append(f"| {a} | {cfg['quotas'][a]} | {len(ids)} | {sum(i not in test for i in ids)} | "
                 f"{sum(i in test for i in ids)} |")
    L += ["", "## Exhibits not picked", "", "| reason | exhibits |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]
    L += ["", "## Picks", "",
          "| id | issuer | agreement date | characters | archetype | split | secured terms | "
          "leveraged terms | rating terms |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["archetype"], r["id"])):
        s = sig.get(r["id"], {})
        L.append(f"| [{r['id']}]({r['url']}) | {r['issuer'][:40]} | {r['agreement_date']} | "
                 f"{int(r['char_count']):,} | {r['archetype']} | "
                 f"{'test' if r['id'] in test else 'train'} | {len(s.get('secured', []))} | "
                 f"{', '.join(s.get('leveraged', []))} | {len(s.get('rating', []))} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from .platform import get_secret
    ap = argparse.ArgumentParser(description="Find and download about 150 credit agreements.")
    ap.add_argument("--out", type=Path, default=Path("work/collect_full"))
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "reports" / "collect_full_mix.md")
    a = ap.parse_args(argv)
    cfg = load_cfg()
    a.out.mkdir(parents=True, exist_ok=True)
    client = EdgarClient(get_secret("SEC_USER_AGENT"), min_interval=cfg["min_interval_s"])
    hits_path = a.out / "hits.json"
    if hits_path.exists():
        hits = json.loads(hits_path.read_text("utf-8"))
    else:
        hits = search(client, cfg)
        hits_path.write_text(json.dumps(hits, ensure_ascii=False), encoding="utf-8")
    m1 = {int(r["cik"]) for r in read_manifest(REPO_ROOT / "reports" / "m1_manifest.csv").values()}
    print(f"{len(hits)} search hits, {sum(keep_hit(h, cfg, m1) is None for h in hits)} kept "
          f"before download, {client.n_requests} requests so far", flush=True)
    rows = pick(client, cfg, hits, a.out, m1)
    parts = split(rows, cfg["test_share"])
    (a.out / "split.json").write_text(json.dumps(parts, indent=1), encoding="utf-8")
    a.report.write_text(mix_report(rows, load_decisions(a.out), parts, cfg), encoding="utf-8")
    print(f"{len(rows)} picked, {client.n_requests} SEC requests; mix in {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
