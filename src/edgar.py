"""Milestone 1: download credit agreements from SEC EDGAR, clean them to text, write a manifest.

SEC fair-access rules: every request carries a User-Agent naming the requester with a contact
email (secret SEC_USER_AGENT, e.g. "Jane Doe jane@example.com"), and stays under 10 requests
per second. We wait at least `min_interval` seconds between requests (default 0.2 s, so at
most 5/s) and back off on 429 / 5xx.

The picks are explicit (configs/collect.yaml: CIK, accession, exhibit file name). Filing date,
form and company name come from EDGAR's submissions API, not from the config, so the manifest
records what EDGAR says. Runs are resumable: a document already on disk whose sha256 matches
the manifest is not fetched again.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import yaml

from .prompts import REPO_ROOT

ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
SUBMISSIONS = "https://data.sec.gov/submissions"
MANIFEST_FIELDS = ["id", "issuer", "cik", "form", "accession", "document", "url",
                   "filing_date", "agreement_date", "title", "archetype", "sha256",
                   "raw_bytes", "char_count", "fetched_at"]


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or REPO_ROOT / "configs" / "collect.yaml").read_text())


# ----------------------------------------------------------------------------- client

class EdgarClient:
    """Rate-limited GET with the SEC User-Agent; retries 429/5xx with exponential backoff."""

    def __init__(self, user_agent: str, min_interval: float = 0.2, retries: int = 5,
                 opener=None, sleep=time.sleep, clock=time.monotonic):
        if not user_agent or "@" not in user_agent:
            raise ValueError("SEC_USER_AGENT must name you and include a contact email, "
                             "e.g. 'Jane Doe jane@example.com'")
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.retries = retries
        self._open = opener or urllib.request.urlopen
        self._sleep, self._clock = sleep, clock
        self._last = None
        self.n_requests = 0

    def _wait(self) -> None:
        if self._last is not None:
            gap = self.min_interval - (self._clock() - self._last)
            if gap > 0:
                self._sleep(gap)
        self._last = self._clock()

    def get(self, url: str) -> bytes:
        for attempt in range(self.retries + 1):
            self._wait()
            self.n_requests += 1
            req = urllib.request.Request(url, headers={
                "User-Agent": self.user_agent, "Accept-Encoding": "gzip"})
            try:
                with self._open(req, timeout=60) as r:
                    body = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise
                self._sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError("unreachable")

    def get_json(self, url: str) -> dict:
        return json.loads(self.get(url))


def archive_url(cik: int | str, accession: str, document: str) -> str:
    return f"{ARCHIVES}/{int(cik)}/{accession.replace('-', '')}/{document}"


def filing_meta(client: EdgarClient, cik: int | str, accession: str) -> dict:
    """Company name, form, and filing date for one accession, from the submissions API."""
    base = client.get_json(f"{SUBMISSIONS}/CIK{int(cik):010d}.json")

    def pages():
        yield base["filings"]["recent"]
        for f in base["filings"].get("files", []):     # older filings, fetched only if needed
            yield client.get_json(f"{SUBMISSIONS}/{f['name']}")

    for p in pages():
        if accession in p["accessionNumber"]:
            i = p["accessionNumber"].index(accession)
            return {"issuer": base["name"], "form": p["form"][i],
                    "filing_date": p["filingDate"][i]}
    raise KeyError(f"accession {accession} not found for CIK {cik}")


# ------------------------------------------------------------------------ html -> text

BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table",
         "center", "blockquote", "pre", "hr", "title", "section", "article"}
CELL = {"td", "th"}
SKIP = {"script", "style", "head", "noscript", "ix:header"}
SGML = re.compile(r"<(TYPE|SEQUENCE|FILENAME|DESCRIPTION)>[^<\n]*", re.I)


VOID = {"br", "hr", "img", "meta", "link", "input", "col", "wbr"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[tuple[str, bool]] = []    # (tag, hidden) of open elements

    @property
    def hidden(self) -> bool:
        return any(h for _, h in self.stack)

    def handle_starttag(self, tag, attrs):
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        if tag not in VOID:
            self.stack.append((tag, tag in SKIP or "display:none" in style))
        if tag in BLOCK:
            self.out.append("\n")
        elif tag in CELL:
            self.out.append(" ")

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        # EDGAR HTML often leaves <p>/<font> unclosed: pop back to the matching open tag,
        # and ignore stray end tags, so one bad tag can't hide (or reveal) the rest
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.out.append(data)


def html_to_text(raw: str) -> str:
    """Readable plain text: blocks become lines, table cells are space-separated."""
    raw = SGML.sub("", raw)
    p = _Text()
    p.feed(raw)
    p.close()
    t = html.unescape("".join(p.out))
    t = t.replace("\xa0", " ").replace("​", "").replace("\r", "")
    lines = [re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in t.split("\n")]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip() + "\n"


DATE = re.compile(r"dated\s+as\s+of\s+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})", re.I)


def agreement_date(text: str, head_chars: int = 5000) -> str | None:
    """ISO date from the first 'dated as of <Month> <d>, <yyyy>' near the top, if any."""
    m = DATE.search(text[:head_chars])
    if not m:
        return None
    s = re.sub(r"\s+", " ", m.group(1).replace(".", "").replace(",", ""))
    for fmt in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def title_of(text: str) -> str:
    """First line that looks like the agreement's title (all caps, mentions AGREEMENT)."""
    for ln in text.split("\n")[:80]:
        if "AGREEMENT" in ln and ln.upper() == ln and len(ln) < 120:
            return ln
    return ""


# --------------------------------------------------------------------------- collect

def read_manifest(path: Path) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def write_manifest(path: Path, rows: list[dict]) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def collect(cfg: dict, out_dir: Path, client: EdgarClient) -> list[dict]:
    """Fetch every pick into out_dir/{raw,text}/ and (re)write out_dir/manifest.csv."""
    out_dir = Path(out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    old = read_manifest(out_dir / "manifest.csv")
    rows = []
    for pick in cfg["picks"]:
        pid = pick["id"]
        raw_p, txt_p = out_dir / "raw" / f"{pid}.htm", out_dir / "text" / f"{pid}.txt"
        prev = old.get(pid)
        if (prev and raw_p.exists() and txt_p.exists() and
                hashlib.sha256(raw_p.read_bytes()).hexdigest() == prev["sha256"]):
            print(f"{pid}: already collected, skipping")
            rows.append(prev)
            continue
        url = archive_url(pick["cik"], pick["accession"], pick["document"])
        meta = filing_meta(client, pick["cik"], pick["accession"])
        body = client.get(url)
        text = html_to_text(body.decode("utf-8", errors="replace"))
        raw_p.write_bytes(body)
        txt_p.write_text(text, encoding="utf-8")
        row = {"id": pid, "issuer": meta["issuer"], "cik": int(pick["cik"]),
               "form": meta["form"], "accession": pick["accession"],
               "document": pick["document"], "url": url,
               "filing_date": meta["filing_date"], "agreement_date": agreement_date(text) or "",
               "title": title_of(text), "archetype": pick.get("archetype", ""),
               "sha256": hashlib.sha256(body).hexdigest(), "raw_bytes": len(body),
               "char_count": len(text),
               "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        print(f"{pid}: {row['issuer']} {row['form']} filed {row['filing_date']}, "
              f"agreement {row['agreement_date'] or '?'}, {row['char_count']:,} chars")
        rows.append(row)
    write_manifest(out_dir / "manifest.csv", rows)
    return rows


def summary(rows: list[dict], chunk_chars: int = 32000, overlap: int = 4000) -> list[dict]:
    """Per document: size and how many chunks it makes at the M3 chunking."""
    from .chunking import chunk_bounds
    return [{"id": r["id"], "issuer": r["issuer"], "agreement_date": r["agreement_date"],
             "filing_date": r["filing_date"], "chars": int(r["char_count"]),
             f"chunks@{chunk_chars // 1000}k": len(chunk_bounds(int(r["char_count"]),
                                                               chunk_chars, overlap))}
            for r in rows]


def run(env, cfg: dict) -> list[dict]:
    """Notebook entry point: Hub pull (resume), collect, Hub push."""
    from .platform import get_secret, pull, push_folder
    root = env.paths.work / cfg["hub_prefix"]
    if env.data_repo:
        n = pull(env.data_repo, cfg["hub_prefix"], env.paths.work)
        print(f"pulled {n} existing files from {env.data_repo}/{cfg['hub_prefix']}")
    client = EdgarClient(get_secret("SEC_USER_AGENT", env.platform),
                         min_interval=cfg.get("min_interval_s", 0.2))
    rows = collect(cfg, root, client)
    print(f"{client.n_requests} SEC requests")
    if env.data_repo:
        push_folder(env.data_repo, root, cfg["hub_prefix"], "m1 collect")
        print(f"pushed {root} -> {env.data_repo}/{cfg['hub_prefix']}")
    return rows
