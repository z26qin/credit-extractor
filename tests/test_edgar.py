"""EDGAR collection without network: fake opener / fake client."""
import hashlib
import io
import json
import urllib.error

import pytest

from src.edgar import (EdgarClient, agreement_date, archive_url, collect, filing_meta,
                       html_to_text, load_cfg, read_manifest, title_of)

DOC = """<DOCUMENT>
<TYPE>EX-10.1
<SEQUENCE>2
<FILENAME>d1dex101.htm
<DESCRIPTION>EX-10.1
<TEXT>
<html><head><title>EX-10.1</title><style>p {color: red}</style></head><body>
<div style="display: none"><ix:header>hidden xbrl</ix:header></div>
<p style="text-align:center"><b>CREDIT AGREEMENT</b></p>
<p>dated as of July&nbsp;16,  2024</p>
<p>among ACME CORP.,&#160;as the Borrower, and BANK OF AMERICA, N.A., as Administrative Agent</p>
<table><tr><td>Section 7.11</td><td>Financial&nbsp;Covenants</td></tr></table>
<p>&ldquo;Maturity Date&rdquo; means July 16, 2029.<br>Next line</p>
<script>var x = 1;</script>
</body></html>
</TEXT>
</DOCUMENT>"""


def test_html_to_text():
    t = html_to_text(DOC)
    assert t.startswith("CREDIT AGREEMENT\n")
    assert "dated as of July 16, 2024" in t
    assert "Section 7.11 Financial Covenants" in t
    assert "“Maturity Date” means July 16, 2029.\nNext line" in t
    for junk in ("EX-10.1", "hidden xbrl", "var x", "color: red", "<", "\xa0"):
        assert junk not in t
    assert "\n\n\n" not in t


def test_unclosed_tags_do_not_hide_the_rest():
    raw = ('<div style="display:none"><p>hidden<font>x</div>'
           '<p>visible one<p>visible two</font></span><div>three</div>')
    t = html_to_text(raw)
    assert "hidden" not in t and "x" not in t
    assert t == "visible one\nvisible two\nthree\n"


# structure of real exhibits: Quanta's cover and TOC, Repay's pricing grid
COVER = """<P ALIGN="center">Dated as of
July&nbsp;16, 2024 </P> <P ALIGN="center">as the
Borrower, </P>"""
TOC = """<table><tr>
<td> <p><b>Section&nbsp;1.01</b></p></td>
<td>&nbsp;</td>
<td><b>Defined Terms</b></td>
<td>&nbsp;&nbsp;</td>
<td><b>1</b></td></tr></table>"""
GRID = """<table><tbody><tr>
<td><p>Level</p></td>
<td><p>First Lien Net Leverage Ratio</p></td>
<td><p>SOFR  Loans and CORRA Loans</p></td></tr>
<tr><td><p>I</p></td><td><p>&gt; 1.90:1.00</p></td><td><p>4.25%</p></td></tr>
<tr><td><p>II</p></td><td><p>&gt; 1.40:1.00 but \u2264 1.90:1.00</p></td><td><p>4.00%</p></td></tr>
</tbody></table>"""


def test_source_newlines_are_wrapping_not_line_breaks():
    assert html_to_text(COVER) == "Dated as of July 16, 2024\n\nas the Borrower,\n"
    assert html_to_text("<pre>a\nb</pre>") == "a\nb\n"


def test_table_rows_stay_on_one_line():
    assert html_to_text(TOC) == "Section 1.01 Defined Terms 1\n"
    assert html_to_text(GRID).split("\n")[:3] == [
        "Level First Lien Net Leverage Ratio SOFR Loans and CORRA Loans",
        "I > 1.90:1.00 4.25%",
        "II > 1.40:1.00 but \u2264 1.90:1.00 4.00%"]


def test_long_layout_cell_keeps_its_paragraphs():
    para = "word " * 80
    t = html_to_text(f"<table><tr><td><p>{para}</p><p>{para}</p></td></tr></table>")
    assert t.count("\n") >= 2 and "\x00" not in t


def test_title_survives_a_wrapped_source_line():
    assert title_of(html_to_text("<p>CREDIT\nAGREEMENT</p>")) == "CREDIT AGREEMENT"


def test_agreement_date_and_title():
    t = html_to_text(DOC)
    assert agreement_date(t) == "2024-07-16"
    assert agreement_date("CREDIT AGREEMENT dated as of Jan. 5 2026 among") == "2026-01-05"
    assert agreement_date("no date here") is None
    assert title_of(t) == "CREDIT AGREEMENT"


def test_archive_url():
    assert archive_url("0001050915", "0001193125-24-182240", "d.htm") == \
        "https://www.sec.gov/Archives/edgar/data/1050915/000119312524182240/d.htm"


class FakeResp(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_client_user_agent_rate_limit_and_retry():
    with pytest.raises(ValueError):
        EdgarClient("no email here")
    now, sleeps, seen = [0.0], [], []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    calls = iter([429, 200, 200])

    def opener(req, timeout):
        seen.append(req.get_header("User-agent"))
        code = next(calls)
        if code != 200:
            raise urllib.error.HTTPError(req.full_url, code, "slow down", {}, None)
        return FakeResp(b"ok")

    c = EdgarClient("Jane Doe jane@example.com", min_interval=0.2, opener=opener,
                    sleep=sleep, clock=lambda: now[0])
    assert c.get("https://www.sec.gov/x") == b"ok"     # 429 -> backoff -> ok
    assert c.get("https://www.sec.gov/y") == b"ok"
    assert seen == ["Jane Doe jane@example.com"] * 3
    assert sleeps[0] == 2                                # backoff after the 429
    assert c.n_requests == 3


def test_client_raises_on_404():
    def opener(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, None)
    c = EdgarClient("a b@c.d", opener=opener, sleep=lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        c.get("https://www.sec.gov/missing")


class FakeClient:
    """Serves one submissions JSON (with an older page) and the fixture document."""

    def __init__(self):
        self.urls = []

    def get_json(self, url):
        self.urls.append(url)
        if url.endswith("CIK0001050915.json"):
            return {"name": "ACME CORP", "filings": {
                "recent": {"accessionNumber": ["0000000000-26-000001"], "form": ["10-K"],
                           "filingDate": ["2026-02-01"]},
                "files": [{"name": "CIK0001050915-submissions-001.json"}]}}
        if url.endswith("submissions-001.json"):
            return {"accessionNumber": ["0001193125-24-182240"], "form": ["8-K"],
                    "filingDate": ["2024-07-22"]}
        raise AssertionError(url)

    def get(self, url):
        self.urls.append(url)
        return DOC.encode()


def test_filing_meta_reads_older_pages():
    c = FakeClient()
    assert filing_meta(c, 1050915, "0001193125-24-182240") == {
        "issuer": "ACME CORP", "form": "8-K", "filing_date": "2024-07-22"}
    with pytest.raises(KeyError):
        filing_meta(c, 1050915, "9999999999-99-999999")


def test_collect_writes_manifest_and_resumes(tmp_path):
    cfg = {"picks": [{"id": "acme-2024", "cik": 1050915, "accession": "0001193125-24-182240",
                      "document": "d1dex101.htm", "archetype": "test"}]}
    c = FakeClient()
    rows = collect(cfg, tmp_path, c)
    r = rows[0]
    assert r["issuer"] == "ACME CORP" and r["form"] == "8-K"
    assert r["filing_date"] == "2024-07-22" and r["agreement_date"] == "2024-07-16"
    assert r["sha256"] == hashlib.sha256(DOC.encode()).hexdigest()
    assert r["char_count"] == len((tmp_path / "text" / "acme-2024.txt").read_text())
    assert r["url"].endswith("/1050915/000119312524182240/d1dex101.htm")
    assert read_manifest(tmp_path / "manifest.csv")["acme-2024"]["title"] == "CREDIT AGREEMENT"
    n = len(c.urls)
    (tmp_path / "text" / "acme-2024.txt").write_text("stale")
    rows = collect(cfg, tmp_path, c)             # second run: nothing fetched, text re-cleaned
    assert len(c.urls) == n
    assert (tmp_path / "text" / "acme-2024.txt").read_text().startswith("CREDIT AGREEMENT")
    assert int(rows[0]["char_count"]) == r["char_count"]
    (tmp_path / "raw" / "acme-2024.htm").write_bytes(b"corrupted")
    collect(cfg, tmp_path, c)                    # hash mismatch: fetched again
    assert len(c.urls) > n


def test_quote_counts(tmp_path):
    from src.edgar import quote_counts
    (tmp_path / "a.txt").write_text('\u201cLender\u201d and "Agent"', encoding="utf-8")
    assert quote_counts(tmp_path) == [{"id": "a", "chars": 20, "straight": 2, "curly": 2}]


def test_collect_config_is_valid():
    cfg = load_cfg()
    ids = [p["id"] for p in cfg["picks"]]
    assert len(ids) == 5 == len(set(ids))
    for p in cfg["picks"]:
        assert {"cik", "accession", "document", "archetype"} <= set(p)
        assert len(p["accession"]) == 20 and p["accession"].count("-") == 2


def test_summary_counts_chunks():
    from src.edgar import summary
    rows = [{"id": "a", "issuer": "A", "agreement_date": "", "filing_date": "",
             "char_count": "60000"}]
    assert summary(rows)[0]["chunks@32k"] == 2      # starts 0, 28000 (< 56000)
