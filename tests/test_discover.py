"""Discovery for the data scale-up, with a fake EDGAR: search paging, filters, text checks,
archetypes, quotas, resume and the split."""
import json
import urllib.parse

from src import discover

CFG = discover.load_cfg()
FILLER = "The Borrower shall comply with this Agreement in all respects. " * 3500   # ~220k chars


def agreement(title="CREDIT AGREEMENT", date="March 3, 2024", terms=(), cover=""):
    defs = " ".join(f"“{t}” means a defined term." for t in terms)
    return (f"<html><body><p>{cover}</p><p>{title}</p><p>dated as of {date}</p>"
            f"<p>among the Borrower and the Lenders</p><p>{defs}</p><p>{FILLER}</p></body></html>")


def hit(cik, doc, file_type="EX-10.1", desc="EXHIBIT 10.1", date="2024-03-05", sics=("3670",)):
    return {"_id": f"0000{cik}-24-000001:{doc}",
            "_source": {"ciks": [f"{cik:010d}"], "display_names": [f"Issuer {cik} Inc (CIK {cik})"],
                        "form": "8-K", "file_type": file_type, "file_description": desc,
                        "file_date": date, "sics": list(sics)}}


class FakeEdgar:
    def __init__(self, hits=(), docs=None, sizes=None):
        self.hits, self.docs, self.sizes = list(hits), docs or {}, sizes or {}
        self.n_requests, self.urls = 0, []

    def get_json(self, url):
        self.n_requests += 1
        self.urls.append(url)
        if url.startswith(discover.SEARCH):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            start = int(q.get("from", ["0"])[0])
            page = self.hits[start:start + discover.PAGE] if q["forms"] == ["8-K"] else []
            total = len(self.hits) if q["forms"] == ["8-K"] else 0
            return {"hits": {"total": {"value": total}, "hits": page}}
        return {"directory": {"item": [{"name": doc, "size": self.sizes.get(doc, 600_000)}
                                       for doc in self.docs]}}

    def get(self, url):
        self.n_requests += 1
        self.urls.append(url)
        return self.docs[url.split("/")[-1]].encode()


def test_search_pages_through_every_hit_once():
    hits = [hit(1000 + i, f"ex{i}.htm") for i in range(230)]
    client = FakeEdgar(hits)
    cfg = dict(CFG, queries=[{"q": '"credit agreement"', "forms": ["8-K", "10-Q"]},
                             {"q": '"term sofr"', "forms": ["8-K"]}])
    rows = discover.search(client, cfg)
    assert len(rows) == 230 and rows[0]["cik"] == 1000 and rows[0]["document"] == "ex0.htm"
    froms = [urllib.parse.parse_qs(urllib.parse.urlparse(u).query).get("from") for u in client.urls]
    assert froms[:3] == [None, ["100"], ["200"]]
    assert '%22credit%20agreement%22' in client.urls[0] and "category=custom" in client.urls[0]


def test_hits_that_are_not_full_agreements_are_dropped_before_download():
    rows = {d: discover.search(FakeEdgar([h]), dict(CFG, queries=[{"q": "x", "forms": ["8-K"]}]))[0]
            for d, h in {
                "ok": hit(1, "creditagreement.htm"),
                "ar": hit(2, "amendedandrestatedcreditagreement.htm"),
                "amend": hit(3, "amendmentno2.htm"),
                "desc": hit(4, "ex101.htm", desc="FIRST AMENDMENT TO CREDIT AGREEMENT"),
                "type": hit(5, "d8k.htm", file_type="8-K"),
                "fund": hit(6, "ex101.htm", sics=("6726",)),
                "reit": hit(7, "ex101.htm", sics=("6798",)),
                "m1": hit(8, "ex101.htm"),
                "ex4": hit(9, "ex41.htm", file_type="EX-4.1"),
            }.items()}
    rows["named"] = dict(rows["ok"], cik=10, issuer="Twin Brook Capital Income Fund (CIK 10)")
    kept = {d for d, h in rows.items() if discover.keep_hit(h, CFG, {8}) is None}
    assert kept == {"ok", "ar", "reit", "ex4"}


def test_text_checks():
    from src.edgar import html_to_text
    ok = html_to_text(agreement())
    assert discover.check_text(ok, CFG) is None
    assert "amendment" in discover.check_text(html_to_text(agreement(
        title="AMENDMENT NO. 3 TO CREDIT AGREEMENT")), CFG)
    assert "before 2023" in discover.check_text(html_to_text(agreement(date="May 1, 2021")), CFG)
    assert "title" in discover.check_text(html_to_text(agreement(title="INDENTURE AGREEMENT")), CFG)
    assert "characters" in discover.check_text(ok[:150_000], CFG)
    restated = html_to_text(agreement(title="AMENDED AND RESTATED CREDIT AGREEMENT"))
    assert discover.check_text(restated, CFG) is None
    assert discover.check_text(html_to_text(agreement(terms=("Net Asset Value",))), CFG) == \
        "a fund or BDC facility"
    bdc = html_to_text(agreement(cover="The Borrower is a business development company."))
    assert discover.check_text(bdc, CFG) == "a fund or BDC facility"
    assert discover.check_text(ok.replace("Agreement", "Agreement. Investment Company Act", 1),
                               CFG) is None


def test_archetype_from_defined_terms():
    assert discover.classify("“Applicable Rate” means")[0] == "investment_grade"
    assert discover.classify("“Security Agreement” means. “Available Amount” means")[0] == \
        "secured_mid_cap"
    archetype, signals = discover.classify(
        "“Collateral Documents” means. “Available Amount” means. “Excess Cash Flow” means")
    assert archetype == "leveraged" and signals["leveraged"] == ["Available Amount", "Excess Cash Flow"]
    assert discover.defined('"Ratings" means', discover.RATING) == ["Ratings"]


SECURED_LEV = ("Collateral Documents", "Available Amount", "Excess Cash Flow")


def fake_market():
    """Six issuers: two of each archetype, one amendment first for issuer 13, one fund."""
    plan = {11: (), 12: (), 13: ("Security Agreement",), 14: ("Security Agreement",),
            15: SECURED_LEV, 16: SECURED_LEV}
    hits, docs = [], {}
    for cik, terms in plan.items():
        hits.append(hit(cik, f"ca{cik}.htm"))
        docs[f"ca{cik}.htm"] = agreement(terms=terms)
    hits.append(hit(13, "ex13new.htm", date="2025-01-01"))
    docs["ex13new.htm"] = agreement(title="FIRST AMENDMENT TO CREDIT AGREEMENT")
    hits.append(hit(17, "fund.htm", sics=("6726",)))
    return [discover.search(FakeEdgar([h]), dict(CFG, queries=[{"q": "x", "forms": ["8-K"]}]))[0]
            for h in hits], docs


def test_pick_fills_quotas_once_per_issuer_and_resumes(tmp_path):
    hits, docs = fake_market()
    cfg = dict(CFG, quotas={"investment_grade": 1, "secured_mid_cap": 2, "leveraged": 2})
    client = FakeEdgar(docs=docs)
    rows = discover.pick(client, cfg, hits, tmp_path, skip_ciks=set())
    by = {r["cik"]: r["archetype"] for r in rows}
    assert sorted(by.values()).count("investment_grade") == 1
    assert {by.get(13), by.get(14)} == {"secured_mid_cap"} and by.get(15) == by.get(16) == "leveraged"
    decisions = discover.load_decisions(tmp_path)
    assert any(d["status"] == "quota full" for d in decisions)
    assert [d["reason"] for d in decisions if d["key"].endswith("ex13new.htm")] == ["an amendment"]
    assert not any("fund" in d["key"] for d in decisions)
    assert (tmp_path / "text" / f"{rows[0]['id']}.txt").exists()
    n = client.n_requests
    again = discover.pick(client, cfg, hits, tmp_path, skip_ciks=set())
    assert client.n_requests == n and len(again) == len(rows)


def test_a_newer_filter_drops_a_pick_and_its_slot_is_refilled(tmp_path):
    hits, docs = fake_market()
    cfg = dict(CFG, quotas={"investment_grade": 1, "secured_mid_cap": 2, "leveraged": 2})
    rows = discover.pick(FakeEdgar(docs=docs), cfg, hits, tmp_path, skip_ciks=set())
    ig = next(r for r in rows if r["archetype"] == "investment_grade")
    cfg2 = dict(cfg, exclude_issuer=rf"Issuer {ig['cik']} Inc")
    again = discover.pick(FakeEdgar(docs=docs), cfg2, hits, tmp_path, skip_ciks=set())
    ig2 = [r for r in again if r["archetype"] == "investment_grade"]
    assert len(again) == len(rows) and [r["cik"] for r in ig2] == [({11, 12} - {ig["cik"]}).pop()]
    assert not (tmp_path / "text" / f"{ig['id']}.txt").exists()
    md = discover.mix_report(again, discover.load_decisions(tmp_path), discover.split(again, 0.5), cfg2)
    assert "| quota full |" not in md


def test_split_keeps_the_mix_and_is_stable():
    rows = [{"id": f"x-{i}", "cik": i, "archetype": a}
            for i, a in enumerate(["investment_grade"] * 10 + ["leveraged"] * 20)]
    parts = discover.split(rows, 0.2)
    test = [r for r in rows if r["id"] in parts["test"]]
    assert len(test) == 6 and sum(r["archetype"] == "leveraged" for r in test) == 4
    assert discover.split(list(reversed(rows)), 0.2) == parts
    assert set(parts["train"]) | set(parts["test"]) == {r["id"] for r in rows}


def test_mix_report_lists_counts_and_reasons(tmp_path):
    hits, docs = fake_market()
    cfg = dict(CFG, quotas={"investment_grade": 1, "secured_mid_cap": 2, "leveraged": 2})
    rows = discover.pick(FakeEdgar(docs=docs), cfg, hits, tmp_path, skip_ciks=set())
    md = discover.mix_report(rows, discover.load_decisions(tmp_path), discover.split(rows, 0.5), cfg)
    assert "| leveraged | 2 | 2 | 1 | 1 |" in md and "| an amendment | 1 |" in md
    assert json.dumps(rows[0]["url"])[1:-1] in md
