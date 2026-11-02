"""The teacher trial with a fake API: quote checks, the rejection flow, the spend cap and the report."""
import csv
import json

import httpx
import pytest

from src import teacher
from src.chunking import Chunk
from src.prompts import PROMPTS, canon_map, credit_clauses

CLAUSES = credit_clauses()
CANON = canon_map(CLAUSES)
LIENS = "other Liens securing obligations in a total amount not to exceed $25,000,000;"
MATURITY = "“Maturity Date” means the fifth anniversary of the Closing Date."
TEXT = f"(k) {LIENS}\n\n12\n\n(l) Liens for taxes not yet overdue;\n\n{MATURITY}\n"


def cfg(**over):
    c = teacher.load_cfg()
    c.update(chunk_chars=200, overlap=40, workers=2, backoff_s=0, probe_chunks=2, **over)
    return c


def answer(d: dict, finish: str = "stop", tokens_in: int = 1000, tokens_out: int = 200) -> dict:
    return {"choices": [{"message": {"content": json.dumps(d, ensure_ascii=False)},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
                      "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": tokens_in}}


class FakeAPI:
    """Answers from a script, or from `default` once the script runs out; records each request."""

    def __init__(self, script=(), default=None):
        self.script, self.default, self.requests = list(script), default, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        item = self.script.pop(0) if self.script else self.default(body)
        if isinstance(item, int):
            return httpx.Response(item, json={"error": "fake"})
        if isinstance(item, str):
            return httpx.Response(200, json={"choices": [{"message": {"content": item},
                                                          "finish_reason": "stop"}],
                                             "usage": {"prompt_tokens": 1000, "completion_tokens": 50}})
        return httpx.Response(200, json=answer(item))

    def client(self):
        return httpx.Client(base_url="https://fake", transport=httpx.MockTransport(self))


def chunk(text=TEXT):
    return Chunk("doc", 0, 1, 0, len(text), text)


def test_teacher_prompt_adds_rules_and_leaves_the_student_prompt_alone():
    c = teacher.load_cfg()
    prefix = teacher.teacher_prefix(c["rules"])
    student = PROMPTS["credit_schema"][1]
    assert prefix.endswith("RULES:\n" + c["rules"].strip() + "\n\nCONTRACT EXCERPT:\n")
    assert prefix.startswith(student[:-len("CONTRACT EXCERPT:\n")])
    assert "RULES:" not in student


def test_check_keeps_only_quotes_in_the_chunk_under_a_clause_key():
    norm = " ".join(TEXT.split())
    status, passing, failed = teacher.check(json.dumps({
        "Liens General Basket": [LIENS.replace(" ", "\n ", 2)],          # whitespace may differ
        "Maturity Date": [MATURITY.replace("“", '"')],                    # quote marks may not
        "Pricing": ["Liens for taxes not yet overdue;"],                  # not a clause type
        "Applicable Margin": ["Term SOFR plus 2.00%"],                     # not in the chunk
    }), norm, CANON)
    assert status == "ok"
    assert passing == {"Liens General Basket": [LIENS.replace(" ", "\n ", 2)]}
    assert [c for c, _ in failed] == ["Maturity Date", "?Pricing", "Applicable Margin"]
    assert teacher.check("not json", norm, CANON)[0] == "invalid"


def test_a_failed_quote_is_asked_again_and_passing_quotes_are_pooled():
    api = FakeAPI([{"Liens General Basket": [LIENS], "Maturity Date": ["Maturity Date means soon"]},
                   {"Maturity Date": [MATURITY]}])
    with api.client() as client:
        rec = teacher.label_chunk(client, cfg(), chunk(), teacher.Budget(1.0), "P:\n", CANON)
    assert [r["temperature"] for r in api.requests] == [0.0, 0.3]
    assert api.requests[0]["thinking"] == {"type": "disabled"}
    assert rec["kept"] == {"Liens General Basket": [LIENS], "Maturity Date": [MATURITY]}
    assert not rec["left_out"] and len(rec["attempts"]) == 2


def test_a_clause_that_never_passes_leaves_the_chunk_out():
    bad = {"Maturity Date": ["Maturity Date means soon"], "Liens General Basket": [LIENS]}
    api = FakeAPI(default=lambda body: bad)
    with api.client() as client:
        rec = teacher.label_chunk(client, cfg(), chunk(), teacher.Budget(1.0), "P:\n", CANON)
    assert len(api.requests) == 3
    assert rec["left_out"] and rec["reason"] == "failed quotes and no passing one: Maturity Date"
    assert rec["kept"] == {"Liens General Basket": [LIENS]}


def test_no_parsed_answer_leaves_the_chunk_out():
    api = FakeAPI(["nope", "still nope", "{broken"])
    with api.client() as client:
        rec = teacher.label_chunk(client, cfg(), chunk(), teacher.Budget(1.0), "P:\n", CANON)
    assert rec["left_out"] and rec["reason"] == "no answer parsed"


def test_rate_limits_are_retried_and_bad_requests_are_not():
    api = FakeAPI([429, 503, {"Liens General Basket": [LIENS]}])
    with api.client() as client:
        rec = teacher.label_chunk(client, cfg(), chunk(), teacher.Budget(1.0), "P:\n", CANON)
    assert len(api.requests) == 3 and rec["kept"] == {"Liens General Basket": [LIENS]}
    api = FakeAPI([400])
    budget = teacher.Budget(1.0)
    with api.client() as client, pytest.raises(httpx.HTTPStatusError):
        teacher.label_chunk(client, cfg(), chunk(), budget, "P:\n", CANON)
    assert budget.held == 0 and budget.spent == 0


def test_the_budget_counts_the_worst_case_of_requests_in_flight():
    c = cfg()
    worst = teacher.worst_case([{"content": "x" * 3000}], c)
    assert worst == pytest.approx((1000 * 0.3 + 4096 * 1.2) / 1e6)
    budget = teacher.Budget(cap=2.5 * worst)
    assert budget.hold(worst) and budget.hold(worst) and not budget.hold(worst)
    budget.settle(worst, 0.001)
    assert budget.spent == 0.001 and budget.held == pytest.approx(worst)
    api = FakeAPI(default=lambda body: {})
    with api.client() as client:
        assert teacher.label_chunk(client, c, chunk(), teacher.Budget(0.0), "P:\n", CANON) is None
    assert api.requests == []


def test_cost_uses_cache_hits_and_the_peak_price():
    price = teacher.load_cfg()["price_per_million"]
    usage = {"prompt_tokens": 10_000, "prompt_cache_hit_tokens": 4_000,
             "prompt_cache_miss_tokens": 6_000, "completion_tokens": 2_000}
    assert teacher.cost_usd(usage, price) == pytest.approx((4000 * 0.006 + 6000 * 0.3 + 2000 * 1.2) / 1e6)
    assert teacher.cost_usd({"prompt_tokens": 1000}, price) == pytest.approx(1000 * 0.3 / 1e6)


DOCS = {"alpha": ("Clause text that is filler. " * 30) + TEXT, "beta": TEXT + ("More filler here. " * 30)}


def good(body):
    text = body["messages"][1]["content"].split("CONTRACT EXCERPT:\n", 1)[1]
    return {c: [q] for c, q in (("Liens General Basket", LIENS), ("Maturity Date", MATURITY)) if q in text}


def test_run_probes_first_and_stops_if_the_trial_would_pass_the_cap(tmp_path):
    api = FakeAPI(default=good)
    c = cfg(max_usd=0.004, max_tokens=500)   # a chunk costs 0.00054, and there are 12 chunks
    with pytest.raises(SystemExit, match="over the approved"):
        teacher.run(c, DOCS, tmp_path, transport=httpx.MockTransport(api), api_key="k")
    assert len(teacher.load_records(tmp_path)) == 2 == len(api.requests)
    with pytest.raises(SystemExit, match="no chunk could be labeled"):
        teacher.run(cfg(max_usd=0.0), DOCS, tmp_path / "none", transport=httpx.MockTransport(api),
                    api_key="k")
    assert len(api.requests) == 2


def test_run_labels_everything_once_and_a_rerun_spends_nothing(tmp_path):
    api = FakeAPI(default=good)
    c = cfg(max_usd=1.0)
    n = len(teacher.chunks_for(DOCS, c))
    out = teacher.run(c, DOCS, tmp_path, transport=httpx.MockTransport(api), api_key="k")
    assert out["chunks"] == n and out["not_labeled"] == [] and len(api.requests) == n
    assert out["spent_usd"] == pytest.approx(n * (1000 * 0.3 + 200 * 1.2) / 1e6)
    again = teacher.run(c, DOCS, tmp_path, transport=httpx.MockTransport(api), api_key="k")
    assert len(api.requests) == n and again["spent_usd"] == pytest.approx(out["spent_usd"])
    with pytest.raises(SystemExit, match="LLM_API_KEY"):
        teacher.run(c, DOCS, tmp_path, transport=httpx.MockTransport(api))


def test_full_run_report_counts_labels_per_split(tmp_path):
    api = FakeAPI(default=good)
    c = cfg(max_usd=1.0)
    teacher.run(c, DOCS, tmp_path, transport=httpx.MockTransport(api), api_key="k")
    recs = teacher.load_records(tmp_path)
    md = teacher.prelabel_report(c, recs, {"train": ["alpha"], "test": ["beta"]},
                                 len(teacher.chunks_for(DOCS, c)))
    n_alpha = sum(r["doc_id"] == "alpha" for r in recs)
    assert md.startswith("# Teacher pre-labeling") and "(1 train, 1 test)" in md
    assert f"| chunks | {n_alpha} | {len(recs) - n_alpha} |" in md
    assert "| Maturity Date | 1 of 1 | 1 of 1 |" in md and "| MFN Protection | 0 of 1 | 0 of 1 |" in md


def test_report_scores_the_pooled_quotes_against_the_gold_set(tmp_path):
    api = FakeAPI(default=good)
    c = cfg(max_usd=1.0)
    teacher.run(c, DOCS, tmp_path, transport=httpx.MockTransport(api), api_key="k")
    recs = teacher.load_records(tmp_path)
    preds = teacher.preds_from_records(recs)
    assert preds["alpha"] == {"Liens General Basket": [LIENS], "Maturity Date": [MATURITY]}
    gold = {"labeled_by": "test", "documents": {d: {
        "Liens General Basket": [{"text": LIENS, "main": True}],
        "Maturity Date": [{"text": MATURITY, "main": True}]} for d in DOCS}}
    (tmp_path / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    with open(tmp_path / "sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "clause", "condition", "found", "quote"])
        w.writeheader()
        w.writerow({"doc_id": "alpha", "clause": "Liens General Basket", "condition": "c",
                    "found": "True", "quote": "Liens for taxes not yet overdue;"})
    md, passed = teacher.write_report(c, recs, DOCS, tmp_path / "gold.json", tmp_path / "sheet.csv",
                                      len(teacher.chunks_for(DOCS, c)))
    assert passed
    assert "**Gate: pass.** Found F1 1.000" in md and "| Teacher | " in md
    assert f"| {len(recs)} of {len(recs)} | {len(recs)} |" in md
