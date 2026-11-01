import json

import pytest

from src import gold as G
from src import score as S
from src.metrics import guide_counts, match_key, paired_diff, right_quote

CLAUSES = ["Basket", "Margin", "Maturity"]
TEXT = (
    "Section 1.1 Baskets.\n\n"
    "(a) other Indebtedness in an aggregate principal amount not to exceed the greater of\n"
    "\n12\n\n"
    "(x) $50,000,000 and (y) 25.0% of “Consolidated EBITDA” at any time outstanding;\n\n"
    "(b) tax distributions to Parent;\n\n"
    "“Applicable Margin” means 2.50% per annum for each Term Loan until the first date.\n\n"
    "LEGAL_US_E # 189704708.7\n\n"
    "“Maturity Date” means the fifth anniversary of the Closing Date.\n"
)
BASKET = ("other Indebtedness in an aggregate principal amount not to exceed the greater of "
          "(x) $50,000,000 and (y) 25.0% of \"Consolidated EBITDA\" at any time outstanding;")


def row(clause, correct, note="", quote="", found="True", qn="1", doc="d1"):
    return {"doc_id": doc, "clause": clause, "correct": correct, "note": note, "quote": quote,
            "found": found, "quote_no": qn, "condition": "c"}


@pytest.fixture
def src():
    return G.Source(TEXT)


def test_clean_source_removes_page_lines_only():
    out = G.clean_source("a\n12\nb 12\nLEGAL_US_E # 1.7\nc\n$12\n")
    assert out == "a\nb 12\nc\n$12\n"


def test_parse_note_splits_tags_and_main_spans():
    tags, spans = G.parse_note("adjacent; belongs: Margin; span: [main] one; two || three")
    assert tags == ["adjacent", "belongs: Margin"]
    assert spans == [("one; two", True), ("three", False)]
    assert G.parse_note("main") == (["main"], [])


def test_span_crosses_a_page_break_and_uses_opening_and_closing_words(src):
    assert src.span("other Indebtedness in an aggregate ... at any time outstanding;") == (
        BASKET.replace('"Consolidated EBITDA"', "“Consolidated EBITDA”"))
    assert src.span("no such words ... outstanding;") is None
    assert src.span("other Indebtedness ... no such ending") is None


def test_place_allows_small_changes_and_refuses_invented_text(src):
    changed = "With respect to baskets, " + BASKET
    placed = src.place(changed)
    assert placed.startswith("other Indebtedness") and placed.endswith("outstanding;")
    assert src.place("The Borrower may pay any amount from the Available Amount, which shall "
                     "be calculated as the sum of retained cash flow and equity") is None


def test_build_gold_follows_the_guide(src):
    rows = [
        row("Basket", "partial", "span: [main] other Indebtedness ... outstanding;",
            quote="other Indebtedness"),
        row("Basket", "no", "adjacent", quote="tax distributions to Parent;", qn="2"),
        row("Margin", "yes", "main", quote="“Applicable Margin” means 2.50% per annum for each "
                                           "Term Loan until the first date."),
        row("Margin", "no", "belongs: Maturity; span: extra words missing", qn="2",
            quote="“Maturity Date” means the fifth anniversary of the Closing Date."),
        row("Maturity", "missed", "span: [main] “Maturity Date” means", found="False", quote=""),
        row("Basket", "", quote="unlabeled rows are skipped", qn="3"),
    ]
    gold, problems = G.build_gold(rows, {"d1": src}, CLAUSES)
    assert problems == ["d1 | Margin | #2: span not found: extra words missing"]
    assert [p["main"] for p in gold["d1"]["Basket"]] == [True]
    assert gold["d1"]["Margin"][0]["origin"] == "quote"
    assert {p["origin"] for p in gold["d1"]["Maturity"]} == {"belongs", "span"}


def test_build_gold_reports_missing_main_and_missing_span(src):
    rows = [row("Margin", "yes", "", quote="“Applicable Margin” means 2.50% per annum for each "
                                           "Term Loan until the first date."),
            row("Maturity", "partial", "", quote="anything", qn="2")]
    _, problems = G.build_gold(rows, {"d1": src}, CLAUSES)
    assert problems == ["d1 | Maturity | #2: partial needs a span",
                        "d1 | Margin: no main passage"]


def test_right_quote_holds_a_passage_or_overlaps_enough():
    g = [match_key("the greater of $50 and 25% of EBITDA at any time")]
    assert right_quote(match_key("Intro. the greater of $50 and 25% of EBITDA at any time. More"), g)
    assert right_quote(match_key("the greater of $50 and 25% of EBITDA"), g)
    assert not right_quote(match_key("tax distributions to Parent"), g)


def gold_one(text="the basket is $50 at any time", main=True):
    return {"d1": {"Basket": [{"text": text, "main": main}], "Margin": []}}


@pytest.mark.parametrize("quotes,want", [
    (["the basket is $50 at any time"], dict(found_tp=1, main_tp=1)),
    (["tax distributions to Parent"], dict(found_fp=1, found_fn=1, main_fn=1)),
    ([], dict(found_fn=1, main_fn=1)),
])
def test_guide_counts_wrong_only_is_a_false_find_and_a_miss(quotes, want):
    got = guide_counts(gold_one(), {"d1": {"Basket": quotes}}, ["Basket", "Margin"])["d1"]
    assert {k: v for k, v in got.items() if v} == want


def test_guide_counts_absent_clause_with_a_quote_is_a_false_find():
    got = guide_counts(gold_one(), {"d1": {"Basket": ["the basket is $50 at any time"],
                                           "Margin": ["2.50%"]}}, ["Basket", "Margin"])["d1"]
    assert got["found_fp"] == got["main_fp"] == 1


def test_strict_needs_main_passages_only():
    g = {"d1": {"Basket": [{"text": "main term", "main": True},
                           {"text": "other term entirely", "main": False}]}}
    got = guide_counts(g, {"d1": {"Basket": ["the main term here"]}}, ["Basket"])["d1"]
    assert got["main_tp"] == 1


def test_paired_diff_is_zero_for_the_same_model():
    c = {f"d{i}": dict(found_tp=i, found_fp=1, found_fn=0) for i in range(5)}
    assert paired_diff(c, c, "found_", n=50) == (0.0, 0.0, 0.0)


def test_score_and_report_end_to_end(tmp_path):
    gold = {"d1": {"Basket": [{"text": BASKET, "main": True}], "Margin": [], "Maturity": []}}
    good = {"d1": {"Basket": [BASKET.replace('"', "“", 1)]}}
    results = {"A": S.score(gold, good, CLAUSES, {"d1": TEXT}, n_boot=20),
               "B": S.score(gold, {}, CLAUSES, {"d1": TEXT}, n_boot=20)}
    assert results["A"]["found_f1"] == results["A"]["strict_f1"] == 1.0
    assert results["B"]["found_f1"] == 0.0
    assert results["A"]["verbatim"]["rate"] == 0.0        # a page number sits inside the quote
    assert results["A"]["verbatim"]["rate_no_page_lines"] == 1.0
    assert results["B"]["verbatim"]["quotes"] == 0
    md = S.report(gold, results, CLAUSES, "a reviewer", n_boot=20)
    assert "Paired difference from A" in md and "| Basket | 1 |" in md


def test_gold_cli_stops_on_problems_and_writes_when_clean(tmp_path, capsys):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "d1.txt").write_text(TEXT, encoding="utf-8")
    sheet = tmp_path / "sheet.csv"
    fields = "doc_id,clause,condition,found,quote_no,quote,correct,note\n"
    good = 'd1,Maturity Date,c,False,,,missed,span: [main] “Maturity Date” means\n'
    bad = "d1,Maturity Date,c,False,,,missed,\n"
    out = tmp_path / "gold.json"
    args = ["--sheet", str(sheet), "--text-dir", str(text_dir), "--out", str(out),
            "--labeled-by", "a reviewer"]
    sheet.write_text(fields + bad, encoding="utf-8")
    assert G.main(args) == 1 and not out.exists()
    sheet.write_text(fields + good, encoding="utf-8")
    assert G.main(args) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["labeled_by"] == "a reviewer"
    assert S.load_gold(out)["d1"]["Maturity Date"][0]["main"] is True
