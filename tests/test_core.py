import json

import pytest

from src.chunking import chunk_bounds, chunk_text, safe_id
from src.merge import merge_chunks
from src.metrics import detection_report, tok_f1, verbatim_flags, verbatim_rate
from src.parsing import canonicalize, classify, detect_repetition, parse_json_object
from src.prompts import (CUAD_SYSTEM, CUAD_USER_PREFIX, canon_map, load_cuad_clauses,
                         messages, norm_key)


# ---------------------------------------------------------------- prompts

def test_cuad_prompt_is_byte_identical_to_model_card():
    assert CUAD_SYSTEM == (
        "You are a contract review system. Extract the requested clause types from the "
        "contract excerpt. Quote the contract text verbatim. A clause type not present in "
        "this excerpt gets an empty list. Reply with JSON only.")
    assert CUAD_USER_PREFIX == (
        "Extract all 41 CUAD clause types from this contract excerpt as JSON (keys: clause "
        "types, values: lists of verbatim quotes).\n\nCONTRACT EXCERPT:\n")
    m = messages(CUAD_SYSTEM, CUAD_USER_PREFIX, "TEXT")
    assert m[1]["content"].endswith("CONTRACT EXCERPT:\nTEXT")


def test_cuad_clauses_and_canon():
    cl = load_cuad_clauses()
    assert len(cl) == 41 and cl[0] == "Document Name"
    canon = canon_map(cl)
    assert canon[norm_key("document_name")] == "Document Name"
    # CUAD qas ids title-case names ("Ip Ownership Assignment"); must map to CSV casing
    assert canon[norm_key("Ip Ownership Assignment")] == "IP Ownership Assignment"
    assert canon[norm_key("Rofr/Rofo/Rofn")] == "Rofr/Rofo/Rofn"


# ---------------------------------------------------------------- chunking

def test_chunk_bounds_match_upstream_rule():
    # upstream: starts = range(0, max(1, n - overlap), size - overlap)
    for n in (0, 500, 60000, 62272, 249551):
        exp = list(range(0, max(1, n - 8000), 52000))
        assert [s for s, _ in chunk_bounds(n, 60000, 8000)] == exp
    assert chunk_bounds(62272, 60000, 8000) == [(0, 60000), (52000, 62272)]


def test_short_span_always_inside_some_chunk():
    text = "".join(chr(97 + i % 26) for i in range(100_000))
    chunks = chunk_text("d", text, 32000, 4000)
    for start in range(0, 100_000 - 3999, 997):
        end = start + 3999
        assert any(c.start <= start and end <= c.end for c in chunks)
    assert "".join(c.text for c in chunks).startswith(text[:32000])
    assert chunks[-1].end == len(text)


def test_chunk_size_must_exceed_overlap():
    with pytest.raises(ValueError):
        chunk_bounds(10, 5, 5)


def test_safe_id():
    assert safe_id("GOOSEHEADINSURANCE,INC_04/02 x__chunk2") == "GOOSEHEADINSURANCE,INC_04_02_x__chunk2"


# ---------------------------------------------------------------- parsing

def test_parse_plain_fenced_and_wrapped():
    assert parse_json_object('{"a": ["x"]}') == {"a": ["x"]}
    assert parse_json_object('```json\n{"a": []}\n```') == {"a": []}
    assert parse_json_object('Here: {"a": []} done') == {"a": []}
    assert parse_json_object('["not", "object"]') is None
    assert parse_json_object('{"a": ["unterminated') is None


def test_detect_repetition():
    sent = "The Distributor shall not sell competing products in the Territory. "
    looped = '{"Non-Compete": ["' + sent * 12
    assert detect_repetition(looped)
    legit = json.dumps({f"Clause {i}": [] for i in range(41)})
    assert not detect_repetition(legit)
    assert not detect_repetition(sent * 2)


def test_classify():
    assert classify('{"a": []}', "stop")[0] == "ok"
    assert classify('{"a": ["x', "length")[0] == "loop"
    assert classify('{"a": ["' + "same sentence here, again and again. " * 20, "stop")[0] == "loop"
    assert classify("I cannot help with that.", "stop")[0] == "invalid"
    assert classify('{"a": ["x', "repetition")[0] == "loop"


def test_canonicalize_maps_keys_and_drops_unknown():
    canon = canon_map(["Governing Law", "Parties"])
    out, dropped = canonicalize({"governing_law": ["NY law", "", 3, "NY law"],
                                 "PARTIES": "not a list", "Bogus": ["x"]}, canon)
    assert out == {"Governing Law": ["NY law"], "Parties": []}
    assert dropped == 1


# ---------------------------------------------------------------- merge

def test_merge_union_dedup_whitespace():
    cl = ["A", "B"]
    m = merge_chunks([{"A": ["x  y"], "B": []}, {"A": ["x y", "z"], "B": ["w"]}], cl)
    assert m == {"A": ["x  y", "z"], "B": ["w"]}


# ---------------------------------------------------------------- metrics

def test_tok_f1():
    assert tok_f1("a b c", "a b c") == 1.0
    assert tok_f1("a b", "c d") == 0.0
    assert tok_f1("a a b", "a b") == pytest.approx(2 * (2 / 3) * 1 / (2 / 3 + 1))


def test_detection_report_counts():
    cl = ["A", "B", "C"]
    gold = {"d1": {"A": ["alpha beta"], "B": [], "C": ["gamma"]},
            "d2": {"A": [], "B": ["delta"], "C": []}}
    pred = {"d1": {"A": ["alpha  beta"], "B": ["junk"], "C": []},
            "d2": {"A": [], "B": ["delta and more"], "C": []}}
    r = detection_report(gold, pred, cl, n_boot=50)
    # TP: d1/A, d2/B  FP: d1/B  FN: d1/C
    assert r["detection_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["detection_recall"] == pytest.approx(2 / 3, abs=1e-4)
    # strict: d1/A covered, d2/B covered ("delta" in "delta and more"), d1/C FN, d1/B FP
    assert r["strict_f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["decisions"] == 6
    lo, hi = r["detection_f1_ci95"]
    assert 0 <= lo <= hi <= 1


def test_verbatim():
    src = "The Borrower  shall maintain\na Total Net Leverage Ratio."
    q = ["The Borrower shall maintain", "the borrower shall", "made up"]
    assert verbatim_flags(q, src) == [True, False, False]
    assert verbatim_flags(q, src, casefold=True) == [True, True, False]
    assert verbatim_rate([], src) is None
