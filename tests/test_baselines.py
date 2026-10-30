"""Milestone 3 end to end with a fake backend: three conditions, repair, ladder, report."""
import csv
import json
from types import SimpleNamespace

import pytest

import src.generation as generation
from src import baselines
from src.generation import GenResult
from src.runner import load_record, run_stage

DOCS = {
    "acme-2026": ("CREDIT AGREEMENT. The Maturity Date means June 1, 2031. "
                  'Section 7.11 the "Total Net Leverage Ratio" shall not exceed 3.50:1.00. ') * 40,
    "beta-2025": "TERM LOAN CREDIT AGREEMENT. LOOPY text. The Maturity Date means May 1, 2030. " * 30,
}


class Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "|".join(m["content"] for m in msgs) + "|assistant"


class Backend:
    """Credit prompt: quotes the maturity date; acme copies a quoted term unescaped;
    beta loops until repeat_penalty 1.15. CUAD prompt: one CUAD clause."""
    tokenizer = Tok()

    def generate(self, prompts, max_tokens, repeat_penalty=1.0):
        out = []
        for p in prompts:
            if "CLAUSE TYPES:" not in p:
                text = json.dumps({"Document Name": ["CREDIT AGREEMENT"]})
            elif "LOOPY" in p and repeat_penalty < 1.15:
                out.append(GenResult('{"Maturity Date": ["' + "May 1, 2030 again. " * 40,
                                     "length", 10, max_tokens, 1.0, len(prompts)))
                continue
            elif "LOOPY" in p:
                text = json.dumps({"Maturity Date": ["The Maturity Date means May 1, 2030."]})
            else:
                text = ('{"Financial Maintenance Covenant": ["the "Total Net Leverage Ratio" '
                        'shall not exceed 3.50:1.00."], "Maturity Date": '
                        '["The Maturity Date means June 1, 2031."], "Applicable Margin": []}')
            out.append(GenResult(text, "stop", 10, 20, 0.5, len(prompts)))
        return out


@pytest.fixture
def plan(tmp_path, monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: Tok()))
    monkeypatch.setattr(generation, "make_backend", lambda *a, **k: Backend())
    monkeypatch.setattr(baselines, "run_with_fallback",
                        lambda d, stage, chain: run_stage(d, stage, "hf", 0))
    text_dir = tmp_path / "work" / "m1_collect" / "text"
    text_dir.mkdir(parents=True)
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "char_count"])
        w.writeheader()
        for d, t in DOCS.items():
            (text_dir / f"{d}.txt").write_text(t, encoding="utf-8")
            w.writerow({"id": d, "char_count": len(t)})
    cfg = dict(baselines.load_cfg(), manifest=str(tmp_path / "manifest.csv"),
               chunk_chars=2000, overlap=200)
    env = SimpleNamespace(data_repo=None, paths=SimpleNamespace(work=tmp_path / "work"),
                          platform="local", gpus=[], session_minutes=lambda: 1.0)
    return env, baselines.prepare(env, cfg), cfg


def test_manifest_mismatch_is_refused(plan, tmp_path):
    env, _, cfg = plan
    (tmp_path / "work" / "m1_collect" / "text" / "beta-2025.txt").write_text("stale")
    with pytest.raises(ValueError, match="beta-2025"):
        baselines.load_documents(env, cfg)


def test_three_conditions_report_and_review_sheet(plan):
    env, p, cfg = plan
    shas = {json.loads((rd / "run_config.json").read_text())["chunks_sha256"]
            for rd in p["run_dirs"].values()}
    assert list(p["run_dirs"]) == ["a", "b", "c"] and len(shas) == 1

    baselines.run(env, p, cfg, "primary")
    results = baselines.report(env, p, cfg)          # before the ladder
    b = results["b"]["decode"]
    n_acme = len([1 for c in baselines.load_chunks(p["run_dirs"]["b"])
                  if c.doc_id == "acme-2026"])
    assert b["primary_json_valid"] == 0 and b["primary_repaired"] == n_acme
    assert b["primary_loop"] == b["chunks_sent_to_ladder"] == results["b"]["ladder_pending"] > 0
    md = (p["root"] / "baseline_m3.md").read_text()
    assert "**Incomplete:**" in md and "waiting for a ladder rung" in md

    baselines.run(env, p, cfg, "ladder")
    results = baselines.report(env, p, cfg)
    assert results["c"]["ladder_pending"] == 0
    assert results["c"]["decode"]["recovered_by_rung"] == {
        "1.05": 0, "1.15": results["c"]["decode"]["chunks_sent_to_ladder"], "1.25": 0}
    assert results["c"]["verbatim_rate_repaired_chunks"] == 1.0
    assert results["a"]["keys"]["with_all_keys"] == 0         # one CUAD key of 41
    md = (p["root"] / "baseline_m3.md").read_text()
    assert "**Incomplete:**" not in md
    assert "| Maturity Date | 2/2 | 2/2 | 2 | 2 |" in md
    assert md.startswith("# Milestone 3: label-free baselines on 2 credit agreements")
    assert "| Financial Maintenance Covenant | 1/2 | 1/2 | 1 | 1 |" in md
    assert "Document Name (100%)" in md

    with open(p["root"] / "review_sheet.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["condition"] for r in rows} == {"b", "c"}
    assert all(r["correct"] == "" and r["note"] == "" for r in rows)
    assert len(rows) == 2 * 2 * 10                        # every clause once per doc: 1 quote or none
    cov = [r for r in rows if r["clause"] == "Financial Maintenance Covenant"
           and r["found"] == "True"]
    assert [r["doc_id"] for r in cov] == ["acme-2026", "acme-2026"]
    assert cov[0]["quote"] == 'the "Total Net Leverage Ratio" shall not exceed 3.50:1.00.'
    assert cov[0]["from_repaired"] == "True" and cov[0]["verbatim"] == "True"
    assert rows[0]["doc_id"] == "acme-2026" and rows[0]["clause"] == "Financial Maintenance Covenant"
    rec = load_record(p["run_dirs"]["b"], "beta-2025__chunk0")
    assert [a["status"] for a in rec["attempts"]] == ["loop", "loop", "ok"]
