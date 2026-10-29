"""Runner + evaluation end to end with a fake backend (no GPU, no model download)."""
import json

import pytest

import src.generation as generation
from src.chunking import chunk_text
from src.evaluate import evaluate_run, reproduce_report
from src.generation import GenResult
from src.runner import load_record, prepare_run, run_stage, stage_todo

CLAUSES = ["Governing Law", "Parties"]
LOOP = '{"Parties": ["' + "Acme Corp and Beta LLC agree as follows here. " * 30


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "|".join(m["content"] for m in msgs) + "|assistant"


class FakeBackend:
    """Loops on chunk 'd1__chunk0' until repeat_penalty >= 1.15; ok elsewhere."""
    tokenizer = FakeTokenizer()
    calls = []

    def generate(self, prompts, max_tokens, repeat_penalty=1.0):
        out = []
        for p in prompts:
            FakeBackend.calls.append(repeat_penalty)
            if "LOOPY" in p and repeat_penalty < 1.15:
                out.append(GenResult(LOOP, "length", 10, max_tokens, 1.0, len(prompts)))
            else:
                out.append(GenResult(json.dumps({"governing law": ["laws of New York"],
                                                 "Parties": []}),
                                     "stop", 10, 20, 0.5, len(prompts)))
        return out


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: FakeTokenizer()))
    monkeypatch.setattr(generation, "make_backend", lambda *a, **k: FakeBackend())
    FakeBackend.calls = []
    chunks = (chunk_text("d1", "LOOPY governed by the laws of New York.", 1000, 100) +
              chunk_text("d2", "This is governed by the laws of New York state.", 1000, 100))
    cfg = {"model": "fake/model", "prompt": "cuad_native", "max_tokens": 64,
           "ladder": [1.05, 1.15, 1.25], "batch_size": 16, "chunk_chars": 1000,
           "overlap": 100}
    return prepare_run(tmp_path / "run", cfg, chunks), cfg, chunks


def test_primary_then_ladder_and_resume(run_dir):
    rd, cfg, chunks = run_dir
    assert run_stage(rd, "primary", "vllm", 2) == 0
    assert load_record(rd, "d1__chunk0")["final_status"] == "loop"
    assert load_record(rd, "d2__chunk0")["final_status"] == "ok"
    # resume: nothing left to do for primary
    assert stage_todo(rd, chunks, "primary") == []
    assert run_stage(rd, "ladder", "hf", 0) == 0
    rec = load_record(rd, "d1__chunk0")
    assert [a["repeat_penalty"] for a in rec["attempts"]] == [1.0, 1.05, 1.15]
    assert rec["final_status"] == "ok" and rec["final_repeat_penalty"] == 1.15
    # the ok chunk was never retried
    assert len(load_record(rd, "d2__chunk0")["attempts"]) == 1


def test_fingerprint_refuses_mixed_settings(run_dir, tmp_path):
    rd, cfg, chunks = run_dir
    with pytest.raises(SystemExit):
        prepare_run(rd, dict(cfg, max_tokens=999), chunks)
    with pytest.raises(SystemExit):
        prepare_run(rd, cfg, chunks[:1])
    prepare_run(rd, cfg, chunks)   # identical settings: fine


def test_evaluate_run(run_dir):
    rd, cfg, chunks = run_dir
    run_stage(rd, "primary", "vllm", 2)
    run_stage(rd, "ladder", "hf", 0)
    (rd / "stages.jsonl").write_text(
        json.dumps({"stage": "primary", "backend": "vllm", "tp": 2, "n_gpus": 2,
                    "exit_code": 0, "wall_s": 60.0}) + "\n" +
        json.dumps({"stage": "primary", "backend": "vllm", "tp": 1, "n_gpus": 1,
                    "exit_code": 1, "wall_s": 30.0}) + "\n")
    texts = {c.doc_id: c.text for c in chunks}
    gold = {"d1": {"Governing Law": ["laws of New York"], "Parties": []},
            "d2": {"Governing Law": ["the laws of New York state"], "Parties": ["Acme"]}}
    r = evaluate_run(rd, CLAUSES, texts, gold)
    d = r["decode"]
    assert d["primary_json_valid"] == 1 and d["primary_loop"] == 1
    assert d["recovered_by_rung"] == {"1.05": 0, "1.15": 1, "1.25": 0}
    assert d["ladder_recovery_rate"] == 1.0 and d["still_failing"] == []
    assert r["verbatim_rate_chunk_exact"] == 1.0
    assert r["gold"]["detection_recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["gold"]["detection_precision"] == 1.0
    assert r["gpu_time"]["gpu_device_minutes"] == 2.5
    assert r["found_share_per_clause"] == {"Governing Law": 1.0, "Parties": 0.0}
    assert (rd / "latency.csv").exists() and (rd / "eval.json").exists()
    md = reproduce_report({"fake": r}, {"platform": "local", "gpus": [], "n_contracts": 2,
                                         "session_minutes": 1})
    assert "| detection F1 | 0.9 |" in md and "exit 1" in md
    assert "incomplete" not in md


def test_report_marks_unrun_chunks_incomplete(run_dir):
    rd, cfg, chunks = run_dir
    run_stage(rd, "primary", "vllm", 2)
    (rd / "records" / "d2__chunk0.json").unlink()
    texts = {c.doc_id: c.text for c in chunks}
    gold = {d: {"Governing Law": [], "Parties": []} for d in texts}
    md = reproduce_report({"x": evaluate_run(rd, CLAUSES, texts, gold)},
                          {"platform": "local", "gpus": [], "n_contracts": 2,
                           "session_minutes": 0})
    assert "| detection F1 | 0.9 | incomplete (1/2 chunks) |" in md
