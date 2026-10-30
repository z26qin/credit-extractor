"""Runner + evaluation end to end with a fake backend (no GPU, no model download)."""
import json

import pytest

import src.generation as generation
from src.chunking import chunk_text
from src.evaluate import decode_summary, evaluate_run, reproduce_report
from src.generation import GenResult
from src.runner import (add_attempt, load_record, prepare_run, run_stage, save_record,
                        stage_todo)

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


def test_backend_python_uses_vllm_venv_only_for_vllm(tmp_path):
    import sys
    from src.runner import backend_python
    assert backend_python("vllm", tmp_path / "missing") == sys.executable
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").write_text("")
    assert backend_python("vllm", tmp_path) == str(tmp_path / "bin" / "python")
    assert backend_python("hf", tmp_path) == sys.executable


def test_repaired_chunk_skips_the_ladder_but_stays_separate_from_raw_validity(tmp_path):
    chunks = chunk_text("d", "text", 1000, 100)
    cfg = {"model": "m", "prompt": "cuad_native", "max_tokens": 64, "chunk_chars": 1000,
           "overlap": 100}
    rd = prepare_run(tmp_path / "run", cfg, chunks)
    rec = {"chunk_id": chunks[0].chunk_id, "doc_id": "d", "chunk_index": 0, "attempts": [],
           "final_status": None}
    add_attempt(rec, {"status": "repaired", "repeat_penalty": 1.0, "backend": "vllm"})
    save_record(rd, rec)
    assert stage_todo(rd, chunks, "ladder") == []
    s = decode_summary([load_record(rd, chunks[0].chunk_id)], [1.05])
    assert s["primary_json_valid"] == 0 and s["primary_repaired"] == 1
    assert s["chunks_sent_to_ladder"] == 0 and s["final_json_valid_rate"] == 1.0


def test_repair_check_replays_saved_failures(tmp_path):
    from src import reproduce
    chunks = chunk_text("d", 'x add ", and y,") and z', 1000, 100)
    cfg = {"model": "m", "prompt": "cuad_native", "max_tokens": 64, "chunk_chars": 1000,
           "overlap": 100}
    rd = prepare_run(tmp_path / "m0" / "r", cfg, chunks)
    bad = '{"a": ["x add \\", and y,") and z"]}'
    rec = {"chunk_id": chunks[0].chunk_id, "doc_id": "d", "chunk_index": 0, "final_status": None,
           "attempts": [{"text": bad, "finish_reason": "stop", "status": "invalid",
                         "repeat_penalty": 1.0, "backend": "vllm", "n_output_tokens": 9,
                         "latency_s": 1.0}]}
    save_record(rd, rec)
    rows = reproduce.repair_check({"root": tmp_path / "m0", "run_dirs": {"r": rd}})
    assert [(r["ladder"], r["status"], r["quotes"], r["verbatim"]) for r in rows] == [
        ("M0", "repaired", 1, 1.0)]


def test_ladder_check_replays_failed_chunks_without_touching_m0(run_dir, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src import reproduce
    rd, cfg, chunks = run_dir
    run_stage(rd, "primary", "vllm", 2)
    run_stage(rd, "ladder", "vllm", 2)
    before = load_record(rd, "d1__chunk0")
    monkeypatch.setattr(reproduce, "run_with_fallback",
                        lambda d, stage, chain: run_stage(d, stage, "hf", 0))
    env = SimpleNamespace(data_repo=None)
    rows = reproduce.ladder_check(env, {"root": tmp_path / "m0", "run_dirs": {"r": rd}},
                                  {"hub_prefix": "m0"})
    assert load_record(rd, "d1__chunk0") == before
    new = load_record(tmp_path / "m0" / "ladder_check" / "r", "d1__chunk0")
    assert [a["repeat_penalty"] for a in new["attempts"]] == [1.0, 1.05, 1.15]
    assert new["attempts"][0] == before["attempts"][0] and new["final_status"] == "ok"
    assert [r["repeat_penalty"] for r in rows if r["ladder"].startswith("check")] == [1.05, 1.15]
    assert len([r for r in rows if r["ladder"].startswith("M0")]) == 3
    assert rows[-1]["status"] == "ok" and rows[-1]["verbatim"] == 1.0
