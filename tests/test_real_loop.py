"""Step 2 on teacher pre-labels: the examples, the step plan, the answer-only loss on a tiny model,
and scoring with a fake backend."""
import json
from pathlib import Path

import pytest

import src.generation as generation
from src import real_loop, small_loop, train
from src.platform import Env, Paths
from src.prompts import credit_clauses
from src.runner import run_stage
from tests.test_small_loop import Backend, Tok, small_sets, tiny, tiny_cfg  # noqa: F401

CLAUSES = credit_clauses()
FILL = "The parties agree to the terms set out in this Agreement. "


def doc(tag: str) -> tuple[str, str, str]:
    """30,000 characters, so two teacher chunks; two quotes near the start, in that order."""
    first = f"{tag} Maturity Date means the fifth anniversary of the Closing Date."
    second = f"{tag} other Liens securing obligations not to exceed $25,000,000."
    text = FILL * 20 + first + " " + FILL * 5 + second + " " + FILL * 600
    return text[:30_000], first, second


def record(d, i, kept, left_out=False):
    return {"chunk_id": f"{d}__chunk{i}", "doc_id": d, "index": i, "kept": kept,
            "left_out": left_out, "attempts": []}


def fixture_records():
    texts, records = {}, []
    for d in ("a", "b", "c", "d"):
        texts[d], first, second = doc(d)
        # quotes listed out of order: the answer puts them back in text order
        records.append(record(d, 0, {"Liens General Basket": [second], "Maturity Date": [first]}))
        records.append(record(d, 1, {}, left_out=(d == "c")))
    return texts, records


def test_examples_use_train_chunks_only_with_empties_capped():
    texts, records = fixture_records()
    examples, info = real_loop.build_examples(records, texts, ["a", "b", "c"], 0.3, seed=1)
    assert info == {"examples": 3 + 1, "with_quotes": 3, "empty_used": 1, "empty_available": 2,
                    "train_agreements": 3, "skipped_not_train": 2, "skipped_left_out": 1}
    assert not any(x["id"].startswith("d__") for x in examples)
    full = next(x for x in examples if x["id"] == "a__chunk0")
    answer = json.loads(full["answer"])
    assert list(answer) == CLAUSES
    assert answer["Maturity Date"] == [doc("a")[1]] and answer["Liens General Basket"] == [doc("a")[2]]
    assert full["answer"].index("Liens") < full["answer"].index("Maturity")   # key order is the schema's
    assert all(q in x["excerpt"] for x in examples for qs in json.loads(x["answer"]).values() for q in qs)
    assert real_loop.build_examples(records, texts, ["a", "b", "c"], 0.3, seed=1)[0] == examples


def test_a_quote_that_is_not_in_its_chunk_stops_the_build():
    texts, records = fixture_records()
    records[0]["kept"] = {"Maturity Date": ["text that is nowhere"]}
    with pytest.raises(ValueError, match="not in the chunk"):
        real_loop.build_examples(records, texts, ["a"], 0.3, seed=1)


def test_answer_quotes_follow_text_order():
    norm = "alpha one. beta two. gamma three."
    got = json.loads(real_loop.answer_text({"Maturity Date": ["gamma three.", "alpha one."]}, norm, CLAUSES))
    assert got["Maturity Date"] == ["alpha one.", "gamma three."] and got["MFN Protection"] == []


@pytest.mark.parametrize("tokens,steps", [(6_000_000, 124), (12_000_000, 96)])
def test_plan_steps_cuts_the_run_to_the_hour_limit(tokens, steps):
    cfg = real_loop.load_cfg()
    info = {"examples": 1000, "over_length": 10, "tokens": tokens, "short_tokens": 24 * 8000}
    got = real_loop.plan_steps(info, {"seconds_per_step": 160.0, "peak_gpu_mib": 9000}, cfg)
    assert got["full_steps"] == 124 and got["max_steps"] == steps
    assert got["projected_hours"] == pytest.approx(0.0025 * tokens / 3600, abs=0.01)


@pytest.fixture
def fake_launch(tmp_path, monkeypatch):
    calls = []

    def launch(module, args, log):
        calls.append([str(a) for a in args])
        final = Path(args[args.index("--out") + 1]) / "final"
        final.mkdir(parents=True, exist_ok=True)
        (final / "adapter_config.json").write_text("{}", encoding="utf-8")
        (final / "train_stats.json").write_text(json.dumps(
            {"stage": "sft", "steps": 3, "seconds_per_step": 160.0, "peak_gpu_mib": 9000,
             "log": [{"loss": 1.2}, {"loss": 0.4}, {"train_loss": 0.8}]}), encoding="utf-8")
    monkeypatch.setattr(real_loop, "launch", launch)
    monkeypatch.setattr(real_loop, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(small_loop, "SCRATCH", tmp_path / "scratch")
    env = Env(platform="local", paths=Paths(tmp_path / "work", tmp_path / "cache"), secrets={}, gpus=[])
    return env, calls


def test_short_test_and_training_use_the_real_config(fake_launch, tmp_path):
    env, calls = fake_launch
    cfg = real_loop.load_cfg()
    info = {"examples": 1000, "over_length": 10, "tokens": 6_000_000, "short_tokens": 192_000}
    short = real_loop.short_test(env, cfg, tmp_path / "data", info)
    assert short["max_steps"] == 124 and calls[0][calls[0].index("--max-steps") + 1] == "3"
    assert calls[0][calls[0].index("--config") + 1].endswith("real_sft.yaml")
    adapter = real_loop.train(env, cfg, tmp_path / "data", short["max_steps"])
    assert adapter.parent.name == "sft_real" and calls[1][calls[1].index("--max-steps") + 1] == "124"
    real_loop.train(env, cfg, tmp_path / "data", 124)
    assert len(calls) == 2, "a finished adapter is not trained again"


def test_answer_only_loss_matches_the_full_loss(tiny):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(tiny), AutoModelForCausalLM.from_pretrained(tiny)
    enc = train.encode(tok, {"excerpt": "contract text " * 30, "answer": '{"a": ["b"]}'}, 8192)
    batch = train.collate([enc])
    full = model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
    keep = int((batch["labels"] != -100).sum()) + 1
    part = model(input_ids=batch["input_ids"], logits_to_keep=keep).logits
    assert part.shape[1] == keep
    assert torch.allclose(train.answer_loss(part, batch["labels"]), full, atol=1e-6)


def test_answer_only_training_moves_the_adapter_as_full_logits_do(tiny, tiny_cfg, tmp_path):
    import torch
    from safetensors.torch import load_file
    small_sets(tmp_path / "data")
    data = tmp_path / "data" / "train.jsonl"
    torch.manual_seed(0)                 # LoRA A matrices start random
    plain = train.run(tiny_cfg, data, tmp_path / "plain", base=tiny, max_steps=2)
    cfg = dict(tiny_cfg, sft=dict(tiny_cfg["sft"], answer_logits_only=True))
    torch.manual_seed(0)
    answer = train.run(cfg, data, tmp_path / "answer", base=tiny, max_steps=2)
    a, b = (load_file(p / "adapter_model.safetensors") for p in (plain, answer))
    assert a.keys() == b.keys()
    assert max(float((a[k] - b[k]).abs().max()) for k in a) < 1e-5
    assert any(float(a[k].abs().max()) > 0 for k in a if "lora_B" in k), "training moved lora_B"
    stats = json.loads((answer / "train_stats.json").read_text(encoding="utf-8"))
    assert stats["tokens"] > 0 and stats["steps"] == 2


def test_evaluate_and_report_with_a_fake_backend(tmp_path, monkeypatch, fake_launch):
    env, _ = fake_launch
    texts = {"m1": "Intro text.\n\n12\n\nThe Maturity Date is June 1, 2030.\n",
             "m2": "No clause here at all.\n"}
    quote = "The Maturity Date is June 1, 2030."
    gold = {"labeled_by": "test", "documents": {
        "m1": {"Maturity Date": [{"text": quote, "main": True}]}, "m2": {}}}
    (tmp_path / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    monkeypatch.setattr(real_loop, "GOLD_M3", tmp_path / "gold.json")
    monkeypatch.setattr(real_loop, "load_documents", lambda env, cfg: texts)
    answers = {"Maturity Date is June": json.dumps({"Maturity Date": [quote]}),
               "No clause": json.dumps({c: [] for c in CLAUSES})}
    monkeypatch.setattr(generation, "make_backend", lambda *a, **k: Backend(answers))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tok())
    merges = []
    monkeypatch.setattr(small_loop, "merge_adapter", merges.append)
    monkeypatch.setattr(small_loop, "run_with_fallback",
                        lambda run_dir, stage, chain: run_stage(run_dir, stage, chain[0][0], 1))
    cfg = real_loop.load_cfg()
    adapter = real_loop.train(env, cfg, tmp_path / "data", 3)
    results = real_loop.evaluate(env, cfg, adapter)
    assert results["Model A"]["found_f1"] == results["SFT"]["found_f1"] == 1.0 and merges == [adapter]
    info = {"examples": 4, "with_quotes": 3, "empty_used": 1, "empty_available": 2,
            "over_length": 0, "skipped_left_out": 1, "tokens": 30_000, "max_tokens": 8000,
            "train_agreements": 3}
    short = {"seconds_per_step_longest": 160.0, "peak_gpu_mib": 9000, "projected_hours": 4.2,
             "full_steps": 124, "max_steps": 124}
    md = real_loop.write_report(env, cfg, info, short, adapter, results)
    assert "| 4 | 3 | 1 of 2 | 0 | 1 | 30,000 | 8,000 |" in md
    assert "| 3 | 160.0 | 9000 | 1.2 | 0.4 |" in md and "| Found F1 |" in md
    assert (env.paths.work / "real_sft" / "real_sft.md").exists()
