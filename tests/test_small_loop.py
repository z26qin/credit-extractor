"""The small loop: synthetic data, reward, scoring with a fake backend, and real training code
on a tiny random model on CPU (skipped unless the packages in requirements-train.txt exist)."""
import json
import random
from pathlib import Path

import pytest

import src.generation as generation
from src import rlvr, score, small_loop, synth, train
from src.generation import GenResult
from src.gold import clean_source
from src.platform import Env, Paths
from src.prompts import credit_clauses
from src.runner import run_stage

CLAUSES = credit_clauses()
SPEC = {
    "defaults": {"min_chars": 600, "max_chars": 1500, "max_clauses": 4, "min_lookalikes": 2,
                 "max_lookalikes": 6, "empty_share": 0.25, "page_break_share": 0.5},
    "train": {"n": 40},
    "rlvr": {"n": 30, "max_clauses": 2, "max_passage_chars": 1000, "max_quote_chars": 1600},
    "held_out": {"n": 20},
}
BODY = "Each party shall perform its obligations under this supply contract in a timely manner. "
GROUPS = [[f"{i}.{j} {BODY}" for j in range(40)] for i in range(10)]
RIGHT = "other Liens securing obligations in a total amount not to exceed $25,000,000;"
NOISE = "Liens for taxes not yet overdue;"
MATURITY = "“Maturity Date” means the fifth anniversary of the Closing Date."
SOURCE = f"(k) other Liens securing obligations in a total amount not to\n\n12\n\nexceed " \
         f"$25,000,000;\n\n(l) {NOISE}\n\n{MATURITY}\n\nSection 9.04 Notices are in writing.\n"
GOLD = {"Liens General Basket": [{"text": RIGHT, "main": True}],
        "Maturity Date": [{"text": MATURITY, "main": True}]}


@pytest.fixture(scope="module")
def sets():
    return synth.build_sets(synth.load_bank(), GROUPS, SPEC, seed=1)


def quotes(example):
    return [q for qs in json.loads(example["answer"]).values() for q in qs]


def test_sets_have_the_planned_sizes_and_are_reproducible(sets):
    assert {k: len(v) for k, v in sets.items()} == {"train": 40, "rlvr": 30, "held_out": 20}
    assert synth.build_sets(synth.load_bank(), GROUPS, SPEC, seed=1) == sets


def test_every_answer_is_the_full_reward_for_its_own_excerpt(sets):
    for examples in sets.values():
        for x in examples:
            assert rlvr.reward_text(x["answer"], x["gold"], x["excerpt"], CLAUSES) == 1.0


def test_answers_use_all_clause_keys_and_some_excerpts_have_no_clause(sets):
    examples = sets["train"]
    assert all(list(json.loads(x["answer"])) == CLAUSES for x in examples)
    assert any(not x["gold"] for x in examples) and any(x["gold"] for x in examples)


def test_a_page_number_can_break_a_passage_and_cleaning_restores_it():
    rng = random.Random(3)
    text = "one two three four five six seven eight nine ten eleven twelve"
    broken = synth.break_page(text, 47, rng)
    assert "\n\n47\n\n" in broken
    assert " ".join(clean_source(broken).split()) == text


def test_held_out_wording_stays_out_of_training_sets(sets):
    _, held = synth.split_bank(synth.load_bank())
    text = "\n".join(x["excerpt"] for n in ("train", "rlvr") for x in sets[n])
    for templates in held["clauses"].values():
        longest = max(synth.SLOT.split(templates[0]), key=len)
        assert " ".join(longest.split()) not in " ".join(text.split())


def test_rlvr_answers_stay_small_enough_for_the_completion_limit(sets):
    assert max(sum(map(len, quotes(x))) for x in sets["rlvr"]) <= 1600


def test_item_labels_stay_out_of_the_answer():
    slots = synth.Slots(random.Random(0))
    source, answer = slots.fill("{item} other Liens up to {amount};")
    assert source.endswith(answer) and source.startswith("(") and answer.startswith("other Liens")


def test_filler_drops_credit_wording():
    para = "The Supplier shall deliver the Products to the Distributor within thirty days of order."
    contract = {"paragraphs": [{"context": f"{para}\n\nThe Maturity Date of the note is fixed "
                                           f"by the parties in this paragraph of the contract.\n\nx"}]}
    assert synth.filler_groups({"t": contract}, ["t"]) == [[para]]


def test_leak_check_finds_shared_wording():
    bank = {"clauses": {"X": ["alpha beta gamma delta {amount} epsilon"]}, "lookalikes": {},
            "general": []}
    assert synth.leak_check(bank, {"d": "Alpha beta gamma delta."}, words=4)
    assert not synth.leak_check(bank, {"d": "Alpha beta gamma."}, words=4)


@pytest.mark.parametrize("pred,want", [
    ({"Liens General Basket": [RIGHT], "Maturity Date": [MATURITY]}, 1.0),
    ({"Liens General Basket": [RIGHT, NOISE], "Maturity Date": [MATURITY]}, 5 / 6),
    ({"Liens General Basket": [RIGHT]}, 2 / 3),
    ({"Liens General Basket": [NOISE], "Maturity Date": ["Section 9.04 Notices are in writing."]}, 0),
    ({}, 0),
])
def test_reward_scores_right_noisy_partial_and_wrong_answers(pred, want):
    assert rlvr.reward_quotes(pred, GOLD, SOURCE, CLAUSES) == pytest.approx(want)


def test_reward_marks_down_changed_words_without_zeroing_them():
    changed = {"Liens General Basket": [RIGHT.replace("total amount", "aggregate amount")],
               "Maturity Date": [MATURITY.replace("“", '"').replace("”", '"')]}
    assert 0 < rlvr.reward_quotes(changed, GOLD, SOURCE, CLAUSES) < 2 / 3


def test_reward_for_excerpts_with_no_clause():
    assert rlvr.reward_quotes({}, {}, SOURCE, CLAUSES) == 1.0
    assert rlvr.reward_quotes({"Maturity Date": [MATURITY]}, {}, SOURCE, CLAUSES) == 0.0


@pytest.mark.parametrize("raw", ["not json", '{"Maturity Date": []}', '{"a": 1'])
def test_reward_text_needs_valid_json_with_the_schema_keys(raw):
    assert rlvr.reward_text(raw, {}, SOURCE, CLAUSES) == 0.0


def test_reward_func_reads_the_dataset_columns():
    answer = json.dumps({c: [] for c in CLAUSES})
    got = rlvr.reward_func([answer, "oops"], [SOURCE, SOURCE], ["{}", "{}"], prompts=None)
    assert got == [1.0, 0.0]


def test_right_share_and_mean_reward():
    gold = {"d": GOLD}
    preds = {"d": {"Liens General Basket": [RIGHT, NOISE], "Maturity Date": [MATURITY]}}
    assert score.right_share(gold, preds, CLAUSES) == pytest.approx(0.6667)
    assert score.mean_reward(gold, preds, CLAUSES, {"d": SOURCE}) == pytest.approx(0.8333)
    assert score.mean_reward(gold, {}, CLAUSES, {"d": SOURCE}) == 0.0
    assert score.right_share(gold, {}, CLAUSES) is None


@pytest.mark.parametrize("log,ok", [
    ([{"loss": 5.5}, {"train_loss": 5.0}], True),
    ([{"train_loss": -1e-7}], True),
    ([{"loss": float("nan")}], False),
    ([{"train_loss": float("inf")}], False),
    ([{"epoch": 1}], False),
])
def test_check_losses_stops_on_a_broken_loss(log, ok):
    stats = {"stage": "sft", "log": log}
    if ok:
        small_loop.check_losses(stats)
    else:
        with pytest.raises(RuntimeError):
            small_loop.check_losses(stats)


def test_gate_needs_the_minimum_gain():
    cfg = {"gate_min_gain": 0.10}
    held = {"Model A": {"mean_reward": 0.40}, "SFT": {"mean_reward": 0.55}}
    assert small_loop.gate(held, cfg)
    held["SFT"]["mean_reward"] = 0.45
    assert not small_loop.gate(held, cfg)


class Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "|".join(m["content"] for m in msgs) + "|assistant"


class Backend:
    """Answers each excerpt with its known answer. The excerpt with HARD fails until the ladder."""
    tokenizer = Tok()

    def __init__(self, answers):
        self.answers = answers

    def generate(self, prompts, max_tokens, repeat_penalty=1.0):
        out = []
        for p in prompts:
            if "HARD" in p and repeat_penalty == 1.0:
                out.append(GenResult("{", "stop", 10, 1, 1.0, len(prompts)))
                continue
            text = next((a for e, a in self.answers.items() if e in p), "{}")
            out.append(GenResult(text, "stop", 10, 20, 1.0, len(prompts)))
        return out


def test_score_models_and_report_end_to_end(tmp_path, monkeypatch, sets):
    examples = sets["held_out"][:4]
    examples[0] = dict(examples[0], excerpt="HARD " + examples[0]["excerpt"])
    data = tmp_path / "data"
    synth.save_sets({"held_out": examples}, data)
    monkeypatch.setattr(small_loop, "SCRATCH", tmp_path / "scratch")
    backend = Backend({x["excerpt"]: x["answer"] for x in examples})
    monkeypatch.setattr(generation, "make_backend", lambda *a, **k: backend)
    merges = []
    monkeypatch.setattr(small_loop, "merge_adapter", merges.append)

    def in_process(run_dir, stage, chain):
        assert run_stage(run_dir, stage, chain[0][0], 1) == 0
    monkeypatch.setattr(small_loop, "run_with_fallback", in_process)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tok())

    cfg = dict(train.load_cfg(), hub_prefix="loop")
    env = Env(platform="local", paths=Paths(tmp_path / "work", tmp_path / "cache"), secrets={}, gpus=[])
    adapter = tmp_path / "scratch" / "loop" / "sft" / "final"
    adapter.mkdir(parents=True)
    (adapter / "train_stats.json").write_text(json.dumps(
        {"steps": 75, "seconds_per_step": 9.5, "peak_gpu_mib": 9000}), encoding="utf-8")

    held = small_loop.score_models(env, cfg, "held_out", {"SFT": adapter}, data)
    assert held["SFT"]["mean_reward"] == 1.0 and set(merges) == {adapter}
    merges.clear()
    held = small_loop.score_models(env, cfg, "held_out", {"SFT": adapter}, data)
    assert not merges, "a finished run needs no merge on restart"

    held["Model A"] = dict(held["SFT"])
    md = small_loop.write_report(env, cfg, data, held, None, {"rlvr_prompts": 200}, {"SFT": adapter})
    assert "RLVR did not run" in md and "| mean reward |" in md and "| SFT | 75 | 9.5 | 9000 |" in md


@pytest.fixture
def fake_training(tmp_path, monkeypatch):
    """Stands in for the training subprocess: every launch writes an adapter with stats."""
    calls = []

    def launch(module, args, log):
        calls.append((module, [str(a) for a in args]))
        final = Path(args[args.index("--out") + 1]) / "final"
        final.mkdir(parents=True, exist_ok=True)
        (final / "adapter_config.json").write_text("{}", encoding="utf-8")
        stage = "rlvr" if module == "src.rlvr" else "sft"
        (final / "train_stats.json").write_text(json.dumps(
            {"stage": stage, "seconds_per_step": 80.0 if stage == "rlvr" else 9.0,
             "peak_gpu_mib": 9000, "log": [{"train_loss": 1.0}]}), encoding="utf-8")
    monkeypatch.setattr(small_loop, "launch", launch)
    monkeypatch.setattr(small_loop, "SCRATCH", tmp_path / "scratch")
    env = Env(platform="local", paths=Paths(tmp_path / "work", tmp_path / "cache"), secrets={}, gpus=[])
    return env, calls


def test_short_test_projects_rlvr_and_cuts_it_when_too_slow(fake_training, tmp_path):
    env, calls = fake_training
    got = small_loop.short_test(env, train.load_cfg(), tmp_path / "data")
    assert [m for m, _ in calls] == ["src.train", "src.rlvr"]
    assert "--max-steps" in calls[0][1] and "--hub-repo" not in calls[0][1]
    assert got["rlvr_projected_hours"] == pytest.approx(4.44, abs=0.01)
    assert got["rlvr_prompts"] == 100


def test_training_stages_skip_a_finished_adapter(fake_training, tmp_path):
    env, calls = fake_training
    cfg = train.load_cfg()
    sft = small_loop.train_sft(env, cfg, tmp_path / "data")
    rl = small_loop.train_rlvr(env, cfg, tmp_path / "data", sft, 100)
    assert (sft.parent.name, rl.parent.name) == ("sft", "rlvr")
    assert calls[1][1][calls[1][1].index("--prompts") + 1] == "100"
    small_loop.train_sft(env, cfg, tmp_path / "data")
    assert len(calls) == 2


def test_usable_preds_leaves_out_excerpts_that_never_decoded(tmp_path, monkeypatch):
    class Never(Backend):
        def generate(self, prompts, max_tokens, repeat_penalty=1.0):
            return [GenResult("{", "stop", 10, 1, 1.0, len(prompts)) for _ in prompts]
    monkeypatch.setattr(generation, "make_backend", lambda *a, **k: Never({}))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tok())
    monkeypatch.setattr(small_loop, "run_with_fallback",
                        lambda run_dir, stage, chain: run_stage(run_dir, stage, "hf", 1))
    small_loop.run_model(train.load_cfg(), tmp_path / "run", "m", {"d": "some text"})
    assert small_loop.usable_preds(tmp_path / "run", CLAUSES) == {}


# ------------------------------------------------- real training code on a tiny model

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    path = tmp_path_factory.mktemp("tiny")
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(["contract"], trainers.BpeTrainer(
        vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>",
                                   pad_token="<|endoftext|>")
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
                          "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}"
                          "<|im_start|>assistant\n{% endif %}")
    fast.save_pretrained(path)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16, max_position_embeddings=8192,
        eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id)).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny_cfg():
    cfg = train.load_cfg()
    cfg["sft"].update(grad_accum=2, epochs=1, save_steps=2, max_seq_length=8192)
    cfg["rlvr"].update(num_generations=2, max_completion_length=8, save_steps=2)
    return cfg


def small_sets(path):
    spec = {"defaults": dict(SPEC["defaults"], min_chars=100, max_chars=300, min_lookalikes=0,
                             max_lookalikes=1), "train": {"n": 4}, "rlvr": {"n": 3},
            "held_out": {"n": 1}}
    synth.save_sets(synth.build_sets(synth.load_bank(), GROUPS, spec, seed=2), path)


def test_encode_puts_the_loss_on_the_answer_only(tiny):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny)
    x = {"excerpt": "some text", "answer": '{"a": []}'}
    got = train.encode(tok, x, 8192)
    n = len(tok(train.prompt_text(tok, "some text"), add_special_tokens=False)["input_ids"])
    assert got["labels"][:n] == [-100] * n and all(t != -100 for t in got["labels"][n:])
    assert got["labels"][-1] == tok.convert_tokens_to_ids(train.IM_END)
    assert train.encode(tok, x, 10) is None


def test_sft_rlvr_and_merge_run_end_to_end_and_resume(tiny, tiny_cfg, tmp_path):
    small_sets(tmp_path / "data")
    sft_dir = tmp_path / "sft"
    adapter = train.run(tiny_cfg, tmp_path / "data" / "train.jsonl", sft_dir, base=tiny)
    assert (adapter / "adapter_model.safetensors").exists()
    stats = json.loads((adapter / "train_stats.json").read_text(encoding="utf-8"))
    assert stats["steps"] == 2 and stats["examples"] == 4
    assert train.resume_point(sft_dir).endswith("checkpoint-2")

    # a restart from a pulled checkpoint carries on from its step
    (sft_dir / "checkpoint-2").rename(sft_dir / "checkpoint")
    assert train.resume_point(sft_dir).endswith("checkpoint")
    train.run(tiny_cfg, tmp_path / "data" / "train.jsonl", sft_dir, base=tiny, max_steps=3)
    stats = json.loads((adapter / "train_stats.json").read_text(encoding="utf-8"))
    assert stats["steps"] == 3

    rl_dir = tmp_path / "rlvr"
    rl_adapter = rlvr.run(tiny_cfg, tmp_path / "data" / "rlvr.jsonl", adapter, rl_dir, base=tiny)
    stats = json.loads((rl_adapter / "train_stats.json").read_text(encoding="utf-8"))
    assert stats["steps"] == 3 and stats["prompts"] == 3
    assert any("rewards/reward_func/mean" in row for row in stats["log"])
    assert train.resume_point(rl_dir)

    merged = train.merge(tiny, rl_adapter, tmp_path / "merged")
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert AutoModelForCausalLM.from_pretrained(merged).config.hidden_size == 32
    assert AutoTokenizer.from_pretrained(merged).eos_token == train.IM_END
    assert (Path(merged) / "config.json").read_bytes() == (Path(tiny) / "config.json").read_bytes()
    assert (load_file(Path(merged) / "model.safetensors").keys()
            == load_file(Path(tiny) / "model.safetensors").keys())


def test_command_lines_run_each_stage(tiny, tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "load_cfg", lambda: dict(tiny_cfg, base_model=tiny))
    small_sets(tmp_path / "data")
    sft, rl = tmp_path / "sft", tmp_path / "rlvr"
    assert train.main(["--data", str(tmp_path / "data" / "train.jsonl"), "--out", str(sft),
                       "--max-steps", "1"]) == 0
    assert rlvr.main(["--data", str(tmp_path / "data" / "rlvr.jsonl"), "--adapter",
                      str(sft / "final"), "--out", str(rl), "--max-steps", "1"]) == 0
    assert train.main(["--merge", "--adapter", str(rl / "final"), "--out",
                       str(tmp_path / "merged")]) == 0
    assert (tmp_path / "merged" / "config.json").exists()


def test_rlvr_changes_the_adapter_when_rewards_differ(tiny, tiny_cfg, tmp_path, monkeypatch):
    from safetensors.torch import load_file
    small_sets(tmp_path / "data")
    adapter = train.run(tiny_cfg, tmp_path / "data" / "train.jsonl", tmp_path / "sft", base=tiny)
    before = load_file(adapter / "adapter_model.safetensors")
    rng = random.Random(0)
    monkeypatch.setattr(rlvr, "reward_func", lambda completions, **_: [rng.random() for _ in completions])
    cfg = dict(tiny_cfg, rlvr=dict(tiny_cfg["rlvr"], learning_rate=1e-2))
    final = rlvr.run(cfg, tmp_path / "data" / "rlvr.jsonl", adapter, tmp_path / "rlvr", base=tiny)
    after = load_file(final / "adapter_model.safetensors")
    assert any(not (before[k] == after[k]).all() for k in before)
    assert {str(v.dtype) for v in after.values()} == {"torch.float32"}
