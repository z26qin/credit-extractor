"""Step 3: windows, the check set, the reward with its length rule, screening, the sampler, the
whole stage order with fakes, and GRPO with the step 3 settings on a tiny model."""
import json
import sys
import types
from pathlib import Path

import pytest

import src.generation as generation
from src import real_loop, real_rlvr, rlvr, small_loop, train
from src.chunking import Chunk
from src.platform import Env, Paths
from src.prompts import credit_clauses
from src.runner import run_stage
from tests.test_real_loop import doc
from tests.test_small_loop import Backend, Tok, small_sets, tiny, tiny_cfg  # noqa: F401

CLAUSES = credit_clauses()


def chunk(text: str) -> Chunk:
    return Chunk("d", 0, 1, 0, len(text), text)


def test_windows_keep_whole_quotes_and_drop_cut_ones():
    text = list("." * 300)
    a, b = "alpha one two", "beta three four five six seven eight nine ten"
    text[10:10 + len(a)] = a.replace(" ", "\n")       # whitespace differs from the quote
    text[90:90 + len(b)] = b
    text = "".join(text)
    got = real_rlvr.windows_for(chunk(text), {"Maturity Date": [a], "MFN Protection": [b]}, 100, 20)
    # starts 0, 80, 160, 240: the first window cuts b, so it goes, though it holds a
    assert [w["id"] for w in got] == ["d__chunk0__w1", "d__chunk0__w2", "d__chunk0__w3"]
    assert got[0]["refs"] == {"MFN Protection": [b]} and b in got[0]["excerpt"]
    assert got[1]["refs"] == {} and got[2]["refs"] == {}
    with pytest.raises(ValueError, match="not in its chunk"):
        real_rlvr.windows_for(chunk(text), {"Maturity Date": ["nowhere at all"]}, 100, 20)


def windows(n_docs: int = 6) -> list[dict]:
    rows = []
    for i in range(n_docs):
        for j in range(4):
            refs = {"Maturity Date": [f"q{i}{j}"]} if j < 2 else {}
            words = "the Maturity Date is used here" if j == 2 else "nothing of note"
            rows.append({"id": f"d{i}__w{j}", "doc_id": f"d{i}", "excerpt": f"{words} q{i}{j}",
                         "refs": refs})
    return rows


def test_pool_and_check_come_from_different_agreements():
    cfg = real_rlvr.load_cfg()
    w = dict(cfg["windows"], check_agreements=2, pool_quotes=5, pool_empty=10, check_quotes=2,
             check_empty=10)
    ids = [f"d{i}" for i in range(6)]
    pool, check, info = real_rlvr.pick(windows(), ids, w, seed=3)
    assert {x["doc_id"] for x in pool}.isdisjoint({x["doc_id"] for x in check})
    assert len({x["doc_id"] for x in check}) <= 2 and info["check_agreements"] == 2
    empty = [x for x in pool + check if not x["refs"]]
    assert empty and all("Maturity Date" in x["excerpt"] for x in empty), "empty needs a term"
    assert info["pool"] == len(pool) == 5 + 4 and info["pool_empty"] == 4
    assert real_rlvr.pick(windows(), ids, w, seed=3)[0] == pool


def test_exact_by_f1_counts_a_whole_section_as_too_long():
    passage = "other Liens securing obligations not to exceed $25,000,000"
    section = " ".join(["Section 7.02 Liens. The Borrower shall not create any Lien except:"] * 4
                       + [passage + ";"] + ["(m) Liens for taxes not yet due;"] * 4)
    gold = {"Liens General Basket": [{"text": passage, "main": True}]}
    exact = {"Liens General Basket": [passage]}
    whole = {"Liens General Basket": [section]}
    for f1 in (False, True):
        assert rlvr.reward_quotes(exact, gold, section, CLAUSES, f1) == 1.0
    assert rlvr.reward_quotes(whole, gold, section, CLAUSES) == 1.0
    assert rlvr.reward_quotes(whole, gold, section, CLAUSES, exact_by_f1=True) == 0.5


def answer(quotes: dict) -> str:
    return json.dumps({c: quotes.get(c, []) for c in CLAUSES})


def test_screening_keeps_windows_whose_samples_differ():
    rows = {f"w{i}": {"id": f"w{i}", "excerpt": f"text q{i} end",
                      "gold": {"Maturity Date": [{"text": f"q{i}", "main": True}]}}
            for i in range(3)}
    rows["e"] = {"id": "e", "excerpt": "no clause", "gold": {}}
    right, empty = (lambda i: answer({"Maturity Date": [f"q{i}"]})), answer({})
    samples = [
        {"id": "w0", "completions": [right(0)] * 4, "finish": ["stop"] * 4, "tokens": [9] * 4},
        {"id": "w1", "completions": [right(1), empty, empty, "{"], "finish": ["stop"] * 3 + ["length"],
         "tokens": [9, 5, 5, 1]},
        {"id": "w2", "completions": [right(2)] * 3 + [empty], "finish": ["stop"] * 4,
         "tokens": [9] * 4},
        {"id": "e", "completions": [empty] * 4, "finish": ["stop"] * 4, "tokens": [5] * 4}]
    scored = real_rlvr.score_samples(samples, rows, exact_by_f1=True)
    assert [x["rewards"] for x in scored] == [[1.0] * 4, [1.0, 0, 0, 0], [1.0, 1.0, 1.0, 0], [1.0] * 4]
    assert scored[1]["truncated"] == 1 and scored[3]["empty"]
    s = {"min_spread": 0.05, "max_prompts": 5}
    assert real_rlvr.select(scored, s) == ["w1", "w2"]          # equal spread: by id
    assert real_rlvr.select(scored, dict(s, max_prompts=1)) == ["w1"]


@pytest.mark.parametrize("seconds,prompts", [(10.0, 400), (60.0, 120)])
def test_plan_prompts_cuts_the_run_to_the_hour_limit(seconds, prompts):
    cfg = real_rlvr.load_cfg()
    cfg["short_test"]["max_rlvr_hours"] = 2.0
    got = real_rlvr.plan_prompts(400, {"seconds_per_step": seconds, "peak_gpu_mib": 1}, cfg)
    assert got["prompts"] == prompts and got["selected"] == 400


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def get_tokenizer(self):
        return Tok()

    def generate(self, prompts, sp, use_tqdm=False):
        self.calls += 1
        out = types.SimpleNamespace
        return [out(prompt_token_ids=p.split(), outputs=[
            out(text=f"answer {k}", finish_reason="stop", token_ids=[1] * (k + 1))
            for k in range(sp.n)]) for p in prompts]


def test_sampler_writes_each_batch_and_resumes(tmp_path, monkeypatch):
    from src import sample
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tok())
    rows = [{"id": f"x{i}", "excerpt": f"excerpt {i}"} for i in range(5)]
    out, llm = tmp_path / "s.jsonl", FakeLLM()
    assert sample.sample(llm, rows[:3], out, 4, 1.0, 64, batch=2) == 3 and llm.calls == 2
    assert sample.sample(llm, rows, out, 4, 1.0, 64, batch=2) == 2 and llm.calls == 3
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [x["id"] for x in lines] == [f"x{i}" for i in range(5)]
    assert lines[0]["completions"] == [f"answer {k}" for k in range(4)] and lines[0]["tokens"] == [1, 2, 3, 4]
    greedy = tmp_path / "g.jsonl"
    sample.sample(llm, rows[:1], greedy, 4, 0.0, 64, batch=2)
    assert len(json.loads(greedy.read_text())["completions"]) == 1


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    """Train agreements, teacher records and M1 text on disk, with every GPU step faked."""
    work = tmp_path / "work"
    texts, records = {}, []
    for d in ("a", "b", "c", "t"):
        texts[d], first, second = doc(d)
        records.append({"chunk_id": f"{d}__chunk0", "doc_id": d, "index": 0, "left_out": False,
                        "kept": {"Maturity Date": [first], "Liens General Basket": [second]}})
        records.append({"chunk_id": f"{d}__chunk1", "doc_id": d, "index": 1, "left_out": False,
                        "kept": {}})
    (work / "collect_full" / "text").mkdir(parents=True)
    rows = ["id,char_count"] + [f"{d},{len(t)}" for d, t in texts.items()]
    (tmp_path / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for d, t in texts.items():
        (work / "collect_full" / "text" / f"{d}.txt").write_text(t, encoding="utf-8")
    (work / "prelabel_full" / "records").mkdir(parents=True)
    for r in records:
        (work / "prelabel_full" / "records" / f"{r['chunk_id']}.json").write_text(json.dumps(r))
    (tmp_path / "split.json").write_text(json.dumps({"train": ["a", "b", "c"], "test": ["t"]}))

    cfg = real_rlvr.load_cfg()
    cfg["data"].update(manifest=str(tmp_path / "manifest.csv"), split=str(tmp_path / "split.json"))
    cfg["windows"].update(chars=8000, overlap=2000, check_agreements=1, pool_quotes=50,
                          pool_empty=50, check_quotes=10, check_empty=10,
                          hard_negative_terms=["Agreement"])
    cfg["screen"].update(min_prompts=1)
    monkeypatch.setattr(real_rlvr, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(small_loop, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tok())

    def sampler(model, data, out, cfg, n, temperature, log):
        rows = [json.loads(l) for l in Path(data).read_text().splitlines()]
        with open(out, "a", encoding="utf-8") as f:
            for x in rows:
                right = answer({c: [p["text"] for p in ps] for c, ps in x["gold"].items()})
                done = [right] * n if n == 1 or not x["gold"] else [right, right, answer({}), "{"]
                f.write(json.dumps({"id": x["id"], "completions": done,
                                    "finish": ["stop"] * len(done),
                                    "tokens": [len(c) for c in done], "prompt_tokens": 10}) + "\n")

    def fake_launch(module, args, log):
        out = Path(args[args.index("--out") + 1]) / "final"
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text("{}")
        steps = int(args[args.index("--max-steps") + 1]) if "--max-steps" in args else \
            int(args[args.index("--prompts") + 1])
        log_rows = [{"loss": 0.01, "reward": 0.5 + 0.1 * i, "frac_reward_zero_std": 0.2,
                     "completions/mean_length": 200.0} for i in range(5)]
        (out / "train_stats.json").write_text(json.dumps(
            {"stage": "rlvr", "steps": steps, "seconds_per_step": 30.0, "peak_gpu_mib": 20000,
             "log": log_rows}))
        launched.append([str(a) for a in args])
    launched = []
    monkeypatch.setattr(real_rlvr, "run_sampler", sampler)
    monkeypatch.setattr(real_rlvr, "launch", fake_launch)
    monkeypatch.setattr(real_rlvr, "merge_adapter", lambda adapter: None)
    monkeypatch.setattr(small_loop, "merge_adapter", lambda adapter: None)
    env = Env(platform="local", paths=Paths(work, tmp_path / "cache"), secrets={}, gpus=[])
    return env, cfg, launched


def test_stages_run_in_order_with_fakes(stage_env, tmp_path, monkeypatch):
    env, cfg, launched = stage_env
    data, info = real_rlvr.build_data(env, cfg)
    pool = [json.loads(l) for l in (data / "pool.jsonl").read_text().splitlines()]
    check = [json.loads(l) for l in (data / "check.jsonl").read_text().splitlines()]
    assert not any(x["id"].startswith("t__") for x in pool + check), "test agreements stay out"
    assert {x["id"].split("__")[0] for x in pool}.isdisjoint({x["id"].split("__")[0] for x in check})
    assert any(x["gold"] for x in pool) and info["pool"] == len(pool)
    assert real_rlvr.build_data(env, cfg)[1] == info

    sft = tmp_path / "scratch" / "real_sft" / "sft_real" / "final"
    sft.mkdir(parents=True)
    (sft / "adapter_config.json").write_text("{}")
    assert real_rlvr.sft_adapter(env, cfg) == sft
    screened = real_rlvr.screen(env, cfg, data, sft)
    with_quotes = sum(bool(x["gold"]) for x in pool)
    assert screened["selected"] == with_quotes and screened["selected_empty"] == 0
    assert screened["mean_reward_empty"] == 1.0 and screened["truncated_share"] == 0
    assert real_rlvr.screen(env, cfg, data, sft) == screened

    short = real_rlvr.short_test(env, cfg, sft)
    assert launched[0][launched[0].index("--max-steps") + 1] == "4"
    assert launched[0][launched[0].index("--config") + 1].endswith("real_rlvr.yaml")
    assert short["prompts"] == with_quotes
    adapter = real_rlvr.train(env, cfg, sft, short["prompts"])
    assert adapter.parent.name == "rlvr_real" and len(launched) == 2

    # M1: step 2's Model A and SFT decodes already exist; RLVR decodes the same way
    texts = {"m1": "Intro text.\n\n12\n\nThe Maturity Date is June 1, 2030.\n",
             "m2": "No clause here at all.\n"}
    quote = "The Maturity Date is June 1, 2030."
    gold = {"labeled_by": "test", "documents": {
        "m1": {"Maturity Date": [{"text": quote, "main": True}]}, "m2": {}}}
    (tmp_path / "gold.json").write_text(json.dumps(gold))
    monkeypatch.setattr(real_rlvr, "GOLD_M3", tmp_path / "gold.json")
    monkeypatch.setattr(real_rlvr, "load_documents", lambda env, cfg: texts)
    answers = {"Maturity Date is June": answer({"Maturity Date": [quote]}), "No clause": answer({})}
    monkeypatch.setattr(generation, "make_backend", lambda *a, **k: Backend(answers))
    monkeypatch.setattr(small_loop, "run_with_fallback",
                        lambda run_dir, stage, chain: run_stage(run_dir, stage, chain[0][0], 1))
    docs = {d: real_rlvr.clean_source(t) for d, t in texts.items()}
    for name in ("model_a", "sft"):
        small_loop.run_model(cfg, env.paths.work / "real_sft" / "runs" / "m1" / name, "m", docs)
    results = real_rlvr.evaluate(env, cfg, data, adapter)
    assert list(results["m1"]) == ["Model A", "SFT", "RLVR"]
    assert results["m1"]["RLVR"]["strict_f1"] == 1.0
    assert results["check"]["SFT"]["windows"] == len(check)
    assert results["check"]["RLVR"]["mean_reward"] == 1.0

    md = real_rlvr.write_report(env, cfg, info, screened, short, adapter, results)
    assert "| RLVR | +0.000 (+0.000 to +0.000) | +0.000 (+0.000 to +0.000) |" in md
    assert f"| {with_quotes} | 30.0 | 20000 | 0.6 | 0.8 | 0.2 | 200.0 |" in md
    assert (env.paths.work / "real_rlvr" / "real_rlvr.md").exists()


def test_screening_stops_when_too_few_windows_differ(stage_env, tmp_path):
    env, cfg, _ = stage_env
    data, _ = real_rlvr.build_data(env, cfg)
    cfg["screen"].update(min_prompts=10_000)
    sft = tmp_path / "sft_real" / "final"
    with pytest.raises(SystemExit, match="have reward spread"):
        real_rlvr.screen(env, cfg, data, sft)


def test_time_limit_stops_training_after_its_hours():
    pytest.importorskip("transformers")
    from transformers import TrainerControl
    limit, control = rlvr.time_limit(0.5), TrainerControl()
    limit.on_train_begin(None, None, control)
    limit.on_step_end(None, None, control)
    assert not control.should_training_stop and not limit.stopped
    limit.start -= 0.5 * 3600 + 1
    limit.on_step_end(None, None, control)
    assert control.should_training_stop and limit.stopped
    none, control = rlvr.time_limit(None), TrainerControl()
    none.on_train_begin(None, None, control)
    none.start -= 10 ** 6
    none.on_step_end(None, None, control)
    assert not control.should_training_stop


def test_grpo_runs_with_the_step_3_settings(tiny, tiny_cfg, tmp_path):
    small_sets(tmp_path / "data")
    adapter = train.run(tiny_cfg, tmp_path / "data" / "train.jsonl", tmp_path / "sft", base=tiny)
    cfg = real_rlvr.load_cfg()
    cfg["rlvr"].update(num_generations=2, max_completion_length=8, save_steps=2)
    assert cfg["rlvr"]["exact_by_token_f1"] and cfg["rlvr"]["mask_truncated_completions"]
    final = rlvr.run(cfg, tmp_path / "data" / "rlvr.jsonl", adapter, tmp_path / "rlvr", base=tiny,
                     prompts=2, max_hours=1.0)
    stats = json.loads((final / "train_stats.json").read_text(encoding="utf-8"))
    assert stats["steps"] == 2 and stats["prompts"] == 2 and stats["stopped_by_time"] is False
    summary = real_rlvr._train_summary(stats)
    assert summary["reward_first"] is not None and summary["zero_spread_share"] is not None
