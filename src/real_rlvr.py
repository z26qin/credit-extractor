"""Step 3: RLVR on real agreements, GRPO from the step 2 adapter, checked on the 5 M1 agreements.

notebooks/06_real_rlvr.ipynb calls these in order, with configs/real_rlvr.yaml. Nobody has
labeled the train agreements, so the reward checks each quote against the teacher's kept quotes
for the same text. That can remove the student's own mistakes, such as quoting a whole
incremental section or repeating the items of a definition, but not mistakes it shares with the
teacher. Prompts are 8,000-character windows of the teacher's chunks, so 4 samples and their
backward pass fit a T4. A screening pass samples every candidate window with the SFT model and
keeps those whose rewards differ, because the small loop showed that a prompt whose samples all
score the same changes nothing. Every stage resumes from the Hub, and finished work is skipped.

On a GPU VM, `python -m src.real_rlvr [--max-rlvr-hours H]` runs the same stages in order
(scripts/nebius_rlvr.sh).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
from pathlib import Path

from .baselines import load_documents
from .chunking import Chunk, chunk_bounds, chunk_text
from .gold import clean_source
from .metrics import paired_diff
from .parsing import norm_ws
from .platform import Env, pull, push_file, push_folder, start_sync
from .prompts import REPO_ROOT, credit_clauses
from .real_loop import GOLD_M3, answer_text
from .rlvr import reward_text
from .runner import backend_python
from .score import load_gold, report, score
from .small_loop import SCRATCH, _hub_args, _stats, _trained, check_losses, launch, \
    merge_adapter, merged_path, run_model, stream, usable_preds
from .synth import load_set
from .teacher import load_cfg as teacher_cfg
from .teacher import load_docs, load_records
from .train import IM_END, load_cfg as _load_cfg

CONFIG = REPO_ROOT / "configs" / "real_rlvr.yaml"


def load_cfg() -> dict:
    return _load_cfg(CONFIG)


def _save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


# ----------------------------------------------------------------------------- data

def quote_spans(text: str, quotes: list[str]) -> list[tuple[int, int]]:
    """Where each quote sits in `text`, across any run of whitespace, as the teacher checked it."""
    spans = []
    for q in quotes:
        m = re.search(r"\s+".join(map(re.escape, q.split())), text)
        if m is None:
            raise ValueError(f"a kept quote is not in its chunk: {q[:80]}")
        spans.append(m.span())
    return spans


def windows_for(chunk: Chunk, kept: dict[str, list[str]], size: int, overlap: int) -> list[dict]:
    """Windows of one teacher chunk, each with the kept quotes that sit wholly inside it.

    A window that cuts through a kept quote is dropped, since its right answer is unclear.
    """
    placed = [(c, q, span) for c, qs in kept.items()
              for q, span in zip(qs, quote_spans(chunk.text, qs))]
    out = []
    for i, (a, b) in enumerate(chunk_bounds(len(chunk.text), size, overlap)):
        if any(s < b and e > a and not (a <= s and e <= b) for _, _, (s, e) in placed):
            continue
        refs: dict[str, list[str]] = {}
        for c, q, (s, e) in placed:
            if a <= s and e <= b:
                refs.setdefault(c, []).append(q)
        out.append({"id": f"{chunk.chunk_id}__w{i}", "doc_id": chunk.doc_id,
                    "excerpt": chunk.text[a:b], "refs": refs})
    return out


def build_windows(records: list[dict], texts: dict[str, str], train_ids: list[str],
                  w: dict) -> list[dict]:
    """Every window of every train chunk the teacher kept. Test agreements never enter."""
    tcfg = teacher_cfg()
    by_id = {r["chunk_id"]: r for r in records}
    out = []
    for d in sorted(train_ids):
        for c in chunk_text(d, clean_source(texts[d]), tcfg["chunk_chars"], tcfg["overlap"]):
            r = by_id.get(c.chunk_id)
            if r is not None and not r["left_out"]:
                out += windows_for(c, r["kept"], w["chars"], w["overlap"])
    return out


def as_example(w: dict) -> dict:
    """A window as src.rlvr reads it, with every teacher quote as a main passage."""
    return {"id": w["id"], "excerpt": w["excerpt"],
            "gold": {c: [{"text": q, "main": True} for q in qs] for c, qs in w["refs"].items()}}


def pick(windows: list[dict], train_ids: list[str], w: dict,
         seed: int) -> tuple[list[dict], list[dict], dict]:
    """The screening pool and the check set, drawn from different train agreements.

    An empty window enters only if it holds a search term, so it tests restraint where that is
    hard: near a clause, or where a sentence only uses its term.
    """
    rng = random.Random(seed)
    docs = sorted(train_ids)
    rng.shuffle(docs)
    check_docs = set(docs[:w["check_agreements"]])

    def draw(rows: list[dict], n_quotes: int, n_empty: int) -> list[dict]:
        quotes = [x for x in rows if x["refs"]]
        empty = [x for x in rows if not x["refs"]
                 and any(t in x["excerpt"] for t in w["hard_negative_terms"])]
        return (rng.sample(quotes, min(n_quotes, len(quotes)))
                + rng.sample(empty, min(n_empty, len(empty))))

    pool = draw([x for x in windows if x["doc_id"] not in check_docs],
                w["pool_quotes"], w["pool_empty"])
    check = draw([x for x in windows if x["doc_id"] in check_docs],
                 w["check_quotes"], w["check_empty"])
    info = {"windows": len(windows), "windows_with_quotes": sum(bool(x["refs"]) for x in windows),
            "check_agreements": len(check_docs), "pool": len(pool),
            "pool_empty": sum(not x["refs"] for x in pool), "check": len(check),
            "check_empty": sum(not x["refs"] for x in check)}
    return pool, check, info


def answer_tokens(windows: list[dict], base: str) -> list[int]:
    """Tokens of each window's teacher answer, as the model would have to write it."""
    from transformers import AutoTokenizer
    tok, clauses = AutoTokenizer.from_pretrained(base), credit_clauses()
    return [len(tok(answer_text(w["refs"], norm_ws(w["excerpt"]), clauses) + IM_END,
                    add_special_tokens=False)["input_ids"]) for w in windows]


def build_data(env: Env, cfg: dict) -> tuple[Path, dict]:
    """pool.jsonl (windows to screen), check.jsonl and info.json, on the Hub."""
    out = env.paths.work / cfg["hub_prefix"] / "data"
    if env.data_repo:
        pull(env.data_repo, f"{cfg['hub_prefix']}/data", env.paths.work)
    if (out / "info.json").exists():
        return out, json.loads((out / "info.json").read_text("utf-8"))
    d, w = cfg["data"], cfg["windows"]
    if env.data_repo:
        pull(env.data_repo, f"{d['collect_prefix']}/text", env.paths.work)
        pull(env.data_repo, f"{d['prelabel_prefix']}/records", env.paths.work)
    texts = load_docs(env.paths.work / d["collect_prefix"] / "text", REPO_ROOT / d["manifest"])
    records = load_records(env.paths.work / d["prelabel_prefix"])
    train_ids = json.loads((REPO_ROOT / d["split"]).read_text("utf-8"))["train"]
    windows = build_windows(records, texts, train_ids, w)
    with_quotes = [x for x in windows if x["refs"]]
    lengths = answer_tokens(with_quotes, cfg["base_model"])
    limit = cfg["rlvr"]["max_completion_length"]
    too_long = {x["id"] for x, n in zip(with_quotes, lengths) if n > limit}
    pool, check, info = pick([x for x in windows if x["id"] not in too_long], train_ids, w,
                             cfg["seed"])
    q = statistics.quantiles(lengths, n=100)
    info.update(answer_over_limit=len(too_long), answer_tokens_median=statistics.median(lengths),
                answer_tokens_p99=round(q[98]), answer_tokens_max=max(lengths))
    _save(out / "pool.jsonl", [as_example(x) for x in pool])
    _save(out / "check.jsonl", [as_example(x) for x in check])
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if env.data_repo:
        push_folder(env.data_repo, out, f"{cfg['hub_prefix']}/data", message="rlvr windows")
    print(json.dumps(info), flush=True)
    return out, info


def sft_adapter(env: Env, cfg: dict) -> Path:
    """Step 2's adapter, from the model repo."""
    prefix = f"{cfg['sft_prefix']}/{cfg['sft_stage']}/final"
    if env.model_repo:
        pull(env.model_repo, prefix, SCRATCH, repo_type="model")
    final = SCRATCH / prefix
    if not (final / "adapter_config.json").exists():
        raise FileNotFoundError(f"step 2 adapter not found at {prefix}; run notebook 05 first")
    return final


# ------------------------------------------------------------------------ screening

def run_sampler(model: Path, data: Path, out: Path, cfg: dict, n: int, temperature: float,
                log: Path) -> None:
    """src.sample in the vLLM venv, on both GPUs."""
    py = backend_python("vllm")
    if py == sys.executable:
        raise RuntimeError("no vLLM venv: sampling needs it; run the install cell again")
    s = cfg["sample"]
    cmd = [py, "-m", "src.sample", "--model", model, "--data", data, "--out", out, "--n", n,
           "--temperature", temperature, "--max-tokens", cfg["rlvr"]["max_completion_length"],
           "--tp", s["tp"], "--max-model-len", s["max_model_len"], "--batch", s["batch"]]
    stream([str(c) for c in cmd], dict(os.environ, TOKENIZERS_PARALLELISM="false"), log)


def score_samples(samples: list[dict], windows: dict[str, dict], exact_by_f1: bool) -> list[dict]:
    """The training reward of each sampled answer, and its spread per window."""
    clauses = credit_clauses()
    out = []
    for s in samples:
        x = windows[s["id"]]
        r = [reward_text(c, x["gold"], x["excerpt"], clauses, exact_by_f1)
             for c in s["completions"]]
        out.append({"id": s["id"], "empty": not x["gold"], "rewards": [round(v, 4) for v in r],
                    "mean": statistics.fmean(r), "spread": statistics.pstdev(r),
                    "truncated": s["finish"].count("length"),
                    "tokens": statistics.fmean(s["tokens"])})
    return out


def select(scored: list[dict], s: dict) -> list[str]:
    """Windows whose samples differ, widest spread first, at most `max_prompts`."""
    keep = sorted((x for x in scored if x["spread"] >= s["min_spread"]),
                  key=lambda x: (-x["spread"], x["id"]))
    return [x["id"] for x in keep[:s["max_prompts"]]]


def screen_dir(env: Env, cfg: dict) -> Path:
    return env.paths.work / cfg["hub_prefix"] / "screen"


def check_dir(env: Env, cfg: dict) -> Path:
    return env.paths.work / cfg["hub_prefix"] / "check"


def _enough(stats: dict, cfg: dict) -> dict:
    if stats["selected"] < cfg["screen"]["min_prompts"]:
        raise SystemExit(f"only {stats['selected']} windows have reward spread, under "
                         f"{cfg['screen']['min_prompts']}: RLVR would have little to learn from")
    return stats


def screen(env: Env, cfg: dict, data: Path, sft: Path) -> dict:
    """Sample each pool window as GRPO would and keep those whose rewards differ (rlvr.jsonl).

    SFT's greedy answers on the check set are taken here too, while its merged model is on disk.
    """
    root, check = screen_dir(env, cfg), check_dir(env, cfg)
    prefix = cfg["hub_prefix"]
    if env.data_repo:
        pull(env.data_repo, f"{prefix}/screen", env.paths.work)
        pull(env.data_repo, f"{prefix}/check", env.paths.work)
    if (root / "screen.json").exists():
        return _enough(json.loads((root / "screen.json").read_text("utf-8")), cfg)
    rl, pool = cfg["rlvr"], load_set(data / "pool.jsonl")
    merge_adapter(sft)
    syncs = [start_sync(env.data_repo, d, f"{prefix}/{d.name}",
                        every_minutes=cfg["sync_every_minutes"])
             for d in (root, check)] if env.data_repo else []
    for d in (root, check):
        d.mkdir(parents=True, exist_ok=True)
    try:
        run_sampler(merged_path(sft), data / "pool.jsonl", root / "samples.jsonl", cfg,
                    rl["num_generations"], rl["temperature"], SCRATCH / "logs" / "screen.log")
        run_sampler(merged_path(sft), data / "check.jsonl", check / "sft.jsonl", cfg, 1, 0.0,
                    SCRATCH / "logs" / "check_sft.log")
    finally:
        for s in syncs:
            s.trigger().result()
            s.stop()
    scored = score_samples(load_set(root / "samples.jsonl"), {x["id"]: x for x in pool},
                           rl["exact_by_token_f1"])
    ids = select(scored, cfg["screen"])
    by_id = {x["id"]: x for x in pool}
    _save(root / "scored.jsonl", scored)
    _save(root / "rlvr.jsonl", [by_id[i] for i in ids])
    picked = set(ids)
    n_samples = sum(len(x["rewards"]) for x in scored)
    stats = {"screened": len(scored), "screened_empty": sum(x["empty"] for x in scored),
             "mean_reward": round(statistics.fmean(x["mean"] for x in scored), 4),
             "mean_reward_quotes": round(statistics.fmean(
                 x["mean"] for x in scored if not x["empty"]), 4),
             "mean_reward_empty": round(statistics.fmean(
                 x["mean"] for x in scored if x["empty"]), 4) if any(x["empty"] for x in scored)
             else None,
             "with_spread": sum(x["spread"] >= cfg["screen"]["min_spread"] for x in scored),
             "selected": len(ids), "selected_empty": sum(x["empty"] for x in scored
                                                         if x["id"] in picked),
             "truncated_share": round(sum(x["truncated"] for x in scored) / n_samples, 4),
             "mean_answer_tokens": round(statistics.fmean(x["tokens"] for x in scored), 1)}
    (root / "screen.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    if env.data_repo:
        push_folder(env.data_repo, root, f"{prefix}/screen", message="screening")
    shutil.rmtree(merged_path(sft), ignore_errors=True)    # 8 GB, and RLVR's merge comes later
    print(json.dumps(stats), flush=True)
    return _enough(stats, cfg)


# ------------------------------------------------------------------------- training

def plan_prompts(selected: int, stats: dict, cfg: dict) -> dict:
    """Prompts for the full run: all selected windows, or fewer if they would pass the limit.

    The short test runs the windows with the longest answers, so the projection errs long.
    """
    hours = stats["seconds_per_step"] * selected / 3600
    limit = cfg["short_test"]["max_rlvr_hours"]
    prompts = selected if hours <= limit else math.floor(selected * limit / hours)
    return {"seconds_per_step_longest": stats["seconds_per_step"],
            "peak_gpu_mib": stats["peak_gpu_mib"], "projected_hours": round(hours, 2),
            "selected": selected, "prompts": prompts}


def short_test(env: Env, cfg: dict, sft: Path) -> dict:
    """A few GRPO steps on the windows SFT answered at greatest length: memory and pace."""
    path = env.paths.work / cfg["hub_prefix"] / "short_test.json"
    if env.data_repo:
        pull(env.data_repo, f"{cfg['hub_prefix']}/short_test", SCRATCH)
    pulled = SCRATCH / cfg["hub_prefix"] / "short_test" / "short_test.json"
    if pulled.exists():
        return json.loads(pulled.read_text(encoding="utf-8"))
    root, steps = screen_dir(env, cfg), cfg["short_test"]["steps"]
    rows = load_set(root / "rlvr.jsonl")
    tokens = {x["id"]: x["tokens"] for x in load_set(root / "scored.jsonl")}
    short = SCRATCH / "short_rlvr_real" / "short.jsonl"
    _save(short, sorted(rows, key=lambda x: -tokens[x["id"]])[:steps])
    out = SCRATCH / "short_rlvr_real"
    launch("src.rlvr", ["--config", CONFIG, "--data", short, "--adapter", sft, "--out", out,
                        "--max-steps", steps], SCRATCH / "logs" / "short_rlvr_real.log")
    stats = _stats(out / "final")
    check_losses(stats)
    result = plan_prompts(len(rows), stats, cfg)
    print(json.dumps(result), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if env.data_repo:
        push_file(env.data_repo, path, f"{cfg['hub_prefix']}/short_test/short_test.json")
    return result


def train(env: Env, cfg: dict, sft: Path, prompts: int) -> Path:
    out, prefix = _trained(env, cfg, "rlvr_real")
    if not (out / "final" / "adapter_config.json").exists():
        launch("src.rlvr", ["--config", CONFIG, "--data", screen_dir(env, cfg) / "rlvr.jsonl",
                            "--adapter", sft, "--out", out, "--prompts", prompts,
                            *_hub_args(env, prefix)], SCRATCH / "logs" / "rlvr_real.log")
    return out / "final"


# -------------------------------------------------------------------------- scoring

def check_rewards(samples: Path, check_rows: list[dict], exact_by_f1: bool) -> dict:
    """Mean greedy reward on the check set, overall and for windows with and without quotes."""
    scored = score_samples(load_set(samples), {x["id"]: x for x in check_rows}, exact_by_f1)

    def mean(rows: list[dict]) -> float | None:
        return round(statistics.fmean(x["mean"] for x in rows), 4) if rows else None
    return {"windows": len(scored), "mean_reward": mean(scored),
            "mean_reward_quotes": mean([x for x in scored if not x["empty"]]),
            "mean_reward_empty": mean([x for x in scored if x["empty"]]),
            "zero_reward": sum(x["mean"] == 0 for x in scored),
            "truncated": sum(x["truncated"] for x in scored)}


def evaluate(env: Env, cfg: dict, data: Path, rlvr: Path) -> dict:
    """RLVR on the 5 M1 agreements, decoded as step 2 decoded Model A and SFT, and on the check
    set. Model A and SFT are scored from step 2's records, not decoded again."""
    prefix, sft_runs = cfg["hub_prefix"], env.paths.work / cfg["sft_prefix"] / "runs" / "m1"
    root, check = env.paths.work / prefix / "runs" / "m1", check_dir(env, cfg)
    if env.data_repo:
        pull(env.data_repo, f"{cfg['sft_prefix']}/runs/m1", env.paths.work)
        pull(env.data_repo, f"{prefix}/runs/m1", env.paths.work)
        pull(env.data_repo, f"{prefix}/check", env.paths.work)
    raw = load_documents(env, cfg)
    docs = {d: clean_source(t) for d, t in raw.items()}
    gold, clauses = load_gold(GOLD_M3), credit_clauses()
    check_rows = load_set(data / "check.jsonl")
    syncs = [start_sync(env.data_repo, d, p, every_minutes=cfg["sync_every_minutes"])
             for d, p in ((root, f"{prefix}/runs/m1"), (check, f"{prefix}/check"))] \
        if env.data_repo else []
    try:
        run_model(cfg, root / "rlvr", rlvr, docs)
        merge_adapter(rlvr)
        run_sampler(merged_path(rlvr), data / "check.jsonl", check / "rlvr.jsonl", cfg, 1, 0.0,
                    SCRATCH / "logs" / "check_rlvr.log")
    finally:
        for s in syncs:
            s.trigger().result()
            s.stop()
    m1 = {label: score(gold, usable_preds(run_dir, clauses), clauses, raw)
          for label, run_dir in (("Model A", sft_runs / "model_a"), ("SFT", sft_runs / "sft"),
                                 ("RLVR", root / "rlvr"))}
    f1 = cfg["rlvr"]["exact_by_token_f1"]
    return {"m1": m1, "check": {name: check_rewards(check / f"{name.lower()}.jsonl", check_rows, f1)
                                for name in ("SFT", "RLVR")}}


def _train_summary(stats: dict) -> dict:
    """First and last logged rewards, and how often a group of samples had no spread."""
    logs = [r for r in stats["log"] if "reward" in r]
    head, tail = logs[:3], logs[-3:]

    def mean(rows: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rows if key in r]
        return round(statistics.fmean(vals), 4) if vals else None
    return {"reward_first": mean(head, "reward"), "reward_last": mean(tail, "reward"),
            "zero_spread_share": mean(logs, "frac_reward_zero_std"),
            "completion_tokens": mean(logs, "completions/mean_length")}


def write_report(env: Env, cfg: dict, info: dict, screened: dict, short: dict, rlvr: Path,
                 results: dict) -> str:
    """reports/real_rlvr.md from what ran. It lives on the Hub until it is committed."""
    s, t = _stats(rlvr), _train_summary(_stats(rlvr))
    w, rl, m1 = cfg["windows"], cfg["rlvr"], results["m1"]
    raw = json.loads(GOLD_M3.read_text("utf-8"))
    L = ["# RLVR on the teacher's pre-labels", "",
         f"GRPO on `{cfg['base_model']}` from the step 2 adapter, with the teacher's kept quotes as "
         f"the answers. Prompts are {w['chars']:,}-character windows of the teacher's train chunks. "
         "Scored on the 5 M1 agreements on step 2's chunks, and on a check set of windows from "
         f"{info['check_agreements']} train agreements that screening and training never saw.", "",
         "## Data", "",
         "| windows | with quotes | answer over the limit | pool | of which empty | check | "
         "of which empty | answer tokens, median / p99 |", "|---|---|---|---|---|---|---|---|",
         f"| {info['windows']:,} | {info['windows_with_quotes']:,} | {info['answer_over_limit']} | "
         f"{info['pool']} | {info['pool_empty']} | {info['check']} | {info['check_empty']} | "
         f"{info['answer_tokens_median']} / {info['answer_tokens_p99']} |", "",
         "## Screening, measured", "",
         f"SFT sampled each pool window {rl['num_generations']} times at temperature "
         f"{rl['temperature']}, as GRPO does, and each answer got the training reward.", "",
         "| screened | mean reward | with quotes | empty | with spread | selected | of which empty "
         "| samples cut at the limit |", "|---|---|---|---|---|---|---|---|",
         f"| {screened['screened']} | {screened['mean_reward']} | {screened['mean_reward_quotes']} | "
         f"{screened['mean_reward_empty']} | {screened['with_spread']} | {screened['selected']} | "
         f"{screened['selected_empty']} | {screened['truncated_share']:.1%} |", "",
         "## Training, measured", "",
         f"Short test on the {cfg['short_test']['steps']} selected windows with the longest "
         f"answers: {short['seconds_per_step_longest']} seconds per prompt, peak "
         f"{short['peak_gpu_mib']} MiB, projected {short['projected_hours']} hours for all "
         f"{short['selected']}, so {short['prompts']} prompts.", "",
         "| prompts | seconds per prompt | peak GPU MiB | reward, first 3 logs | last 3 logs | "
         "groups with no spread | answer tokens |", "|---|---|---|---|---|---|---|",
         f"| {s['steps']} | {s['seconds_per_step']} | {s['peak_gpu_mib']} | {t['reward_first']} | "
         f"{t['reward_last']} | {t['zero_spread_share']} | {t['completion_tokens']} |", "",
         "## Check set, greedy", "",
         "| model | mean reward | with quotes | empty | reward 0 | cut at the limit |",
         "|---|---|---|---|---|---|"]
    for name, c in results["check"].items():
        L.append(f"| {name} | {c['mean_reward']} | {c['mean_reward_quotes']} | "
                 f"{c['mean_reward_empty']} | {c['zero_reward']} | {c['truncated']} |")
    L += ["", "## The 5 M1 agreements", "",
          "Five agreements cannot separate these models, and they shaped the labeling rules. "
          "Model A and SFT are step 2's decodes; RLVR was decoded the same way.", "",
          report(load_gold(GOLD_M3), m1, credit_clauses(), raw["labeled_by"]).split("\n", 2)[2],
          "Paired difference from SFT:", "",
          "| model | Found F1 difference (95% CI) | Strict F1 difference (95% CI) |",
          "|---|---|---|",
          "| RLVR | " + " | ".join(
              "{:+.3f} ({:+.3f} to {:+.3f})".format(*paired_diff(
                  m1["RLVR"]["counts"], m1["SFT"]["counts"], p)) for p in ("found_", "main_"))
          + " |", ""]
    md = "\n".join(L)
    path = env.paths.work / cfg["hub_prefix"] / "real_rlvr.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    if env.data_repo:
        push_file(env.data_repo, path, f"{cfg['hub_prefix']}/real_rlvr.md", message="report")
    print(md)
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step 3, every stage in order, as the notebook runs.")
    ap.add_argument("--max-rlvr-hours", type=float, help="override short_test.max_rlvr_hours")
    a = ap.parse_args(argv)
    from .platform import setup
    env = setup(REPO_ROOT)
    if not (env.data_repo and env.gpus):
        raise SystemExit("needs HF_TOKEN and a GPU")
    cfg = load_cfg()
    if a.max_rlvr_hours:
        cfg["short_test"]["max_rlvr_hours"] = a.max_rlvr_hours
    data, info = build_data(env, cfg)
    sft = sft_adapter(env, cfg)
    screened = screen(env, cfg, data, sft)
    short = short_test(env, cfg, sft)
    adapter = train(env, cfg, sft, short["prompts"])
    results = evaluate(env, cfg, data, adapter)
    write_report(env, cfg, info, screened, short, adapter, results)
    print(f"session wall time: {env.session_minutes():.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
