"""Step 2 of the parked plan: SFT on the teacher's pre-labels, checked on the 5 M1 agreements.

notebooks/05_real_sft.ipynb calls these in order, with configs/real_sft.yaml. The data come from
the Hub: agreement text from src/discover.py and teacher records from src/teacher.py. Training,
merging and scoring reuse the small loop's subprocess, merge and runner code. Every stage
resumes from the Hub, and finished work is skipped.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .baselines import load_documents
from .chunking import chunk_text
from .gold import clean_source
from .parsing import norm_ws
from .platform import Env, pull, push_file, push_folder, start_sync
from .prompts import REPO_ROOT, credit_clauses
from .score import load_gold, report, score
from .small_loop import SCRATCH, _hub_args, _stats, _trained, check_losses, launch, run_model, \
    usable_preds
from .teacher import load_cfg as teacher_cfg
from .teacher import load_docs, load_records
from .train import encode, load_cfg as _load_cfg

CONFIG = REPO_ROOT / "configs" / "real_sft.yaml"
GOLD_M3 = REPO_ROOT / "reports" / "gold_m3.json"


def load_cfg() -> dict:
    return _load_cfg(CONFIG)


# ----------------------------------------------------------------------------- data

def answer_text(kept: dict[str, list[str]], chunk_norm: str, clauses: list[str]) -> str:
    """Every clause key, each with its quotes in the order they appear in the chunk."""
    return json.dumps({c: sorted(kept.get(c, []), key=lambda q: chunk_norm.find(norm_ws(q)))
                       for c in clauses}, ensure_ascii=False, indent=2)


def build_examples(records: list[dict], texts: dict[str, str], train_ids: list[str],
                   empty_share: float, seed: int) -> tuple[list[dict], dict]:
    """One example per train chunk the teacher kept, with empty chunks capped at `empty_share`.

    The chunks are cut exactly as the teacher saw them, and every kept quote must be in its
    chunk. Test and left-out chunks never enter.
    """
    clauses, tcfg = credit_clauses(), teacher_cfg()
    train = set(train_ids)
    chunks = {c.chunk_id: c for d in sorted(train) for c in
              chunk_text(d, clean_source(texts[d]), tcfg["chunk_chars"], tcfg["overlap"])}
    with_quotes, empty = [], []
    skipped = {"not train": 0, "left out": 0}
    for r in sorted(records, key=lambda r: r["chunk_id"]):
        if r["doc_id"] not in train:
            skipped["not train"] += 1
            continue
        if r["left_out"]:
            skipped["left out"] += 1
            continue
        text = chunks[r["chunk_id"]].text
        norm = norm_ws(text)
        missing = [q for qs in r["kept"].values() for q in qs if norm_ws(q) not in norm]
        if missing:
            raise ValueError(f"{r['chunk_id']}: a kept quote is not in the chunk: {missing[0][:80]}")
        ex = {"id": r["chunk_id"], "excerpt": text, "answer": answer_text(r["kept"], norm, clauses)}
        (with_quotes if r["kept"] else empty).append(ex)
    rng = random.Random(seed)
    n_empty = min(len(empty), math.floor(len(with_quotes) * empty_share / (1 - empty_share)))
    examples = with_quotes + rng.sample(empty, n_empty)
    rng.shuffle(examples)
    return examples, {"examples": len(examples), "with_quotes": len(with_quotes),
                      "empty_used": n_empty, "empty_available": len(empty),
                      "train_agreements": len(train), **{f"skipped_{k.replace(' ', '_')}": v
                                                          for k, v in skipped.items()}}


def token_lengths(examples: list[dict], base: str, max_len: int) -> list[int | None]:
    """Prompt plus answer tokens, as training encodes them; None if over the limit."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    return [len(e["input_ids"]) if (e := encode(tok, x, max_len)) else None for x in examples]


def _save(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def build_data(env: Env, cfg: dict) -> tuple[Path, dict]:
    """train.jsonl, short.jsonl (the longest examples that fit) and info.json, on the Hub."""
    out = env.paths.work / cfg["hub_prefix"] / "data"
    if env.data_repo:
        pull(env.data_repo, f"{cfg['hub_prefix']}/data", env.paths.work)
    if (out / "info.json").exists():
        return out, json.loads((out / "info.json").read_text("utf-8"))
    d = cfg["data"]
    if env.data_repo:
        pull(env.data_repo, f"{d['collect_prefix']}/text", env.paths.work)
        pull(env.data_repo, f"{d['prelabel_prefix']}/records", env.paths.work)
    texts = load_docs(env.paths.work / d["collect_prefix"] / "text", REPO_ROOT / d["manifest"])
    records = load_records(env.paths.work / d["prelabel_prefix"])
    parts = json.loads((REPO_ROOT / d["split"]).read_text("utf-8"))
    examples, info = build_examples(records, texts, parts["train"], d["empty_share"], cfg["seed"])
    lengths = token_lengths(examples, cfg["base_model"], cfg["sft"]["max_seq_length"])
    fit = [(n, x) for n, x in zip(lengths, examples) if n is not None]
    n_short = cfg["short_test"]["steps"] * cfg["sft"]["grad_accum"]
    short = [x for _, x in sorted(fit, key=lambda t: -t[0])[:n_short]]
    info.update(over_length=len(examples) - len(fit), tokens=sum(n for n, _ in fit),
                max_tokens=max(n for n, _ in fit), short_tokens=sum(sorted(
                    (n for n, _ in fit), reverse=True)[:n_short]))
    out.mkdir(parents=True, exist_ok=True)
    _save(out / "train.jsonl", examples)
    _save(out / "short.jsonl", short)
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if env.data_repo:
        push_folder(env.data_repo, out, f"{cfg['hub_prefix']}/data", message="sft data")
    print(json.dumps(info), flush=True)
    return out, info


# ------------------------------------------------------------------------- training

def plan_steps(info: dict, stats: dict, cfg: dict) -> dict:
    """Steps for the full run: the whole epoch, or fewer if it would pass `max_train_hours`."""
    sft, t = cfg["sft"], cfg["short_test"]
    seconds_per_token = stats["seconds_per_step"] / (info["short_tokens"] / t["steps"])
    full_steps = math.ceil((info["examples"] - info["over_length"]) * sft["epochs"] /
                           sft["grad_accum"])
    hours = seconds_per_token * info["tokens"] * sft["epochs"] / 3600
    steps = full_steps if hours <= t["max_train_hours"] else \
        math.floor(full_steps * t["max_train_hours"] / hours)
    return {"seconds_per_step_longest": stats["seconds_per_step"],
            "peak_gpu_mib": stats["peak_gpu_mib"], "projected_hours": round(hours, 2),
            "full_steps": full_steps, "max_steps": steps}


def short_test(env: Env, cfg: dict, data: Path, info: dict) -> dict:
    """A few steps on the longest examples: memory at its worst, and the pace for the plan."""
    path = env.paths.work / cfg["hub_prefix"] / "short_test.json"
    if env.data_repo:
        pull(env.data_repo, f"{cfg['hub_prefix']}/short_test", SCRATCH)
    pulled = SCRATCH / cfg["hub_prefix"] / "short_test" / "short_test.json"
    if pulled.exists():
        return json.loads(pulled.read_text(encoding="utf-8"))
    out = SCRATCH / "short_real"
    launch("src.train", ["--config", CONFIG, "--data", data / "short.jsonl", "--out", out,
                         "--max-steps", cfg["short_test"]["steps"]], SCRATCH / "logs" / "short_real.log")
    stats = _stats(out / "final")
    check_losses(stats)
    result = plan_steps(info, stats, cfg)
    print(json.dumps(result), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if env.data_repo:
        push_file(env.data_repo, path, f"{cfg['hub_prefix']}/short_test/short_test.json")
    return result


def train(env: Env, cfg: dict, data: Path, max_steps: int) -> Path:
    out, prefix = _trained(env, cfg, "sft_real")
    if not (out / "final" / "adapter_config.json").exists():
        launch("src.train", ["--config", CONFIG, "--data", data / "train.jsonl", "--out", out,
                             "--max-steps", max_steps, *_hub_args(env, prefix)],
               SCRATCH / "logs" / "sft_real.log")
    return out / "final"


# -------------------------------------------------------------------------- scoring

def evaluate(env: Env, cfg: dict, adapter: Path) -> dict[str, dict]:
    """Model A and the new SFT on the 5 M1 agreements, decoded on the chunks the student saw."""
    root = env.paths.work / cfg["hub_prefix"] / "runs" / "m1"
    hub_path = f"{cfg['hub_prefix']}/runs/m1"
    if env.data_repo:
        pull(env.data_repo, hub_path, env.paths.work)
    raw = load_documents(env, cfg)
    docs = {d: clean_source(t) for d, t in raw.items()}
    gold, clauses = load_gold(GOLD_M3), credit_clauses()
    sched = start_sync(env.data_repo, root, hub_path, every_minutes=cfg["sync_every_minutes"]) \
        if env.data_repo else None
    out = {}
    try:
        for label, model in (("Model A", cfg["base_model"]), ("SFT", adapter)):
            run_dir = root / label.lower().replace(" ", "_")
            run_model(cfg, run_dir, model, docs)
            out[label] = score(gold, usable_preds(run_dir, clauses), clauses, raw)
    finally:
        if sched is not None:
            sched.trigger().result()
            sched.stop()
    return out


def write_report(env: Env, cfg: dict, info: dict, short: dict, adapter: Path,
                 results: dict[str, dict]) -> str:
    """reports/real_sft.md from what ran. It lives on the Hub until it is committed."""
    s, sft = _stats(adapter), cfg["sft"]
    losses = [row["loss"] for row in s["log"] if "loss" in row]
    raw = json.loads(GOLD_M3.read_text("utf-8"))
    L = ["# SFT on the teacher's pre-labels", "",
         f"QLoRA on `{cfg['base_model']}`, trained on teacher labels of the "
         f"{info['train_agreements']} train agreements and scored on the 5 M1 agreements. "
         "Training and scoring use 24,000-character chunks with 4,000 overlap and page numbers "
         "removed.", "",
         "## Data", "",
         "| examples | with quotes | empty, used of available | over the length limit | "
         "left-out chunks skipped | tokens | longest |", "|---|---|---|---|---|---|---|",
         f"| {info['examples']} | {info['with_quotes']} | {info['empty_used']} of "
         f"{info['empty_available']} | {info['over_length']} | {info['skipped_left_out']} | "
         f"{info['tokens']:,} | {info['max_tokens']:,} |", "",
         "## Training, measured", "",
         f"Short test on the {cfg['short_test']['steps'] * sft['grad_accum']} longest examples: "
         f"{short['seconds_per_step_longest']} seconds per step, peak {short['peak_gpu_mib']} MiB, "
         f"projected {short['projected_hours']} hours for one epoch, so {short['max_steps']} of "
         f"{short['full_steps']} steps.", "",
         "| steps | seconds per step | peak GPU MiB | first loss | last loss |", "|---|---|---|---|---|",
         f"| {s['steps']} | {s['seconds_per_step']} | {s['peak_gpu_mib']} | "
         f"{losses[0] if losses else '-'} | {losses[-1] if losses else '-'} |", "",
         "## The 5 M1 agreements", "",
         "Five agreements cannot separate these models, and they shaped the labeling rules. "
         "Model A ran again on these chunks, so both rows are comparable.", "",
         report(load_gold(GOLD_M3), results, credit_clauses(), raw["labeled_by"]).split("\n", 2)[2]]
    md = "\n".join(L)
    path = env.paths.work / cfg["hub_prefix"] / "real_sft.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    if env.data_repo:
        push_file(env.data_repo, path, f"{cfg['hub_prefix']}/real_sft.md", message="report")
    print(md)
    return md
