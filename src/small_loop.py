"""The small SFT and RLVR loop, stage by stage (notebooks/04_small_loop.ipynb).

Training runs in its own venv on one GPU, as a subprocess. Scoring merges each adapter into a
full model and sends it through the M3 runner, so it is decoded exactly as Model A was. Every
stage can restart: adapters and run records live on the Hub, and finished work is skipped.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .baselines import RUN_KEYS, load_documents
from .chunking import chunk_text
from .cuad import download_cuad, load_contracts
from .parsing import OK_STATUSES
from .platform import Env, pull, push_file, push_folder, start_sync
from .prompts import REPO_ROOT, credit_clauses
from .runner import (DEFAULT_CHAIN, DEFAULT_LADDER, LADDER_CHAIN, load_chunks, load_record,
                     prepare_run, run_with_fallback, stage_todo)
from .score import load_gold, preds_from_run, preds_from_sheet, report, score
from .synth import SET_NAMES, build_sets, filler_groups, load_bank, load_set, save_sets

SCRATCH = Path(tempfile.gettempdir()) / "small_loop"     # weights stay out of /kaggle/working
TRAIN_VENV = Path(os.environ.get("CREDIT_TRAIN_VENV", "/tmp/train-env"))
GOLD_M3 = REPO_ROOT / "reports" / "gold_m3.json"
REVIEW_SHEET = REPO_ROOT / "reports" / "review_sheet.csv"
LABELED_BY_GENERATOR = "the synthetic generator, so these answers are exact"


def train_python(venv: Path = TRAIN_VENV) -> str:
    py = venv / "bin" / "python"
    return str(py) if py.exists() else sys.executable


def launch(module: str, args: list, log: Path) -> None:
    """Run a training module on one GPU, streaming its output to the notebook and a log file."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    stream([train_python(), "-m", module, *map(str, args)], env, log)


def stream(cmd: list[str], env: dict, log: Path) -> None:
    """Run a subprocess from the repo root, copying its output to the notebook and `log`."""
    log.parent.mkdir(parents=True, exist_ok=True)
    module = cmd[2] if len(cmd) > 2 and cmd[1] == "-m" else cmd[0]
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                lf.write(line)
                print(line, end="", flush=True)
            code = proc.wait()
        except KeyboardInterrupt:           # a stopped cell must free the GPU too
            proc.terminate()
            proc.wait()
            raise
    if code:
        raise RuntimeError(f"{module} failed (exit {code}); see {log}")


# ----------------------------------------------------------------------------- data

def build_data(env: Env, cfg: dict) -> Path:
    """The three synthetic sets, from a fixed seed, so a restart rebuilds the same excerpts."""
    out = env.paths.work / cfg["hub_prefix"] / "data"
    if all((out / f"{n}.jsonl").exists() for n in SET_NAMES):
        return out
    cuad = download_cuad(SCRATCH / "cuad")
    titles = json.loads((REPO_ROOT / "data" / "split_manifest.json").read_text("utf-8"))["train"]
    groups = filler_groups(load_contracts(cuad), titles)
    save_sets(build_sets(load_bank(), groups, cfg["synth"], cfg["seed"]), out)
    if env.data_repo:
        push_folder(env.data_repo, out, f"{cfg['hub_prefix']}/data", message="synthetic sets")
    return out


# ------------------------------------------------------------------------- training

def _stats(adapter: Path) -> dict:
    return json.loads((adapter / "train_stats.json").read_text(encoding="utf-8"))


def check_losses(stats: dict) -> None:
    """Stop before hours of training if fp16 has already turned the loss into NaN or infinity."""
    losses = [float(row[k]) for row in stats["log"] for k in ("loss", "train_loss") if k in row]
    if not losses or not all(math.isfinite(x) for x in losses):
        raise RuntimeError(f"{stats['stage']} loss is not finite: {losses}")


def short_test(env: Env, cfg: dict, data: Path) -> dict:
    """A few real steps of each stage: seconds per step, peak memory, and the RLVR size to use."""
    path = env.paths.work / cfg["hub_prefix"] / "short_test.json"
    if env.data_repo:
        pull(env.data_repo, f"{cfg['hub_prefix']}/short_test", SCRATCH)
    pulled = SCRATCH / cfg["hub_prefix"] / "short_test" / "short_test.json"
    if pulled.exists():
        return json.loads(pulled.read_text(encoding="utf-8"))
    t = cfg["short_test"]
    sft_out, rlvr_out = SCRATCH / "short_sft", SCRATCH / "short_rlvr"
    launch("src.train", ["--data", data / "train.jsonl", "--out", sft_out,
                         "--max-steps", t["sft_steps"]], SCRATCH / "logs" / "short_sft.log")
    launch("src.rlvr", ["--data", data / "rlvr.jsonl", "--adapter", sft_out / "final",
                        "--out", rlvr_out, "--max-steps", t["rlvr_steps"]],
           SCRATCH / "logs" / "short_rlvr.log")
    sft, rlvr = _stats(sft_out / "final"), _stats(rlvr_out / "final")
    check_losses(sft)
    check_losses(rlvr)
    planned = cfg["rlvr"]["prompts"]
    hours = (rlvr["seconds_per_step"] or 0) * planned / 3600
    result = {"sft_seconds_per_step": sft["seconds_per_step"],
              "rlvr_seconds_per_step": rlvr["seconds_per_step"],
              "sft_peak_gpu_mib": sft["peak_gpu_mib"], "rlvr_peak_gpu_mib": rlvr["peak_gpu_mib"],
              "rlvr_projected_hours": round(hours, 2),
              "rlvr_prompts": planned if hours <= t["max_rlvr_hours"] else t["fewer_prompts"]}
    print(json.dumps(result), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if env.data_repo:
        push_file(env.data_repo, path, f"{cfg['hub_prefix']}/short_test/short_test.json")
    return result


def _trained(env: Env, cfg: dict, stage: str) -> tuple[Path, str]:
    """Local folder for a stage, with whatever the Hub already has for it."""
    prefix = f"{cfg['hub_prefix']}/{stage}"
    if env.model_repo:
        pull(env.model_repo, prefix, SCRATCH, repo_type="model")
    return SCRATCH / prefix, prefix


def _hub_args(env: Env, prefix: str) -> list[str]:
    return ["--hub-repo", env.model_repo, "--hub-prefix", prefix] if env.model_repo else []


def train_sft(env: Env, cfg: dict, data: Path) -> Path:
    out, prefix = _trained(env, cfg, "sft")
    if not (out / "final" / "adapter_config.json").exists():
        launch("src.train", ["--data", data / "train.jsonl", "--out", out,
                             *_hub_args(env, prefix)], SCRATCH / "logs" / "sft.log")
    return out / "final"


def train_rlvr(env: Env, cfg: dict, data: Path, sft: Path, prompts: int) -> Path:
    out, prefix = _trained(env, cfg, "rlvr")
    if not (out / "final" / "adapter_config.json").exists():
        launch("src.rlvr", ["--data", data / "rlvr.jsonl", "--adapter", sft, "--out", out,
                            "--prompts", prompts, *_hub_args(env, prefix)],
               SCRATCH / "logs" / "rlvr.log")
    return out / "final"


def merged_path(adapter: Path) -> Path:
    return SCRATCH / "merged" / adapter.parent.name


def merge_adapter(adapter: Path) -> None:
    """Build the full model for the runner, once, in the scratch folder."""
    out = merged_path(adapter)
    if not (out / "config.json").exists():
        launch("src.train", ["--merge", "--adapter", adapter, "--out", out],
               SCRATCH / "logs" / f"merge_{adapter.parent.name}.log")


# -------------------------------------------------------------------------- scoring

def scoring_set(env: Env, cfg: dict, name: str, data: Path) -> tuple[dict, dict[str, str], str]:
    """(gold, source text per document, who labeled it) for `held_out` or `real`."""
    if name == "held_out":
        examples = load_set(data / "held_out.jsonl")
        return ({x["id"]: x["gold"] for x in examples}, {x["id"]: x["excerpt"] for x in examples},
                LABELED_BY_GENERATOR)
    raw = json.loads(GOLD_M3.read_text(encoding="utf-8"))
    return load_gold(GOLD_M3), load_documents(env, cfg), raw["labeled_by"]


def run_model(cfg: dict, run_dir: Path, model: str | Path, docs: dict[str, str]) -> None:
    """Decode every document with the M3 runner. Finished chunks are skipped.

    `model` is a Hub id, or an adapter folder that is merged only if there is work to do.
    """
    adapter = model if isinstance(model, Path) else None
    chunks = [c for d, t in docs.items()
              for c in chunk_text(d, t, cfg["chunk_chars"], cfg["overlap"])]
    run_cfg = {k: cfg[k] for k in RUN_KEYS}
    run_cfg.update(model=str(merged_path(adapter) if adapter else model), prompt="credit_schema",
                   docs=list(docs))
    prepare_run(run_dir, run_cfg, chunks)
    ladder = tuple(cfg.get("ladder", DEFAULT_LADDER))
    for stage, chain in (("primary", DEFAULT_CHAIN), ("ladder", LADDER_CHAIN)):
        if stage_todo(run_dir, chunks, stage, ladder):
            if adapter:
                merge_adapter(adapter)
            run_with_fallback(run_dir, stage, chain)


def usable_preds(run_dir: Path, clauses: list[str]) -> dict:
    """Quotes per document, leaving out documents where no chunk gave a usable decode."""
    ok = set()
    for c in load_chunks(run_dir):
        rec = load_record(run_dir, c.chunk_id)
        if rec is not None and rec["final_status"] in OK_STATUSES:
            ok.add(c.doc_id)
    return {d: q for d, q in preds_from_run(run_dir, clauses).items() if d in ok}


def score_models(env: Env, cfg: dict, name: str, models: dict[str, str | Path],
                 data: Path) -> dict[str, dict]:
    """Decode and score each model on one set. The model "m3" is read from the review sheet."""
    root = env.paths.work / cfg["hub_prefix"] / "runs" / name
    hub_path = f"{cfg['hub_prefix']}/runs/{name}"
    if env.data_repo:
        pull(env.data_repo, hub_path, env.paths.work)
    gold, sources, _ = scoring_set(env, cfg, name, data)
    clauses = credit_clauses()
    sched = None
    if env.data_repo:
        sched = start_sync(env.data_repo, root, hub_path,
                           every_minutes=cfg.get("sync_every_minutes", 3))
    out = {}
    try:
        for label, model in models.items():
            if model == "m3":
                out[label] = score(gold, preds_from_sheet(REVIEW_SHEET, "c"), clauses, sources)
                continue
            run_dir = root / label.lower().replace(" ", "_")
            run_model(cfg, run_dir, model, sources)
            out[label] = score(gold, usable_preds(run_dir, clauses), clauses, sources)
    finally:
        if sched is not None:
            sched.trigger().result()
            sched.stop()
    return out


def gate(held_out: dict[str, dict], cfg: dict) -> bool:
    """RLVR goes ahead only if SFT clearly beats Model A on wording neither has seen."""
    gain = held_out["SFT"]["mean_reward"] - held_out["Model A"]["mean_reward"]
    passed = gain >= cfg["gate_min_gain"]
    print(f"gate: SFT {held_out['SFT']['mean_reward']:.3f}, Model A "
          f"{held_out['Model A']['mean_reward']:.3f}, gain {gain:+.3f}, needs "
          f"{cfg['gate_min_gain']:+.2f}: {'pass' if passed else 'FAIL, RLVR skipped'}", flush=True)
    return passed


def write_report(env: Env, cfg: dict, data: Path, held_out: dict, real: dict | None,
                 short: dict, adapters: dict[str, Path]) -> str:
    """reports/small_loop.md from what actually ran. It lives on the Hub until it is committed."""
    clauses = credit_clauses()
    parts = ["# Small SFT and RLVR loop", "",
             "Model A, SFT and RLVR on the synthetic held-out set"
             + (", and on the 5 M1 agreements." if real else ". RLVR did not run: see the gate."),
             "", "## Held-out synthetic excerpts", "",
             report(*scoring_set(env, cfg, "held_out", data)[:1], held_out, clauses,
                    LABELED_BY_GENERATOR).split("\n", 2)[2]]
    if real:
        gold, _, labeled_by = scoring_set(env, cfg, "real", data)
        parts += ["## The 5 M1 agreements", "", "Five agreements cannot separate these models, "
                  "and they shaped the labeling rules. Read this as a direction only.", "",
                  report(gold, real, clauses, labeled_by).split("\n", 2)[2]]
    parts += ["## Training, measured", "", "| stage | steps | seconds per step | peak GPU MiB |",
              "|---|---|---|---|"]
    for stage, adapter in adapters.items():
        s = _stats(adapter)
        parts.append(f"| {stage} | {s['steps']} | {s['seconds_per_step']} | {s['peak_gpu_mib']} |")
    parts += ["", f"Short test: `{json.dumps(short)}`", ""]
    md = "\n".join(parts)
    path = env.paths.work / cfg["hub_prefix"] / "small_loop.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    if env.data_repo:
        push_file(env.data_repo, path, f"{cfg['hub_prefix']}/small_loop.md", message="report")
    print(md)
    return md
