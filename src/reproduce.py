"""Milestone 0 orchestration: Model B on a CUAD test subset. Used by notebooks/00_reproduce."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import cuad
from .evaluate import evaluate_run, reproduce_report
from .platform import Env, pull, push_folder, start_sync
from .prompts import REPO_ROOT, load_cuad_clauses
from .runner import DEFAULT_CHAIN, LADDER_CHAIN, prepare_run, run_with_fallback


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or REPO_ROOT / "configs" / "reproduce.yaml").read_text())


def prepare(env: Env, cfg: dict) -> dict:
    """Download CUAD, pull finished outputs from the Hub, and build/validate the run dirs."""
    root = env.paths.work / cfg["hub_prefix"]
    if env.data_repo:
        n = pull(env.data_repo, cfg["hub_prefix"], env.paths.work)
        print(f"pulled {n} existing files from {env.data_repo}/{cfg['hub_prefix']}")
    contracts = cuad.load_contracts(cuad.download_cuad(env.paths.work / "cuad"))
    titles = cuad.sample_titles(cuad.test_titles(), cfg["n_contracts"])
    clauses = load_cuad_clauses()
    run_dirs = {}
    for r in cfg["runs"]:
        chunks = cuad.build_chunks(contracts, titles, r["chunk_chars"], r["overlap"])
        run_cfg = {k: cfg[k] for k in ("model", "prompt", "max_tokens", "ladder",
                                       "batch_size", "max_model_len_cap", "vllm_kwargs")}
        run_cfg.update(chunk_chars=r["chunk_chars"], overlap=r["overlap"], titles=titles)
        run_dirs[r["name"]] = prepare_run(root / r["name"], run_cfg, chunks)
        print(f"{r['name']}: {len(chunks)} chunks over {len(titles)} contracts")
    return {"root": root, "run_dirs": run_dirs, "titles": titles, "clauses": clauses,
            "contracts": contracts}


def run(env: Env, plan: dict, cfg: dict, chain=DEFAULT_CHAIN,
        ladder_chain=LADDER_CHAIN) -> dict:
    """Primary pass then retry ladder for every run; outputs sync to the Hub as they land."""
    sched = None
    if env.data_repo:
        sched = start_sync(env.data_repo, plan["root"], cfg["hub_prefix"],
                           every_minutes=cfg.get("sync_every_minutes", 3))
    used = {}
    try:
        for name, rd in plan["run_dirs"].items():
            used[name] = {"primary": run_with_fallback(rd, "primary", chain),
                          "ladder": run_with_fallback(rd, "ladder", ladder_chain)}
    finally:
        if sched is not None:
            sched.trigger().result()        # final push even if a stage raised
            sched.stop()
    return used


def report(env: Env, plan: dict, cfg: dict) -> dict:
    texts = {t: cuad.contract_text(plan["contracts"][t]) for t in plan["titles"]}
    gold = {t: cuad.gold_for(plan["contracts"][t], plan["clauses"]) for t in plan["titles"]}
    results = {name: evaluate_run(rd, plan["clauses"], texts, gold)
               for name, rd in plan["run_dirs"].items()}
    meta = {"platform": env.platform, "gpus": [g["name"] for g in env.gpus],
            "n_contracts": len(plan["titles"]),
            "session_minutes": round(env.session_minutes(), 1)}
    md = reproduce_report(results, meta)
    (plan["root"] / "report.md").write_text(md, encoding="utf-8")
    (plan["root"] / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if env.data_repo:
        push_folder(env.data_repo, plan["root"], cfg["hub_prefix"], "m0 report")
    print(md)
    return results
