"""Milestone 0 orchestration: Model B on a CUAD test subset. Used by notebooks/00_reproduce."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import cuad
from .evaluate import evaluate_run, reproduce_report
from .metrics import verbatim_flags
from .parsing import json_error, parse_json_object
from .platform import Env, pull, push_folder, start_sync
from .prompts import REPO_ROOT, load_cuad_clauses
from .runner import (DEFAULT_CHAIN, LADDER_CHAIN, load_chunks, load_config, load_record,
                     prepare_run, run_with_fallback, save_record)


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


def _attempt_row(run: str, label: str, a: dict, text: str) -> dict:
    d = parse_json_object(a["text"])
    quotes = [q for v in (d or {}).values() if isinstance(v, list) for q in v
              if isinstance(q, str)]
    flags = verbatim_flags(quotes, text)
    return {"run": run, "ladder": label, "repeat_penalty": a["repeat_penalty"],
            "backend": a["backend"], "status": a["status"], "finish_reason": a["finish_reason"],
            "output_tokens": a["n_output_tokens"], "latency_s": round(a["latency_s"], 1),
            "quotes": len(quotes), "verbatim": round(sum(flags) / len(flags), 4) if flags else None}


def ladder_check(env: Env, plan: dict, cfg: dict, chain=(("hf", 0),)) -> list[dict]:
    """Replay M0's failed chunks through the transformers ladder, in separate run dirs.

    M0's retries fell back to vLLM after transformers ran out of memory. This proves the
    fixed path on the real hardware and compares the two penalties on the same chunks. The
    M0 run dirs are only read, so M0's reported results don't change.
    """
    root = plan["root"] / "ladder_check"
    rows, errors = [], {}
    for name, rd in plan["run_dirs"].items():
        run_cfg = {k: v for k, v in load_config(rd).items()
                   if k not in ("chunks_sha256", "n_chunks")}
        chunks = load_chunks(rd)
        failed = [c for c in chunks if load_record(rd, c.chunk_id)["attempts"][0]["status"] != "ok"]
        if not failed:
            continue
        new = prepare_run(root / name, run_cfg, failed)
        for c in failed:
            if load_record(new, c.chunk_id) is None:
                rec = load_record(rd, c.chunk_id)
                first = rec["attempts"][0]
                save_record(new, dict(rec, attempts=[first], final_status=first["status"],
                                      final_repeat_penalty=first["repeat_penalty"]))
        run_with_fallback(new, "ladder", chain)
        for c in failed:
            old, cur = load_record(rd, c.chunk_id), load_record(new, c.chunk_id)
            errors[f"{name}:{c.chunk_id}"] = json_error(old["attempts"][0]["text"])
            rows += [_attempt_row(name, "M0 (vLLM full-context)", a, c.text)
                     for a in old["attempts"]]
            rows += [_attempt_row(name, "check (HF 64-token window)", a, c.text)
                     for a in cur["attempts"][1:]]
    root.mkdir(parents=True, exist_ok=True)
    (root / "ladder_check.json").write_text(json.dumps({"rows": rows, "primary_json_errors": errors},
                                                       indent=2), encoding="utf-8")
    if env.data_repo:
        push_folder(env.data_repo, root, f"{cfg['hub_prefix']}/ladder_check", "m0 ladder check")
    for k, e in errors.items():
        print(k, json.dumps(e, indent=1, ensure_ascii=False))
    return rows
