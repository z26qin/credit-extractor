"""Milestone 3: three label-free baselines on the M1 credit agreements. Used by
notebooks/03_baselines.

Every condition decodes the same chunks. All primary passes run before any ladder, so a
slow ladder never holds back the core numbers, and the report can be written in between.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from .chunking import chunk_text
from .edgar import read_manifest
from .evaluate import evaluate_run, final_output, gpu_time, latency_rows
from .metrics import verbatim_flags
from .parsing import OK_STATUSES, norm_ws, parse_output
from .platform import Env, pull, push_folder, start_sync
from .prompts import REPO_ROOT, canon_map, clauses_for, load_credit_schema, norm_key
from .runner import (DEFAULT_CHAIN, LADDER_CHAIN, load_chunks, load_config, load_record,
                     prepare_run, run_with_fallback, stage_todo)

RUN_KEYS = ("max_tokens", "ladder", "batch_size", "max_model_len_cap", "vllm_kwargs",
            "chunk_chars", "overlap")
REVIEW_FIELDS = ("doc_id", "clause", "condition", "model", "found", "quote_no", "quote",
                 "chars", "verbatim", "chunks", "from_repaired", "correct", "note")


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or REPO_ROOT / "configs" / "baselines.yaml").read_text())


def load_documents(env: Env, cfg: dict) -> dict[str, str]:
    """M1's cleaned text from the Hub, checked against the manifest committed after review."""
    if env.data_repo:
        pull(env.data_repo, cfg["m1_prefix"], env.paths.work)
    manifest = read_manifest(REPO_ROOT / cfg["manifest"])
    docs = {}
    for doc_id, row in manifest.items():
        text = (env.paths.work / cfg["m1_prefix"] / "text" / f"{doc_id}.txt").read_text(
            encoding="utf-8")
        if len(text) != int(row["char_count"]):
            raise ValueError(f"{doc_id}: {len(text)} chars on the Hub, manifest says "
                             f"{row['char_count']}; rerun notebook 01 or update the manifest")
        docs[doc_id] = text
    return docs


def prepare(env: Env, cfg: dict) -> dict:
    """Pull finished outputs and M1 text, then build one fingerprinted run dir per condition."""
    root = env.paths.work / cfg["hub_prefix"]
    if env.data_repo:
        n = pull(env.data_repo, cfg["hub_prefix"], env.paths.work)
        print(f"pulled {n} existing files from {env.data_repo}/{cfg['hub_prefix']}")
    docs = load_documents(env, cfg)
    chunks = [c for d, t in docs.items()
              for c in chunk_text(d, t, cfg["chunk_chars"], cfg["overlap"])]
    conditions = {c["id"]: c for c in load_credit_schema()["conditions"]}
    run_dirs = {}
    for cid, cond in conditions.items():
        run_cfg = {k: cfg[k] for k in RUN_KEYS}
        run_cfg.update(model=cond["model"], prompt=cond["prompt"], docs=list(docs))
        run_dirs[cid] = prepare_run(root / cid, run_cfg, chunks)
    print(f"{len(chunks)} chunks over {len(docs)} documents, the same for conditions "
          f"{', '.join(conditions)}")
    return {"root": root, "run_dirs": run_dirs, "docs": docs, "conditions": conditions}


def run(env: Env, plan: dict, cfg: dict, stage: str, chain=None) -> dict:
    """One stage for every condition in order; outputs sync to the Hub as they land."""
    chain = chain or (DEFAULT_CHAIN if stage == "primary" else LADDER_CHAIN)
    sched = None
    if env.data_repo:
        sched = start_sync(env.data_repo, plan["root"], cfg["hub_prefix"],
                           every_minutes=cfg.get("sync_every_minutes", 3))
    try:
        return {cid: run_with_fallback(rd, stage, chain) for cid, rd in plan["run_dirs"].items()}
    finally:
        if sched is not None:
            sched.trigger().result()        # final push even if a stage raised
            sched.stop()


# ----------------------------------------------------------------------------- scoring

def merged_quotes(run_dir: Path, clauses: list[str]) -> dict:
    """{doc: {clause: [(quote, [chunk index], from_repaired)]}}, union per clause, deduplicated."""
    canon = canon_map(clauses)
    out: dict = {}
    for c in load_chunks(run_dir):
        rec = load_record(run_dir, c.chunk_id)
        doc = out.setdefault(c.doc_id, {cl: {} for cl in clauses})
        got, _ = final_output(rec, canon)
        for cl, qs in (got or {}).items():
            for q in qs:
                entry = doc[cl].setdefault(norm_ws(q), [q, [], False])
                entry[1].append(c.index)
                entry[2] = entry[2] or rec["final_status"] == "repaired"
    return {d: {cl: [tuple(e) for e in m.values()] for cl, m in doc.items()}
            for d, doc in out.items()}


def key_coverage(run_dir: Path, clauses: list[str]) -> dict:
    """How closely usable outputs follow the format: every clause key present, no others."""
    want = {norm_key(c) for c in clauses}
    n = full = extra = 0
    for c in load_chunks(run_dir):
        rec = load_record(run_dir, c.chunk_id)
        if not rec or rec["final_status"] not in OK_STATUSES:
            continue
        a = [a for a in rec["attempts"] if a["status"] in OK_STATUSES][0]
        keys = {norm_key(k) for k in parse_output(a["text"])}
        n += 1
        full += want <= keys
        extra += bool(keys - want)
    return {"usable_outputs": n, "with_all_keys": full, "with_unknown_keys": extra}


def score(plan: dict) -> dict:
    """evaluate_run for every condition, plus the extras this report needs."""
    results = {}
    for cid, rd in plan["run_dirs"].items():
        cfg = load_config(rd)
        clauses = clauses_for(cfg["prompt"])
        r = evaluate_run(rd, clauses, plan["docs"])
        chunks = load_chunks(rd)
        r["ladder_pending"] = len(stage_todo(rd, chunks, "ladder", tuple(cfg["ladder"])))
        r["primary_missing"] = len(stage_todo(rd, chunks, "primary"))
        r["keys"] = key_coverage(rd, clauses)
        r["merged"] = merged_quotes(rd, clauses)
        lat = latency_rows(chunks, [load_record(rd, c.chunk_id) for c in chunks])
        lad = [x["latency_s"] for x in lat if x["attempt"] > 0]
        r["latency_ladder_s"] = {"n": len(lad), "mean": round(sum(lad) / len(lad), 1)
                                 if lad else None, "max": max(lad) if lad else None}
        stages = gpu_time(rd)["stages"]
        r["wall_minutes_by_stage"] = {s: round(sum(x["wall_s"] for x in stages
                                                   if x["stage"] == s) / 60, 1)
                                      for s in ("primary", "ladder")}
        results[cid] = r
    return results


def review_rows(plan: dict, results: dict) -> list[dict]:
    """One row per merged quote for the credit-schema conditions, one per clause not found."""
    rows = []
    for cid, r in results.items():
        if r["prompt"] != "credit_schema":
            continue       # condition a answers CUAD's clauses, which are not reviewed here
        for doc_id, by_clause in r["merged"].items():
            for clause, quotes in by_clause.items():
                base = {"doc_id": doc_id, "clause": clause, "condition": cid,
                        "model": r["model"], "correct": "", "note": ""}
                if not quotes:
                    rows.append(dict(base, found=False, quote_no="", quote="", chars="",
                                     verbatim="", chunks="", from_repaired=""))
                    continue
                flags = verbatim_flags([q for q, _, _ in quotes], plan["docs"][doc_id],
                                       casefold=True, fold_quotes=True)
                for i, ((q, idx, rep), v) in enumerate(zip(quotes, flags), 1):
                    rows.append(dict(base, found=True, quote_no=i, quote=q, chars=len(q),
                                     verbatim=v, chunks=";".join(map(str, idx)),
                                     from_repaired=rep))
    order = {c: i for i, c in enumerate(clauses_for("credit_schema"))}
    docs = list(plan["docs"])
    rows.sort(key=lambda x: (docs.index(x["doc_id"]), order[x["clause"]], x["condition"],
                             x["quote_no"] or 0))
    return rows


# ----------------------------------------------------------------------------- report

def _rate(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.1%})" if n else "n/a"


def baseline_report(plan: dict, results: dict, meta: dict) -> str:
    conds = plan["conditions"]
    ids = list(results)
    first = next(iter(plan["run_dirs"].values()))
    n_chunks = len(load_chunks(first))
    L = [f"# Milestone 3: label-free baselines on {len(plan['docs'])} credit agreements", "",
         f"- Platform: {meta.get('platform')}, GPUs: {meta.get('gpus')}",
         f"- Written: {meta.get('written_at')}, session wall time so far: "
         f"{meta.get('session_minutes')} min",
         f"- Documents: {', '.join(plan['docs'])}",
         f"- Chunks: {n_chunks} at {load_config(first)['chunk_chars']} chars / "
         f"{load_config(first)['overlap']} overlap, identical for every condition "
         f"(sha256 {load_config(first)['chunks_sha256'][:12]})", "",
         "No labels: nothing here says whether an extraction is right. That is what "
         "`review_sheet.csv` is for.", ""]
    incomplete = [cid for cid in ids if results[cid]["primary_missing"]
                  or results[cid]["ladder_pending"]]
    if incomplete:
        L += ["**Incomplete:** " + "; ".join(
            f"condition {cid}: {results[cid]['primary_missing']} chunks without a primary "
            f"decode, {results[cid]['ladder_pending']} waiting for a ladder rung"
            for cid in incomplete) + ". Numbers below cover what has run.", ""]

    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(str(fn(results[c])) for c in ids) + " |")

    L += ["## Decode stability, verbatim copying, cost", "",
          "| metric | " + " | ".join(f"{c}: {conds[c]['label']}" for c in ids) + " |",
          "|---|" + "---|" * len(ids)]
    d = lambda r: r["decode"]
    n = lambda r: r["decode"]["chunks_with_record"]
    row("model", lambda r: f"`{r['model']}`")
    row("prompt", lambda r: f"`{r['prompt']}`")
    row("primary decodes", n)
    row("JSON valid as generated", lambda r: _rate(d(r)["primary_json_valid"], n(r)))
    row("valid after quote repair", lambda r: _rate(d(r)["primary_repaired"], n(r)))
    row("loops (cap or repetition)", lambda r: _rate(d(r)["primary_loop"], n(r)))
    row("other invalid", lambda r: _rate(d(r)["primary_invalid_other"], n(r)))
    row("sent to ladder", lambda r: d(r)["chunks_sent_to_ladder"])
    row("recovered by rung", lambda r: d(r)["recovered_by_rung"])
    row("ladder recovery", lambda r: d(r)["ladder_recovery_rate"])
    row("still failing (ladder done)", lambda r: len(d(r)["still_failing"]) - r["ladder_pending"])
    row("waiting for a ladder rung", lambda r: r["ladder_pending"])
    row("usable after everything", lambda r: d(r)["final_json_valid_rate"])
    row("usable outputs with every clause key",
        lambda r: _rate(r["keys"]["with_all_keys"], r["keys"]["usable_outputs"]))
    row("usable outputs with unknown keys",
        lambda r: _rate(r["keys"]["with_unknown_keys"], r["keys"]["usable_outputs"]))
    row("verbatim, exact vs own chunk", lambda r: r["verbatim_rate_chunk_exact"])
    row("verbatim, casefolded vs document", lambda r: r["verbatim_rate_casefolded"])
    row("same, curly and straight quote marks equal",
        lambda r: r["verbatim_rate_quotes_folded"])
    row("verbatim, quotes from repaired chunks", lambda r: r["verbatim_rate_repaired_chunks"])
    row("quotes after merge", lambda r: r["total_predicted_quotes_merged"])
    row("primary latency per chunk, mean / max s",
        lambda r: f"{r['latency_primary_s']['mean']} / {r['latency_primary_s']['max']}")
    row("ladder latency per attempt, mean / max s (n)",
        lambda r: f"{r['latency_ladder_s']['mean']} / {r['latency_ladder_s']['max']} "
                  f"({r['latency_ladder_s']['n']})")
    row("wall minutes, primary / ladder",
        lambda r: f"{r['wall_minutes_by_stage']['primary']} / "
                  f"{r['wall_minutes_by_stage']['ladder']}")
    row("GPU-device minutes", lambda r: r["gpu_time"]["gpu_device_minutes"])
    row("backends that completed a stage", lambda r: ", ".join(sorted(
        {f"{s['stage']}={s['backend']}:tp{s['tp']}" for s in r["gpu_time"]["stages"]
         if s.get("exit_code", 0) == 0})) or "none")
    row("failed backend attempts", lambda r: ", ".join(
        f"{s['stage']}={s['backend']}:tp{s['tp']} (exit {s['exit_code']})"
        for s in r["gpu_time"]["stages"] if s.get("exit_code", 0) != 0) or "none")
    L += ["", "Primary latency is batch wall time / batch size for vLLM (throughput), and per "
          "request for transformers. Ladder attempts run one at a time on transformers, unless "
          "a vLLM ladder backend is listed above.", ""]

    credit = [c for c in ids if results[c]["prompt"] == "credit_schema"]
    if credit:
        docs = list(plan["docs"])
        L += ["## Credit clauses found", "",
              "Documents with at least one quote, then quotes after merge, per condition. "
              "Found is not correct: see the review sheet.", "",
              "| clause | " + " | ".join(f"{c}: docs found" for c in credit) + " | " +
              " | ".join(f"{c}: quotes" for c in credit) + " |",
              "|---|" + "---|" * (2 * len(credit))]
        for cl in clauses_for("credit_schema"):
            found = [sum(bool(results[c]["merged"].get(dc, {}).get(cl)) for dc in docs)
                     for c in credit]
            nq = [sum(len(results[c]["merged"].get(dc, {}).get(cl, [])) for dc in docs)
                  for c in credit]
            L.append(f"| {cl} | " + " | ".join(f"{k}/{len(docs)}" for k in found) + " | " +
                     " | ".join(map(str, nq)) + " |")
        L += ["", "Quotes per document (" + " / ".join(credit) + "):", "",
              "| clause | " + " | ".join(docs) + " |", "|---|" + "---|" * len(docs)]
        for cl in clauses_for("credit_schema"):
            L.append(f"| {cl} | " + " | ".join(
                " / ".join(str(len(results[c]["merged"].get(dc, {}).get(cl, [])))
                           for c in credit) for dc in docs) + " |")
        L.append("")
    for c in ids:
        if results[c]["prompt"] == "cuad_native":
            share = {k: v for k, v in results[c]["found_share_per_clause"].items() if v}
            L += [f"## Condition {c}: CUAD clauses found", "",
                  f"{len(share)} of 41 CUAD clause types have a quote in at least one "
                  "document: " + (", ".join(f"{k} ({v:.0%})" for k, v in
                                           sorted(share.items(), key=lambda x: -x[1]))
                                  or "none") + ".", ""]
    return "\n".join(L) + "\n"


def report(env: Env, plan: dict, cfg: dict) -> dict:
    """Write baseline_m3.md, review_sheet.csv and results.json, and push them to the Hub."""
    import time
    results = score(plan)
    meta = {"platform": env.platform, "gpus": [g["name"] for g in env.gpus],
            "session_minutes": round(env.session_minutes(), 1),
            "written_at": time.strftime("%Y-%m-%d %H:%M")}
    md = baseline_report(plan, results, meta)
    root = plan["root"]
    (root / "baseline_m3.md").write_text(md, encoding="utf-8")
    rows = review_rows(plan, results)
    with open(root / "review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    slim = {c: {k: v for k, v in r.items() if k != "merged"} for c, r in results.items()}
    (root / "results.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    if env.data_repo:
        push_folder(env.data_repo, root, cfg["hub_prefix"], "m3 report")
    print(md)
    print(f"review sheet: {len(rows)} rows -> {root / 'review_sheet.csv'}")
    return results
