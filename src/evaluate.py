"""Score a run directory: decode stability, latency, GPU time, and (if gold) detection F1."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .merge import merge_chunks
from .metrics import detection_report, verbatim_flags
from .parsing import canonicalize, parse_json_object
from .prompts import canon_map
from .runner import load_chunks, load_config, load_record

# Model B card / upstream results/eval_cuad-extractor_short.json (102 contracts, 150
# chunks, Ollama Q4_K_M, RTX 5090). Reference only - never produced by this repo.
CARD_REFERENCE = {
    "detection_f1": 0.900, "detection_f1_ci95": [0.887, 0.913],
    "detection_precision": 0.901, "detection_recall": 0.899,
    "overlap_tokenF1_on_TP": 0.938, "verbatim_rate_casefolded": 0.972,
    "strict_f1_3seed_mean": 0.678, "chunks_needing_ladder": "9/150 (6.0%)",
}


def final_output(rec: dict | None, canon: dict) -> tuple[dict | None, int]:
    """Canonicalized output of the chunk's successful attempt (None if every attempt failed)."""
    if not rec or rec.get("final_status") != "ok":
        return None, 0
    ok = [a for a in rec["attempts"] if a["status"] == "ok"][0]
    return canonicalize(parse_json_object(ok["text"]), canon)


def decode_summary(records: list[dict | None], ladder: list[float]) -> dict:
    present = [r for r in records if r]
    prim = [r["attempts"][0] for r in present]
    n = len(prim)
    s = {
        "chunks_total": len(records), "chunks_with_record": n,
        "primary_json_valid": sum(a["status"] == "ok" for a in prim),
        "primary_loop": sum(a["status"] == "loop" for a in prim),
        "primary_invalid_other": sum(a["status"] == "invalid" for a in prim),
        "recovered_by_rung": {str(rp): 0 for rp in ladder},
        "still_failing": [],
        "ladder_backends": sorted({a["backend"] for r in present for a in r["attempts"][1:]}),
    }
    s["primary_json_valid_rate"] = round(s["primary_json_valid"] / n, 4) if n else None
    s["primary_loop_rate"] = round(s["primary_loop"] / n, 4) if n else None
    for r in present:
        if r["attempts"][0]["status"] == "ok":
            continue
        if r["final_status"] == "ok":
            s["recovered_by_rung"][str(r["final_repeat_penalty"])] += 1
        else:
            s["still_failing"].append(r["chunk_id"])
    failed = n - s["primary_json_valid"]
    s["ladder_recovery_rate"] = (round((failed - len(s["still_failing"])) / failed, 4)
                                 if failed else None)
    s["final_json_valid_rate"] = round(sum(r["final_status"] == "ok" for r in present) / n,
                                       4) if n else None
    return s


def latency_rows(chunks, records) -> list[dict]:
    rows = []
    for c, r in zip(chunks, records):
        if not r:
            continue
        for i, a in enumerate(r["attempts"]):
            rows.append({"chunk_id": c.chunk_id, "chars": len(c.text), "attempt": i,
                         "repeat_penalty": a["repeat_penalty"], "backend": a["backend"],
                         "tp": a.get("tp"), "batch_size": a["batch_size"],
                         "status": a["status"], "finish_reason": a["finish_reason"],
                         "prompt_tokens": a["n_prompt_tokens"],
                         "output_tokens": a["n_output_tokens"],
                         "latency_s": round(a["latency_s"], 2)})
    return rows


def gpu_time(run_dir: Path) -> dict:
    p = Path(run_dir) / "stages.jsonl"
    stages = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    wall = sum(s["wall_s"] for s in stages) / 60
    dev = sum(s["wall_s"] * max(1, s["n_gpus"]) for s in stages) / 60
    return {"stages": stages, "stage_wall_minutes": round(wall, 1),
            "gpu_device_minutes": round(dev, 1)}


def evaluate_run(run_dir: Path, clauses: list[str], doc_texts: dict[str, str],
                 gold: dict[str, dict] | None = None) -> dict:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    chunks = load_chunks(run_dir)
    records = [load_record(run_dir, c.chunk_id) for c in chunks]
    canon = canon_map(clauses)
    per_doc: dict[str, list[dict]] = {}
    dropped = 0
    chunk_quotes, chunk_verbatim = 0, 0
    for c, r in zip(chunks, records):
        out, d = final_output(r, canon)
        dropped += d
        per_doc.setdefault(c.doc_id, [])
        if out is not None:
            per_doc[c.doc_id].append(out)
            qs = [q for v in out.values() for q in v]
            chunk_quotes += len(qs)
            chunk_verbatim += sum(verbatim_flags(qs, c.text))
    merged = {d: merge_chunks(outs, clauses) for d, outs in per_doc.items()}
    doc_q, doc_v = 0, 0
    for d, m in merged.items():
        qs = [q for v in m.values() for q in v]
        doc_q += len(qs)
        doc_v += sum(verbatim_flags(qs, doc_texts[d], casefold=True))
    result = {
        "run_dir": str(run_dir), "model": cfg["model"], "prompt": cfg["prompt"],
        "chunk_chars": cfg["chunk_chars"], "overlap": cfg["overlap"],
        "max_tokens": cfg["max_tokens"], "documents": len(merged),
        "decode": decode_summary(records, cfg.get("ladder", [])),
        "dropped_prediction_keys": dropped,
        "verbatim_rate_chunk_exact": round(chunk_verbatim / chunk_quotes, 4)
        if chunk_quotes else None,
        "verbatim_rate_casefolded": round(doc_v / doc_q, 4) if doc_q else None,
        "total_predicted_quotes_merged": doc_q,
        "gpu_time": gpu_time(run_dir),
        "found_share_per_clause": {
            cl: round(sum(bool(m[cl]) for m in merged.values()) / len(merged), 4)
            for cl in clauses} if merged else {},
    }
    if gold is not None:
        result["gold"] = detection_report({d: gold[d] for d in merged}, merged, clauses)
    lat = latency_rows(chunks, records)
    prim = [r["latency_s"] for r in lat if r["attempt"] == 0]
    result["latency_primary_s"] = {
        "mean": round(sum(prim) / len(prim), 2) if prim else None,
        "max": max(prim) if prim else None,
        "note": "vLLM: batch wall time / batch size (amortized); HF: per request"}
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if lat:
        with open(run_dir / "latency.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(lat[0]))
            w.writeheader()
            w.writerows(lat)
    return result


def reproduce_report(results: dict[str, dict], meta: dict) -> str:
    """Markdown comparing reproduction run(s) against the model card's numbers."""
    L = ["# Milestone 0: Model B reproduction on CUAD (subset)", "",
         f"- Platform: {meta.get('platform')}, GPUs: {meta.get('gpus')}",
         f"- Contracts: {meta.get('n_contracts')} of 102 official test contracts "
         f"(first by md5(title))",
         f"- Session wall time so far: {meta.get('session_minutes')} min", "",
         "Model card numbers are on all 102 test contracts with Ollama Q4_K_M on an RTX 5090;",
         "ours are fp16 on T4 on a subset, so CIs are much wider. Compare with that in mind.", "",
         "| metric | model card (102 contracts) | " +
         " | ".join(results) + " |",
         "|---|---|" + "---|" * len(results)]

    def g(r, k):
        d = r["decode"]
        if d["chunks_with_record"] < d["chunks_total"]:   # never show partial runs as results
            return f"incomplete ({d['chunks_with_record']}/{d['chunks_total']} chunks)"
        v = r.get("gold", {}).get(k, r.get(k))
        return "n/a" if v is None else v
    rows = [("detection F1", "detection_f1", CARD_REFERENCE["detection_f1"]),
            ("F1 95% CI", "detection_f1_ci95", CARD_REFERENCE["detection_f1_ci95"]),
            ("precision", "detection_precision", CARD_REFERENCE["detection_precision"]),
            ("recall", "detection_recall", CARD_REFERENCE["detection_recall"]),
            ("quote overlap (token F1 on TP)", "overlap_tokenF1_on_TP",
             CARD_REFERENCE["overlap_tokenF1_on_TP"]),
            ("strict F1", "strict_f1", f"{CARD_REFERENCE['strict_f1_3seed_mean']} (3-seed mean)"),
            ("verbatim rate (casefolded, vs contract)", "verbatim_rate_casefolded",
             CARD_REFERENCE["verbatim_rate_casefolded"])]
    for label, k, ref in rows:
        L.append(f"| {label} | {ref} | " + " | ".join(str(g(r, k)) for r in results.values()) + " |")
    L.append("| chunks needing ladder | " + CARD_REFERENCE["chunks_needing_ladder"] + " | " +
             " | ".join(f"{r['decode']['chunks_with_record'] - r['decode']['primary_json_valid']}"
                        f"/{r['decode']['chunks_with_record']}" for r in results.values()) + " |")
    L += ["", "## Decode stability and latency", ""]
    for name, r in results.items():
        d, gt = r["decode"], r["gpu_time"]
        L += [f"### {name}", "",
              f"- chunk size {r['chunk_chars']} / overlap {r['overlap']}, "
              f"max output tokens {r['max_tokens']}",
              f"- primary JSON valid: {d['primary_json_valid']}/{d['chunks_with_record']}, "
              f"loops: {d['primary_loop']}, other invalid: {d['primary_invalid_other']}",
              f"- recovered by ladder rung: {d['recovered_by_rung']}; still failing: "
              f"{d['still_failing'] or 'none'}; ladder backends: {d['ladder_backends'] or 'n/a'}",
              f"- verbatim (exact after whitespace norm, vs own chunk): "
              f"{r['verbatim_rate_chunk_exact']}",
              f"- dropped unknown keys: {r['dropped_prediction_keys']}",
              f"- primary latency per chunk: mean {r['latency_primary_s']['mean']} s, "
              f"max {r['latency_primary_s']['max']} s ({r['latency_primary_s']['note']})",
              f"- stage wall time {gt['stage_wall_minutes']} min, "
              f"GPU-device minutes {gt['gpu_device_minutes']}",
              f"- backends that completed a stage: " + ", ".join(sorted(
                  {f"{s['stage']}={s['backend']}:tp{s['tp']}" for s in gt['stages']
                   if s.get('exit_code', 0) == 0})),
              f"- failed backend attempts: " + (", ".join(
                  f"{s['stage']}={s['backend']}:tp{s['tp']} (exit {s['exit_code']})"
                  for s in gt['stages'] if s.get('exit_code', 0) != 0) or "none"), ""]
    return "\n".join(L) + "\n"
