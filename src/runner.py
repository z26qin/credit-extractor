"""Resumable prediction runs: primary greedy pass + repeat-penalty retry ladder.

A run directory holds everything for one (model, prompt, chunk set) condition:
  run_config.json   fingerprint; a run dir built under other settings refuses to continue
  chunks.jsonl      the exact chunks (shared across conditions by copying the same file)
  records/<id>.json one per chunk: every decode attempt + the final status
  stages.jsonl      wall time / backend / #GPUs per stage invocation (GPU-minute tracking)
  logs/             subprocess logs

GPU stages run in a SUBPROCESS (`python -m src.runner ...`) so vLLM gets a clean CUDA
context and frees the GPUs on exit; `run_with_fallback` walks the backend chain
(vLLM TP=2 -> vLLM TP=1 -> transformers) until one completes, resuming where the
previous one stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .chunking import Chunk, safe_id
from .parsing import classify
from .prompts import PROMPTS, REPO_ROOT, messages

DEFAULT_LADDER = (1.05, 1.15, 1.25)     # Model B card: 1.05, escalating to 1.15 / 1.25
DEFAULT_CHAIN = (("vllm", 2), ("vllm", 1), ("hf", 0))
# ladder: HF first (Ollama-style 64-token window penalty); vLLM's full-context penalty
# is only a fallback and is flagged in every attempt record via `backend`
LADDER_CHAIN = (("hf", 0), ("vllm", 2))


# ----------------------------------------------------------------- run directory io

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def prepare_run(run_dir: Path, config: dict, chunks: list[Chunk]) -> Path:
    """Create or validate a run dir. Raises if it was built under different settings."""
    run_dir = Path(run_dir)
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    chunk_lines = "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks)
    cfg = dict(config, chunks_sha256=_sha(chunk_lines), n_chunks=len(chunks))
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old != cfg:
            diff = {k: (old.get(k), cfg.get(k)) for k in set(old) | set(cfg)
                    if old.get(k) != cfg.get(k)}
            raise SystemExit(f"{run_dir} was built under different settings: {diff}. "
                             f"Use a new run dir or delete this one.")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "chunks.jsonl").write_text(chunk_lines, encoding="utf-8")
    return run_dir


def load_config(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "run_config.json").read_text(encoding="utf-8"))


def load_chunks(run_dir: Path) -> list[Chunk]:
    lines = (Path(run_dir) / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [Chunk(**json.loads(l)) for l in lines if l.strip()]


def record_path(run_dir: Path, chunk_id: str) -> Path:
    return Path(run_dir) / "records" / f"{safe_id(chunk_id)}.json"


def load_record(run_dir: Path, chunk_id: str) -> dict | None:
    p = record_path(run_dir, chunk_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:      # torn write from a disconnect: redo the chunk
        return None


def save_record(run_dir: Path, rec: dict) -> None:
    p = record_path(run_dir, rec["chunk_id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)                # atomic: the Hub sync never sees half a file


def add_attempt(rec: dict, attempt: dict) -> dict:
    rec["attempts"].append(attempt)
    if attempt["status"] == "ok" or rec.get("final_status") != "ok":
        rec["final_status"] = attempt["status"]
        rec["final_repeat_penalty"] = attempt["repeat_penalty"]
    return rec


def stage_todo(run_dir: Path, chunks: list[Chunk], stage: str,
               ladder: tuple[float, ...] = DEFAULT_LADDER) -> list[tuple[Chunk, float]]:
    """(chunk, repeat_penalty) pairs still to decode for this stage.

    ladder: only the NEXT untried rung for each chunk that is not ok yet, so rungs are
    applied in order (1.05 for everyone first, then 1.15 for the holdouts, ...).
    """
    todo = []
    for c in chunks:
        rec = load_record(run_dir, c.chunk_id)
        if stage == "primary":
            if rec is None:
                todo.append((c, 1.0))
        elif stage == "ladder":
            if rec is None or rec["final_status"] == "ok":
                continue
            tried = {a["repeat_penalty"] for a in rec["attempts"]}
            nxt = [rp for rp in ladder if rp not in tried]
            if nxt:
                todo.append((c, nxt[0]))
        else:
            raise ValueError(stage)
    return todo


# ----------------------------------------------------------------- stage execution

def _max_model_len(tokenizer, prompts: list[str], max_tokens: int, cap: int) -> int:
    longest = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    need = longest + max_tokens + 64
    return min(cap, ((need + 1023) // 1024) * 1024)


def run_stage(run_dir: Path, stage: str, backend_kind: str, tp: int = 1) -> int:
    """Run one stage in-process. Returns the number of chunks still pending."""
    from .generation import make_backend, render_prompts
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    chunks = load_chunks(run_dir)
    ladder = tuple(cfg.get("ladder", DEFAULT_LADDER))
    system, user_prefix = PROMPTS[cfg["prompt"]]
    t_start = time.time()
    backend = None
    n_done = 0
    while True:
        todo = stage_todo(run_dir, chunks, stage, ladder)
        if not todo:
            break
        if backend is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(cfg["model"])
            all_prompts = render_prompts(tok, [messages(system, user_prefix, c.text)
                                               for c, _ in todo])
            mml = _max_model_len(tok, all_prompts, cfg["max_tokens"],
                                 cfg.get("max_model_len_cap", 65536))
            backend = make_backend(backend_kind, cfg["model"], tp=tp, max_model_len=mml,
                                   **(cfg.get("vllm_kwargs", {}) if backend_kind == "vllm" else {}))
            print(f"[{stage}] backend={backend_kind} tp={tp} max_model_len={mml} "
                  f"todo={len(todo)}", flush=True)
        # one penalty per batch; batches sized for incremental Hub sync
        rp = todo[0][1]
        bs = cfg.get("batch_size", 16) if backend_kind == "vllm" else 1
        batch = [c for c, r in todo if r == rp][:bs]
        prompts = render_prompts(backend.tokenizer,
                                 [messages(system, user_prefix, c.text) for c in batch])
        results = backend.generate(prompts, cfg["max_tokens"], repeat_penalty=rp)
        for c, res in zip(batch, results):
            status, _ = classify(res.text, res.finish_reason)
            rec = load_record(run_dir, c.chunk_id) or {
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "chunk_index": c.index,
                "model": cfg["model"], "prompt": cfg["prompt"], "attempts": [],
                "final_status": None}
            add_attempt(rec, dict(res.to_dict(), status=status, repeat_penalty=rp,
                                  backend=backend_kind, tp=tp, stage=stage))
            save_record(run_dir, rec)
            n_done += 1
            print(f"[{stage}] {c.chunk_id[:70]} rp={rp} -> {status} "
                  f"({res.n_output_tokens} tok, {res.latency_s:.1f}s)", flush=True)
    print(f"[{stage}] {n_done} decodes in {time.time() - t_start:.0f}s", flush=True)
    return len(stage_todo(run_dir, chunks, stage, ladder))


def log_stage(run_dir: Path, entry: dict) -> None:
    with open(Path(run_dir) / "stages.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_with_fallback(run_dir: Path, stage: str, chain=DEFAULT_CHAIN) -> str:
    """Run a stage in a subprocess, walking the backend chain until one finishes.

    Returns the "backend:tp" label that completed the stage. Output is streamed to the
    notebook and tee'd to run_dir/logs/.
    """
    from .platform import gpu_info
    run_dir = Path(run_dir)
    (run_dir / "logs").mkdir(exist_ok=True)
    n_visible = len(gpu_info())
    for kind, tp in chain:
        label = f"{kind}:{tp}"
        log = run_dir / "logs" / f"{stage}_{kind}_tp{tp}_{int(time.time())}.log"
        cmd = [sys.executable, "-m", "src.runner", "--run-dir", str(run_dir),
               "--stage", stage, "--backend", kind, "--tp", str(tp)]
        print(f"==> {stage}: trying {label}", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                lf.write(line)
                print(line, end="", flush=True)
            code = proc.wait()
        # every attempt counts toward GPU time, including ones that crashed
        log_stage(run_dir, {"stage": stage, "backend": kind, "tp": tp, "exit_code": code,
                            "n_gpus": tp if kind == "vllm" else n_visible,
                            "wall_s": round(time.time() - t0, 1),
                            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if code == 0:
            print(f"==> {stage}: completed with {label}", flush=True)
            return label
        print(f"==> {stage}: {label} failed (exit {code}), log: {log}", flush=True)
    raise RuntimeError(f"all backends failed for stage {stage}; see {run_dir / 'logs'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", choices=["primary", "ladder"], required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], required=True)
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args(argv)
    pending = run_stage(Path(a.run_dir), a.stage, a.backend, a.tp)
    print(f"[{a.stage}] pending after stage: {pending}", flush=True)
    return 0 if pending == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
