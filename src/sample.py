"""Sample answers for excerpts with vLLM, n per excerpt, saved as each batch finishes.

    python -m src.sample --model <dir> --data windows.jsonl --out samples.jsonl --n 4 \
        --temperature 1.0 --max-tokens 1024 [--tp 2] [--max-model-len 4096] [--batch 100]

Runs in the vLLM venv, started by src/real_rlvr.py. The prompt is the one training uses, and
excerpts already in --out are skipped, so a restart resumes. Temperature 0 gives one greedy
answer per excerpt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .synth import load_set
from .train import prompt_text


def done_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {json.loads(line)["id"] for line in out.read_text("utf-8").splitlines() if line.strip()}


def sample(llm, rows: list[dict], out: Path, n: int, temperature: float, max_tokens: int,
           batch: int, seed: int = 0) -> int:
    """Append one line per excerpt to `out`; returns how many were sampled."""
    from vllm import SamplingParams
    sp = SamplingParams(n=1 if temperature == 0 else n, temperature=temperature, top_p=1.0,
                        max_tokens=max_tokens, seed=seed)
    tok = llm.get_tokenizer()
    todo = [x for x in rows if x["id"] not in done_ids(out)]
    t0 = time.time()
    for start in range(0, len(todo), batch):
        part = todo[start:start + batch]
        outs = llm.generate([prompt_text(tok, x["excerpt"]) for x in part], sp, use_tqdm=False)
        with open(out, "a", encoding="utf-8") as f:
            for x, o in zip(part, outs):
                f.write(json.dumps({
                    "id": x["id"], "completions": [c.text for c in o.outputs],
                    "finish": [c.finish_reason for c in o.outputs],
                    "tokens": [len(c.token_ids) for c in o.outputs],
                    "prompt_tokens": len(o.prompt_token_ids)}, ensure_ascii=False) + "\n")
        print(f"{start + len(part)}/{len(todo)} excerpts, {time.time() - t0:.0f} s", flush=True)
    return len(todo)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sample answers for excerpts with vLLM.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args(argv)
    rows = load_set(a.data)
    if not [x for x in rows if x["id"] not in done_ids(a.out)]:
        print(f"all {len(rows)} excerpts already sampled", flush=True)
        return 0
    import torch
    from vllm import LLM
    tp = min(a.tp, max(1, torch.cuda.device_count()))      # one GPU on a VM, two on Kaggle
    llm = LLM(model=a.model, dtype="float16", tensor_parallel_size=tp,
              max_model_len=a.max_model_len, gpu_memory_utilization=0.90, seed=0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sample(llm, rows, a.out, a.n, a.temperature, a.max_tokens, a.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
