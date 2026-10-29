# credit-extractor

A small local model that extracts provisions from credit/loan agreements. It starts from
[Ihteshamstar/qwen3-4b-cuad-extractor](https://huggingface.co/Ihteshamstar/qwen3-4b-cuad-extractor)
(Model B: Qwen3-4B-Instruct-2507 + QLoRA on CUAD) and compares it with the plain base
model (Model A) before deciding whether to train.

This is a fork of [Ihtesham-star/cuad_llm_finetuning](https://github.com/Ihtesham-star/cuad_llm_finetuning).
Their original README is in [UPSTREAM_README.md](UPSTREAM_README.md). Their `pipeline/`,
`demo/` and `results/` are kept unmodified as reference. **Upstream code has no license**,
so `src/` is an independent implementation and does not import it. See
[docs/repo_notes.md](docs/repo_notes.md).

## Status

| milestone | state |
|---|---|
| M0 fork + reproduce Model B on CUAD | code ready, `notebooks/00_reproduce.ipynb` **not run yet** |
| M1 collect 5 credit agreements | not started |
| M2 credit schema + prompts | not started |
| M3 label-free baselines | not started |

## Layout

```
src/platform.py    Kaggle/Colab/local detection, paths, secrets, private-Hub sync helpers
src/prompts.py     prompt strings (Model B's, byte-identical) + clause canon
src/chunking.py    char windows with overlap
src/parsing.py     JSON parsing, loop detection, key canonicalization
src/merge.py       per-document union of chunk outputs (deduplicated)
src/metrics.py     detection F1 + bootstrap CI, overlap, strict F1, verbatim rate
src/generation.py  vLLM + transformers backends, Ollama-style windowed repeat penalty
src/runner.py      resumable primary pass + retry ladder, backend fallback chain, GPU time
src/evaluate.py    scores a run dir, writes eval.json / latency.csv, markdown report
src/cuad.py        CUAD download + official test split
src/reproduce.py   M0 orchestration (used by the notebook)
configs/           run configs (reproduce.yaml)
notebooks/         thin notebooks: clone repo, pip install, call src/
tests/             CPU unit tests:  python -m pytest -q
```

## How runs work

- **Notebooks run unchanged on Kaggle and Colab.** `src.platform.setup()` detects the
  platform and sets the work dir (`/kaggle/working/work` or `/content/work`) and the HF
  cache. It loads `HF_TOKEN`, `SEC_USER_AGENT` and `NEBIUS_API_KEY` from Kaggle secrets,
  Colab `userdata`, or the environment.
- **Artifact store:** a private HF dataset repo `<hf-user>/credit-extractor-data` and a
  private model repo `<hf-user>/credit-extractor-adapters`. Both are created on first use,
  and setup refuses to use them if they turn out to be public.
- **Resumable:** each condition has its own run dir with `run_config.json` (a
  fingerprint), `chunks.jsonl`, and one `records/<chunk>.json` per chunk holding every
  decode attempt. At startup the notebook pulls the run dir from the Hub. Finished chunks
  are skipped, and a run dir built under different settings refuses to continue.
  `CommitScheduler` pushes new files every 3 minutes, and records are written atomically.
- **GPU stages run in a subprocess** (`python -m src.runner …`), so each backend gets a
  clean CUDA context. The backend chain is vLLM TP=2 → vLLM TP=1 → transformers. Every
  attempt, including failed ones, is logged to `stages.jsonl` with wall time and GPU count.

## Decisions (and why)

- **No code copied from upstream.** The repo has no license (see repo notes). The prompt
  strings come from the Apache-2.0 model card.
- **fp16 everywhere, no flash-attention-2.** T4 has no bf16 support. vLLM uses
  `dtype=float16`. transformers uses SDPA with `device_map="auto"`.
- **Unconstrained generation.** Upstream used Ollama `format: json`. We decode without a
  JSON grammar so that JSON validity is a real, measurable metric in M3. The model was
  trained to emit JSON only. Parsing tolerates code fences and surrounding prose.
- **Output cap 12,288 tokens**, the same as upstream `num_predict`. A primary decode that
  hits the cap, or whose tail is one substring repeated 5 or more times, counts as a
  *loop*. A decode that finishes normally but isn't a JSON object counts as *invalid*.
  Both go to the ladder, as upstream retried every failed chunk.
- **Retry ladder: 1.05 → 1.15 → 1.25 at temperature 0.** Each chunk tries the next rung
  only if the previous one failed, and every attempt is kept. Ollama's `repeat_penalty`,
  which upstream used, only penalizes the **last 64 tokens**. vLLM's and transformers'
  built-in `repetition_penalty` penalize *every* prompt and output token. For verbatim
  copying that penalizes almost everything the model should copy, so it hardly singles out
  the loop. The ladder therefore runs on transformers with a custom 64-token windowed
  penalty (`WindowedRepeatPenalty`), the same for every model and condition. vLLM with its
  native penalty is only a fallback, and `backend` is recorded on every attempt. In
  transformers, a retry that loops again is stopped early once the loop is detected. It
  still counts as a failure, so metrics are unchanged and T4 time is saved.
- **Latency:** vLLM rows report batch wall time divided by batch size, which is amortized
  rather than true single-request latency. transformers rows report true per-request
  time. `latency.csv` has one row per decode attempt.
- **GPU minutes:** `stage_wall_minutes` is the sum of subprocess wall time, model loading
  included, and is what counts against Kaggle's weekly quota. `gpu_device_minutes` is
  wall time × GPUs used.
- **M0 sample:** the first 10 of the 102 official test contracts sorted by md5(title).
  This is the same rule upstream used for its data curve, and the sample happens to
  include INGEVITY, which needed the 1.15 rung upstream. That gives 17 chunks at
  60k/8k and 24 chunks at 32k/4k.
- **Chunking for credit agreements: 32k chars with 4k overlap.** The longest CUAD gold
  span is 3.9k chars, so 4k covers spans of that size. M0 also runs CUAD at 32k/4k to
  check that Model B's F1 holds at the chunk size planned for M3.
- **Partial runs are never reported as results.** If any chunk is missing, every gold
  metric in the report shows `incomplete (k/n chunks)`.
- **vLLM unpinned for the first run.** The notebook prints the installed versions, and
  the first version that works on T4 gets pinned in `requirements-gpu.txt`.
