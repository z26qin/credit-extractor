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
| M0 fork + reproduce Model B on CUAD | **ran 2026-10-29**: F1 0.883 (60k) / 0.892 (32k), card's 0.900 inside both CIs; see [reports/m0_reproduce.md](reports/m0_reproduce.md). Ladder check ran 2026-10-30: the memory fix works on T4 x2, and the one failure is an unescaped quote that the 64-token ladder did not recover, so M3 repairs quotes first |
| M1 collect 5 credit agreements | **done 2026-10-30**: 5 documents on the Hub, cleaned text reviewed after a cleaner fix, manifest in [reports/m1_manifest.csv](reports/m1_manifest.csv) |
| M2 credit schema + prompts | **written 2026-10-30, not run**: 10 clause types in [configs/credit_schema.yaml](configs/credit_schema.yaml), exact prompts in [reports/m2_prompts.md](reports/m2_prompts.md) |
| M3 label-free baselines | not started |

## Layout

```
src/platform.py    Kaggle/Colab/local detection, paths, secrets, private-Hub sync helpers
src/prompts.py     prompt strings (Model B's, byte-identical), credit prompt, clause canon
src/chunking.py    char windows with overlap
src/parsing.py     JSON parsing, loop detection, key canonicalization
src/merge.py       per-document union of chunk outputs (deduplicated)
src/metrics.py     detection F1 + bootstrap CI, overlap, strict F1, verbatim rate
src/generation.py  vLLM + transformers backends, Ollama-style windowed penalty, slice prefill
src/runner.py      resumable primary pass + retry ladder, backend fallback chain, GPU time
src/evaluate.py    scores a run dir, writes eval.json / latency.csv, markdown report
src/cuad.py        CUAD download + official test split
src/reproduce.py   M0 orchestration (used by the notebook)
src/edgar.py       M1: rate-limited EDGAR client, HTML -> text, manifest
configs/           run configs (reproduce.yaml, collect.yaml, credit_schema.yaml)
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
- **Quote repair before the ladder (changed after M0).** Upstream's grammar forces valid
  JSON. Without it, a copied quote broke the JSON in M0, and the ladder did not fix that in 6 of 6 attempts at up to 12 minutes each. So a decode that finishes
  normally but fails to parse gets one deterministic repair: a `"` ends a string only if a
  key colon, list end or next string follows, and any other `"` is escaped. A decode saved
  this way has status *repaired* and skips the ladder. Loops, and decodes that still don't
  parse, go to the ladder. This differs from upstream, which retried every failed chunk.
  The raw JSON valid count is unchanged by repair, and repaired counts and the verbatim rate
  of repaired chunks are reported separately. The same rule applies to every condition.
- **Retry ladder: 1.05 → 1.15 → 1.25 at temperature 0.** Each chunk tries the next rung
  only if the previous one failed, and every attempt is kept. Ollama's `repeat_penalty`,
  which upstream used, only penalizes the **last 64 tokens**. vLLM's and transformers'
  built-in `repetition_penalty` penalize *every* prompt and output token. For verbatim
  copying that penalizes almost everything the model should copy, so it hardly singles out
  the loop. The ladder therefore runs on transformers with a custom 64-token windowed
  penalty (`WindowedRepeatPenalty`), the same for every model and condition. Prompts over
  512 tokens enter the cache in slices (`prefill_cache`), because a single pass over 8-14k
  tokens needs 5.6-19 GiB for attention on T4. Slicing gives identical output. vLLM with its
  native penalty is only a fallback, and `backend` is recorded on every attempt. In
  transformers, a retry that loops again is stopped early once the loop is detected. It
  still counts as a failure, so metrics are unchanged and T4 time is saved.
- **Credit prompt (M2).** The system turn is Model B's, byte for byte. The user turn keeps
  Model B's first sentence with the count and name changed, and adds a one-sentence
  definition per clause, because the 10 credit clauses are not in B's weights. Condition b is
  therefore outside the prompt shape B was trained on, by design. Models A and B get the
  same text. Both chat templates end at `<|im_start|>assistant\n` with no think block, and
  neither reacts to `enable_thinking`, so no thinking flag is set.
- **Condition a outputs CUAD's 41 clauses, not the 10 credit ones.** It is the control for
  decode stability and verbatim rate. Per-clause found share compares b and c only.
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
- **vLLM lives in its own venv (`/tmp/vllm-env`).** On Kaggle's first run, `pip install
  vllm` swapped the preinstalled torch for a CUDA 13.0 build and left torchaudio at CUDA
  12.8. transformers imports torchaudio, so every backend failed, the transformers
  fallback included. The notebook now installs vLLM with `uv pip install
  --torch-backend=auto` into a separate venv, where uv picks the torch build for the GPU
  driver. vLLM subprocesses use that interpreter (`runner.backend_python`), and
  transformers keeps the platform's own Python. `stages.jsonl` records which interpreter
  each attempt used.
- **vLLM pinned to 0.32.0**, the version that ran M0 on T4 x2.

### Milestone 1 (credit agreements)

- **Five full, original credit agreements, no amendments.** Amendments that only list
  changes would make "found / not found" depend on what the amendment touches. Amendments
  with a conformed copy attached (for example ZoomInfo or Tronox) double the text and mix
  two documents. Commitment letters and transaction support agreements mention the same
  terms but aren't agreements. The picks cover:
  - an investment-grade bank revolver (Quanta 2024),
  - a bank deal with defined Serta and Chewy provisions (ExlService 2026),
  - a secured mid-cap deal with a "Material Intellectual Property" (J.Crew) blocker
    (Angi 2025),
  - a leveraged revolver plus term loan (Repay 2026),
  - a cov-lite, term-loan-only agreement from a tech issuer with no financial
    covenant at all, because the revolver is a separate ABL (VIAVI 2025).

  Some clause types should be absent from some documents. For example, the IG revolver
  should have no builder basket or MFN. That makes "found share" mean something in M3.
  VIAVI replaced the first pick, AlTi 2026 (private credit), on review: the set had no
  agreement without a maintenance covenant. The alternates are listed in
  `configs/collect.yaml`. The picks came from EDGAR
  full-text search, and their lengths and key terms were checked in a browser.
- **Facts come from EDGAR, not the config.** Issuer, form and filing date come from the
  submissions API. Agreement date is the first "dated as of …" in the first 5,000 chars.
  sha256 is over the raw exhibit bytes. `char_count` is the length of the cleaned text,
  which is what gets chunked.
- **Cleaning:** stdlib `HTMLParser`. Block tags become line breaks, source newlines inside
  text become spaces (they are only wrapping), and each table row stays on one line even
  when cells contain `<p>`, so a pricing grid keeps its rows. Cells over 500 chars keep
  their breaks, because those are layout tables holding paragraphs. Script, style and `display:none` content is dropped, along with the
  EDGAR SGML header (`<TYPE>`, `<SEQUENCE>`, …). Entities, nbsp and runs of spaces are
  normalized. Unclosed tags (common in EDGAR HTML) can't hide the rest of the document.
  Page headers and footers and the table of contents are **kept**, because removing them
  heuristically could delete real text. Verbatim checks run against this same text.
- **Politeness:** at most 5 requests/s (SEC allows 10/s). Back off on 429 and 5xx. Every
  request carries `SEC_USER_AGENT`. One collection is about 15 requests.
- **Manifest to GitHub:** this machine has no HF token, so the notebook prints
  `manifest.csv` at the end, and that output is committed unchanged as
  `reports/m1_manifest.csv`.
- **Size (measured):** 2,864,929 cleaned chars, which makes 104 chunks at 32k/4k per
  condition and 312 primary decodes for the three M2 conditions, before any ladder retries.
- **Re-running M1 makes no SEC requests.** If the raw file's sha256 matches, the text is
  rebuilt from it, so a cleaner fix doesn't re-download anything. The first run's
  `title: AGREEMENT` for ExlService came from the same wrapping bug.
