# Notes on the upstream repo (Ihtesham-star/cuad_llm_finetuning)

Read at commit `f9c9553` (2026-10-29), together with the HF model card for
`Ihteshamstar/qwen3-4b-cuad-extractor` and its `lora/train_config.json`.
Everything below comes from reading code and docs. Nothing upstream was run.

## License

- **The code repo has no LICENSE file, and its README grants no license.** By default that
  means all rights reserved. GitHub's terms let us view and fork it. They do not let us
  copy or redistribute the code.
- The **model weights** (HF repo) are **Apache-2.0** per the model card, and so is the
  base model Qwen3-4B-Instruct-2507.
- `data/category_descriptions.csv` and `data/split_manifest.json` come from CUAD (The
  Atticus Project, **CC BY 4.0**).
- What we do about it: `src/` is our own implementation, written from the documented
  behavior in the model card and README. It does not import or copy `pipeline/`. The prompt
  strings come from the Apache-2.0 model card. `pipeline/`, `demo/` and `results/` stay
  untouched in the fork as reference only. To reuse upstream code directly, ask the author
  to add a license.

## Data format

- Source: the official CUAD `data.zip` → `CUADv1.json` (SQuAD-style: one `context` per
  contract, `qas` ids `"<title>__<Clause Name>"`, answers with `answer_start`).
- The split is pinned in `data/split_manifest.json`: 408 train / 102 test titles.
- `1_convert.py` writes one JSONL row per contract chunk: `{name: "<title>__chunk<i>",
  title, chunk_index, n_chunks, messages: [system, user, assistant]}`. The assistant
  message is `json.dumps({clause: [quotes]})` with **all 41 keys**. Quotes are
  whitespace-normalized and deduplicated, and an empty list means absent. About 70% of
  (contract, clause) pairs are absences, and those empty lists are part of the training
  target.
- Clause names are matched to the CSV canon with a normalized key (lowercase
  alphanumerics), because qas ids title-case the names ("Ip Ownership Assignment" →
  "IP Ownership Assignment").

## Prompt (training = serving)

- System: "You are a contract review system. Extract the requested clause types … Reply
  with JSON only."
- User: "Extract all 41 CUAD clause types from this contract excerpt as JSON (keys: clause
  types, values: lists of verbatim quotes).\n\nCONTRACT EXCERPT:\n" + chunk.
- No clause definitions in the prompt; the definitions live in the weights. The zero-shot
  baseline (`USER_DEFS`) adds the 41 official definitions instead.
- The training text has the empty `<think>` block removed, so the assistant turn starts
  right after `<|im_start|>assistant\n`. The HF `chat_template.jinja` shipped with the
  model produces exactly that generation prompt. I checked this locally by rendering
  with the model's tokenizer.
- **Observation (unverified impact):** called with `think: false` the way `2_predict.py`
  calls it, `demo/Modelfile` would append `" /no_think"` to the user turn and add an empty
  think block, and the training text has neither. The Modelfile actually used for the
  published eval (`models/Modelfile`) is not in the repo, so the serving prompt behind
  the reported numbers can't be confirmed. We render with the HF template, which matches
  the training text byte for byte.

## Chunking and merge

- 60,000-char windows with 8,000 overlap. Starts are
  `range(0, max(1, n - overlap), size - overlap)`. The longest CUAD gold span is 3,884
  chars, which is under the overlap, so no span is ever cut. Gold spans go to a chunk only
  if they fit inside it by character offset.
- 102 test contracts → 150 chunks.
- At eval time, chunk predictions are merged per contract: union per clause type over
  whitespace-normalized quotes.
- Context sizing (Ollama): `num_ctx = min(40960, max(8192, ceil((chars/4.5 + 800 +
  num_predict)/2048)*2048))`, with `num_predict = 12288`. The README warns that Ollama
  truncates over-long prompts silently. On our 10-contract sample the real rate is about
  4.6 chars/token, and the longest 60k chunk is 14.5k prompt tokens.

## Decoding and the retry ladder

- Ollama, `format: "json"` (grammar-constrained), temperature 0, `num_predict` 12288.
- A primary decode that fails (repetition loop hits the cap → invalid JSON) is never
  cached. `2b_fallback.py` re-decodes only the missing chunks with `repeat_penalty` 1.05,
  then 1.15 (1.25 per the card for rare holdouts), still at temperature 0. Every
  fallback is recorded in `_run_config.json`.
- Published seed-42 run: 9/150 chunks failed the primary decode. 7 were recovered at
  1.05 (HEMISPHERX c0, VALENCE c0, Quaker c0, Soupman c0+c1, Monsanto c2, Goosehead c0),
  2 at 1.15 (Goosehead c2, **INGEVITY c0**, which is in our 10-contract sample), and none
  were left failing.
- Ollama's `repeat_penalty` only looks back `repeat_last_n` = 64 tokens (Ollama default;
  the repo does not override it). That detail matters for our reimplementation: see the
  README, "Retry ladder".
- Prediction caches carry a config fingerprint and refuse to mix settings.

## Metrics (`3_evaluate.py`, `3b_strict_eval.py`)

- Detection P/R/F1 over (contract, clause) decisions (102 × 41 = 4,182). TP = gold and
  prediction both non-empty, FP = prediction only, FN = gold only. Coverage is enforced:
  the script refuses to score with missing chunks unless `--allow-partial` is passed.
- Bootstrap 95% CI: 1,000 resamples of contracts, `random.seed(42)`, percentiles
  `fs[24]`, `fs[974]`.
- Quote overlap: mean over TPs of the max token-F1 between any predicted and gold quote.
- Verbatim rate: share of merged predicted quotes that are a substring of the
  whitespace-normalized, **casefolded** full contract text.
- Strict F1 (ContractEval-style): TP only if every gold span is a casefolded substring of
  some predicted quote. There is also a lenient "any span" variant.
- Unknown keys in the output are dropped and counted (7 in the published run).

## Training recipe (`4_train.py`, `lora/train_config.json`)

| | value |
|---|---|
| base | `unsloth/Qwen3-4B-Instruct-2507`, loaded 4-bit (QLoRA) via Unsloth |
| LoRA | r=16, alpha=32, dropout 0, targets q/k/v/o/gate/up/down, Unsloth gradient checkpointing |
| data | 619 chunk examples from 408 contracts (1 over-length example excluded: 28,353 tokens) |
| max_seq_length | **24,576** (30,720 ran out of memory) |
| optimizer | adamw_8bit, lr 2e-4 linear, warmup_ratio 0.06, 2 epochs |
| batch | per-device 1 × grad accumulation 8 = effective 8 |
| loss | `train_on_responses_only` (assistant JSON only) |
| precision | not set in the script; Unsloth picks bf16 on GPUs that support it, so presumably bf16 on the 5090 (inferred, not stated). The merged weights on the Hub are **bf16** (`config.json` `torch_dtype: bfloat16`, and vLLM logs a bf16 → fp16 cast for every tensor), even though the export step is called `merged_16bit` |
| checkpoints | every 50 steps, keep 1 |
| hardware / time | 1× RTX 5090 (32 GB), 44.6 min, final loss 0.0835 |
| seeds | 42 (published), 43, 44; data curve n = 50/150/300/408 |
| export | `save_pretrained_merged(..., "merged_16bit")` → `convert_hf_to_gguf.py` f16 → `ollama create --quantize q4_K_M` |

A length guard tokenizes every example, because a character-based estimate missed the
over-length one. The launch scripts hardcode Windows/WSL paths.

## Reported results (model card; not reproduced yet)

- Detection F1 0.900 [0.887–0.913], P 0.901, R 0.899, overlap 0.938, verbatim 97.2%
  (seed 42, 102 contracts, Q4_K_M via Ollama).
- 3 seeds: 0.8995 ± 0.0017. Strict F1, 3-seed mean: 0.678.
- From the serving bench, as stated on the card: vLLM BF16 0.900, FP8 0.895.
- Qwen3-14B zero-shot with definitions: 0.816.

## Checked on the first Kaggle run (2026-10-29)

- Weights are bf16. T4 has no bf16, so we load them as fp16. That keeps the exponent
  range narrower than the model was trained in, and an fp16 overflow would show up as
  garbage or invalid JSON, which the decode stats would catch. The model card's numbers
  came from a Q4_K_M GGUF, so neither precision is the card's exactly.
- transformers warns "incorrect regex pattern … fix_mistral_regex" when loading B's
  tokenizer. That is a false positive: B's `tokenizer.json` has the same split regex,
  vocab and merges as `Qwen/Qwen3-4B-Instruct-2507`. Only the ByteLevel
  `trim_offsets` / decoder flags differ, and those don't change ids. I checked locally
  that encoding a mixed legal/JSON/unicode string gives identical ids and decoded text.

## What this means for credit agreements

- The model has only ever seen the CUAD key set and the "41 CUAD clause types" prompt.
  Condition (b) in Milestone 2 (new keys, same shape) is out of distribution by
  construction.
- Training at 24k tokens on an RTX 5090 does not carry over to T4 as-is: 15 GB, no bf16,
  slower. Milestone 3's 32k-char chunks (≈ 7–8k tokens, plus a JSON target) are much
  shorter, which is what makes a T4 recipe plausible. See `docs/next_steps.md`, written
  in Milestone 3.
