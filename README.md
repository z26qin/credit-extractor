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
| M2 credit schema + prompts | **approved 2026-10-30**: 10 clause types in [configs/credit_schema.yaml](configs/credit_schema.yaml), exact prompts in [reports/m2_prompts.md](reports/m2_prompts.md) |
| M3 label-free baselines | **ran 2026-10-30 to 31**: numbers and observations in [reports/baseline_m3.md](reports/baseline_m3.md). All 312 decodes usable after the ladder. Model B ignored the credit schema (0 quotes), and Model A gave 271 quotes in [reports/review_sheet.csv](reports/review_sheet.csv) labeled under [docs/labeling_guide.md](docs/labeling_guide.md). Claude labeled all five agreements on 2026-11-01 at the user's request: 61 of 271 quotes are right, Found F1 0.829, Strict F1 0.657 ([reports/score_m3.md](reports/score_m3.md)) |
| M4 small SFT + small RLVR on synthetic data | **ran once on Kaggle T4 x2 on 2026-11-02, in about 6.5 hours**: [reports/small_loop.md](reports/small_loop.md). SFT passed the gate with a held-out mean reward of 0.982, against 0.675 for Model A. On the 5 M1 agreements, Strict F1 went from 0.657 (Model A) to 0.778 (SFT) and 0.806 (RLVR), a direction only. RLVR added little. Plan in [docs/next_steps.md](docs/next_steps.md), run in [notebooks/04_small_loop.ipynb](notebooks/04_small_loop.ipynb). No Nebius |
| Teacher trial, the first step of the parked plan | **ran 2026-11-02 on DeepSeek's own API, for $0.25 at peak price**: [reports/teacher_trial.md](reports/teacher_trial.md). Passed its gate on the 5 M1 agreements with Found F1 0.951 and 80% of quotes right, against 0.90 and 60% needed. Full pre-labeling is not approved |
| Data scale-up, step 1a of the parked plan | **collected 2026-11-02**: 150 credit agreements from 150 issuers, found by EDGAR full-text search with 2,460 SEC requests, not counting a first run stopped to add the fund filter: 30 investment grade, 60 secured mid-cap and 60 leveraged, split by CIK hash into 120 train and 30 test. Text on the Hub under `collect_full/`. Manifest in [reports/collect_full_manifest.csv](reports/collect_full_manifest.csv), and the mix and archetypes, which you have not reviewed yet, in [reports/collect_full_mix.md](reports/collect_full_mix.md) |
| Teacher pre-labeling, step 1c of the parked plan | **ran 2026-11-02, approved by the user at about $7.70 at peak price**: [reports/prelabel_full.md](reports/prelabel_full.md). All 4,430 chunks of the 150 agreements were labeled for $7.56 at peak price, about half that off-peak. 2,415 quotes were kept, and 9 chunks are left out of training. Records are on the Hub under `prelabel_full/`. Nobody has reviewed the labels yet |
| SFT on the pre-labels, step 2 of the parked plan | **ran once on Kaggle T4 x2 on 2026-11-02**: [reports/real_sft.md](reports/real_sft.md). 146 steps on 1,447 examples from the 120 train agreements (about 81% of one epoch) in 6.0 hours, then scoring in about an hour. On the 5 M1 agreements, against Model A on the same chunks, Found F1 went from 0.756 to 0.951 and Strict F1 from 0.542 to 0.776, with paired intervals above zero, a direction only on 5 agreements. The teacher is still ahead on Strict F1 (0.829) and right quotes (80% against 55%). Run in [notebooks/05_real_sft.ipynb](notebooks/05_real_sft.ipynb), [configs/real_sft.yaml](configs/real_sft.yaml). No Nebius. Scoring on the 30 test agreements waits for your review of them |
| RLVR on the pre-labels, step 3 | **code written, CPU tests pass, nothing has run on a GPU yet**: [src/real_rlvr.py](src/real_rlvr.py), [configs/real_rlvr.yaml](configs/real_rlvr.yaml), [scripts/nebius_rlvr.sh](scripts/nebius_rlvr.sh). The user approved one run on a Nebius H100 on 2026-11-02, estimated at about $9-11 with a cap of $18. A local dry run of the data step gives 13,592 windows of 8,000 characters from the train agreements, 1,458 with a teacher quote, and prompts of at most 3,412 tokens |
| TRL + vLLM check, before vLLM sampling in GRPO | **passed on Kaggle T4 x2 on 2026-11-03, second run**: [docs/trl_vllm_check.md](docs/trl_vllm_check.md), [notebooks/07_check_trl_vllm.ipynb](notebooks/07_check_trl_vllm.ipynb). vLLM 0.32.0 needs transformers below 5.20, so the one venv TRL's vLLM mode needs pins 5.19.0 ([requirements-rl-vllm.txt](requirements-rl-vllm.txt)). Colocate and server mode both keep vLLM's log probs within 0.003 of the trainer's. On a T4, server mode with Model A is about 1.5-1.7x as fast per step as transformers' generate. Colocate with Model A is not tested; step 3 keeps vLLM off |

## Layout

```
src/platform.py    Kaggle/Colab/local detection, paths, secrets, private-Hub sync helpers
src/prompts.py     prompt strings (Model B's, byte-identical), credit prompt, clause canon
src/baselines.py   M3 orchestration, report, review sheet
src/gold.py        labeled review sheet -> gold set (docs/labeling_guide.md)
src/score.py       scores models against a gold set
src/synth.py       synthetic credit excerpts with known answers
src/train.py       small SFT (QLoRA), adapter merge
src/rlvr.py        verifiable reward, small RLVR (TRL GRPO)
src/small_loop.py  M4 orchestration: data, training, gate, scoring, report
src/teacher.py     teacher pre-labeling trial: DeepSeek labels, checked quotes, spend cap
src/discover.py    data scale-up: EDGAR full-text search, filters, archetypes, issuer split
src/real_loop.py   step 2: SFT on the teacher's pre-labels, scored on the 5 M1 agreements
src/real_rlvr.py   step 3: RLVR on windows of the train agreements, against the teacher's labels
src/sample.py      vLLM sampling, n answers per excerpt, resumable (step 3 screening)
src/check_vllm.py  checks GRPO with vLLM sampling (colocate, server) against transformers' generate
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
configs/           run configs (reproduce.yaml, collect.yaml, credit_schema.yaml, baselines.yaml)
docs/              upstream notes, next steps, labeling guide
notebooks/         thin notebooks: clone repo, pip install, call src/
scripts/           nebius_rlvr.sh: step 3 on one Nebius GPU VM
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
  JSON. Without it, a copied quote broke the JSON in M0, and the ladder did not fix that in
  6 of 6 attempts at up to 12 minutes each. So a decode that finishes normally but fails to
  parse gets one deterministic repair: a `"` ends a string only if a
  key colon, list end or next string follows, and any other `"` is escaped. A decode saved
  this way has status *repaired* and skips the ladder. Loops, and decodes that still don't
  parse, go to the ladder. This differs from upstream, which retried every failed chunk.
  The raw JSON valid count is unchanged by repair, and repaired counts and the verbatim rate
  of repaired chunks are reported separately. The same rule applies to every condition.
  Notebook 00c parsed all 10 saved failures this way (see the M0 report).
- **Curly quotes.** The M1 text has 6,503 curly double quotes and no straight ones. A model
  that swaps one for the other fails the plain verbatim check although the wording matches,
  so M3 reports the verbatim rate both ways, and the review sheet uses the tolerant one.
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
- **M3 run order.** All three primary passes finish before any ladder, and a report is
  written in between, so a slow ladder never holds back the core numbers. Stopping a cell
  also stops the backend subprocess, and finished chunks are kept. `prepare` refuses to run
  if the Hub text doesn't match the sizes in the reviewed `reports/m1_manifest.csv`.
- **Review sheet.** One row per merged quote for conditions b and c, sorted so both
  conditions sit together per document and clause. A clause with no quote gets one row with
  `found=False`, so misses can be reviewed too. Condition a is left out because its output
  uses CUAD's clause types. `docs/labeling_guide.md` sets the rules for `correct` and `note`.
- **Small plan (decided 2026-11-01).** The next step is a small SFT loop and a small RLVR loop
  on synthetic data, so the compute plan below is parked and no Nebius call is planned.
- **Training stack.** The brief reuses upstream's Unsloth recipe. M4 keeps the recipe's settings
  and uses transformers, peft, bitsandbytes and TRL directly, in their own venv
  (`requirements-train.txt`). Unsloth needs a GPU and caps transformers at 5.17, and this stack
  lets the whole loop run in CPU tests on a tiny model. The cost may be speed on a T4, which the
  notebook's short test measures.
- **Compute after M3 (decided 2026-10-30, parked).** Teacher pre-labeling uses a DeepSeek model on
  Nebius Token Factory with the existing `NEBIUS_API_KEY`. The calls need no GPU, so they
  can run on Kaggle with no accelerator. Nebius GPU VMs stay a training fallback. Every
  Nebius call needs a cost estimate and approval first. Token Factory's terms (effective
  2026-10-22) leave outputs with the customer and bar using them to build a service that
  competes with Nebius's, and they have no explicit rule against training a model on
  outputs. Each model's own license also applies. The chosen model is
  `deepseek-ai/DeepSeek-V4.1-Flash-1017`, listed as MIT at $0.30 input and $1.20 output per
  million tokens. Only public EDGAR text is sent. Nothing has run.
- **Teacher trial (decided 2026-11-02).** After M4, the user approved about $0.5-0.7 for the
  parked plan's teacher trial and chose DeepSeek's own API, the one their other projects use,
  instead of Nebius Token Factory. The model is the same, DeepSeek-V4.1-Flash-1017 (named
  `deepseek-flash` there), at the same peak list price of $0.30 input and $1.20 output per million
  tokens. The key comes from `LLM_API_KEY` in the environment. The trial runs on the local
  machine, because it needs no GPU and the key is kept there. Thinking is off, because the
  approved estimate assumed no hidden reasoning. `max_usd` in `configs/teacher.yaml` is a hard
  cap: no request starts if the spend so far plus its worst case would pass it. Only public
  EDGAR text is sent. DeepSeek's Open Platform Terms of Service (section 4.2, read 2026-11-02)
  allow using inputs and outputs for "training other models (such as model distillation)", as
  long as the use is legal and follows the terms. The model weights are MIT.
- **Data scale-up (2026-11-02).** `src/discover.py` finds the agreements by EDGAR full-text
  search, so no pick is made by hand. It keeps EX-10 and EX-4 exhibits, drops amendments,
  fund and BDC facilities, and texts outside 200,000 to 1,000,000 characters or dated before
  2023, and takes one agreement per issuer, visiting issuers in the order of a hash of their
  CIK. The archetype comes from defined terms: unsecured is investment grade, and secured with
  two or more leveraged-loan terms (such as `Available Amount` or `Excess Cash Flow`) is
  leveraged. On the 5 M1 agreements, this matches the hand labels. It runs on the local
  machine, like the teacher trial: it needs no GPU, and the outputs go to the Hub as before.
  `SEC_USER_AGENT` comes from the environment.
- **SFT on the pre-labels (step 2, 2026-11-02).** The M4 recipe, with these changes. Examples
  are the teacher's 24,000-character chunks with page numbers removed, so `max_seq_length` is
  8,192, the parked plan's Path 1. Full-vocabulary logits for every position of such a prompt
  need about 4 GB in fp32 plus their gradient, so the trainer projects only the answer's
  positions (`answer_trainer` in `src/train.py`). The loss is the same: on a tiny model, the
  adapter after 2 steps matches full-logit training within 1e-5. One epoch instead of two, about
  6 T4 hours, and the short test cuts the steps if one epoch would pass 6.5 hours, so scoring
  fits the same 12-hour session. Empty chunks are capped at 30% of the examples, and 434 of 2,495
  were sampled. Scoring uses the 5 M1 agreements, the only gold set, on the same chunks, and
  Model A runs again on them so both rows compare.
- **RLVR on the pre-labels (step 3, 2026-11-02).** GRPO from the step 2 adapter, with the
  teacher's kept quotes as the answers, since the train agreements have no human labels. It can
  remove the student's own mistakes (in M1, 28 of its 52 wrong quotes did not overlap a teacher
  quote) but not the teacher's. Prompts are 8,000-character windows of the teacher's 24,000-
  character train chunks; a window that cuts a kept quote is dropped, so its answer is clear.
  The pool is 600 windows with quotes and 200 without, and the empty ones must hold a search
  term, so they test restraint where it is hard. A check set of 100 windows comes from 12 other
  train agreements. Screening samples each pool window 4 times with SFT and keeps up to 400
  windows whose rewards differ, because M4 showed that groups with no spread teach nothing. The
  reward is M4's, with one change: for its exactness factor, a quote counts as right only by
  token F1 of at least 0.5, so a whole section that holds the right passage counts as too long.
  On a GPU with at least 40 GB and bf16 the base trains in bf16, not 4-bit; the adapter moves
  between the two, since LoRA only adds to the base. Scoring reuses step 2's Model A and SFT
  decodes on M1 and decodes RLVR the same way.
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
