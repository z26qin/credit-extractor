# Next steps

Plan only: nothing here has run, and every time and cost below is an estimate unless it says
measured. Every Nebius call needs a cost estimate and approval first.

## Decisions from Milestone 3

- **Train with SFT.** Model A has 61 right quotes of 271. The usual fault is real text from the
  wrong provision: 145 of the 210 wrong quotes are in the agreement, so a verbatim reward would
  score them well, and 21 of the 61 right quotes fail the verbatim check. Teaching what counts
  needs labeled examples, and the copying errors can be fixed after decoding.
- **RLVR stays out of this plan.** Reconsider it only if, after SFT, the model finds the right
  clauses and copying is still the main error after the restore step. If it still finds wrong
  or missing clauses, add data instead.
- **Start from Model A.** Model B gave no credit quotes (`reports/score_m3.md`).
- **The M3 gold set is Claude's labels** (`reports/gold_m3.json`). The user has not reviewed it.
  It is good for the trial and for development, not as the final test.
- **Upstream code has no license,** so training is our own code written from the documented
  recipe. `pipeline/` stays reference only (`docs/repo_notes.md`).

## 0. Cheap check before spending

The prompt gives each clause one sentence, and Model A's noise looks a lot like what the
labeling guide's "does not count" lists rule out. Run Model A once with those rules in the
prompt, on the 5 M1 agreements, and score it with `src.score`.

- Where and time: Kaggle T4, about 1 hour.
- If it reaches the teacher trial gate in section 1, stop and rethink whether training is worth
  it. If not, it still tells us which rules to put in the teacher prompt.

## 1. Data scale-up

About 155 agreements: 120 train, 30 test and the 5 M1 agreements as a development set. The 5 M1
agreements shaped the rules, so they cannot be test data, and they are not trained on.

### Collection

Runs on a Kaggle or Colab CPU, about 15 minutes for about 800 requests at 5 per second.

1. M1's picks were found by hand. New `src/discover.py` queries EDGAR full-text search for
   8-K, 10-Q and 10-K exhibits titled as credit agreements, from 2023 on, so they use SOFR.
2. Keep full agreements, not amendments, of 200,000 to 1,000,000 characters, with one agreement
   per issuer.
3. Aim for the M1 mix: about 30 investment-grade, 60 secured mid-cap, and 60 leveraged or
   cov-lite. Classify from the text, and print the mix for review.
4. Write `configs/collect_full.yaml` and reuse `src/edgar.py`, which is resumable and already
   carries `SEC_USER_AGENT`.
5. Split by issuer, by hash of the CIK, with the archetype mix kept in the test set.

### Text and chunks

- **Remove page numbers and footers before chunking,** using `clean_source` from `src/gold.py`.
  They break quotes across pages, and removing them lifts Model A's verbatim rate from 0.683 to
  0.727 (measured).
- **Use 24,000-character chunks with 4,000 overlap for training and for the test.** The brief's
  ~8K limit cannot hold a 32,000-character chunk: that is about 7,500 tokens of input, plus 1,000
  to 2,500 of answer. A 24,000-character chunk is about 5,700 tokens with the prompt, which leaves
  about 2,500 for the answer. The longest valid M3 output was 2,529 tokens on the longer chunks.
- That gives about 29 chunks per agreement, or about 4,500 in all. The M3 run is unaffected.

### Teacher pre-labeling

Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017` on Nebius Token Factory, Base tier. The price list
shows no batch discount, so the estimate assumes none. The teacher prompt adds the guide's
"counts" and "does not count" rules to the student prompt, about 700 tokens. Students are trained
and served on the plain `credit_schema` prompt, because the prompt seen in training must be the
one used in serving.

Rejection sampling, per chunk:

1. Generate once at temperature 0.
2. Check each quote: the JSON is valid, the clause name is in the schema, and the quote is
   in this chunk.
3. Quotes that fail do not enter the training set.
4. Regenerate a chunk that has a failed quote, up to 2 more times, at a temperature slightly
   above 0. Repeating the same settings gives essentially the same result.
5. Keep only quotes that pass.

Implementation choices:

- The retry temperature starts at 0.3. The trial below can change it.
- Step 5 keeps the union of passing quotes across attempts, deduplicated after whitespace
  normalization, so a retry cannot lose a good quote.
- The check is an exact match after whitespace normalization, with curly and straight quote
  marks kept distinct, so training targets copy the source exactly. If the trial shows the
  teacher swaps quote marks often, add a step that restores the source text.
- A chunk is left out of the training set if no attempt gives valid JSON, or if a clause has
  failed quotes and no passing one. Otherwise that clause's empty list would teach the model
  the clause is absent although the teacher found one.
- Each run reports the reject rate per clause type and the number of chunks left out,
  because failures tend to be the longer, harder chunks.

Limits:

- The check confirms format and copying, not correctness. A verbatim quote under the wrong
  clause passes.
- It cannot find missed clauses, because an empty list passes.
- Retries do not use a repeat penalty. On the one chunk checked in Milestone 0, higher
  penalties lowered the verbatim rate (`reports/m0_reproduce.md`).
- The test agreements use the reviewed labels as answers, not this flow's output.
- M3 shows why the first limit matters: most of Model A's wrong quotes were real text.

Trial: run the flow on the 5 M1 agreements and score it against `reports/gold_m3.json` before any
larger run. Proposed gate: Found F1 of at least 0.90, and at least 60% of accepted quotes right,
against 0.829 and 23% for Model A. Below that, fix the teacher prompt and run the trial again.

Optional: sample each chunk twice, treat quotes found both times as high confidence, and send
disagreements to review first. This roughly doubles the cost.

Estimated cost at the 2026-10-30 list price ($0.30 input, $1.20 output per million tokens),
assuming about 6,400 input tokens and 1,050 to 2,100 output tokens per 24,000-character chunk:

| run | chunks | cost |
|---|---|---|
| trial on the 5 M1 agreements | about 146 | about $0.5-0.7 |
| 150 agreements | about 4,400 | about $14-20 |
| 150 agreements, if 10% of chunks retry once | about 4,800 | about $15-22 |

### Test set review

You review all 30 test agreements yourself, under `docs/labeling_guide.md`, and search every
clause type even when a pre-label looks right. Allow 45 to 60 minutes per agreement, so about 23
to 30 hours in all. Pre-labels are only a starting point, and no model output enters the test
labels unreviewed.

## 2. Training

Both paths use new `src/train.py`, with the same data and prompt, on Model A
(`Qwen/Qwen3-4B-Instruct-2507`).

- **Examples:** one per chunk, with the 10 clause keys and empty lists for absent clauses. Keep
  every chunk with a quote, and hold all-empty chunks to at most 30% of the set, because empty
  answers teach the model when to say nothing. Expect about 2,500 examples from 3,500 chunks.
- **Recipe from the model card:** QLoRA on a 4-bit base, r=16, alpha=32, dropout 0, all linear
  modules, `adamw_8bit`, learning rate 2e-4, warmup 6%, 2 epochs, batch 1 with 8 accumulation
  steps. That is about 625 optimizer steps.
- **Length guard:** tokenize every example and drop any over the limit, reporting how many.
- **Pass:** the development set runs after training, scored with `src.score`.

### Path 1, default and free: Kaggle T4

- Unsloth QLoRA with fp16 and `max_seq_length` of 8,192.
- Save a checkpoint to the private model repo every 25 optimizer steps. On start, pull the latest
  and resume, so a run continues across the 12-hour limit with Save & Run All. Stop cleanly at
  11.5 hours.
- **Runtime estimate:** 6 to 26 hours in 1 to 3 sessions. Upstream took 2.16 s per example on an
  RTX 5090 (measured by upstream). Assuming a T4 is 3 to 7 times slower, with shorter sequences
  saving up to 40% at the low end, 5,000 example passes take 6 to 21 hours, plus overhead. An
  8K sequence may not fit in T4 memory at all.
- **Measure first:** a 30-minute test with 50 real steps records seconds per step and peak memory.
  - It runs out of memory: try 6,000 tokens with shorter chunks, then move to Path 2.
  - It projects over 24 session hours: ask before moving to Path 2.
  - Quota: M3 used at least 8.7 session hours. If T4 x2 counts double, which Kaggle does not
    confirm, a long run takes two weeks of quota.

### Path 2, fallback and paid: Nebius GPU VM

Used only with your OK, after a fresh estimate. One command runs the original recipe with bf16
and `max_seq_length` of 24,576, on the same data:

```bash
python -m src.train --preset nebius
```

Estimated at the 2026-10-30 list price, for one run including setup. These are not measured, and
the disk is extra:

| VM | price per hour | hours | cost |
|---|---|---|---|
| L40S 48 GB, on demand | $1.55 | 7-12 | about $11-19 |
| L40S 48 GB, spot | about $0.75 | 7-12 | about $5-9 |
| H100 | $4.50 | 4-6 | about $18-27 |

Default to L40S on demand. Spot is cheaper and can stop mid-run, which the Hub checkpoints cover.
Nebius does not list an A100.

Setup script, to be committed as `scripts/nebius_train.sh`. It has not been run:

```bash
#!/usr/bin/env bash
# Run on a fresh VM. HF_TOKEN and INSTANCE_ID come from the environment, never from this file.
set -euo pipefail
: "${HF_TOKEN:?}" "${INSTANCE_ID:?}"
stop_vm() { nebius compute instance stop --id "$INSTANCE_ID"; }
trap stop_vm EXIT                                     # stop on success and on failure
sudo shutdown -h +840                                 # hard limit of 14 hours as a backstop
git clone -b "${BRANCH:-main}" https://github.com/z26qin/credit-extractor.git
cd credit-extractor
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-train.txt
python -m src.train --preset nebius
```

`src/train.py`, `requirements-train.txt` and the Nebius CLI setup do not exist yet. The VM needs a
service account that may stop it. Auto-shutdown uses the API because Nebius says a shutdown from
inside the VM can be treated as a failure and restarted, so `shutdown` is only the backstop. A
stopped VM stops GPU billing, but disks keep billing, so delete the VM and its disk after checking
that the adapter is on the Hub.
See [Nebius compute pricing](https://docs.nebius.com/compute/resources/pricing).

## 3. Evaluation

New `notebooks/07_eval.ipynb` on Kaggle runs each model on the 30 test agreements, with the same
chunks, prompt, temperature 0 and repeat-penalty ladder, then calls `src.score`.

- **Models:** Model A, the fine-tuned model, and Model B. Model B goes last, because its M3 score
  is 0 and it can be stopped if quota runs short.
- **Metrics:** Found and Strict F1 overall and per clause type, the upstream detection and strict
  F1, verbatim rate, and 95% bootstrap intervals over agreements. Comparisons use paired
  bootstrap, resampling the same agreements for both models.
- **Token cap:** 4,096 for every model, and the transformers ladder stops early on repetition. At
  12,288, four failed retries took 54% of the M3 run.
- **Runtime estimate:** about 5 hours per model for the first pass, scaled from Model A's measured
  47 minutes, plus up to 5 hours of ladder. So 15 to 30 hours for three models, over 2 or 3
  sessions.

Proposed pass criteria, which need your OK:

| check | target |
|---|---|
| Strict F1 | paired difference from the better of Model A and Model B has a CI above 0 |
| per clause type | no clause more than 0.10 below that baseline's F1 |
| verbatim rate | at least 0.97, quote marks and case ignored |
| usable outputs | at least 99%, and loops at most 5% |

## 4. Export

New `notebooks/08_export.ipynb` runs only if the fine-tuned model passes.

1. Merge the adapter into the fp16 base, on CPU.
2. Convert with llama.cpp `convert_hf_to_gguf.py` to f16, then `llama-quantize` to Q4_K_M. Put the
   GGUF in the private model repo.
3. Re-run the evaluation on the GGUF with the same chunks and ladder. Target: Strict F1 falls by
   at most 0.02. About 5 to 10 hours, on Kaggle or locally.
4. Serve locally with this Modelfile. It is our own, because `demo/Modelfile` is unlicensed. The
   template has no think block, to match the training text, and the penalty is off by default,
   because Ollama's default of 1.1 would change the output.

```
FROM ./credit-extractor-q4_k_m.gguf
SYSTEM """You are a contract review system. Extract the requested clause types from the contract excerpt. Quote the contract text verbatim. A clause type not present in this excerpt gets an empty list. Reply with JSON only."""
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0
PARAMETER repeat_penalty 1.0
PARAMETER repeat_last_n 64
PARAMETER num_ctx 12288
PARAMETER num_predict 4096
```

```bash
ollama create credit-extractor -f Modelfile
```

The caller sends the user turn from `credit_schema`, chunks at 24,000 characters, merges per
agreement, runs the ladder by passing `repeat_penalty`, and restores quotes to the source text.

## Order and approvals

| step | where | cost | needs your OK |
|---|---|---|---|
| 0. Prompt check | Kaggle | about 1 hour of quota | no |
| 1a. Collection | Kaggle or Colab CPU | free | no, you review the mix |
| 1b. Teacher trial | Token Factory | about $0.5-0.7 | **yes** |
| 1c. Full pre-labeling | Token Factory | about $14-22 | **yes** |
| 1d. Test set review | you | about 23-30 hours of your time | it is your work |
| 2a. Short test | Kaggle | about 30 minutes of quota | no |
| 2b. Training, Path 1 | Kaggle | 6-26 hours of quota | no |
| 2c. Training, Path 2 | Nebius VM | about $5-27 | **yes**, only if Path 1 fails |
| 3. Evaluation | Kaggle | 15-30 hours of quota | pass criteria |
| 4. Export | Kaggle or local | 5-10 hours | no |
