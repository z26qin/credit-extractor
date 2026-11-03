# Next steps

This plan ran once, on Kaggle GPU T4 x2 on 2026-11-02, and the results are in
`reports/small_loop.md`. Times marked measured come from that run. The rest of this file is the
plan as written before it ran. No step used Nebius or cost money.

The goal is one small SFT loop and one small RLVR loop on Model A, end to end, on free Kaggle
GPU, trained on synthetic data. The larger plan (about 155 real agreements, teacher pre-labeling,
30 reviewed test agreements, a Nebius VM path and GGUF export) is parked. It is in git at commit
`b9b3fe8` if it is wanted later.

## Decisions

- **Train on synthetic data only.** Code builds each example, so its answer is known exactly.
  That removes EDGAR collection, teacher labels and test review.
- **RLVR is back, in a small form.** Milestone 3 ruled it out because a verbatim reward would
  score wrong quotes well: 145 of the 210 wrong quotes are real agreement text. On synthetic data
  the reward compares each quote with the known answer, so wrong text scores low. Only 5 real
  agreements have answers, and they are not trained on, so RLVR trains on synthetic data only.
- **Start from Model A.** Model B gave no credit quotes (`reports/score_m3.md`).
- **Keep the M3 prompt and chunks.** Training uses the exact `credit_schema` prompt, and real
  agreements keep M3's 32,000-character chunks. Model A's measured Found F1 of 0.829 and Strict F1
  of 0.657 stay the baseline, so Model A does not rerun on real text.
- **Our own code.** Upstream code has no license, so `src/train.py` follows the documented recipe
  in `docs/repo_notes.md`. RLVR uses `GRPOTrainer` from TRL.
- **No Unsloth.** Training uses transformers, peft and bitsandbytes directly, with the same recipe.
  Unsloth needs a GPU and caps transformers at 5.17, while this stack lets the whole loop run in
  CPU tests. It may be slower on a T4, and the short test will show the real pace.

## 1. Synthetic data

`src/synth.py` builds the data from a fixed seed, on CPU inside the notebook.

**Text bank,** in `configs/synth_bank.yaml`, written from the clause definitions and not from the
gold passages:

- **60 clause passages:** 6 templates for each of the 10 clause types, with slots for amounts,
  percentages, dates and names. They were meant to follow the gold set's lengths (median 362
  characters, 90th percentile 1,430), but the training answers came out shorter: median 255, 90th
  percentile 417 and longest 563 (measured after the run).
- **43 look-alikes,** aimed at M3's wrong quotes: neighboring basket items (`adjacent`, 107 wrong
  quotes), sentences that only use a term (`mention`, 29), and headings (`heading`, 17).
- **10 general passages** that belong to no clause type (`unrelated`, 36).
- **Filler:** paragraphs from the 408 CUAD training contracts, loaded by `src/cuad.py`. They are
  commercial contracts, and any paragraph using terms such as "Indebtedness" or "Applicable
  Margin" is dropped.

**Examples:**

- One excerpt holds 0 to 4 clause passages and 2 to 6 look-alikes, shuffled into filler, for a
  total of 3,000 to 13,000 characters.
- 25% of excerpts have no clause passage, because M3's main fault was quoting too much.
- Passages carry curly quote marks and some have a page number inside, the two copying faults M3
  showed. Answers keep the curly marks and leave the page numbers out.
- The answer is JSON with all 10 clause keys and empty lists for absent clauses. The training text
  is the prompt M3 rendered plus this answer.

**Sets:**

| set | examples | use |
|---|---|---|
| train | 300 | SFT |
| RLVR | 200 | shorter excerpts: 2,000 to 5,000 characters, at most 2 clause passages of up to 1,000 characters each, so answers fit in 512 tokens |
| held out | 60 | scoring only |

One template per clause type, 10 look-alikes, 2 general passages and a fifth of the filler contracts
appear only in the held-out set, so that set tests wording the model has not seen. The leak check
finds no run of 15 words from the bank in the 5 M1 agreements (measured).

## 2. Small SFT

`src/train.py` runs QLoRA on Model A.

- **Recipe from the model card:** 4-bit base, r=16, alpha=32, dropout 0, the same linear layers,
  `adamw_8bit`, learning rate 2e-4, warmup 6%, 2 epochs, batch 1 with 8 accumulation steps.
- **Changes for T4:** fp16 and `max_seq_length` of 4,096. With the real tokenizer, 1 of the 300
  examples is over the limit and gets dropped (measured).
- **Size:** about 600 example passes, or 75 optimizer steps. An average example is 2,478 tokens
  (measured).
- **Saving:** the adapter goes to the private model repo every 25 steps, and a restart resumes
  from the latest one.
- **Time:** 61 minutes, at 49.1 seconds per step, with a 12.5 GB memory peak (measured). The plan
  estimated 0.5 to 2 hours.

**Gate:** SFT's mean reward (section 3) on the held-out set must beat Model A's by at least 0.10. If
it does not, the notebook stops, because RLVR would be building on a setup that does not work.

## 3. Small RLVR

`src/rlvr.py` starts from the SFT adapter and calls TRL's `GRPOTrainer`. We write only the data and
the reward.

**Reward,** computed by code for each sampled answer:

| answer | reward |
|---|---|
| not valid JSON, or the keys are not the 10 clause types | 0 |
| excerpt has no clause and nothing is quoted | 1 |
| otherwise | F × (0.5 + 0.5 × C) |

- **F** is the average of Found F1 and Strict F1 over the 10 clause types.
- **C** is the share of quotes that are both right and verbatim in the excerpt, once page numbers
  are removed. Curly and straight quote marks count as different here, to teach exact copying.

F alone is not enough: Found and Strict work per clause, so extra wrong quotes beside a right one
cost nothing. A trial of the formula with the repo's functions gave these scores (measured, on
hand-written cases):

| answer | reward |
|---|---|
| all right and verbatim | 1.0 |
| right, plus one quote from the neighboring item | 0.833 |
| one of two clauses found | 0.667 |
| right clauses with changed words or quote marks | 0.417 |
| real text from the wrong provision | 0 |
| empty, when a clause is there | 0 |

Model A's M3 output scores a mean of 0.422 on the 5 M1 agreements, taking each agreement as one
excerpt (measured). The reward reuses `parse_output`, `canonicalize`, `guide_counts`, `right_quote`,
`verbatim_flags` and `clean_source`.

**Settings:**

- 200 prompts, 4 sampled answers each, one prompt per optimizer step.
- `num_generations=4`, `max_completion_length=512`, `beta=0` (no reference model), learning rate
  1e-5, batch 1 with 4 accumulation steps.
- The adapter goes to the private model repo every 50 steps.
- TRL turns QLoRA adapter weights into bf16, which a T4 cannot compute with, so `src/rlvr.py` turns
  them back to fp32 before training, as in SFT.

**Time:** 131 minutes for 200 prompts, at 39.2 seconds per prompt, with a 7.7 GB memory peak
(measured). The plan estimated 1 to 3 hours. On 88% of prompts all 4 answers got the same reward,
so most steps changed nothing (measured).

## 4. Comparison

Each adapter is merged into a local folder in the notebook, then goes through the M3 runner and
`src.score`, with temperature 0 and the same repeat-penalty ladder. The token cap is 4,096, because
the longest valid M3 output was 2,529 tokens.

| set | Model A | SFT | RLVR |
|---|---|---|---|
| 60 held-out excerpts | runs | runs | runs |
| 5 M1 agreements | already measured | runs | runs |

**Time (measured):** 5 to 6 minutes per model on the held-out set. On the real agreements, SFT took
87 minutes and RLVR took 68, because some chunks fell back to the slow transformers ladder. Model A
took 52 minutes in M3.

Results go to `reports/small_loop.md`: Found F1, Strict F1, verbatim rate, share of quotes that are
right and mean reward for each model, plus measured training times. The share of right quotes
matters most on real text, because Found and Strict do not see extra wrong quotes. Model A's is 24%
by the automatic rule (64 of 271). Apart from the gate in section 2, there are no pass criteria.

## How it runs

One notebook, `notebooks/04_small_loop.ipynb`, with Save & Run All on Kaggle GPU T4 x2: data, short
test, SFT, gate, RLVR, comparison. Every stage resumes from the Hub. Training uses one of the two
GPUs, and the comparison uses both, as in M3.

**Short test:** 10 SFT steps and 3 RLVR steps, about 20 minutes, recording seconds per step and peak
memory.

- Out of memory: the notebook stops. Lower `max_seq_length` to 3,000 and shorten the excerpts in
  `configs/small_loop.yaml` to match.
- A loss that is NaN or infinite: the notebook stops.
- RLVR projects over 4 hours: it cuts to 100 prompts on its own.

**Total:** about 6.5 session hours, with one Save & Run All and no restart (measured). The plan
estimated 4 to 10.

## What this will show

- **It can show** that both loops run, and whether each beats Model A on synthetic wording it has
  not seen.
- **It cannot show** that the model is better on real agreements. Five agreements are too few to
  separate the models, and their gold set is Claude's labels, which the user has not reviewed. The
  same agreements also shaped the labeling rules.
- **Main risk:** the model learns the bank's wording rather than the clauses. The real-agreement
  row is the only direct check, and a gap there is a finding, not a failure of the loop.

## What it showed

Measured, from `reports/small_loop.md`:

- **Both loops ran end to end** on free Kaggle GPU, in one session and with no restart.
- **SFT works on synthetic wording it has not seen.** On the held-out set, its mean reward was
  0.982 against 0.675 for Model A, and its Strict F1 was 0.983 against 0.701.
- **RLVR added almost nothing.** SFT already scored near 1.0 on the synthetic prompts, so most
  groups of 4 answers had no reward spread. On the held-out set RLVR and SFT tie.
- **On the 5 M1 agreements the direction is good.** Strict F1 rose from 0.657 for Model A to 0.778
  for SFT and 0.806 for RLVR, and the share of right quotes rose from 24% to 56% and 57%. Almost
  all of this gain comes from fewer wrong quotes: 271 quotes became 113, while right quotes
  stayed at about 63.
- **Some real quotes stop too early,** at the colon before a flattened pricing grid. DeepSeek
  stops at the same points in the teacher trial, so this is not clearly an effect of the short
  synthetic answers.

## After the small loop: teacher trial

On 2026-11-02 the parked plan's teacher trial ran, on DeepSeek's own API, as the user chose
(README, Decisions). It passed its gate on the 5 M1 agreements: Found F1 0.951 and 80% of
quotes right, against 0.90 and 60% needed. It cost $0.25 at the peak price. Results are in
`reports/teacher_trial.md`. Full pre-labeling (about $14-22 in the parked plan) needs its own
estimate and approval.

The same day, `src/discover.py` collected 150 more agreements (README, Status). They make about
4,435 chunks at 24,000 characters with 4,000 overlap. At the trial's measured $0.00174 per chunk
at peak price, full pre-labeling would cost about $7.70 at peak, or about $3.90 off-peak, plus
retries. The user approved it, and it ran the same day for $7.56 at peak price, off-peak, in 14
minutes (`reports/prelabel_full.md`). Next in the parked plan is your review of the 30 test
agreements (step 1d), then training on the 120 train agreements (step 2).

Step 2 comes first, at the user's request: `notebooks/05_real_sft.ipynb` trains on the 1,447
examples from the train agreements and scores the result on the 5 M1 agreements, the only gold
set so far (README, Decisions). Scoring on the 30 test agreements waits for step 1d.

It ran the same day on Kaggle (`reports/real_sft.md`, measured): 146 steps, about 81% of one
epoch, in 6.0 hours. On the 5 M1 agreements, against Model A on the same chunks, Found F1 went
from 0.756 to 0.951 and Strict F1 from 0.542 to 0.776. The student matches the teacher's Found
F1 but not its Strict F1 (0.829) or its share of right quotes (80% against 55%). Most of the extra
wrong quotes are whole incremental sections quoted sentence by sentence, mention-only Maturity
Date sentences, and builder-basket items quoted again on their own.

## Step 3: RLVR on the pre-labels

Written 2026-11-02, before any GPU run. The design is in the README (Decisions). On Kaggle it
would take about 7 hours with training capped at 4.5 (estimate). The user approved one run on a
Nebius H100 instead:

| part | hours (estimate) |
|---|---|
| VM start, two venvs, model download | about 0.5 |
| screening 800 windows x 4 samples, and SFT on the check set | about 0.25 |
| short test, then GRPO on up to 400 windows, capped at 2 hours | about 1-1.5 |
| RLVR merge, M1 decode, check set | about 0.25 |

At $4.50 an hour on demand that is about $9-11. The VM is stopped from outside at 4 hours, so
it cannot pass $18, and it is deleted with its disk once the adapter and report are on the Hub.

## Checked so far

Measured on CPU, with `tests/test_small_loop.py`:

- The real SFT, restart, RLVR and merge code runs end to end on a tiny random model, through the
  same command lines the notebook uses.
- Every synthetic answer scores a reward of exactly 1.0 on its own excerpt.
- RLVR changes the adapter when rewards differ, keeps it in fp32, and samples with the cache on.
- Scoring, the gate and the report run end to end with a fake model.

Checked on the T4 run of 2026-11-02 (measured): 4-bit loading, fp16 training, memory, speed, and
vLLM loading a merged model all worked.
