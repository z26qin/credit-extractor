# Next steps

Plan only: nothing here has run, and every time below is an estimate unless it says measured.
No step uses Nebius or costs money.

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

## 1. Synthetic data

New `src/synth.py` builds the data from a fixed seed, on CPU inside the notebook.

**Text bank,** in new `configs/synth_bank.yaml`, written from the clause definitions and not from
the gold passages:

- **60 clause passages:** 6 templates for each of the 10 clause types, with slots for amounts,
  percentages, dates and names. Lengths follow the gold set: median 362 characters, 90th
  percentile 1,430 (measured).
- **40 look-alikes,** aimed at M3's wrong quotes: neighboring basket items (`adjacent`, 107 wrong
  quotes), cross-references (`mention`, 29), headings (`heading`, 17), and text that belongs to
  another clause type (`belongs`, 18).
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

One template per clause type and 8 look-alikes appear only in the held-out set, so that set tests
wording the model has not seen. A leak check confirms that no run of 15 words from the bank appears
in the 5 M1 agreements.

## 2. Small SFT

New `src/train.py` runs Unsloth QLoRA on Model A.

- **Recipe from the model card:** 4-bit base, r=16, alpha=32, dropout 0, all linear modules,
  `adamw_8bit`, learning rate 2e-4, warmup 6%, 2 epochs, batch 1 with 8 accumulation steps.
- **Changes for T4:** fp16 and `max_seq_length` of 4,096. Examples over the limit are dropped and
  counted.
- **Size:** 600 example passes, or 75 optimizer steps. An average example is about 2,500 tokens.
- **Saving:** the adapter goes to the private model repo every 25 steps, and a restart resumes
  from the latest one.
- **Time:** 0.5 to 2 hours. This assumes a T4 is 3 to 7 times slower than upstream's RTX 5090, at
  about half the sequence length.

**Gate:** SFT's mean reward (section 3) on the held-out set must beat Model A's by at least 0.10. If
it does not, the notebook stops, because RLVR would be building on a setup that does not work.

## 3. Small RLVR

New `src/rlvr.py` starts from the SFT adapter and calls TRL's `GRPOTrainer`. We write only the data
and the reward.

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

**Time:** 1 to 3 hours, or 20 to 50 seconds per prompt. This assumes 15 to 30 tokens per second on
excerpts of about 1,500 tokens. The M3 transformers ladder reached only about 3 tokens per second
on much longer context (measured), so the short test below checks this first.

## 4. Comparison

Each adapter is merged into a local folder in the notebook, then goes through the M3 runner and
`src.score`, with temperature 0 and the same repeat-penalty ladder. The token cap is 4,096, because
the longest valid M3 output was 2,529 tokens.

| set | Model A | SFT | RLVR |
|---|---|---|---|
| 60 held-out excerpts | runs | runs | runs |
| 5 M1 agreements | already measured | runs | runs |

**Time:** about 30 minutes for the held-out set. On real agreements, 1 to 2 hours per adapter;
Model A took 52 minutes (measured).

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

- Out of memory: lower `max_seq_length` to 3,000 and shorten the excerpts to match.
- RLVR projects over 4 hours: cut to 100 prompts.

**Total:** 4 to 10 session hours, against about 30 GPU hours a week. If T4 x2 counts double, which
Kaggle does not confirm, that is 8 to 20 hours of quota, and it still fits in one week.

## What this will show

- **It can show** that both loops run, and whether each beats Model A on synthetic wording it has
  not seen.
- **It cannot show** that the model is better on real agreements. Five agreements are too few to
  separate the models, and their gold set is Claude's labels, which the user has not reviewed. The
  same agreements also shaped the labeling rules.
- **Main risk:** the model learns the bank's wording rather than the clauses. The real-agreement
  row is the only direct check, and a gap there is a finding, not a failure of the loop.

## To build

- `configs/synth_bank.yaml` and `src/synth.py`
- `src/train.py`, `src/rlvr.py` and `requirements-train.txt`, with `unsloth` and `trl` pinned
- `configs/small_loop.yaml`, and `src/baselines.py` taking its conditions from the config
- `src/score.py` reporting the share of right quotes and the mean reward
- `notebooks/04_small_loop.ipynb`
- CPU tests for the generator and the reward
