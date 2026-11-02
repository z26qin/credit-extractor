# SFT on the teacher's pre-labels

QLoRA on `Qwen/Qwen3-4B-Instruct-2507`, trained on teacher labels of the 120 train agreements and scored on the 5 M1 agreements. Training and scoring use 24,000-character chunks with 4,000 overlap and page numbers removed.

## Data

| examples | with quotes | empty, used of available | over the length limit | left-out chunks skipped | tokens | longest |
|---|---|---|---|---|---|---|
| 1447 | 1013 | 434 of 2495 | 0 | 5 | 8,465,700 | 8,088 |

## Training, measured

Short test on the 24 longest examples: 211.83 seconds per step, peak 10571 MiB, projected 8.06 hours for one epoch, so 146 of 181 steps.

| steps | seconds per step | peak GPU MiB | first loss | last loss |
|---|---|---|---|---|
| 146 | 148.49 | 10571 | 0.10832469463348389 | 0.011955894529819489 |

## The 5 M1 agreements

Five agreements cannot separate these models, and they shaped the labeling rules. Model A ran again on these chunks, so both rows are comparable.

Gold set labeled by: Claude, 2026-11-01, at the user's request; not reviewed by the user. 41 clause decisions have gold text, out of 50.

| metric | Model A | SFT |
|---|---|---|
| Found F1 | 0.756 (0.622-0.886) | 0.951 (0.882-1.000) |
| Strict F1 | 0.542 (0.413-0.667) | 0.776 (0.635-0.909) |
| upstream detection F1 | 0.951 | 1.000 |
| upstream strict F1 | 0.170 | 0.536 |
| mean reward | 0.360 | 0.678 |
| share of quotes right | 0.2222 | 0.5517 |
| quotes | 243 | 116 |
| verbatim, quote marks and case ignored | 0.6914 | 0.9138 |
| same, page numbers removed | 0.7407 | 1.0 |

Brackets are 95% bootstrap intervals over agreements.

| clause | agreements with it | Model A: Found | Model A: Strict | SFT: Found | SFT: Strict |
|---|---|---|---|---|---|
| Financial Maintenance Covenant | 4 | 0.667 | 0.750 | 1.000 | 0.857 |
| Incremental Facility | 4 | 1.000 | 0.857 | 1.000 | 0.857 |
| MFN Protection | 2 | 0.800 | 0.000 | 1.000 | 0.000 |
| Restricted Payments General Basket | 4 | 0.500 | 0.667 | 0.750 | 0.857 |
| Available Amount / Builder Basket | 3 | 1.000 | 0.800 | 1.000 | 1.000 |
| Debt Incurrence General Basket | 5 | 0.444 | 0.000 | 0.800 | 0.750 |
| Liens General Basket | 5 | 0.444 | 0.000 | 1.000 | 1.000 |
| Applicable Margin | 5 | 0.800 | 0.571 | 1.000 | 0.333 |
| Maturity Date | 5 | 1.000 | 0.889 | 1.000 | 0.889 |
| Liability Management Protections | 4 | 1.000 | 0.000 | 1.000 | 0.400 |

Paired difference from Model A, same agreement resamples:

| model | Found F1 difference (95% CI) | Strict F1 difference (95% CI) |
|---|---|---|
| SFT | +0.195 (+0.028 to +0.342) | +0.234 (+0.122 to +0.322) |

## Checked after the run

Added by hand on 2026-11-02, from the raw records on the Hub (`real_sft/runs/m1/`) and the
teacher trial's records. Everything here is measured unless it says otherwise.

- **The numbers above are right.** Scoring the raw records again on the local machine, with the
  committed gold set and the M1 text, gives the same M1 section, character for character.
- **Run:** one Save & Run All on Kaggle GPU T4 x2, Version 1, with no restart. Training took
  21,686 seconds (6.0 hours) for 146 steps, about 81% of one epoch, so about 279 of the 1,447
  examples were not seen. Scoring took 31 minutes for each model. All 144 chunks gave a usable
  answer for both models: Model A's all on vLLM, SFT's 143 on vLLM and 1 on the transformers
  ladder.
- **Model A scores lower here than in M3** (Found F1 0.756 against 0.829, Strict F1 0.542
  against 0.657). M3 used 32,000-character chunks and kept page numbers. This was not looked
  into further. Compare SFT with the Model A row in this report, which ran on the same chunks.
- **Against the teacher on the same chunks** (`reports/teacher_trial.md`): Found F1 is the same,
  0.951. Strict F1 is 0.776 against 0.829, and the share of right quotes is 55% against 80%. SFT
  gives 116 quotes, 52 of them wrong. The teacher gave 92, 18 of them wrong.
- **Against the synthetic SFT from M4** (`reports/small_loop.md`, 32,000-character chunks, so a
  direction only): Found F1 rose from 0.857 to 0.951. Strict F1 (0.778 there, 0.776 here) and
  the share of right quotes (56%, 55%) are about the same.
- **Where the 52 wrong quotes are:**
  - Incremental Facility, 19. In exlservice-2026, 15 of SFT's 17 quotes are wrong: SFT quotes
    the procedural sentences of the incremental section one by one (notice to the agent, lenders
    choosing to join, the agent's notice afterward). The teacher gave 5 quotes there, 4 right.
  - Maturity Date, 13: sentences that only use the term (repaying loans on the maturity date,
    commitments ending on it) and the "Senior Notes Maturity Date".
  - Available Amount / Builder Basket, 9, all in angi-2025. SFT quotes the whole definition,
    which is right, and then 9 of its lettered items again as separate quotes, which each score
    as wrong.
  - The other 11 are spread over 6 clause types.
- **Many of the teacher's errors carry over.** 16 of the teacher's 18 wrong quotes overlap an
  SFT quote (one inside the other, or token F1 of at least 0.5), and 24 of SFT's 52 wrong
  quotes overlap a teacher quote. The 9 builder-basket pieces count here because they sit inside
  the teacher's one right quote.
- **Applicable Margin Strict fell from 0.571 to 0.333, although Found is 1.000.** Strict needs
  one quote that holds the whole main passage. SFT now copies the pricing grids, but it stops a
  long definition at the end of the grid or splits it into two quotes. In angi-2025, exlservice-2026
  and repay-2026, SFT stops at the same point as the teacher. In viavi-2025 its two quotes
  together hold the whole passage. Model A had longer quotes in angi-2025 and passed there.

Not done: scoring on the 30 test agreements, which waits for your review of them, and any
review of the teacher's labels on the train agreements.
