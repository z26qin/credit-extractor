# Teacher trial

`deepseek-flash` at `https://api.deepseek.com` labeled the 5 M1 agreements in 24,000-character chunks with 4,000 overlap, page numbers removed, thinking off, at temperature 0, with up to 2 more answers at 0.3 for a chunk with a failed quote.

**Gate: pass.** Found F1 0.951 against 0.90 needed, and share of quotes right 0.8043 against 0.60 needed.

## Spend

| chunks labeled | requests | input tokens | of them cached | output tokens | cost at peak price |
|---|---|---|---|---|---|
| 144 of 144 | 146 | 905,413 | 177,280 | 25,855 | $0.2505 |

Cap $0.70. Off-peak hours bill half the peak price. 1.6 seconds per request on average.

## Checks

- Chunks retried: 1. Left out of training: 1. Stopped by the budget: 0.
- Answers that did not parse: 0. Answers cut off at 4096 tokens: 0. Quotes under a key that is not a clause type: 0.

| clause | quotes passing | quotes failing the check | share failing |
|---|---|---|---|
| Financial Maintenance Covenant | 8 | 0 | 0.00 |
| Incremental Facility | 17 | 3 | 0.15 |
| MFN Protection | 5 | 0 | 0.00 |
| Restricted Payments General Basket | 4 | 0 | 0.00 |
| Available Amount / Builder Basket | 6 | 0 | 0.00 |
| Debt Incurrence General Basket | 6 | 0 | 0.00 |
| Liens General Basket | 7 | 0 | 0.00 |
| Applicable Margin | 16 | 0 | 0.00 |
| Maturity Date | 18 | 0 | 0.00 |
| Liability Management Protections | 12 | 0 | 0.00 |

Passing and failing are counted per answer, so a chunk asked again counts twice.

## Scores

"Teacher" is every passing quote. "Kept chunks only" leaves out the chunks that would not enter training. Five agreements are few, and they shaped the labeling rules, so read this as a direction.

Gold set labeled by: Claude, 2026-11-01, at the user's request; not reviewed by the user. 41 clause decisions have gold text, out of 50.

| metric | Model A | Teacher | Teacher, kept chunks only |
|---|---|---|---|
| Found F1 | 0.829 (0.740-0.905) | 0.951 (0.857-1.000) | 0.951 (0.857-1.000) |
| Strict F1 | 0.657 (0.492-0.744) | 0.829 (0.725-0.918) | 0.812 (0.708-0.909) |
| upstream detection F1 | 0.902 | 1.000 | 1.000 |
| upstream strict F1 | 0.333 | 0.633 | 0.633 |
| mean reward | 0.422 | 0.814 | 0.806 |
| share of quotes right | 0.2362 | 0.8043 | 0.8022 |
| quotes | 271 | 92 | 91 |
| verbatim, quote marks and case ignored | 0.6827 | 0.8913 | 0.8901 |
| same, page numbers removed | 0.7269 | 1.0 | 1.0 |

Brackets are 95% bootstrap intervals over agreements.

| clause | agreements with it | Model A: Found | Model A: Strict | Teacher: Found | Teacher: Strict | Teacher, kept chunks only: Found | Teacher, kept chunks only: Strict |
|---|---|---|---|---|---|---|---|
| Financial Maintenance Covenant | 4 | 0.889 | 0.750 | 1.000 | 0.857 | 1.000 | 0.857 |
| Incremental Facility | 4 | 1.000 | 0.667 | 1.000 | 0.857 | 1.000 | 0.857 |
| MFN Protection | 2 | 0.800 | 0.000 | 1.000 | 0.667 | 1.000 | 0.667 |
| Restricted Payments General Basket | 4 | 0.750 | 0.857 | 0.750 | 0.857 | 0.750 | 0.857 |
| Available Amount / Builder Basket | 3 | 0.857 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 |
| Debt Incurrence General Basket | 5 | 0.571 | 0.571 | 1.000 | 1.000 | 1.000 | 1.000 |
| Liens General Basket | 5 | 0.667 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| Applicable Margin | 5 | 1.000 | 0.889 | 0.800 | 0.333 | 0.800 | 0.333 |
| Maturity Date | 5 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 0.889 |
| Liability Management Protections | 4 | 0.667 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |

Paired difference from Model A, same agreement resamples:

| model | Found F1 difference (95% CI) | Strict F1 difference (95% CI) |
|---|---|---|
| Teacher | +0.122 (-0.049 to +0.260) | +0.172 (-0.002 to +0.365) |
| Teacher, kept chunks only | +0.122 (-0.049 to +0.260) | +0.155 (-0.002 to +0.319) |

## Notes added by hand

Added on 2026-11-02 after reading the records. Everything above was written by `src.teacher`.

- **When and what it cost.** The run took about a minute, from 13:49 to 13:50 UTC on a Monday.
  That is off-peak for DeepSeek, so the bill should be about half the $0.25 recorded. The
  DeepSeek account is the only place to confirm it. The probe on 3 chunks projected $0.29 for
  the whole trial.
- **Most chunks have nothing to quote.** 94 of the 144 answers kept no quote, so the mean
  answer was 177 tokens, far below the 1,050 to 2,100 the estimate assumed.
- **Left-out chunk:** `repay-2026__chunk9`. All 3 answers quoted a "Ratio Incremental Amount"
  definition that does not match the chunk text, so the chunk would not enter training.
- **Compared with the small loop** on the same 5 agreements (`reports/small_loop.md`), the teacher
  is ahead: Found F1 0.951 against 0.857 for SFT and 0.892 for RLVR, and 80% of quotes right
  against 56% and 57%.
- **Weak spots.** For Applicable Margin, the definitions in viavi-2025 and exlservice-2026 stop
  at the colon before a flattened pricing grid, which the gold passage includes. SFT stops at the
  same two points. For Liability Management Protections, every quote is right, but in each of the 4
  agreements that have it, at least one main passage (one per J.Crew, Chewy or Serta type) is
  not inside any quote, so Strict F1 is 0. One Restricted Payments basket, in viavi-2025, is missed.
- **Why this may flatter the teacher.** Its prompt carries the labeling rules the gold set was
  built under, and the same 5 agreements shaped those rules. Claude labeled the gold set, and the
  user has not reviewed it.
