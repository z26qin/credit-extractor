# Teacher pre-labeling

`deepseek-flash` at `https://api.deepseek.com` labeled 150 agreements (120 train, 30 test) in 24,000-character chunks with 4,000 overlap, page numbers removed, thinking off, at temperature 0, with up to 2 more answers at 0.3 for a chunk with a failed quote.

These are pre-labels. Test labels come from your review, and no pre-label enters them unreviewed.

## Spend

| chunks labeled | requests | input tokens | of them cached | output tokens | cost at peak price |
|---|---|---|---|---|---|
| 4430 of 4430 | 4490 | 27,903,892 | 5,483,261 | 668,166 | $7.5611 |

Cap $8.50. Off-peak hours bill half the peak price. 1.5 seconds per request on average.

## Checks

- Chunks retried: 35. Left out of training: 9. Stopped by the budget: 0.
- Answers that did not parse: 1. Answers cut off at 4096 tokens: 0. Quotes under a key that is not a clause type: 0.

| clause | quotes passing | quotes failing the check | share failing |
|---|---|---|---|
| Financial Maintenance Covenant | 352 | 4 | 0.01 |
| Incremental Facility | 490 | 30 | 0.06 |
| MFN Protection | 73 | 3 | 0.04 |
| Restricted Payments General Basket | 103 | 0 | 0.00 |
| Available Amount / Builder Basket | 134 | 10 | 0.07 |
| Debt Incurrence General Basket | 155 | 3 | 0.02 |
| Liens General Basket | 153 | 0 | 0.00 |
| Applicable Margin | 358 | 12 | 0.03 |
| Maturity Date | 474 | 11 | 0.02 |
| Liability Management Protections | 178 | 7 | 0.04 |

Passing and failing are counted per answer, so a chunk asked again counts twice.

## What the labels hold

| | train | test |
|---|---|---|
| agreements | 120 | 30 |
| chunks | 3513 | 917 |
| chunks with a quote | 1013 | 261 |
| chunks left out | 5 | 4 |
| quotes kept | 1910 | 505 |

Agreements with at least one quote for each clause type, after merging chunks:

| clause | train | test |
|---|---|---|
| Financial Maintenance Covenant | 109 of 120 | 26 of 30 |
| Incremental Facility | 97 of 120 | 27 of 30 |
| MFN Protection | 33 of 120 | 8 of 30 |
| Restricted Payments General Basket | 66 of 120 | 16 of 30 |
| Available Amount / Builder Basket | 38 of 120 | 8 of 30 |
| Debt Incurrence General Basket | 93 of 120 | 24 of 30 |
| Liens General Basket | 98 of 120 | 26 of 30 |
| Applicable Margin | 118 of 120 | 29 of 30 |
| Maturity Date | 119 of 120 | 30 of 30 |
| Liability Management Protections | 55 of 120 | 15 of 30 |

## Notes added by hand

Added on 2026-11-02 after reading the records. Everything above was written by `src.teacher`.

- **Approval and cost.** The user approved this run at about $7.70 at peak price, plus retries,
  and it ran under an $8.50 cap. A probe on 20 chunks projected $7.61. The run took 14 minutes,
  from 15:42 to 15:56 UTC on a Monday. That is off-peak for DeepSeek, so the bill should be
  about $3.78, half the $7.56 recorded. Only the DeepSeek account can confirm it.
- **Left-out chunks.** All 9 have a clause type whose quotes never matched the chunk text: 2 for
  Available Amount, 2 for Liability Management Protections, 2 for Incremental Facility, and 1
  each for Maturity Date, Debt Incurrence General Basket and MFN Protection.
- **A recall check.** 43 of the 150 agreements define a builder-basket term, such as `Available
  Amount` or `Cumulative Credit` (`reports/collect_full_mix.md`). The teacher quotes the clause
  type in 41 of them, and in 5 agreements that define none of these terms.
- **By archetype,** an agreement has quotes for 8.7 clause types on average if leveraged, 6.2 if
  secured mid-cap and 4.8 if investment grade.
- **Not checked:** whether these quotes are right. The trial on the 5 M1 agreements found 80% of
  quotes right (`reports/teacher_trial.md`). Only the review of the 30 test agreements will
  measure the rest.
