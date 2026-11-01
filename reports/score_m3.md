# Scores on 5 agreements

Gold set labeled by: Claude, 2026-11-01, at the user's request; not reviewed by the user. 41 clause decisions have gold text, out of 50.

| metric | Model A | Model B |
|---|---|---|
| Found F1 | 0.829 (0.740-0.905) | 0.000 (0.000-0.000) |
| Strict F1 | 0.657 (0.492-0.744) | 0.000 (0.000-0.000) |
| upstream detection F1 | 0.902 | 0.000 |
| upstream strict F1 | 0.333 | 0.000 |
| quotes | 271 | 0 |
| verbatim, quote marks and case ignored | 0.6827 | None |
| same, page numbers removed | 0.7269 | None |

Brackets are 95% bootstrap intervals over agreements.

| clause | agreements with it | Model A: Found | Model A: Strict | Model B: Found | Model B: Strict |
|---|---|---|---|---|---|
| Financial Maintenance Covenant | 4 | 0.889 | 0.750 | 0.000 | 0.000 |
| Incremental Facility | 4 | 1.000 | 0.667 | 0.000 | 0.000 |
| MFN Protection | 2 | 0.800 | 0.000 | 0.000 | 0.000 |
| Restricted Payments General Basket | 4 | 0.750 | 0.857 | 0.000 | 0.000 |
| Available Amount / Builder Basket | 3 | 0.857 | 0.667 | 0.000 | 0.000 |
| Debt Incurrence General Basket | 5 | 0.571 | 0.571 | 0.000 | 0.000 |
| Liens General Basket | 5 | 0.667 | 0.750 | 0.000 | 0.000 |
| Applicable Margin | 5 | 1.000 | 0.889 | 0.000 | 0.000 |
| Maturity Date | 5 | 1.000 | 0.750 | 0.000 | 0.000 |
| Liability Management Protections | 4 | 0.667 | 0.000 | 0.000 | 0.000 |

Paired difference from Model A, same agreement resamples:

| model | Found F1 difference (95% CI) | Strict F1 difference (95% CI) |
|---|---|---|
| Model B | -0.829 (-0.905 to -0.740) | -0.657 (-0.744 to -0.492) |

Automatic right-quote rule against the labels: 59 right quotes kept, 2 lost, 5 wrong quotes passed, 205 rejected.
