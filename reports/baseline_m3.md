# Milestone 3: label-free baselines on 5 credit agreements

**Final report.** All primary passes and ladders ran. The generated section below is the
notebook's report as written on 2026-10-31 at 19:52, checked against the copy on the Hub.
[`review_sheet.csv`](review_sheet.csv) is the notebook's final sheet, 330 rows, copied from the
Hub's `m3_baselines/`.

## Observations

Written 2026-10-31 from the reports, run logs, raw records and review sheet, and updated
2026-11-01 after the review below.

1. **Model B ignored the credit schema.** Condition b has 0 quotes even after the ladder. Its
   raw records show why: all 104 usable outputs have exactly CUAD's 41 keys and no credit key.
   The 49 empty answers are all 291 tokens long, the length of a 41-key empty answer. The
   `0/5` in the table says nothing about the agreements, and this is the risk of a model
   trained on one prompt shape.
2. **Model A follows the schema.** It has one loop in 104 chunks, 97 of 104 outputs have every
   key, and it found quotes for all 10 clause types. Found is not correct, so this says nothing
   about quality yet.
3. **Model A copies poorly but rarely invents.** With quote marks and case ignored, 68.3% of
   its quotes are in the document, so 86 of 271 are not. The review tagged 3 of the 86 as
   invented. The others are real text with small changes, such as quote marks, page numbers
   left in the M1 text, dropped clause numbers and added lead-in words. 21 of the 61 right
   quotes are in this group, so a verbatim check without a restore step would discard a third
   of the right answers.
4. **Quote marks explain part of the gap.** Treating curly and straight marks as equal lifts
   condition a from 0.8556 to 0.9747, and condition c from 0.6052 to 0.6827. The M1 text has
   only curly marks, so the models are writing straight ones. A step that restores quotes to
   the source text is likely to be needed if a trained model keeps doing this.
5. **Conditions a and c are different tasks.** Condition a's 0.9747 is on CUAD clauses and
   condition c's 0.6827 is on credit clauses, so this is not a fair copying comparison.
6. **Starting model.** The brief says to start from B if it clearly beats A here, otherwise
   from A. Model B did not beat A on the credit schema, so the default is A, subject to the
   review.
7. **Found counts that looked wrong, as the review settled them:**
   - `viavi-2025` has no financial covenant, and all 7 of its Financial Maintenance Covenant
     quotes are wrong.
   - All 5 agreements have a Debt Incurrence General Basket. Model A missed `angi-2025`,
     `repay-2026` and `quanta-2024`.
   - In `exlservice-2026`, 1 of 19 Liens quotes is the basket. In `angi-2025`, 1 of 17
     Restricted Payments quotes is.
   - `quanta-2024` is an investment-grade term loan. It has no incremental, MFN, restricted
     payments or builder terms, but it does have debt and liens baskets that Model A missed.
8. **The ladder recovered every failed chunk.** All 10 recovered, so all 312 decodes are usable:
   6 at 1.05, 2 at 1.15 and 2 at 1.25.
9. **Recovered chunks did not behave alike.** Condition a gained 37 quotes, and all 37 are in
   the document with quote marks and case ignored (233 of 240 before, 270 of 277 after). Its
   CUAD clause types with a quote rose from 15 to 25. Condition c gained 4 quotes from
   `exlservice-2026` chunk 0, which holds the table of contents. The review tagged 3 of the 4
   as invented, and these are the only invented quotes in the sheet. One is an Available Amount
   quote, although the agreement never uses the term. The fourth is the real pricing grid,
   which fails the verbatim check because a page number sits inside it.
10. **Most of the runtime is four failed retries.** Stage times add up to 519 minutes (8.7
    hours): 188 for the primary passes and 331 for the ladder. That is 1,038.5 GPU-device
    minutes (17.3 hours). They leave out setup and one 214 s attempt from a session that stopped
    early. Three logged attempts reached the 12,288-token cap and took about 70 minutes each.
    The longest attempt in condition a took 4,246 s, so it very likely hit the cap too. Those
    four took 283 minutes, which is 85% of the ladder and 54% of the whole run. The other
    logged attempts took 132-246 s.
11. **My estimates were low.** I estimated about 2.1 hours for the primary passes and 0.5-3 hours
    for the ladder, and they took 3.1 and 5.5 hours. The M0 attempts behind the ladder estimate
    never reached the cap, so the "about 13 minutes" per chunk in the notebook was wrong.
12. **The token cap should come down in later runs.** The longest output that was not a loop is
    2,529 tokens across the 312 primary decodes, and no recovered attempt in the logs exceeds
    1,076. A 4,096 cap would have cut no valid output seen here, and each failed retry would
    stop after about 15-25 minutes (estimate). M3 keeps 12,288 so the three conditions stay
    comparable.
13. **Loops fall on different chunks.** Condition a: quanta 10, exlservice 5, angi 16, repay 21.
    Condition b: quanta 10, exlservice 5, viavi 19, 20, 21. Condition c: exlservice 0.
14. **Resume worked on Kaggle.** The ladder ran across two sessions. The second one skipped all
    312 primary decodes and the finished ladder of condition a, then continued with condition b.

## Review results

Labeled 2026-11-01 by Claude at the user's request, under `docs/labeling_guide.md`, reading each
quote against the agreement text. These are not the user's labels. Every span was checked against
the source. Scores come from `src.score` against `reports/gold_m3.json`, and the full output is in
[`score_m3.md`](score_m3.md). Condition c only, since Model B gave no credit quotes.

| agreement | quotes | right | not found: absent / missed |
|---|---|---|---|
| quanta-2024 | 7 | 4 | 4 / 2 |
| exlservice-2026 | 64 | 10 | 1 / 0 |
| angi-2025 | 54 | 13 | 0 / 1 |
| repay-2026 | 49 | 12 | 0 / 1 |
| viavi-2025 | 97 | 22 | 0 / 0 |
| total | 271 | 61 | 5 / 4 |

Right means `yes` (52) or `partial` (9). The 210 wrong quotes are nearby provisions the rules
exclude (107), text that belongs to no clause (36), mentions of a term (29), right text under the
wrong clause (18), headings and cross-references (17), and invented text (3).

| clause | agreements that have it | quotes | right | Found F1 | Strict F1 |
|---|---|---|---|---|---|
| Financial Maintenance Covenant | 4 | 15 | 5 | 0.889 | 0.750 |
| Incremental Facility | 4 | 48 | 7 | 1.000 | 0.667 |
| MFN Protection | 2 | 11 | 7 | 0.800 | 0.000 |
| Restricted Payments General Basket | 4 | 47 | 3 | 0.750 | 0.857 |
| Available Amount / Builder Basket | 3 | 24 | 13 | 0.857 | 0.667 |
| Debt Incurrence General Basket | 5 | 12 | 2 | 0.571 | 0.571 |
| Liens General Basket | 5 | 39 | 3 | 0.667 | 0.750 |
| Applicable Margin | 5 | 33 | 6 | 1.000 | 0.889 |
| Maturity Date | 5 | 28 | 7 | 1.000 | 0.750 |
| Liability Management Protections | 4 | 14 | 8 | 0.667 | 0.000 |

| metric | F1 | 95% CI over agreements |
|---|---|---|
| Found, the guide's rule | 0.829 | 0.740-0.905 |
| Strict, the guide's rule | 0.657 | 0.492-0.744 |
| detection, upstream rule | 0.902 | |
| strict, upstream rule | 0.333 | |

These replace the 0.805 and 0.714 first given on 2026-11-01, which came from a quick script. That
script used the labels instead of the automatic right-quote rule, and it compared `yes` quotes
with themselves, so it did not count copying errors against Strict. With five agreements, the
intervals are rough.

What the review shows:

- **Noise is the main problem.** 1 quote in 4.4 is right. Model A finds the right area and then
  quotes everything near it.
- **General baskets are the weakest clauses.** Model A has the right text in 8 of the 14 places
  they exist. All three are wrong in `repay-2026`.
- **Liability management scores low on Strict** because most agreements hold more than one
  protection, and Model A rarely quotes all of them.
- **One open call:** the Drop Down Provision in `exlservice-2026` covers all material property,
  not only intellectual property, and that agreement has no unrestricted subsidiaries. It is
  counted as `jcrew` and tagged `unsure`.

## Generated report

- Platform: kaggle, GPUs: ['Tesla T4', 'Tesla T4']
- Written: 2026-10-31 19:52, session wall time so far: 235.0 min
- Documents: quanta-2024, exlservice-2026, angi-2025, repay-2026, viavi-2025
- Chunks: 104 at 32000 chars / 4000 overlap, identical for every condition (sha256 c140f8ccd495)

No labels: nothing here says whether an extraction is right. That is what `review_sheet.csv` is for.

### Decode stability, verbatim copying, cost

| metric | a: Model B, native CUAD prompt | b: Model B, credit schema prompt | c: Model A, credit schema prompt |
|---|---|---|---|
| model | `Ihteshamstar/qwen3-4b-cuad-extractor` | `Ihteshamstar/qwen3-4b-cuad-extractor` | `Qwen/Qwen3-4B-Instruct-2507` |
| prompt | `cuad_native` | `credit_schema` | `credit_schema` |
| primary decodes | 104 | 104 | 104 |
| JSON valid as generated | 100/104 (96.2%) | 99/104 (95.2%) | 103/104 (99.0%) |
| valid after quote repair | 0/104 (0.0%) | 0/104 (0.0%) | 0/104 (0.0%) |
| loops (cap or repetition) | 4/104 (3.8%) | 5/104 (4.8%) | 1/104 (1.0%) |
| other invalid | 0/104 (0.0%) | 0/104 (0.0%) | 0/104 (0.0%) |
| sent to ladder | 4 | 5 | 1 |
| recovered by rung | {'1.05': 3, '1.15': 0, '1.25': 1} | {'1.05': 2, '1.15': 2, '1.25': 1} | {'1.05': 1, '1.15': 0, '1.25': 0} |
| ladder recovery | 1.0 | 1.0 | 1.0 |
| still failing (ladder done) | 0 | 0 | 0 |
| waiting for a ladder rung | 0 | 0 | 0 |
| usable after everything | 1.0 | 1.0 | 1.0 |
| usable outputs with every clause key | 102/104 (98.1%) | 0/104 (0.0%) | 97/104 (93.3%) |
| usable outputs with unknown keys | 1/104 (1.0%) | 104/104 (100.0%) | 7/104 (6.7%) |
| verbatim, exact vs own chunk | 0.8606 | None | 0.6107 |
| verbatim, casefolded vs document | 0.8556 | None | 0.6052 |
| same, curly and straight quote marks equal | 0.9747 | None | 0.6827 |
| verbatim, quotes from repaired chunks | None | None | None |
| quotes after merge | 277 | 0 | 271 |
| primary latency per chunk, mean / max s | 38.83 / 52.45 | 39.2 / 152.98 | 25.25 / 54.47 |
| ladder latency per attempt, mean / max s (n) | 959.6 / 4245.69 (6) | 1548.6 / 4331.34 (9) | 205.4 / 205.44 (1) |
| wall minutes, primary / ladder | 71.1 / 96.5 | 70.2 / 229.9 | 46.9 / 4.6 |
| GPU-device minutes | 335.2 | 600.2 | 103.1 |
| backends that completed a stage | ladder=hf:tp0, primary=vllm:tp2 | ladder=hf:tp0, primary=vllm:tp2 | ladder=hf:tp0, primary=vllm:tp2 |
| failed backend attempts | none | none | none |

Primary latency is batch wall time / batch size for vLLM (throughput), and per request for transformers. Ladder attempts run one at a time on transformers, unless a vLLM ladder backend is listed above.

### Credit clauses found

Documents with at least one quote, then quotes after merge, per condition. Found is not correct: see the review sheet.

| clause | b: docs found | c: docs found | b: quotes | c: quotes |
|---|---|---|---|---|
| Financial Maintenance Covenant | 0/5 | 5/5 | 0 | 15 |
| Incremental Facility | 0/5 | 4/5 | 0 | 48 |
| MFN Protection | 0/5 | 3/5 | 0 | 11 |
| Restricted Payments General Basket | 0/5 | 4/5 | 0 | 47 |
| Available Amount / Builder Basket | 0/5 | 4/5 | 0 | 24 |
| Debt Incurrence General Basket | 0/5 | 2/5 | 0 | 12 |
| Liens General Basket | 0/5 | 4/5 | 0 | 39 |
| Applicable Margin | 0/5 | 5/5 | 0 | 33 |
| Maturity Date | 0/5 | 5/5 | 0 | 28 |
| Liability Management Protections | 0/5 | 5/5 | 0 | 14 |

Quotes per document (b / c):

| clause | quanta-2024 | exlservice-2026 | angi-2025 | repay-2026 | viavi-2025 |
|---|---|---|---|---|---|
| Financial Maintenance Covenant | 0 / 2 | 0 / 2 | 0 / 2 | 0 / 2 | 0 / 7 |
| Incremental Facility | 0 / 0 | 0 / 12 | 0 / 10 | 0 / 11 | 0 / 15 |
| MFN Protection | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 3 | 0 / 7 |
| Restricted Payments General Basket | 0 / 0 | 0 / 4 | 0 / 17 | 0 / 13 | 0 / 13 |
| Available Amount / Builder Basket | 0 / 0 | 0 / 1 | 0 / 3 | 0 / 3 | 0 / 17 |
| Debt Incurrence General Basket | 0 / 0 | 0 / 7 | 0 / 0 | 0 / 0 | 0 / 5 |
| Liens General Basket | 0 / 0 | 0 / 19 | 0 / 2 | 0 / 3 | 0 / 15 |
| Applicable Margin | 0 / 2 | 0 / 7 | 0 / 7 | 0 / 9 | 0 / 8 |
| Maturity Date | 0 / 2 | 0 / 9 | 0 / 8 | 0 / 3 | 0 / 6 |
| Liability Management Protections | 0 / 1 | 0 / 3 | 0 / 4 | 0 / 2 | 0 / 4 |

### Condition a: CUAD clauses found

25 of 41 CUAD clause types have a quote in at least one document: Document Name (100%), Parties (100%), Agreement Date (100%), Governing Law (100%), Change of Control (100%), Anti-Assignment (100%), Minimum Commitment (100%), Volume Restriction (100%), Audit Rights (100%), Cap on Liability (100%), Insurance (100%), Effective Date (40%), Expiration Date (20%), Most Favored Nation (20%), Competitive Restriction Exception (20%), Termination for Convenience (20%), Rofr/Rofo/Rofn (20%), Price Restrictions (20%), Joint IP Ownership (20%), Non-Transferable License (20%), Affiliate License-Licensee (20%), Post-Termination Services (20%), Uncapped Liability (20%), Liquidated Damages (20%), Covenant Not to Sue (20%).
