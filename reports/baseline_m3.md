# Milestone 3: label-free baselines on 5 credit agreements

**Interim report.** The primary passes ran, and the ladder has not. The generated section below
is the notebook's report as written on 2026-10-30 at 20:40. Its ladder rows read 0 because no
ladder ran, not because the ladder failed. The final notebook report replaces this section.
`review_sheet.csv` is on the Hub under `m3_baselines/` and is not in `reports/` yet.

## Observations

Written 2026-10-31 from the report and run log below. Nothing in this section comes from
reading raw outputs or the review sheet, and no quote has been judged yet.

1. **Model B ignored the credit schema.** Condition b has 0 quotes. All 99 usable outputs have
   unknown keys, and none has every credit key. Its empty answers are 291 tokens, the same as
   condition a's 41-key empty answer, while condition c's 10-key empty answer is 82 tokens. So
   Model B very likely kept answering in CUAD's 41 keys. That is an inference from these
   counts, because the raw outputs were not opened. The `0/5` in the table says nothing about
   the agreements, and this is the risk of a model trained on one prompt shape.
2. **Model A follows the schema.** It has one loop in 104 chunks, 96 of 103 usable outputs have
   every key, and it found quotes for all 10 clause types. Found is not correct, so this says
   nothing about quality yet.
3. **Model A copies poorly.** With quote marks and case ignored, 69.3% of its quotes are in the
   document, so about 31% are not. The cause is unknown until the sheet is read.
4. **Quote marks explain part of the gap.** Treating curly and straight marks as equal lifts
   condition a from 0.8458 to 0.9708, and condition c from 0.6142 to 0.6929. The M1 text has
   only curly marks, so the models are writing straight ones. The restore step in
   `docs/next_steps.md` is likely to be needed, and the pre-labeling trial will show whether
   the teacher does the same.
5. **Conditions a and c are different tasks.** Condition a's 0.9708 is on CUAD clauses and
   condition c's 0.6929 is on credit clauses, so this is not a fair copying comparison.
6. **Starting model.** The brief says to start from B if it clearly beats A here, otherwise
   from A. Model B did not beat A on the credit schema, so the default is A, subject to the
   review.
7. **Found counts that look wrong against the M1 manifest:**
   - `viavi-2025` is described as having no financial covenant, but condition c has 7 Financial
     Maintenance Covenant quotes.
   - Debt Incurrence General Basket is found in 2 of 5 agreements and Liens General Basket in
     4 of 5. Secured deals usually have both, so `angi-2025` and `repay-2026` may be misses.
   - A general basket is usually one clause, but `exlservice-2026` has 19 Liens quotes and
     `angi-2025` has 17 Restricted Payments quotes. Many may be other exceptions.
   - `quanta-2024` has none in six clause types, which fits its description as an
     investment-grade revolver with few baskets.
8. **Loops are the main runtime cost.** Batches with no loop ran at 17-22 s per chunk, batches
   with one loop at 50-55 s, and the one batch with three loops at 153 s. The primary passes
   took 188 minutes, against my estimate of about 2.1 hours.
9. **The token cap could come down in later runs.** The longest output that was not a loop is
   2,529 tokens across all 312 decodes, so a 4,096 cap would have cut no valid output here. M3
   keeps 12,288 so the three conditions stay comparable.
10. **Loops fall on different chunks.** Condition a: quanta 10, exlservice 5, angi 16, repay 21.
    Condition b: quanta 10, exlservice 5, viavi 19, 20, 21. Condition c: exlservice 0.

Not yet run or done: the ladder (a 4, b 5, c 1 chunks), the final report, and the review of
`review_sheet.csv` using `docs/labeling_guide.md`.

## Generated report (primary passes only)

- Platform: kaggle, GPUs: ['Tesla T4', 'Tesla T4']
- Written: 2026-10-30 20:40, session wall time so far: 188.3 min
- Documents: quanta-2024, exlservice-2026, angi-2025, repay-2026, viavi-2025
- Chunks: 104 at 32000 chars / 4000 overlap, identical for every condition (sha256 c140f8ccd495)

No labels: nothing here says whether an extraction is right. That is what `review_sheet.csv` is for.

**Incomplete:** condition a: 0 chunks without a primary decode, 4 waiting for a ladder rung; condition b: 0 chunks without a primary decode, 5 waiting for a ladder rung; condition c: 0 chunks without a primary decode, 1 waiting for a ladder rung. Numbers below cover what has run.

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
| recovered by rung | {'1.05': 0, '1.15': 0, '1.25': 0} | {'1.05': 0, '1.15': 0, '1.25': 0} | {'1.05': 0, '1.15': 0, '1.25': 0} |
| ladder recovery | 0.0 | 0.0 | 0.0 |
| still failing (ladder done) | 0 | 0 | 0 |
| waiting for a ladder rung | 4 | 5 | 1 |
| usable after everything | 0.9615 | 0.9519 | 0.9904 |
| usable outputs with every clause key | 99/100 (99.0%) | 0/99 (0.0%) | 96/103 (93.2%) |
| usable outputs with unknown keys | 0/100 (0.0%) | 99/99 (100.0%) | 7/103 (6.8%) |
| verbatim, exact vs own chunk | 0.8508 | None | 0.6196 |
| verbatim, casefolded vs document | 0.8458 | None | 0.6142 |
| same, curly and straight quote marks equal | 0.9708 | None | 0.6929 |
| verbatim, quotes from repaired chunks | None | None | None |
| quotes after merge | 240 | 0 | 267 |
| primary latency per chunk, mean / max s | 38.83 / 52.45 | 39.2 / 152.98 | 25.25 / 54.47 |
| ladder latency per attempt, mean / max s (n) | None / None (0) | None / None (0) | None / None (0) |
| wall minutes, primary / ladder | 71.1 / 0.0 | 70.2 / 0.0 | 46.9 / 0.0 |
| GPU-device minutes | 142.2 | 140.4 | 93.8 |
| backends that completed a stage | primary=vllm:tp2 | primary=vllm:tp2 | primary=vllm:tp2 |
| failed backend attempts | none | none | none |

Primary latency is batch wall time / batch size for vLLM (throughput), and per request for transformers. Ladder attempts run one at a time on transformers, unless a vLLM ladder backend is listed above.

### Credit clauses found

Documents with at least one quote, then quotes after merge, per condition. Found is not correct: see the review sheet.

| clause | b: docs found | c: docs found | b: quotes | c: quotes |
|---|---|---|---|---|
| Financial Maintenance Covenant | 0/5 | 5/5 | 0 | 15 |
| Incremental Facility | 0/5 | 4/5 | 0 | 47 |
| MFN Protection | 0/5 | 3/5 | 0 | 11 |
| Restricted Payments General Basket | 0/5 | 4/5 | 0 | 46 |
| Available Amount / Builder Basket | 0/5 | 3/5 | 0 | 23 |
| Debt Incurrence General Basket | 0/5 | 2/5 | 0 | 12 |
| Liens General Basket | 0/5 | 4/5 | 0 | 39 |
| Applicable Margin | 0/5 | 5/5 | 0 | 32 |
| Maturity Date | 0/5 | 5/5 | 0 | 28 |
| Liability Management Protections | 0/5 | 5/5 | 0 | 14 |

Quotes per document (b / c):

| clause | quanta-2024 | exlservice-2026 | angi-2025 | repay-2026 | viavi-2025 |
|---|---|---|---|---|---|
| Financial Maintenance Covenant | 0 / 2 | 0 / 2 | 0 / 2 | 0 / 2 | 0 / 7 |
| Incremental Facility | 0 / 0 | 0 / 11 | 0 / 10 | 0 / 11 | 0 / 15 |
| MFN Protection | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 3 | 0 / 7 |
| Restricted Payments General Basket | 0 / 0 | 0 / 3 | 0 / 17 | 0 / 13 | 0 / 13 |
| Available Amount / Builder Basket | 0 / 0 | 0 / 0 | 0 / 3 | 0 / 3 | 0 / 17 |
| Debt Incurrence General Basket | 0 / 0 | 0 / 7 | 0 / 0 | 0 / 0 | 0 / 5 |
| Liens General Basket | 0 / 0 | 0 / 19 | 0 / 2 | 0 / 3 | 0 / 15 |
| Applicable Margin | 0 / 2 | 0 / 6 | 0 / 7 | 0 / 9 | 0 / 8 |
| Maturity Date | 0 / 2 | 0 / 9 | 0 / 8 | 0 / 3 | 0 / 6 |
| Liability Management Protections | 0 / 1 | 0 / 3 | 0 / 4 | 0 / 2 | 0 / 4 |

### Condition a: CUAD clauses found

15 of 41 CUAD clause types have a quote in at least one document: Document Name (100%), Parties (100%), Agreement Date (100%), Governing Law (100%), Change of Control (100%), Anti-Assignment (100%), Volume Restriction (100%), Audit Rights (100%), Insurance (100%), Minimum Commitment (80%), Cap on Liability (80%), Effective Date (40%), Expiration Date (20%), Most Favored Nation (20%), Termination for Convenience (20%).

Review sheet: 327 rows.
