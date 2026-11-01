# Milestone 3: label-free baselines on 5 credit agreements

**Final report.** All primary passes and ladders ran. The generated section below is the
notebook's report as written on 2026-10-31 at 19:52. `review_sheet.csv` has 330 rows and is on
the Hub under `m3_baselines/`. It is not in `reports/` yet, because this machine is not logged
in to the private Hub repo.

## Observations

Written 2026-10-31 from the reports and run logs of both sessions. Nothing in this section comes
from reading raw outputs or the review sheet, and no quote has been judged yet.

1. **Model B ignored the credit schema.** Condition b has 0 quotes even after the ladder. All
   104 outputs have unknown keys, and none has every credit key. Its empty answers are 291
   tokens, the same as condition a's 41-key empty answer, while condition c's 10-key empty
   answer is 82 tokens. So Model B very likely kept answering in CUAD's 41 keys. That is an
   inference from these counts, because the raw outputs were not opened. The `0/5` in the table
   says nothing about the agreements, and this is the risk of a model trained on one prompt
   shape.
2. **Model A follows the schema.** It has one loop in 104 chunks, 97 of 104 outputs have every
   key, and it found quotes for all 10 clause types. Found is not correct, so this says nothing
   about quality yet.
3. **Model A copies poorly.** With quote marks and case ignored, 68.3% of its quotes are in the
   document, so about 32% are not. The cause is unknown until the sheet is read.
4. **Quote marks explain part of the gap.** Treating curly and straight marks as equal lifts
   condition a from 0.8556 to 0.9747, and condition c from 0.6052 to 0.6827. The M1 text has
   only curly marks, so the models are writing straight ones. The restore step in
   `docs/next_steps.md` is likely to be needed, and the pre-labeling trial will show whether
   the teacher does the same.
5. **Conditions a and c are different tasks.** Condition a's 0.9747 is on CUAD clauses and
   condition c's 0.6827 is on credit clauses, so this is not a fair copying comparison.
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
8. **The ladder recovered every failed chunk.** All 10 recovered, so all 312 decodes are usable:
   6 at 1.05, 2 at 1.15 and 2 at 1.25.
9. **Recovered chunks did not behave alike.** Condition a gained 37 quotes, and all 37 are in
   the document with quote marks and case ignored (233 of 240 before, 270 of 277 after). Its
   CUAD clause types with a quote rose from 15 to 25. Condition c gained 4 quotes, all in
   `exlservice-2026`, and none of the 4 is in the document (185 of 267 before, 185 of 271
   after). These counts are worked out from the reported rates.
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

Not yet done: the review of `review_sheet.csv` using `docs/labeling_guide.md`.

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

Review sheet: 330 rows.
