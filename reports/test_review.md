# Review of the 30 test agreements

Step 1d of the parked plan, done on 2026-11-03 by Claude at the user's request, not by the user.
No model was run and nothing was paid for. The labels are in
[test_review_sheet.csv](test_review_sheet.csv), under [the labeling guide](../docs/labeling_guide.md),
and `python -m src.gold` turned them into [gold_test.json](gold_test.json) with no problems:
394 gold passages over 30 agreements.

## What was reviewed

- **Every teacher quote**: 447 quotes from the full pre-labeling run (`prelabel_full/`), merged
  across chunks, one row each. All 447 are verbatim. Each got `yes`, `partial` or `no` with a tag.
- **Every clause type with no quote**: 91 agreement and clause pairs. Each was searched with the
  guide's search terms (a regex per clause type), and the hits were read. 83 are `absent` and 8
  are `missed`, with the text added as a span.
- **Some extra searches**, where the quotes looked incomplete: missing main passages (for example
  Outfront's term loan margin, Steris's right to increase commitments, LCI's revolver maturity)
  and other liability management types (Alnylam, DigitalOcean, Huntsman, Outfront).

How the review was done, which matters for how far to trust it:

- **The agreements were not read in full.** They run 300,000 to 960,000 characters each. Each
  quote was read, and for each missing clause type the search hits were read. A clause that the
  search terms miss, or a main passage hidden beside a right quote, could still be missing.
  Clause types with a right quote were not searched again in most agreements, so recall is
  probably overstated, as the guide warns.
- **The labels start from the teacher's quotes.** So the teacher's score below is flattered, as
  Model A's was on M1.
- **16 rows are tagged `unsure`.** Their notes say why. The main ones are listed under Rule
  calls.

## The teacher against these labels

Scored with `src.score`, as on M1. 217 of the 300 clause decisions have gold text.

| metric | Teacher |
|---|---|
| Found F1 | 0.981 (0.968-0.993) |
| Strict F1 | 0.967 (0.951-0.981) |
| share of quotes right (scorer's rule) | 0.848 |
| quotes labeled `yes` | 374 of 447 (84%), plus 3 `partial` |

| clause | agreements with it | Found | Strict | quotes yes / partial / no |
|---|---|---|---|---|
| Financial Maintenance Covenant | 26 | 1.000 | 1.000 | 62 / 0 / 0 |
| Incremental Facility | 27 | 1.000 | 0.962 | 77 / 3 / 25 |
| MFN Protection | 8 | 1.000 | 1.000 | 11 / 0 / 0 |
| Restricted Payments General Basket | 17 | 0.970 | 0.970 | 17 / 0 / 1 |
| Available Amount / Builder Basket | 8 | 1.000 | 1.000 | 15 / 0 / 5 |
| Debt Incurrence General Basket | 24 | 1.000 | 1.000 | 25 / 0 / 4 |
| Liens General Basket | 26 | 1.000 | 1.000 | 28 / 0 / 0 |
| Applicable Margin | 29 | 1.000 | 0.983 | 58 / 0 / 4 |
| Maturity Date | 30 | 1.000 | 0.983 | 56 / 0 / 31 |
| Liability Management Protections | 22 | 0.811 | 0.743 | 25 / 0 / 0 |

Brackets are 95% bootstrap intervals over agreements. These are higher than the teacher's M1
scores (Found 0.951, Strict 0.829), but the two are not comparable: M1's gold was built from
Model A's quotes and the M1 agreements shaped the rules, while this gold was built from the
teacher's own quotes.

## Where the teacher goes wrong

- **It misses liability management protections.** 7 of the 8 misses are here: serta-type
  consent rights in Amphastar, Beyond, Coeur, Macerich, ServiceNow and Veris, and both jcrew and
  chewy in Outfront. These sit in the amendments section, as one item in a list of "each
  affected Lender" consents, far from the other clauses. In Alnylam and DigitalOcean it found
  jcrew but missed a chewy guarantor-release limit, which is a main passage, so Strict counts it.
  In Huntsman it missed a second jcrew passage, which is not main.
- **The 8th miss is a general basket gated by a ratio**: OneWater's "other Restricted Payments
  ... not to exceed $50,000,000" if leverage is at most 1.50:1.00.
- **The 70 wrong quotes are nearly all text next to the clause, not invented text:**
  - Maturity Date, 31: 24 sentences that only use the term (commitments end on the Maturity
    Date, loans are repaid on it) and 7 limits on other debt or amortization dates.
  - Incremental Facility, 25: 21 mechanics or limits on incremental terms (minimum amounts,
    who may lend, maturity and amortization limits, incremental equivalent debt), 2 bare
    definitions and 2 loan modification offers, moved to Maturity Date as extension terms.
  - Available Amount, 5: lettered deductions inside the definition, quoted again on their own,
    as SFT did on M1.
  - Debt, 4: non-guarantor caps, a purchase money basket and a prepayment definition.
  - Applicable Margin, 4: a bare cross-reference to the grid, fee rules outside it and incremental
    margins set elsewhere.
  - Restricted Payments, 1: a share buyback basket.
- **3 quotes are cut off** (Conduent and Lazard incremental clauses, Macerich's increase right).
  Each is `partial`, with the full passage as a span.

## Rule calls

Settled here, without a guide change. They are tagged `unsure` where the guide did not settle
them, so they can be decided together.

- **Liquidity and asset tests count as financial maintenance covenants**: Rubicon's minimum
  availability, and Macerich's floating rate debt cap and unencumbered pool value.
- **A capped basket gated by a ratio is still the general basket** (OneWater, Merit, Solv). An
  uncapped one is not (Coeur's clause (h)), nor is Conduent's $10,000,000-plus-contributed-equity
  debt basket.
- **Pricing set only by reference to another agreement is absent** (Conduent's Applicable Rate
  is "the Newco Applicable Rate").
- **A promise to add "customary most favored nation" pricing counts as MFN**, with no threshold
  (Alnylam, `unsure`). Revolving margin MFNs count (Amphastar, Merit). An `All-In Yield`
  definition used only for repricing is not MFN (CACI, Phreesia, `unsure`).
- **Increases of a sublimit are not incremental facilities** (United's Canadian and ROW maximums,
  `unsure`).
- **Agreements whose debt or lien covenant has no general basket** are `absent` for that clause:
  Beyond (only a $10,000,000 subordinated debt basket), LCI (ratio debt and a non-guarantor cap),
  Merit (no general lien basket).

## Not done

- The user has not reviewed these labels. The plan meant this review for the user, at about 45 to
  60 minutes an agreement; this one was a search-based pass and is weaker than that.
- No model has been scored on these agreements yet. Scoring Model A, SFT and RLVR here needs a
  GPU run, which needs its own estimate and approval.
