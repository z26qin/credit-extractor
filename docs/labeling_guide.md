# Labeling guide

Rules for filling `correct` and `note` in `reports/review_sheet.csv`. The same rules apply to
the 30 test agreements, so a label means the same thing everywhere.

Drafted 2026-10-31 from the definitions in `configs/credit_schema.yaml`, before any quote was
reviewed. The clause rules are a first draft, and the search terms have not been tried on the
agreements. Change them after the first agreement and log each change under
[Rule changes](#rule-changes).

## The M3 sheet

The final sheet has 330 rows. The earlier 327-row sheet lacks 4 `exlservice-2026` quotes, and
`quote_no` can differ in that agreement, so copy any labels across by quote text.

| rows | count | what to do |
|---|---|---|
| condition c, `found=True` | 271 | label each quote |
| condition c, `found=False` | 9 | search the agreement, then label `absent` or `missed` |
| condition b | 50 | skip: Model B answered in CUAD keys, so these rows say nothing about the agreement |

All five agreements were labeled on 2026-11-01 by Claude at the user's request, not by the user.

Order of work:

1. Start with `repay-2026`: 49 quotes across most clause types, plus one clause to search for.
2. Stop after it, fix the rules that did not fit, and relabel what they affect.
3. Do the other four agreements. `viavi-2025` is the largest, with 97 quotes.

For each agreement and clause type, read the rules below, then the rows from top to bottom.
Open the filing from the `url` column of `reports/m1_manifest.csv` and search for the first
few words of a quote to see its context.

After the last row of a clause type, check two things:

- If every quote is `no`, search the agreement as for a `found=False` row. Put any text you
  find in a `span:` on the last row.
- If you noticed a key term that no row quotes, add it in a `span:` on the last row.

On this sheet a clause type with one right quote gets no full search, so recall may be
overstated.

On the 30 test agreements, search every clause type, even when a quote is right. Reviewing only
what a model proposed hides what it missed.

## Values for `correct`

| value | rows | meaning |
|---|---|---|
| `yes` | `found=True` | The right clause type, and the quote states a key term in full. |
| `partial` | `found=True` | The right provision, but a key term is cut off or unrelated text is included. Give the right text in `span:`. |
| `no` | `found=True` | The wrong clause type, or the quote states no term. Add a tag. |
| `absent` | `found=False` | You searched and the agreement does not have this clause. |
| `missed` | `found=False` | The agreement has this clause. Give the text in `span:`. |

## General rules

1. **Substance, not mention.** A quote counts if it states the term itself: the level, cap,
   date, condition or protection. A sentence that only uses a defined term is `no`, tag
   `mention`. "Term SOFR plus the Applicable Margin" is not an Applicable Margin quote.
2. **Definitions count** when they carry the substance. A definition can be the main text of a
   clause.
3. **Several quotes can be right.** Judge each one alone. A quote that covers one term in full
   is `yes`, even if other terms sit in other quotes. Near-duplicates from chunk overlap are
   labeled separately.
4. **Right text, wrong clause.** Label `no` and add `belongs: <clause name>`.
5. **Headings do not count.** Table of contents entries, section titles, cross-references and
   exhibit titles are `no`, tag `heading`.
6. **`verbatim=False`.** Judge these by meaning, and do not fix the text. If the quote states
   terms the agreement does not have, label `no`, tag `invented`. About 32% of condition c's
   quotes are in this group.
7. **Flattened tables.** A pricing grid or covenant table reads as a run of values in the
   text. It is `yes` if the values are there.
8. **Weak terms still count.** Judge what the clause is, not how strong it is. A blocker with
   a large exception is still a blocker.
9. **Spans are short.** A `span:` is the smallest complete passage: the sentence, clause or
   table that states the term, not the whole section. Strict F1 needs a prediction to contain
   every main passage, so an oversized span makes it unreachable.
10. **Cut off or separate.** If a quote states one term in full but cuts off another key term
    of the same sentence or clause, it is `partial`, and its span is the whole sentence or
    clause. If the missing term sits in a separate passage, such as a pricing table or another
    definition, the quote is `yes` and the missing passage goes in a span on the last row.
11. **Main passages.** Each clause type that an agreement has gets at least one main passage,
    named under **Main** in its rules below. Mark it with the tag `main` on a `yes` row, or by
    starting its span with `[main] `. Other right passages are gold but not main.
12. **Not sure.** Pick what you would want the model to do and add the tag `unsure`. These rows
    get settled together and logged as rule changes.

## Rules per clause type

The search terms are starting points.

### Financial Maintenance Covenant

- **Counts:** the covenant that sets a ratio and its level, including a table of levels; the
  condition that springs the test, such as a revolver usage trigger; a temporary step-up after
  an acquisition.
- **Does not count:** ratio conditions on a basket or incremental (incurrence tests); a
  requirement to be in pro forma compliance; ratio definitions on their own; equity cure;
  compliance certificates; the event of default for a breach.
- **Main:** each covenant with its level.
- **Search:** `Financial Covenant`, `Leverage Ratio`, `Interest Coverage`,
  `Fixed Charge Coverage`, `Springing`

### Incremental Facility

- **Counts:** the right to add term loans or increase commitments; the size limit, including
  definitions such as `Incremental Cap`; the conditions on using it.
- **Does not count:** incremental equivalent debt under the debt covenant; refinancing
  facilities; extensions of existing loans; mechanics such as notices, joinders and
  allocations; limits on the terms of incremental loans, such as matching the existing
  terms or a minimum maturity. A quote that is only the MFN test goes under MFN Protection.
- **Main:** the right to add loans or commitments.
- **Search:** `Incremental`, `Increase`, `Additional Commitments`, `Accordion`

### MFN Protection

- **Counts:** the test comparing the yield or margin of new debt with the existing term loans,
  and its threshold; its sunset; its carve-outs.
- **Does not count:** yield definitions on their own, such as `All-In Yield`; a most favored
  lender clause on covenants; other pricing terms.
- **Main:** the test with its threshold.
- **Search:** `All-In Yield`, `Effective Yield`, `Most Favored`, `MFN`, `basis points`

### Restricted Payments General Basket

- **Counts:** the general-purpose exception capped by a dollar amount or a share of EBITDA or
  assets. This is usually one clause per agreement.
- **Does not count:** the other exceptions, including tax distributions, management equity
  repurchases, ratio-based capacity and Available Amount spending; the covenant's lead-in.
- **Main:** the basket.
- **Search:** `Restricted Payment`, `other Restricted Payments`, `greater of`

### Available Amount / Builder Basket

- **Counts:** the definition, with its starter amount and build-up parts; a permission to
  spend it on restricted payments, investments or junior debt prepayments.
- **Does not count:** other uses of the word, such as `Available Tenor` and
  `Available Commitment`; passing references, such as a reduction of another basket.
- **Main:** the definition.
- **Search:** `Available Amount`, `Cumulative Credit`, `Cumulative Amount`,
  `Available Basket`, `Builder`

### Debt Incurrence General Basket

- **Counts:** the general-purpose exception to the debt covenant capped by a dollar amount or
  a share of EBITDA or assets. This is usually one clause per agreement.
- **Does not count:** the specific exceptions, including capital leases, purchase money debt,
  acquired debt, ratio debt, incremental equivalent debt, refinancing debt, intercompany debt
  and non-guarantor caps.
- **Main:** the basket.
- **Search:** `other Indebtedness`, `additional Indebtedness`, `Permitted Indebtedness`,
  `greater of`

### Liens General Basket

- **Counts:** the general-purpose exception to the liens covenant capped by a dollar amount or
  a share of EBITDA or assets. This is usually one clause per agreement.
- **Does not count:** the specific permitted liens, including tax, landlord, deposit, purchase
  money and existing liens, and liens securing ratio debt.
- **Main:** the basket.
- **Search:** `Permitted Liens`, `other Liens`, `additional Liens`, `greater of`

### Applicable Margin

- **Counts:** the definition with its rates; the pricing grid, including fee columns in the
  same grid; the rule that sets the grid level; fixed pricing until a first compliance
  certificate.
- **Does not count:** interest clauses that only add the margin; benchmark floors; default
  interest; commitment or letter of credit fee clauses outside the grid.
- **Main:** each passage that states the rates, one per facility where they differ.
- **Search:** `Applicable Margin`, `Applicable Rate`, `Applicable Percentage`, `Pricing Level`

### Maturity Date

- **Counts:** the definition or clause that gives the final date for each facility; a
  springing maturity; extension terms.
- **Does not count:** sentences that only use the term; amortization dates; limits on other
  debt, such as "not earlier than the Latest Maturity Date"; a `Latest Maturity Date`
  definition with no date.
- **Main:** the final maturity date of each facility.
- **Search:** `Maturity Date`, `Termination Date`, `Springing`, `Extension`

### Liability Management Protections

Tag every `yes` and `partial` with `jcrew`, `chewy` or `serta`.

- **Counts:**
  - `jcrew`: a limit on moving material intellectual property to unrestricted subsidiaries, or
    on those subsidiaries owning it.
  - `chewy`: a limit on releasing a guarantor that stops being wholly owned.
  - `serta`: a requirement for each affected lender's consent to subordinate the loans or
    liens, and the exceptions to it.
- **Does not count:** other amendment voting rules; ordinary guarantee release terms; ordinary
  conditions for designating an unrestricted subsidiary. Other lender protections, such as
  limits on open market buybacks or pro rata sharing, are `no` with the tag `other_lme`, so we
  can decide later whether the schema should include them.
- **Main:** one passage for each type the agreement has.
- **Search:** `Material Intellectual Property`, `Unrestricted Subsidiary`, `wholly`,
  `subordinat`

## Format of `note`

Put tags first, separated by `;`, and spans last.

| tag | use |
|---|---|
| `mention` | uses the term but states no term |
| `heading` | a heading, contents entry or cross-reference |
| `adjacent` | a nearby provision the rules exclude |
| `unrelated` | text that belongs to no clause type, such as EBITDA add-backs |
| `belongs: <clause name>` | the right text under the wrong clause type |
| `invented` | states terms the agreement does not have |
| `jcrew`, `chewy`, `serta`, `other_lme` | liability management type |
| `unsure` | the rules do not settle it |
| `main` | this `yes` row's quote is a main passage |
| `span: <text>` | text to add to the gold set for this agreement and clause type |

A `span:` can sit on any row, whatever its `correct` value. Copy the text from the agreement.
For a long span, give the opening and closing words with ` ... ` between them. Separate several
spans with ` || `, and start a main passage with `[main] `.

Examples:

```
adjacent
belongs: Debt Incurrence General Basket
serta
unsure; adjacent
span: [main] any other Indebtedness in an aggregate principal amount ... and 10% of Consolidated EBITDA
```

## How labels become the gold set

`python -m src.gold` does this. It writes no gold file until every row can be placed.

- The gold quotes for an agreement and clause type are its `yes` quotes, every `span:`, and
  quotes moved there by `belongs:`.
- `no` quotes are dropped. A `partial` quote is replaced by its `span:`.
- An empty result means the clause is absent, which is why that needs a real search: without
  one, a missed clause would enter the gold set as absent.
- Gold text is cut from the agreement with page numbers and footers removed. `yes` and
  `belongs:` rows with `verbatim=False` are matched to it, and any the script cannot place
  come back for review.
- Each span must be found in the source text. Any that is not found comes back for review.

## How the gold set is scored

Decided 2026-11-01 after the first agreement. The upstream metrics are reported as well, so
results stay comparable with Model B's card.

- **Found:** a clause counts as found only if at least one predicted quote is right. A quote
  is right if it holds a gold passage or has token F1 of at least 0.5 against one. A clause
  with only wrong quotes counts as both a false find and a miss. Upstream counts any quote
  as found. `src.score` uses this rule for every model, so unlabeled models compare fairly.
  On the M3 sheet it agrees with the labels on 264 of 271 quotes, which flatters it, since
  those labels built the gold set.
- **Strict:** a clause counts as right if every main passage sits inside some predicted
  quote. Upstream requires every gold passage, which the extra spans make almost
  unreachable.
- **Matching** drops quote marks and ignores case. A quote that changes any other word
  fails Strict, even if it is `yes`, because Strict is also a test of copying.

## Rule changes

| date | change | rows affected |
|---|---|---|
| 2026-11-01 | Added the tag `unrelated` | 36 rows |
| 2026-11-01 | Limits on incremental loan terms do not count | 3 rows |
| 2026-11-01 | Added rule 10, cut off or separate | none |
| 2026-11-01 | Added main passages, the tag `main` and `[main] ` spans | all labeled rows |
| 2026-11-01 | Found and Strict use main passages and right quotes | scoring only |
| 2026-11-01 | `src.score` applies the right-quote rule to every model, labeled or not | scoring only |
