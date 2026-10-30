# Next steps

Plan only: nothing here has run. The rest of the plan (data scale-up, training paths,
evaluation, export) is written after the Milestone 3 review.

## Teacher pre-labeling

Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017` on Nebius Token Factory. Every Nebius call
needs a cost estimate and approval first.

Rejection sampling, per chunk:

1. Generate once at temperature 0.
2. Check each quote: the JSON is valid, the clause name is in the schema, and the quote is
   in this chunk.
3. Quotes that fail do not enter the training set.
4. Regenerate a chunk that has a failed quote, up to 2 more times, at a temperature slightly
   above 0. Repeating the same settings gives essentially the same result.
5. Keep only quotes that pass.

Implementation choices:

- The retry temperature starts at 0.3. The trial below can change it.
- Step 5 keeps the union of passing quotes across attempts, deduplicated after whitespace
  normalization, so a retry cannot lose a good quote.
- The check is an exact match after whitespace normalization, with curly and straight quote
  marks kept distinct, so training targets copy the source exactly. If the trial shows the
  teacher swaps quote marks often, add a step that restores the source text.
- A chunk is left out of the training set if no attempt gives valid JSON, or if a clause has
  failed quotes and no passing one. Otherwise that clause's empty list would teach the model
  the clause is absent although the teacher found one.
- Each run reports the reject rate per clause type and the number of chunks left out,
  because failures tend to be the longer, harder chunks.

Limits:

- The check confirms format and copying, not correctness. A verbatim quote under the wrong
  clause passes.
- It cannot find missed clauses, because an empty list passes.
- Retries do not use a repeat penalty. On the one chunk checked in Milestone 0, higher
  penalties lowered the verbatim rate (`reports/m0_reproduce.md`).
- The test agreements use the reviewed labels as answers, not this flow's output.

Trial: run the flow on the 5 M1 agreements and compare it with the `correct` column of
`reports/review_sheet.csv`, to see whether rejected quotes were really wrong and how many
accepted quotes are right.

Optional: sample each chunk twice, treat quotes found both times as high confidence, and send
disagreements to review first. This roughly doubles the cost.

Estimated cost at the 2026-10-30 list price ($0.30 input, $1.20 output per million tokens),
assuming 1,500-3,000 output tokens per chunk:

| run | cost |
|---|---|
| 5 agreements (104 chunks) | about $0.4-0.6 |
| 150 agreements (about 3,120 chunks) | about $12-18 |
| 150 agreements, if 10% of chunks retry once | about $13-20 |
