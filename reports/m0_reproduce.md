# Milestone 0: Model B reproduction on CUAD (subset)

The first part is the notebook's report, unchanged. The review notes come after it.
`latency.csv` and the per-chunk records are in the private Hub repo under `m0_reproduce/`.

- Platform: kaggle, GPUs: ['Tesla T4', 'Tesla T4']
- Contracts: 10 of 102 official test contracts (first by md5(title))
- Session wall time so far: 37.5 min

Model card numbers are on all 102 test contracts with Ollama Q4_K_M on an RTX 5090;
ours are fp16 on T4 on a subset, so CIs are much wider. Compare with that in mind.

| metric | model card (102 contracts) | cuad60k | cuad32k |
|---|---|---|---|
| detection F1 | 0.9 | 0.8826 | 0.8916 |
| F1 95% CI | [0.887, 0.913] | [0.8198, 0.935] | [0.8286, 0.9431] |
| precision | 0.901 | 0.9397 | 0.9407 |
| recall | 0.899 | 0.8321 | 0.8473 |
| quote overlap (token F1 on TP) | 0.938 | 0.9655 | 0.9623 |
| strict F1 | 0.678 (3-seed mean) | 0.7042 | 0.7222 |
| verbatim rate (casefolded, vs contract) | 0.972 | 0.977 | 0.9817 |
| chunks needing ladder | 9/150 (6.0%) | 1/17 | 1/24 |

## Decode stability and latency

### cuad60k

- chunk size 60000 / overlap 8000, max output tokens 12288
- primary JSON valid: 16/17, loops: 0, other invalid: 1
- recovered by ladder rung: {'1.05': 0, '1.15': 1, '1.25': 0}; still failing: none; ladder backends: ['vllm']
- verbatim (exact after whitespace norm, vs own chunk): 0.969
- dropped unknown keys: 0
- primary latency per chunk: mean 43.71 s, max 55.72 s (vLLM: batch wall time / batch size (amortized); HF: per request)
- stage wall time 23.0 min, GPU-device minutes 45.7
- backends that completed a stage: ladder=vllm:tp2, primary=vllm:tp2
- failed backend attempts: primary=vllm:tp2 (exit 1), primary=vllm:tp1 (exit 1), primary=hf:tp0 (exit 1), ladder=hf:tp0 (exit 1)

### cuad32k

- chunk size 32000 / overlap 4000, max output tokens 12288
- primary JSON valid: 23/24, loops: 0, other invalid: 1
- recovered by ladder rung: {'1.05': 0, '1.15': 1, '1.25': 0}; still failing: none; ladder backends: ['vllm']
- verbatim (exact after whitespace norm, vs own chunk): 0.9825
- dropped unknown keys: 0
- primary latency per chunk: mean 22.45 s, max 30.9 s (vLLM: batch wall time / batch size (amortized); HF: per request)
- stage wall time 15.2 min, GPU-device minutes 30.3
- backends that completed a stage: ladder=vllm:tp2, primary=vllm:tp2
- failed backend attempts: ladder=hf:tp0 (exit 1)

# Review notes (2026-10-30)

**Result:** Model B reproduces on T4 with fp16 within sampling noise. Both F1 values are
below the card's 0.900, but both CIs contain it. This is 10 contracts, so the CI is about
±0.06: it rules out a large gap, not a small one.

**Worth watching:**
- Precision is about 0.04 above the card and recall about 0.05 to 0.07 below it, in both
  runs. The two runs share the same 10 contracts, so that is one observation, not two.
  The cause can't be separated here: the sample, fp16 versus Q4_K_M, or no JSON grammar.
- 32k/4k chunking did no worse than 60k/8k (0.8916 vs 0.8826, heavily overlapping CIs) and
  took about two thirds of the stage time. That supports using 32k/4k in M3.

**Open problems (updated after the Kaggle logs, 2026-10-30):**
1. **The ladder did not run as designed.** `hf:0` ran out of GPU memory in both runs, inside
   attention, asking for 5.63 GiB (32k) and 19.27 GiB (60k). That is about 3.4x, roughly
   quadratic in prompt length, so one pass over the whole prompt builds a full T x T matrix.
   Retries therefore fell back to vLLM, whose penalty covers the full context instead of
   Ollama's 64-token window. The 1.15 recoveries above are not the planned semantics.
   **Fix:** `prefill_cache` feeds long prompts in 512-token slices. Output is token-identical
   to a single pass locally (transformers 4.57 and 5.22). **Now proven on T4 x2:** all 6
   decodes in the ladder check below finished without running out of memory.
2. **The one failure is not a repetition loop, and it isn't the chunk upstream reported.**
   Both runs failed on El Pollo Loco chunk 0 (`stop` at 914 / 1,632 tokens). The output ends
   cleanly with `...[]}`, so the break is inside the JSON. Rung 1.05 failed too and 1.15
   passed. El Pollo is not in upstream's list of 9 failures, and INGEVITY c0, which is in
   our sample and needed 1.15 upstream, passed first time here. We decode without Ollama's
   JSON grammar, which would have forced valid JSON. **Confirmed for the first attempt in
   both runs:** the break is an unescaped double quote. See the ladder check below.

**Reading the time figures:** the cuad60k stage time includes the crashed attempts from
the first session (the failed primary attempts in its list). Latency is amortized over a
batch, so it is throughput, not single-request latency.

**Rough M3 time (estimate, not measured):** 104 chunks × 22.45 s is about 39 min of primary
decoding per condition, plus model load, so about 2 hours for three conditions before
retries. Credit agreements may produce different output lengths than CUAD.

# Ladder check (notebook 00b, ran 2026-10-30, Kaggle T4 x2)

The one failed chunk (El Pollo Loco chunk 0) replayed through the transformers ladder with
the 64-token windowed penalty. M0's vLLM rows are copied from M0's records, not rerun.

| run | ladder | penalty | backend | status | finish | output tokens | latency s | quotes | verbatim |
|---|---|---|---|---|---|---|---|---|---|
| cuad60k | M0 (vLLM full-context) | 1.00 | vllm | invalid | stop | 1632 | 43.0 | 0 | - |
| cuad60k | M0 (vLLM full-context) | 1.05 | vllm | invalid | stop | 1632 | 115.8 | 0 | - |
| cuad60k | M0 (vLLM full-context) | 1.15 | vllm | ok | stop | 1918 | 94.8 | 34 | 0.8824 |
| cuad60k | check (HF 64-token window) | 1.05 | hf | invalid | stop | 1535 | 555.1 | 0 | - |
| cuad60k | check (HF 64-token window) | 1.15 | hf | invalid | stop | 1675 | 606.7 | 0 | - |
| cuad60k | check (HF 64-token window) | 1.25 | hf | invalid | stop | 1962 | 715.0 | 0 | - |
| cuad32k | M0 (vLLM full-context) | 1.00 | vllm | invalid | stop | 914 | 18.2 | 0 | - |
| cuad32k | M0 (vLLM full-context) | 1.05 | vllm | invalid | stop | 964 | 47.7 | 0 | - |
| cuad32k | M0 (vLLM full-context) | 1.15 | vllm | ok | stop | 978 | 35.4 | 20 | 0.9500 |
| cuad32k | check (HF 64-token window) | 1.05 | hf | invalid | stop | 902 | 197.5 | 0 | - |
| cuad32k | check (HF 64-token window) | 1.15 | hf | invalid | stop | 1154 | 253.9 | 0 | - |
| cuad32k | check (HF 64-token window) | 1.25 | hf | invalid | stop | 1517 | 338.3 | 0 | - |

Parse errors of the first attempt in each run (`Expecting ',' delimiter`):

- cuad60k, position 5720: `add \", and specifically Section 2.20 hereof,") and as long as`.
  The first inner quote is escaped and the second is not.
- cuad32k, position 2229: `add ", "and specifically Section 2.20 hereof,") and as long as`.
  Neither is escaped, and an extra `"` appears after the comma, so the output looks like
  it closed the string and opened a new one.

**What this shows:**
- The memory fix works on T4 x2, including the 60k prompt that needed 19.27 GiB before.
- The failure is a JSON escaping error, not a repetition loop: every attempt stopped
  normally. Only the first attempt's error was printed, so the cause of the later failures
  is not confirmed.
- The designed ladder recovered 0 of 6 attempts here. vLLM's full-context penalty recovered
  this chunk at 1.15 in both runs. That is one chunk, and a penalty does not address
  escaping directly, so it does not show vLLM's penalty is better.
- The two recovered outputs copied less exactly: verbatim 0.8824 and 0.9500, against 0.969
  and 0.9825 for those runs overall. One chunk each.

**Cost:** transformers ran at about 2.75 tok/s at 60k and 4.5 tok/s at 32k. All three rungs
took 31.3 min at 60k and 13.2 min at 32k for this one chunk.

**M3 estimate (not measured):** if credit agreements fail at M0's CUAD rate (2 of 41 chunks)
and nothing recovers, that is about 15 of 312 chunks × 13.2 min, or about 3.3 hours of
retries on top of about 2 hours of primary decoding.

# Quote repair check (notebook 00c, ran 2026-10-30, Kaggle, no GPU)

The repair applied to every saved failed decode. Decoding did not run again, so `output
tokens` and `latency s` are the original attempts' values.

| run | ladder | penalty | backend | status | output tokens | latency s | quotes | verbatim |
|---|---|---|---|---|---|---|---|---|
| cuad60k | M0 | 1.00 | vllm | repaired | 1632 | 43.0 | 26 | 1.0000 |
| cuad60k | M0 | 1.05 | vllm | repaired | 1632 | 115.8 | 26 | 1.0000 |
| cuad60k | check | 1.05 | hf | repaired | 1535 | 555.1 | 24 | 1.0000 |
| cuad60k | check | 1.15 | hf | repaired | 1675 | 606.7 | 26 | 1.0000 |
| cuad60k | check | 1.25 | hf | repaired | 1962 | 715.0 | 40 | 0.8500 |
| cuad32k | M0 | 1.00 | vllm | repaired | 914 | 18.2 | 18 | 1.0000 |
| cuad32k | M0 | 1.05 | vllm | repaired | 964 | 47.7 | 19 | 1.0000 |
| cuad32k | check | 1.05 | hf | repaired | 902 | 197.5 | 17 | 1.0000 |
| cuad32k | check | 1.15 | hf | repaired | 1154 | 253.9 | 21 | 0.9048 |
| cuad32k | check | 1.25 | hf | repaired | 1517 | 338.3 | 31 | 0.8387 |

Double quotes in the cleaned M1 credit agreements:

| document | chars | straight | curly |
|---|---|---|---|
| angi-2025 | 661,145 | 0 | 1,488 |
| exlservice-2026 | 499,265 | 0 | 1,115 |
| quanta-2024 | 348,218 | 0 | 806 |
| repay-2026 | 660,536 | 0 | 1,418 |
| viavi-2025 | 695,765 | 0 | 1,676 |

**What this shows:**
- The repair parsed all 10 saved failures, including the 6 the ladder could not fix. On the
  first attempts every quote was verbatim: 26 of 26 at 60k and 18 of 18 at 32k. That means
  the text is in the chunk, not that the extraction is right, and it is one chunk.
- vLLM's 1.15 attempts had 30 of 34 and 19 of 20 verbatim, so repair kept slightly fewer
  verbatim quotes at 60k (26 vs 30). Whether those extra quotes are correct was not checked.
- Across both chunkings, more penalty meant more quotes and fewer verbatim ones, and 1.25
  was at 0.85 and 0.84. This is one chunk, but it fits the concern that a penalty disturbs
  copying.
- The credit agreements contain no straight double quotes, so M0's exact failure cannot come
  from copying them. It could still appear if a model turns curly quotes into straight ones.
  That swap would also lower the plain verbatim rate, so M3 also reports it with curly and
  straight quote marks treated as equal, and the review sheet's `verbatim` column uses that
  version.
