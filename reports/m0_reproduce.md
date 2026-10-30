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

**Open problems:**
1. **The ladder did not run as designed.** `hf:0` exited 1 in both runs, so the retries fell
   back to vLLM, whose repetition penalty covers the full context instead of Ollama's
   64-token window. The one recovery at 1.15 is therefore not the planned semantics. Cause
   unknown: `HFBackend` runs end to end locally (tiny random Qwen3, transformers 5.22,
   torch 2.15, with and without the penalty), so the traceback from Kaggle is needed.
   Until that is fixed there is also no working transformers fallback.
2. **The failed chunks were "invalid", not "loop".** The card describes greedy repetition
   loops; here each run had one decode that ended without hitting the cap but was not
   valid JSON. We don't use Ollama's JSON grammar, which may be the difference. Not yet
   inspected. Which chunk failed is not in this report (upstream's was INGEVITY c0).

**Reading the time figures:** the cuad60k stage time includes the crashed attempts from
the first session (the failed primary attempts in its list). Latency is amortized over a
batch, so it is throughput, not single-request latency.

**Rough M3 time (estimate, not measured):** 104 chunks × 22.45 s is about 39 min of primary
decoding per condition, plus model load, so about 2 hours for three conditions before
retries. Credit agreements may produce different output lengths than CUAD.
