# Small SFT and RLVR loop

Model A, SFT and RLVR on the synthetic held-out set, and on the 5 M1 agreements.

## Held-out synthetic excerpts

Gold set labeled by: the synthetic generator, so these answers are exact. 114 clause decisions have gold text, out of 600.

| metric | Model A | SFT | RLVR |
|---|---|---|---|
| Found F1 | 0.858 (0.817-0.896) | 0.987 (0.969-1.000) | 0.987 (0.972-1.000) |
| Strict F1 | 0.701 (0.623-0.772) | 0.983 (0.965-0.996) | 0.982 (0.966-0.996) |
| upstream detection F1 | 0.875 | 0.987 | 0.987 |
| upstream strict F1 | 0.560 | 0.983 | 0.982 |
| mean reward | 0.675 | 0.982 | 0.988 |
| share of quotes right | 0.7571 | 0.9826 | 0.9912 |
| quotes | 140 | 115 | 113 |
| verbatim, quote marks and case ignored | 0.8571 | 0.9217 | 0.9204 |
| same, page numbers removed | 0.8714 | 0.9913 | 0.9912 |

Brackets are 95% bootstrap intervals over agreements.

| clause | agreements with it | Model A: Found | Model A: Strict | SFT: Found | SFT: Strict | RLVR: Found | RLVR: Strict |
|---|---|---|---|---|---|---|---|
| Financial Maintenance Covenant | 19 | 0.973 | 0.643 | 1.000 | 0.973 | 1.000 | 0.973 |
| Incremental Facility | 8 | 0.762 | 0.143 | 1.000 | 1.000 | 1.000 | 1.000 |
| MFN Protection | 12 | 1.000 | 0.737 | 1.000 | 1.000 | 1.000 | 1.000 |
| Restricted Payments General Basket | 10 | 0.737 | 0.429 | 0.947 | 0.947 | 0.889 | 0.889 |
| Available Amount / Builder Basket | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Debt Incurrence General Basket | 14 | 0.750 | 0.593 | 1.000 | 1.000 | 1.000 | 1.000 |
| Liens General Basket | 6 | 0.909 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 |
| Applicable Margin | 13 | 0.743 | 0.743 | 0.929 | 0.929 | 0.963 | 0.963 |
| Maturity Date | 11 | 0.815 | 0.815 | 1.000 | 1.000 | 1.000 | 1.000 |
| Liability Management Protections | 10 | 0.947 | 0.947 | 1.000 | 1.000 | 1.000 | 1.000 |

Paired difference from Model A, same agreement resamples:

| model | Found F1 difference (95% CI) | Strict F1 difference (95% CI) |
|---|---|---|
| SFT | +0.129 (+0.093 to +0.166) | +0.281 (+0.215 to +0.354) |
| RLVR | +0.129 (+0.093 to +0.167) | +0.281 (+0.215 to +0.354) |

## The 5 M1 agreements

Five agreements cannot separate these models, and they shaped the labeling rules. Read this as a direction only.

Gold set labeled by: Claude, 2026-11-01, at the user's request; not reviewed by the user. 41 clause decisions have gold text, out of 50.

| metric | Model A | SFT | RLVR |
|---|---|---|---|
| Found F1 | 0.829 (0.740-0.905) | 0.857 (0.806-0.891) | 0.892 (0.812-0.958) |
| Strict F1 | 0.657 (0.492-0.744) | 0.778 (0.706-0.843) | 0.806 (0.773-0.838) |
| upstream detection F1 | 0.902 | 0.952 | 0.964 |
| upstream strict F1 | 0.333 | 0.603 | 0.635 |
| mean reward | 0.422 | 0.609 | 0.639 |
| share of quotes right | 0.2362 | 0.5575 | 0.5741 |
| quotes | 271 | 113 | 108 |
| verbatim, quote marks and case ignored | 0.6827 | 0.7168 | 0.713 |
| same, page numbers removed | 0.7269 | 0.8496 | 0.8704 |

Brackets are 95% bootstrap intervals over agreements.

| clause | agreements with it | Model A: Found | Model A: Strict | SFT: Found | SFT: Strict | RLVR: Found | RLVR: Strict |
|---|---|---|---|---|---|---|---|
| Financial Maintenance Covenant | 4 | 0.889 | 0.750 | 0.889 | 0.750 | 1.000 | 0.667 |
| Incremental Facility | 4 | 1.000 | 0.667 | 1.000 | 0.857 | 1.000 | 1.000 |
| MFN Protection | 2 | 0.800 | 0.000 | 0.800 | 0.500 | 0.800 | 0.800 |
| Restricted Payments General Basket | 4 | 0.750 | 0.857 | 1.000 | 1.000 | 1.000 | 1.000 |
| Available Amount / Builder Basket | 3 | 0.857 | 0.667 | 1.000 | 0.800 | 1.000 | 0.800 |
| Debt Incurrence General Basket | 5 | 0.571 | 0.571 | 0.800 | 0.889 | 0.800 | 0.889 |
| Liens General Basket | 5 | 0.667 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| Applicable Margin | 5 | 1.000 | 0.889 | 0.600 | 0.571 | 0.800 | 0.571 |
| Maturity Date | 5 | 1.000 | 0.750 | 0.667 | 0.750 | 0.889 | 0.750 |
| Liability Management Protections | 4 | 0.667 | 0.000 | 0.889 | 0.333 | 0.667 | 0.333 |

Paired difference from Model A, same agreement resamples:

| model | Found F1 difference (95% CI) | Strict F1 difference (95% CI) |
|---|---|---|
| SFT | +0.028 (-0.058 to +0.115) | +0.121 (+0.040 to +0.231) |
| RLVR | +0.062 (-0.027 to +0.182) | +0.149 (+0.071 to +0.281) |

## Training, measured

| stage | steps | seconds per step | peak GPU MiB |
|---|---|---|---|
| SFT | 75 | 49.14 | 12506 |
| RLVR | 200 | 39.16 | 7721 |

Short test: `{"sft_seconds_per_step": 51.28, "rlvr_seconds_per_step": 31.54, "sft_peak_gpu_mib": 12506, "rlvr_peak_gpu_mib": 5800, "rlvr_projected_hours": 1.75, "rlvr_prompts": 200}`

## Checked after the run

Everything above was written by the notebook. This section was added by hand on 2026-11-02,
from the raw records on the Hub.

- **The tables match the records.** Each table line was rebuilt from the decode records with
  `src.score`, and all 52 match. Bootstrap intervals were left out of that check, because they are
  random draws.
- **The run:** one Save & Run All on Kaggle GPU T4 x2. The synthetic data reached the Hub at
  03:10 UTC and this report at 09:31 UTC, so about 6 hours 20 minutes, plus the install before it.
- **Stage times:** SFT trained for 61 minutes, and RLVR for 131 minutes. Each held-out decode took
  5 to 6 minutes with vLLM. The real agreements took longer because some chunks went to the
  transformers ladder: SFT took 36 minutes with vLLM plus 51 minutes on the ladder, and RLVR took
  35 minutes plus 33 minutes.
- **Decode failures:** none on the held-out set; Model A had 3 repaired outputs there. On the real
  agreements, 3 SFT chunks needed retries, and one of them, `repay-2026__chunk13`, was still not
  valid JSON after all of them, so its quotes are missing from SFT's row. 2 RLVR chunks needed
  retries, and both ended valid.
- **RLVR had little to learn from.** On 88% of its 200 prompts, all 4 sampled answers got the same
  reward, which gives GRPO no gradient. Mean training reward was 0.965 over the first 50 steps and
  0.982 over the last 50. RLVR minus SFT, same agreement resamples:

  | set | Found F1 | Strict F1 |
  |---|---|---|
  | 60 held-out excerpts | -0.000 (-0.011 to +0.013) | -0.000 (-0.012 to +0.013) |
  | 5 M1 agreements | +0.034 (+0.000 to +0.073) | +0.028 (-0.030 to +0.076) |

- **What changed on real text is fewer wrong quotes.** The right quotes were counted with the
  automatic rule:

  | model | quotes | right | wrong, a cut-short piece of the right passage | other wrong |
  |---|---|---|---|---|
  | Model A | 271 | 64 | 4 | 203 |
  | SFT | 113 | 63 | 8 | 42 |
  | RLVR | 108 | 62 | 6 | 40 |

  Two of SFT's cut-short quotes are Applicable Rate definitions. In viavi-2025, SFT stops at 523
  of the gold passage's 2,012 characters, and in exlservice-2026 at 329 of 1,073. Both stop at the
  colon before a flattened pricing grid, which the gold passage includes.
- **Correction, 2026-11-02, after the teacher trial (`reports/teacher_trial.md`).** An earlier
  version of this section blamed short synthetic answers for those two quotes. No synthetic
  training answer is longer than 563 characters, which is true, but DeepSeek, which saw no
  synthetic data, stops the same two quotes at the same points. So the cause is more likely how
  models treat a grid after a definition than the synthetic data.
- Both SFT and RLVR miss angi-2025's Maturity Date.
