# RLVR on the teacher's pre-labels

GRPO on `Qwen/Qwen3-4B-Instruct-2507` from the step 2 adapter, with the teacher's kept quotes as the answers. Prompts are 8,000-character windows of the teacher's train chunks. Scored on the 5 M1 agreements on step 2's chunks, and on a check set of windows from 12 train agreements that screening and training never saw.

## Data

| windows | with quotes | answer over the limit | pool | of which empty | check | of which empty | answer tokens, median / p99 |
|---|---|---|---|---|---|---|---|
| 13,592 | 1,458 | 9 | 800 | 200 | 100 | 25 | 211 / 868 |

## Screening, measured

SFT sampled each pool window 4 times at temperature 1.0, as GRPO does, and each answer got the training reward.

| screened | mean reward | with quotes | empty | with spread | selected | of which empty | samples cut at the limit |
|---|---|---|---|---|---|---|---|
| 800 | 0.7415 | 0.7211 | 0.8025 | 394 | 394 | 47 | 0.2% |

## Training, measured

Short test on the 4 selected windows with the longest answers: 58.04 seconds per prompt, peak 10815 MiB, so 6.35 hours for all 394, an upper bound since those answers are the longest. Training took the selected windows in random order, with a limit of 2.0 hours, which stopped it.

| prompts | seconds per prompt | peak GPU MiB | reward, first 3 logs | last 3 logs | groups with no spread | answer tokens |
|---|---|---|---|---|---|---|
| 373 | 19.29 | 11790 | 0.7241 | 0.9475 | 0.5129 | 285.6358 |

## Check set, greedy

| model | mean reward | with quotes | empty | reward 0 | cut at the limit |
|---|---|---|---|---|---|
| SFT | 0.7616 | 0.7488 | 0.8 | 9 | 0 |
| RLVR | 0.8879 | 0.8639 | 0.96 | 6 | 0 |

## The 5 M1 agreements

Five agreements cannot separate these models, and they shaped the labeling rules. Model A and SFT are step 2's decodes; RLVR was decoded on the same chunks with the same settings, but by transformers instead of vLLM (see below).

Gold set labeled by: Claude, 2026-11-01, at the user's request; not reviewed by the user. 41 clause decisions have gold text, out of 50.

| metric | Model A | SFT | RLVR |
|---|---|---|---|
| Found F1 | 0.756 (0.622-0.886) | 0.951 (0.882-1.000) | 0.951 (0.882-1.000) |
| Strict F1 | 0.542 (0.413-0.667) | 0.776 (0.635-0.909) | 0.829 (0.761-0.899) |
| upstream detection F1 | 0.951 | 1.000 | 1.000 |
| upstream strict F1 | 0.170 | 0.536 | 0.610 |
| mean reward | 0.360 | 0.678 | 0.778 |
| share of quotes right | 0.2222 | 0.5517 | 0.759 |
| quotes | 243 | 116 | 83 |
| verbatim, quote marks and case ignored | 0.6914 | 0.9138 | 0.8434 |
| same, page numbers removed | 0.7407 | 1.0 | 1.0 |

Brackets are 95% bootstrap intervals over agreements.

| clause | agreements with it | Model A: Found | Model A: Strict | SFT: Found | SFT: Strict | RLVR: Found | RLVR: Strict |
|---|---|---|---|---|---|---|---|
| Financial Maintenance Covenant | 4 | 0.667 | 0.750 | 1.000 | 0.857 | 1.000 | 0.857 |
| Incremental Facility | 4 | 1.000 | 0.857 | 1.000 | 0.857 | 1.000 | 0.857 |
| MFN Protection | 2 | 0.800 | 0.000 | 1.000 | 0.000 | 1.000 | 0.667 |
| Restricted Payments General Basket | 4 | 0.500 | 0.667 | 0.750 | 0.857 | 0.750 | 0.857 |
| Available Amount / Builder Basket | 3 | 1.000 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 |
| Debt Incurrence General Basket | 5 | 0.444 | 0.000 | 0.800 | 0.750 | 0.800 | 0.889 |
| Liens General Basket | 5 | 0.444 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Applicable Margin | 5 | 0.800 | 0.571 | 1.000 | 0.333 | 1.000 | 0.571 |
| Maturity Date | 5 | 1.000 | 0.889 | 1.000 | 0.889 | 1.000 | 0.889 |
| Liability Management Protections | 4 | 1.000 | 0.000 | 1.000 | 0.400 | 1.000 | 0.400 |

Paired difference from Model A, same agreement resamples:

| model | Found F1 difference (95% CI) | Strict F1 difference (95% CI) |
|---|---|---|
| SFT | +0.195 (+0.028 to +0.342) | +0.234 (+0.122 to +0.322) |
| RLVR | +0.195 (+0.028 to +0.342) | +0.286 (+0.152 to +0.409) |

Paired difference from SFT:

| model | Found F1 difference (95% CI) | Strict F1 difference (95% CI) |
|---|---|---|
| RLVR | +0.000 (+0.000 to +0.000) | +0.052 (-0.043 to +0.144) |

## Checked after the run

Added by hand on 2026-11-03, from the raw records on the Hub (`real_rlvr/`, `real_sft/runs/m1/`)
and the VM's log. Everything here is measured unless it says otherwise.

- **The numbers above are right.** Scoring the raw records again on the local machine gives the
  same M1 section, character for character, the same two check-set rows, and the same paired
  difference. The sentence that RLVR "was decoded the same way" was wrong in the generated report
  and is corrected above.
- **RLVR's M1 decode ran on transformers, not vLLM.** vLLM failed to start twice: its FlashInfer
  sampler tried to compile, and the VM has neither ninja nor nvcc. The switch that turns it off
  was set only for `src.sample`, so the check set ran on vLLM and the M1 decode fell back to
  transformers (742 seconds, all 144 chunks usable). Step 2 decoded Model A and SFT with vLLM.
  Both are greedy on the same chunks, but the two engines can pick different tokens where two are
  close, so a small part of the M1 difference may come from the engine. `scripts/nebius_rlvr.sh`
  now sets the switch for every process.
- **Run:** one Nebius H100 VM, from 00:06:57 to 02:55:30 UTC, 2.81 hours, about $12.64 at $4.50
  an hour (the approved cap was $18). The first 29 minutes were three failed starts (vLLM's CUDA
  13 wheel on a CUDA 12.8 driver, a missing `Python.h`, FlashInfer's sampler in screening) and a
  restart after the short test cut training to 124 windows. The run that finished took 137
  minutes; training took 7,210 seconds and stopped at the 2-hour limit after 373 of the 394
  selected windows. The VM and its boot disk are deleted.
- **Training reward rose, but it is a weak signal.** Each step is a new window, so the logged
  reward depends on which windows came up. Over thirds of the 74 logs: reward 0.716, 0.771,
  0.827; groups whose 4 samples all scored the same, 38%, 50%, 64%. Answers were never cut at the
  1,024-token limit.
- **Check set** (100 windows from 12 train agreements that screening and training never saw):
  RLVR scored higher than SFT on 26 windows, the same on 70 and lower on 4. On the 25 windows with
  no quote the mean reward rose from 0.80 to 0.96, so RLVR gives fewer quotes where there are
  none. The answers here are the teacher's, so this measures how close RLVR gets to the teacher.
- **On M1, Strict F1 rose by 0.052, and the 95% interval (-0.043 to +0.144) includes zero.**
  Found F1 is unchanged. RLVR gives 83 quotes against SFT's 116, and about 63 of each model's
  quotes are right (0.759 x 83 and 0.552 x 116), so RLVR mostly dropped wrong quotes rather than
  finding new right ones. MFN Protection Strict rose from 0.000 to 0.667, Applicable Margin from
  0.333 to 0.571 and Debt Incurrence from 0.750 to 0.889; no clause fell.
- **Fewer of RLVR's quotes are verbatim once quote marks and case are ignored (84% against
  91%), but all are once page numbers are removed.** So the gap is page numbers left inside
  quotes. This was not looked into further.
- **Not done:** the 30 test agreements have not been scored or reviewed, and the gold set is still
  Claude's, not reviewed by the user.
