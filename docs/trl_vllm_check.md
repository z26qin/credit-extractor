# Checking TRL + vLLM before a paid run uses it

Written 2026-11-03. Nothing here has run on a GPU yet. The notebook below is the check, and it
runs on free Kaggle GPUs.

## Why

Step 3's GRPO samples its answers with transformers' `generate`. On the Nebius H100 on
2026-11-03, the short test took 58 seconds per step on the 4 windows with the longest answers (815
tokens on average), and most of that was sampling: about 20 tokens a second per answer, worked
out from those numbers, not timed alone. On the same GPU vLLM sampled the first 100 screening
windows, 400 answers of about 224 tokens, in 14 seconds: about 6,400 tokens a second. TRL can
sample with vLLM inside GRPO (`use_vllm`), which could cut each step to a fraction.

## What stood in the way

TRL's vLLM mode needs vLLM in the same Python as training. This repo keeps them in separate
venvs (`requirements-train.txt`, `requirements-gpu.txt`), because vLLM's torch pin once broke the
platform Python on Kaggle. Found on 2026-11-03 from the package metadata, before any GPU run:

- vLLM 0.32.0 requires transformers >=5.16.1,<5.20. Training pins 5.22.1, so the two do not
  install together as they are.
- TRL 1.17.1 supports vLLM 0.22.0 to 0.32.x, so vLLM 0.32.0 is in range.
- `requirements-rl-vllm.txt` is one venv for both, with transformers 5.19.0, the newest release
  vLLM accepts.
- All 157 CPU tests pass with transformers 5.19.0 and torch 2.13.0 (measured), so the training
  code does not depend on 5.22.1. That covers CPU only.

## What TRL does with vLLM

Read from TRL 1.17.1's source:

- Before each step's sampling it merges the LoRA adapter, sends the weights to vLLM, and unmerges.
  It also does this before the first step, so sampling starts from the step 2 adapter, not from
  the bare base.
- `colocate` starts vLLM inside the training process from the model's own files, with no dtype.
  These Qwen3 files name bf16, which a T4 cannot run, so the check gives a T4 a copy whose config
  names fp16. On an H100 bf16 is fine.
- For a 4-bit base, colocate asks vLLM for in-flight bitsandbytes quantization, and the merged
  weights it sends are still 4-bit. Whether vLLM takes those correctly is unclear, so in vLLM
  modes training loads the base unquantized (`quantize: false` in `src/rlvr.py`). On a T4 that
  is fp16 Model A, 8 GB, which the server check tests for memory.
- `server` streams the merged weights by NCCL to `trl vllm-serve` on another GPU.
- With vLLM it logs `sampling/sampling_logp_difference/mean`, the mean gap between the token log
  probabilities vLLM sampled with and the trainer's. A small gap means the trained weights reached
  vLLM; a stale copy shows as a large one.

## The checks

`notebooks/07_check_trl_vllm.ipynb` on Kaggle T4 x2, branch `trl-vllm-check`. Each check runs 4
GRPO steps of 4 samples on the same 4 step 3 windows, picked for median answer length, with a
learning rate 5x step 3's so the weights move enough for a stale copy to show.

| check | mode | model | tests |
|---|---|---|---|
| baseline | transformers' generate | Model A, 4-bit, step 2 adapter, GPU 0 | the speed to beat on Kaggle |
| colocate | vLLM in the training process | Qwen3-0.6B, fresh adapter, GPU 0 | the H100 code path: install, weight sync, sampling |
| server | `trl vllm-serve` on GPU 1 | Model A, fp16, step 2 adapter, GPU 0 | the Kaggle path, and whether fp16 Model A trains in 15 GB |

A check passes if it finishes its steps, its answers earn some reward, and the worst logged gap is
at most 0.1. The report, with seconds per step and the speed against the baseline, goes to the
Hub under `checks/trl_vllm/`. Estimated at 30 to 45 minutes of Kaggle time, not measured.

It does not test colocate memory with Model A, which needs a GPU bigger than a T4, nor runs longer
than 4 steps, nor whether vLLM sampling changes what GRPO learns.

## After it passes

1. Set `rlvr.use_vllm: true` in `configs/real_rlvr.yaml`, with `vllm_mode: colocate` on an H100
   or `server` on Kaggle T4 x2, and keep the transformers path as the fallback.
2. Install `requirements-rl-vllm.txt` into one venv where training runs, in
   `scripts/nebius_rlvr.sh` and the step 3 notebook.
3. The paid run's own short test then exercises colocate with Model A in its first 4 steps, on
   the real memory, before the long part starts.
