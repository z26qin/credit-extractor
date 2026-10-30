"""Generation backends: vLLM (batched, preferred) and transformers (fallback).

Both backends get the SAME pre-rendered prompt strings (the model's own chat template,
add_generation_prompt=True), greedy decoding, fp16, and the same output cap.

Repeat penalty. Model B's documented ladder (1.05 -> 1.15 -> 1.25) was measured with
Ollama, whose repeat_penalty only looks at the last `repeat_last_n` = 64 tokens. vLLM's
and transformers' built-in repetition_penalty instead penalize every token in the prompt
and output - for verbatim extraction that penalizes nearly every token the model should
copy, so it barely singles out the looping span. `WindowedRepeatPenalty` reproduces the
Ollama semantics for transformers; the ladder therefore runs on the HF backend by
default (see README, "Retry ladder").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

from .parsing import detect_repetition

OLLAMA_REPEAT_LAST_N = 64
# transformers on T4: one attention pass over a whole 8-14k token prompt needs a T x T matrix
# per layer (5.6 / 19 GiB, OOM on Kaggle), so long prompts enter the cache in slices
PREFILL_TOKENS = 512


@dataclass
class GenResult:
    text: str
    finish_reason: str | None      # "stop" | "length" | "repetition" (HF early stop)
    n_prompt_tokens: int
    n_output_tokens: int
    latency_s: float               # per-request wall time (HF) or batch wall / batch size (vLLM)
    batch_size: int

    def to_dict(self) -> dict:
        return asdict(self)


def render_prompts(tokenizer, message_lists: list[list[dict]]) -> list[str]:
    return [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in message_lists]


def apply_windowed_penalty(scores, input_ids, penalty: float, last_n: int):
    """Ollama/llama.cpp-style repeat penalty over the last `last_n` tokens (in place).

    scores: [B, V] float tensor, input_ids: [B, L] long tensor. Positive logits are
    divided by `penalty`, negative ones multiplied (same rule as llama.cpp / HF).
    """
    import torch
    if penalty == 1.0:
        return scores
    window = input_ids[:, -last_n:]
    picked = torch.gather(scores, 1, window)
    picked = torch.where(picked > 0, picked / penalty, picked * penalty)
    scores.scatter_(1, window, picked)
    return scores


class WindowedRepeatPenalty:
    """transformers LogitsProcessor wrapper around apply_windowed_penalty."""

    def __init__(self, penalty: float, last_n: int = OLLAMA_REPEAT_LAST_N):
        self.penalty, self.last_n = penalty, last_n

    def __call__(self, input_ids, scores):
        return apply_windowed_penalty(scores, input_ids, self.penalty, self.last_n)


class RepetitionStop:
    """Stop an HF decode once its output tail is a detected repetition loop.

    Saves T4 time on retries that loop again; the result is classified as a loop either
    way, so this does not change any metric versus running to the cap.
    """

    def __init__(self, tokenizer, prompt_len: int, every: int = 32):
        self.tok, self.prompt_len, self.every = tokenizer, prompt_len, every
        self.fired = False

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        n_new = input_ids.shape[1] - self.prompt_len
        hit = False
        if n_new > 0 and n_new % self.every == 0:
            tail = self.tok.decode(input_ids[0, self.prompt_len:][-1500:],
                                   skip_special_tokens=True)
            hit = detect_repetition(tail)
        self.fired = self.fired or hit
        return torch.full((input_ids.shape[0],), hit, dtype=torch.bool,
                          device=input_ids.device)


def prefill_cache(model, ids, slice_tokens: int = PREFILL_TOKENS):
    """Run all but the last prompt token through the model in slices; return the cache.

    generate() then continues from it. Each slice costs slice x T attention memory instead
    of T x T. The output is the same as a single pass, because the attention result for a
    token does not depend on how the prompt was split.
    """
    from transformers import DynamicCache
    cache = DynamicCache()
    for start in range(0, ids.shape[1] - 1, slice_tokens):
        end = min(start + slice_tokens, ids.shape[1] - 1)
        model(input_ids=ids[:, start:end], past_key_values=cache, use_cache=True)
    return cache


class VLLMBackend:
    name = "vllm"

    def __init__(self, model: str, tensor_parallel_size: int, max_model_len: int,
                 gpu_memory_utilization: float = 0.90, enforce_eager: bool = False):
        from vllm import LLM
        self.llm = LLM(model=model, dtype="float16", tensor_parallel_size=tensor_parallel_size,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=enforce_eager, seed=0)
        self.tokenizer = self.llm.get_tokenizer()
        self.tp = tensor_parallel_size

    def generate(self, prompts: list[str], max_tokens: int,
                 repeat_penalty: float = 1.0) -> list[GenResult]:
        from vllm import SamplingParams
        # NB: vLLM's repetition_penalty is full-context (not Ollama's 64-token window)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            repetition_penalty=repeat_penalty, seed=0)
        t0 = time.time()
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        per = (time.time() - t0) / max(1, len(prompts))
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(GenResult(text=c.text, finish_reason=c.finish_reason,
                                 n_prompt_tokens=len(o.prompt_token_ids),
                                 n_output_tokens=len(c.token_ids), latency_s=per,
                                 batch_size=len(prompts)))
        return res


class HFBackend:
    name = "hf"

    def __init__(self, model: str):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", attn_implementation="sdpa",
            **{dtype_kw: torch.float16})
        self.model.eval()
        self.tp = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def generate(self, prompts: list[str], max_tokens: int,
                 repeat_penalty: float = 1.0) -> list[GenResult]:
        from transformers import LogitsProcessorList, StoppingCriteriaList
        res = []
        for p in prompts:
            enc = self.tokenizer(p, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(self.model.device)
            n_prompt = ids.shape[1]
            stop = RepetitionStop(self.tokenizer, n_prompt)
            procs = LogitsProcessorList(
                [WindowedRepeatPenalty(repeat_penalty)] if repeat_penalty != 1.0 else [])
            t0 = time.time()
            with self.torch.inference_mode():
                cache = (prefill_cache(self.model, ids)
                         if n_prompt > PREFILL_TOKENS else None)
                out = self.model.generate(
                    input_ids=ids, attention_mask=enc["attention_mask"].to(ids.device),
                    past_key_values=cache,
                    max_new_tokens=max_tokens, do_sample=False, temperature=None,
                    top_p=None, top_k=None, logits_processor=procs,
                    stopping_criteria=StoppingCriteriaList([stop]),
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            dt = time.time() - t0
            new = out[0, n_prompt:]
            n_out = int(new.shape[0])
            eos = set(self.model.generation_config.eos_token_id
                      if isinstance(self.model.generation_config.eos_token_id, list)
                      else [self.model.generation_config.eos_token_id])
            if stop.fired:
                fr = "repetition"
            elif n_out and int(new[-1]) in eos:
                fr = "stop"
            else:
                fr = "length" if n_out >= max_tokens else "stop"
            res.append(GenResult(text=self.tokenizer.decode(new, skip_special_tokens=True),
                                 finish_reason=fr, n_prompt_tokens=n_prompt,
                                 n_output_tokens=n_out, latency_s=dt, batch_size=1))
        return res


def make_backend(kind: str, model: str, tp: int = 1, max_model_len: int = 32768,
                 **kw):
    if kind == "vllm":
        return VLLMBackend(model, tp, max_model_len, **kw)
    if kind == "hf":
        return HFBackend(model)
    raise ValueError(f"unknown backend {kind!r}")
