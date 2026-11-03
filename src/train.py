"""Small SFT: QLoRA on Model A, on the synthetic excerpts (docs/next_steps.md).

Follows the recipe in docs/repo_notes.md, written from its documented settings, with fp16 and a
shorter sequence limit for a T4. Upstream used Unsloth. This uses transformers, peft and
bitsandbytes directly, because Unsloth needs a GPU and caps transformers, and this way the whole
loop can be tested on CPU with a tiny model. Runs in its own venv (requirements-train.txt), one
GPU, started by src/small_loop.py.

    python -m src.train --data train.jsonl --out <dir> [--max-steps N]
    python -m src.train --merge --adapter <dir>/final --out <dir>      # adapter -> full model
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import yaml

from .generation import render_prompts
from .platform import push_folder
from .prompts import PROMPTS, REPO_ROOT, messages
from .synth import load_set

IM_END = "<|im_end|>"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# copied from the base after a merge, so the merged model is served exactly as Model A was
BASE_FILES = ["config.json", "generation_config.json", "tokenizer*", "vocab.json", "merges.txt",
              "chat_template*", "special_tokens_map.json", "added_tokens.json"]


def load_cfg(path: Path | None = None) -> dict:
    path = path or REPO_ROOT / "configs" / "small_loop.yaml"
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def prompt_text(tok, excerpt: str) -> str:
    """The prompt M3 sent for condition c, byte for byte."""
    system, prefix = PROMPTS["credit_schema"]
    return render_prompts(tok, [messages(system, prefix, excerpt)])[0]


def encode(tok, example: dict, max_len: int) -> dict | None:
    """Token ids with loss on the answer only, or None if the example is over the limit."""
    prompt = tok(prompt_text(tok, example["excerpt"]), add_special_tokens=False)["input_ids"]
    answer = tok(example["answer"] + IM_END, add_special_tokens=False)["input_ids"]
    ids = prompt + answer
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt) + answer}


def collate(rows: list[dict]) -> dict:
    """Batch size is 1, so nothing needs padding."""
    import torch
    return {k: torch.tensor([r[k] for r in rows]) for k in rows[0]}


def answer_loss(logits, labels, num_items: int | None = None):
    """Summed cross-entropy over the labeled tokens, from logits of the last positions only.

    `logits` covers the last `labels`-suffix length + 1 positions: each predicts the next token,
    so the first of them predicts the first answer token. Divided by `num_items` if given.
    """
    import torch.nn.functional as F
    n = labels.shape[1] - logits.shape[1] + 1
    target = labels[:, n:]
    total = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                            target.reshape(-1), ignore_index=-100, reduction="sum")
    return total / (num_items if num_items is not None else (target != -100).sum())


def answer_trainer():
    """A Trainer that projects only the answer's positions onto the vocabulary.

    With a 24,000-character prompt, full logits are about 151,936 x 7,000 values in fp32, near
    4 GB, plus their gradient, too much for a T4. Only the answer has loss, so only its logits
    are needed; the loss is the same, normalized over the accumulated batch as Trainer does.
    """
    from transformers import Trainer

    class AnswerLossTrainer(Trainer):
        loss_is_scaled_for_ga = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            labeled = (labels != -100).nonzero()[:, 1]
            first = int(labeled.min()) if len(labeled) else labels.shape[1] - 1
            if len(labeled) and int(labeled.max()) - first + 1 != len(labeled):
                raise ValueError("labels must be one suffix of the sequence")
            keep = labels.shape[1] - first + 1
            outputs = model(**inputs, logits_to_keep=keep)
            n = num_items_in_batch
            if n is None:
                n = (labels[:, 1:] != -100).sum() * self.current_gradient_accumulation_steps
            loss = answer_loss(outputs.logits, labels, n)
            return (loss, outputs) if return_outputs else loss

    return AnswerLossTrainer


BIG_GPU_GIB = 40           # from this much memory, the base trains in bf16 instead of 4-bit


def big_gpu() -> bool:
    """A GPU with bf16 and room for the full base, such as an H100, unlike a T4."""
    import torch
    return (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            and torch.cuda.get_device_properties(0).total_memory >= BIG_GPU_GIB * 2**30)


def load_base(model_id: str):
    """Model A in 4-bit on a small GPU, in bf16 on a big one, or as it is on CPU, where only the
    tests run. Steps 1 and 2 trained on T4s in 4-bit. An adapter moves between the two, since
    LoRA only adds to the base's weights."""
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(model_id)
    if big_gpu():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0},
                                                     dtype=torch.bfloat16)
        model.enable_input_require_grads()
        return model
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 device_map={"": 0}, dtype=torch.float16)
    return prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})


def precision_args() -> dict:
    """fp16 on a T4, which has no bf16; bf16 on newer GPUs; plain CPU otherwise, also on a Mac
    with MPS. The check is the compute capability, since torch reports emulated bf16 on a T4."""
    import torch
    gpu = torch.cuda.is_available()
    bf16 = gpu and torch.cuda.get_device_capability(0)[0] >= 8
    return {"fp16": gpu and not bf16, "bf16": bf16, "use_cpu": not gpu}


def trainer_callbacks(hub_repo: str | None, hub_prefix: str | None):
    """Timing for the run's stats, and the latest checkpoint on the Hub after every save."""
    from transformers import TrainerCallback

    class Clock(TrainerCallback):
        def __init__(self):
            self.first_step, self.first_at = None, None
            self.last_step, self.last_at = None, None

        def on_step_end(self, args, state, control, **kwargs):
            now = time.time()
            if self.first_at is None:       # the first step includes loading and compiling
                self.first_step, self.first_at = state.global_step, now
            self.last_step, self.last_at = state.global_step, now

        def seconds_per_step(self) -> float | None:
            done = (self.last_step or 0) - (self.first_step or 0)
            return round((self.last_at - self.first_at) / done, 2) if done else None

    class HubCheckpoints(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if hub_repo and folder.exists():
                push_folder(hub_repo, folder, f"{hub_prefix}/checkpoint", repo_type="model",
                            message=f"checkpoint {state.global_step}")

    return Clock(), HubCheckpoints()


def resume_point(out_dir: Path) -> str | None:
    """Latest local checkpoint, else the one pulled from the Hub."""
    numbered = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if numbered:
        return str(numbered[-1])
    pulled = out_dir / "checkpoint"
    return str(pulled) if (pulled / "trainer_state.json").exists() else None


def finish(model, trainer, clock, out_dir: Path, stats: dict, hub_repo: str | None,
           hub_prefix: str | None) -> Path:
    """Save the adapter and the run's stats, and put both on the Hub."""
    import torch
    final = out_dir / "final"
    model.save_pretrained(final)
    stats.update(steps=trainer.state.global_step, seconds_per_step=clock.seconds_per_step(),
                 peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2**20)
                 if torch.cuda.is_available() else None,
                 log=trainer.state.log_history)
    (final / "train_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    if hub_repo:
        push_folder(hub_repo, final, f"{hub_prefix}/final", repo_type="model",
                    message="final adapter")
    print(json.dumps({k: v for k, v in stats.items() if k != "log"}), flush=True)
    return final


def run(cfg: dict, data: Path, out_dir: Path, base: str | None = None,
        max_steps: int | None = None, hub_repo: str | None = None,
        hub_prefix: str | None = None) -> Path:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    base = base or cfg["base_model"]
    sft = cfg["sft"]
    out_dir = Path(out_dir)
    tok = AutoTokenizer.from_pretrained(base)
    examples = load_set(data)
    encoded = [e for e in (encode(tok, x, sft["max_seq_length"]) for x in examples) if e]
    dropped = len(examples) - len(encoded)
    print(f"{len(encoded)} examples, {dropped} over {sft['max_seq_length']} tokens", flush=True)

    model = load_base(base)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=sft["r"], lora_alpha=sft["alpha"], lora_dropout=0.0, target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    total = max_steps or math.ceil(len(encoded) * sft["epochs"] / sft["grad_accum"])
    args = TrainingArguments(
        output_dir=str(out_dir), max_steps=total, seed=cfg["seed"],
        per_device_train_batch_size=1, gradient_accumulation_steps=sft["grad_accum"],
        learning_rate=sft["learning_rate"], lr_scheduler_type="linear",
        warmup_steps=math.ceil(sft["warmup_ratio"] * total),
        optim="adamw_bnb_8bit" if precision_args()["fp16"] else "adamw_torch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=sft["save_steps"], save_total_limit=1,
        logging_steps=5, disable_tqdm=True, report_to="none", remove_unused_columns=False,
        **precision_args())
    clock, hub = trainer_callbacks(hub_repo, hub_prefix)
    trainer_cls = answer_trainer() if sft.get("answer_logits_only") else Trainer
    trainer = trainer_cls(model=model, args=args, train_dataset=encoded, data_collator=collate,
                          callbacks=[clock, hub])
    trainer.train(resume_from_checkpoint=resume_point(out_dir))
    return finish(model, trainer, clock, out_dir,
                  {"stage": "sft", "examples": len(encoded), "dropped": dropped,
                   "tokens": sum(len(e["input_ids"]) for e in encoded)}, hub_repo, hub_prefix)


def merge(base: str, adapter: Path, out_dir: Path) -> Path:
    """Merge an adapter into the base weights on CPU, keeping the base's own config files."""
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, dtype="auto")
    PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().save_pretrained(out_dir)
    if Path(base).is_dir():
        for pattern in BASE_FILES:
            for f in Path(base).glob(pattern):
                shutil.copy(f, out_dir)
    else:
        snapshot_download(base, local_dir=str(out_dir), allow_patterns=BASE_FILES)
    return Path(out_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Small SFT, or merge an adapter.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--hub-repo")
    ap.add_argument("--hub-prefix")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--config", type=Path, help="default configs/small_loop.yaml")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    if a.merge:
        merge(cfg["base_model"], a.adapter, a.out)
    else:
        run(cfg, a.data, a.out, max_steps=a.max_steps, hub_repo=a.hub_repo,
            hub_prefix=a.hub_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
