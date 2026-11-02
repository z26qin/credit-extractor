"""Small RLVR: TRL's GRPOTrainer from the SFT adapter, on the synthetic excerpts.

The reward needs no human: every excerpt's answer is known, so code can check whether each
quote is the right text, and a quote that is real text from the wrong provision scores low.
That is why this works on synthetic data and not on unlabeled agreements (docs/next_steps.md).

    python -m src.rlvr --data rlvr.jsonl --adapter <sft>/final --out <dir> [--max-steps N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gold import clean_source
from .metrics import _prf, guide_counts, match_key, right_quote, verbatim_flags
from .parsing import canonicalize, parse_json_object
from .prompts import canon_map, credit_clauses, norm_key

CLAUSES = credit_clauses()
EXACT_WEIGHT = 0.5         # how much of the reward rests on quotes being right and exact


def reward_quotes(pred: dict[str, list[str]], gold_doc: dict, source: str,
                  clauses: list[str]) -> float:
    """Reward in [0, 1] for one document's quotes.

    Found and Strict work per clause, so they do not see extra wrong quotes beside a right one.
    The second factor does: it is the share of quotes that are both right and verbatim.
    """
    quotes = [(c, q) for c in clauses for q in pred.get(c, [])]
    if not quotes:
        return 0.0 if any(gold_doc.get(c) for c in clauses) else 1.0
    n = guide_counts({"x": gold_doc}, {"x": pred}, clauses)["x"]
    found = _prf(n["found_tp"], n["found_fp"], n["found_fn"])[2]
    strict = _prf(n["main_tp"], n["main_fp"], n["main_fn"])[2]
    text = clean_source(source)
    exact = sum(right_quote(match_key(q), [match_key(p["text"]) for p in gold_doc.get(c, [])])
                and verbatim_flags([q], text)[0] for c, q in quotes)
    return (found + strict) / 2 * (1 - EXACT_WEIGHT + EXACT_WEIGHT * exact / len(quotes))


def reward_text(raw: str, gold_doc: dict, source: str, clauses: list[str]) -> float:
    """Reward for a raw model answer: 0 unless it is valid JSON with exactly the schema's keys."""
    d = parse_json_object(raw)
    canon = canon_map(clauses)
    if d is None or {norm_key(k) for k in d} != set(canon):
        return 0.0
    return reward_quotes(canonicalize(d, canon)[0], gold_doc, source, clauses)


def reward_func(completions, excerpt, gold, **_) -> list[float]:
    """TRL calls this with the dataset's columns. `gold` is a JSON string per row."""
    return [reward_text(c, json.loads(g), e, CLAUSES)
            for c, e, g in zip(completions, excerpt, gold)]


def load_policy(base: str, adapter: Path):
    from peft import PeftModel
    from .train import load_base
    model = load_base(base)
    model.config.use_cache = False
    return PeftModel.from_pretrained(model, str(adapter), is_trainable=True)


def run(cfg: dict, data: Path, adapter: Path, out_dir: Path, base: str | None = None,
        prompts: int | None = None, max_steps: int | None = None, hub_repo: str | None = None,
        hub_prefix: str | None = None) -> Path:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from .synth import load_set
    from .train import finish, precision_args, prompt_text, resume_point, trainer_callbacks

    base = base or cfg["base_model"]
    rl = cfg["rlvr"]
    out_dir = Path(out_dir)
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    rows = [{"prompt": prompt_text(tok, x["excerpt"]), "excerpt": x["excerpt"],
             "gold": json.dumps(x["gold"], ensure_ascii=False)}
            for x in load_set(data)[:prompts or rl["prompts"]]]
    model = load_policy(base, adapter)

    # Each optimizer step takes `num_generations` samples of one prompt, one at a time.
    args = GRPOConfig(
        output_dir=str(out_dir), max_steps=max_steps or len(rows), seed=cfg["seed"],
        per_device_train_batch_size=1, gradient_accumulation_steps=rl["num_generations"],
        num_generations=rl["num_generations"], max_completion_length=rl["max_completion_length"],
        beta=0.0, learning_rate=rl["learning_rate"],
        generation_kwargs={"use_cache": True},
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=rl["save_steps"], save_total_limit=1,
        logging_steps=5, disable_tqdm=True, report_to="none", **precision_args())
    clock, hub = trainer_callbacks(hub_repo, hub_prefix)
    trainer = GRPOTrainer(model=model, reward_funcs=reward_func, args=args,
                          train_dataset=Dataset.from_list(rows), processing_class=tok,
                          callbacks=[clock, hub])
    # TRL turns QLoRA adapter weights into bf16, which a T4 cannot compute with. The recipe keeps
    # them in fp32 under fp16 autocast, as SFT does, and no optimizer exists yet at this point.
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainer.train(resume_from_checkpoint=resume_point(out_dir))
    return finish(model, trainer, clock, out_dir, {"stage": "rlvr", "prompts": len(rows)},
                  hub_repo, hub_prefix)


def main(argv=None) -> int:
    from .train import load_cfg
    ap = argparse.ArgumentParser(description="Small RLVR from the SFT adapter.")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--hub-repo")
    ap.add_argument("--hub-prefix")
    a = ap.parse_args(argv)
    run(load_cfg(), a.data, a.adapter, a.out, prompts=a.prompts, max_steps=a.max_steps,
        hub_repo=a.hub_repo, hub_prefix=a.hub_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
