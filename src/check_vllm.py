"""Checks that TRL's GRPO can sample with vLLM, on free Kaggle GPUs, before a paid run relies on it.

Step 3 sampled with transformers' generate, about 20 tokens a second per answer on an H100, and
that was most of each 58-second step. TRL can sample with vLLM instead, but only with vLLM in the
training venv, which no run here has tried. notebooks/07_check_trl_vllm.ipynb runs three checks
on Kaggle T4 x2, a few GRPO steps each on the same real step 3 windows (docs/trl_vllm_check.md):

1. baseline: transformers' generate, as step 3 would run on Kaggle, for the speed to beat
2. colocate: vLLM inside the training process on one GPU, the mode for an H100. A T4 cannot hold
   Model A twice, so this uses a small Qwen3 with a fresh adapter: it tests the code, not memory
3. server: `trl vllm-serve` with Model A on GPU 1, training Model A from the step 2 adapter on
   GPU 0, the mode for Kaggle

A check passes if it finishes its steps, its answers earn some reward, and the token log
probabilities vLLM sampled with stay close to the trainer's. That last part shows the trained
weights reached vLLM: TRL merges the adapter and sends the weights before each step's sampling.

    python -m src.check_vllm --name colocate --mode colocate --base Qwen/Qwen3-0.6B \
        --data windows.jsonl --out <dir> [--adapter <dir>] [--steps 4]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .platform import Env, pull, push_folder
from .prompts import REPO_ROOT
from .synth import load_set

CONFIG = REPO_ROOT / "configs" / "real_rlvr.yaml"
MAX_LOGP_DIFF = 0.1        # mean absolute per-token log-prob gap, vLLM against the trainer
SMALL_BASE = "Qwen/Qwen3-0.6B"
CHECK_LR = 5e-5            # 5x step 3's, so the weights move enough for a stale copy to show
MODES = ("off", "colocate", "server")


def check_cfg(mode: str, steps: int, port: int = 8000) -> dict:
    """Step 3's settings, with vLLM on or off and logs after every step."""
    from .train import load_cfg
    cfg = load_cfg(CONFIG)
    rl = cfg["rlvr"]
    rl.update(learning_rate=CHECK_LR, logging_steps=1, save_steps=10 * steps)
    if mode != "off":
        rl.update(use_vllm=True, vllm_mode=mode, quantize=False, vllm_server_port=port,
                  vllm_max_model_length=cfg["sample"]["max_model_len"])
    return cfg


def fresh_adapter(base: str, out: Path) -> Path:
    """A new LoRA adapter with step 2's settings, for a base with no trained adapter."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from .train import LORA_TARGETS, load_cfg
    sft = load_cfg(REPO_ROOT / "configs" / "real_sft.yaml")["sft"]
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(base), LoraConfig(
        r=sft["r"], lora_alpha=sft["alpha"], lora_dropout=0.0, target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM"))
    model.save_pretrained(out)
    return out


def fp16_copy(model_id: str, out: Path) -> Path:
    """The model with fp16 in its config. In colocate mode TRL starts vLLM from the model's own
    files and passes no dtype, and a T4 cannot run the bf16 these Qwen3 files name."""
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, local_dir=str(out))
    path = out / "config.json"
    config = json.loads(path.read_text("utf-8"))
    for key in ("torch_dtype", "dtype"):
        if key in config:
            config[key] = "float16"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return out


def summarize(name: str, mode: str, base: str, stats: dict, steps: int,
              generations: int = 4) -> dict:
    """The check's numbers from train_stats.json, and whether it passed."""
    logs = [r for r in stats["log"] if "reward" in r]

    def mean(key: str) -> float | None:
        vals = [float(r[key]) for r in logs if key in r]
        return round(statistics.fmean(vals), 4) if vals else None
    gaps = [float(r["sampling/sampling_logp_difference/mean"]) for r in logs
            if "sampling/sampling_logp_difference/mean" in r]
    length = mean("completions/mean_length")
    seconds = stats.get("seconds_per_step")
    out = {"name": name, "mode": mode, "base": base, "steps": stats["steps"],
           "seconds_per_step": seconds, "peak_gpu_mib": stats.get("peak_gpu_mib"),
           "answer_tokens": length, "reward": mean("reward"),
           "logp_gap_worst": round(max(gaps), 4) if gaps else None,
           "answer_tokens_per_second": round(generations * length / seconds, 1)
           if length and seconds else None}
    problems = []
    if stats["steps"] < steps:
        problems.append(f"stopped after {stats['steps']} of {steps} steps")
    if not out["reward"]:
        problems.append("no answer earned any reward")
    if mode != "off":
        if not gaps:
            problems.append("TRL logged no vLLM log-prob gap, so vLLM may not have sampled")
        elif max(gaps) > MAX_LOGP_DIFF:
            problems.append(f"vLLM's log probs are {max(gaps):.3f} from the trainer's, over "
                            f"{MAX_LOGP_DIFF}: the weights may not have reached vLLM")
    out.update(passed=not problems, problems=problems)
    return out


def run_check(name: str, mode: str, base: str, data: Path, out: Path, steps: int,
              adapter: Path | None = None, port: int = 8000) -> dict:
    """One check, in this process. Needs a GPU, and vLLM in this venv unless mode is off."""
    import torch
    from . import rlvr
    out.mkdir(parents=True, exist_ok=True)
    if adapter is None:
        adapter = fresh_adapter(base, out / "fresh_adapter")
    model = base
    if mode == "colocate" and torch.cuda.get_device_capability(0)[0] < 8:
        model = str(fp16_copy(base, out / "fp16_base"))
    cfg = check_cfg(mode, steps, port)
    stats_path = rlvr.run(cfg, data, adapter, out / "train", base=model,
                          max_steps=steps) / "train_stats.json"
    result = summarize(name, mode, base, json.loads(stats_path.read_text("utf-8")), steps,
                       cfg["rlvr"]["num_generations"])
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


# --------------------------------------------------------------------- orchestration

def pick_windows(rows: list[dict], tokens: dict[str, float], n: int) -> list[dict]:
    """The `n` selected windows whose SFT answers were closest to the median length, so the
    speed each check measures is a typical step's, not the worst."""
    mid = statistics.median(tokens[x["id"]] for x in rows)
    return sorted(rows, key=lambda x: (abs(tokens[x["id"]] - mid), x["id"]))[:n]


def serve(python: str, base: str, port: int, gpu: str, log: Path, timeout: int = 900):
    """`trl vllm-serve` on one GPU, once it answers its health check."""
    import urllib.request
    trl_cli = str(Path(python).parent / "trl")
    cmd = [trl_cli, "vllm-serve", "--model", base, "--tensor-parallel-size", "1",
           "--dtype", "float16", "--max-model-len", "5120", "--gpu-memory-utilization", "0.90",
           "--port", str(port)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, VLLM_USE_FLASHINFER_SAMPLER="0")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT)
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm-serve exited with {proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health/", timeout=5) as r:
                if r.status == 200:
                    return proc
        except OSError:
            pass
        time.sleep(10)
    proc.terminate()
    raise RuntimeError(f"vllm-serve not healthy after {timeout} s; see {log}")


def launch_check(python: str, args: list, gpu: str, log: Path, timeout: int) -> dict | None:
    """src.check_vllm in the combined venv; None if it failed, with the log's tail printed."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false",
               VLLM_USE_FLASHINFER_SAMPLER="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    out = Path(args[args.index("--out") + 1])
    try:
        with open(log, "w", encoding="utf-8") as f:
            code = subprocess.run([python, "-m", "src.check_vllm", *map(str, args)], cwd=REPO_ROOT,
                                  env=env, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = f"timeout after {timeout} s"
    if code == 0 and (out / "result.json").exists():
        return json.loads((out / "result.json").read_text("utf-8"))
    print(f"check failed ({code}); last lines of {log}:", flush=True)
    print("\n".join(log.read_text("utf-8", errors="replace").splitlines()[-25:]), flush=True)
    return None


def versions(python: str) -> dict:
    code = ("import json, torch, transformers, peft, trl\n"
            "try:\n    import vllm; v = vllm.__version__\nexcept Exception as e:\n"
            "    v = f'not importable: {e}'\n"
            "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
            "'transformers': transformers.__version__, 'peft': peft.__version__, "
            "'trl': trl.__version__, 'vllm': v}))")
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-500:]}


def run_all(env: Env, python: str, steps: int = 4) -> dict:
    """The three checks, then the report on the Hub under checks/trl_vllm/."""
    from . import real_rlvr
    cfg = real_rlvr.load_cfg()
    work = env.paths.work / "checks" / "trl_vllm"
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    screen = real_rlvr.screen_dir(env, cfg)
    if env.data_repo:
        pull(env.data_repo, f"{cfg['hub_prefix']}/screen", env.paths.work)
    rows = load_set(screen / "rlvr.jsonl")
    tokens = {x["id"]: x["tokens"] for x in load_set(screen / "scored.jsonl")}
    data = work / "windows.jsonl"
    data.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                            for x in pick_windows(rows, tokens, steps)), encoding="utf-8")
    sft, base = real_rlvr.sft_adapter(env, cfg), cfg["base_model"]

    def check(name, mode, model, gpu, adapter=None, timeout=1800):
        args = ["--name", name, "--mode", mode, "--base", model, "--data", data,
                "--out", work / name, "--steps", steps]
        if adapter:
            args += ["--adapter", adapter]
        print(f"--- {name}: {mode}, {model}", flush=True)
        t0 = time.time()
        result = launch_check(python, args, gpu, logs / f"{name}.log", timeout)
        result = result or {"name": name, "mode": mode, "base": model, "passed": False,
                            "problems": [f"did not finish; see logs/{name}.log"]}
        result["wall_minutes"] = round((time.time() - t0) / 60, 1)
        print(json.dumps(result), flush=True)
        return result

    results = [check("baseline", "off", base, "0", sft),
               check("colocate", "colocate", SMALL_BASE, "0")]
    try:
        server = serve(python, base, 8000, "1", logs / "vllm_serve.log")
    except RuntimeError as e:
        results.append({"name": "server", "mode": "server", "base": base, "passed": False,
                        "problems": [str(e)]})
    else:
        try:
            results.append(check("server", "server", base, "0", sft))
        finally:
            server.terminate()
            server.wait(timeout=120)
    report = {"versions": versions(python), "gpus": [g["name"] for g in env.gpus],
              "steps": steps, "windows": [x["id"] for x in load_set(data)], "checks": results}
    (work / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (work / "report.md").write_text(report_md(report), encoding="utf-8")
    if env.data_repo:
        push_folder(env.data_repo, work, "checks/trl_vllm", message="trl + vllm check")
    print(report_md(report), flush=True)
    return report


def report_md(report: dict) -> str:
    v = report["versions"]
    base = next((c for c in report["checks"] if c["name"] == "baseline" and c.get("passed")), None)
    L = ["# TRL + vLLM check", "",
         f"GPUs: {', '.join(report['gpus'])}. Versions: " +
         ", ".join(f"{k} {v[k]}" for k in v) + ".", "",
         f"Each check ran {report['steps']} GRPO steps of 4 samples on the same "
         f"{len(report['windows'])} step 3 windows, picked for median answer length.", "",
         "| check | mode | model | passed | seconds per step | answer tokens | answer tokens per "
         "second | speed against baseline | log-prob gap, worst | reward | peak GPU MiB |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in report["checks"]:
        speed = (f"{base['seconds_per_step'] / c['seconds_per_step']:.1f}x"
                 if base and c.get("seconds_per_step") else "-")
        L.append(f"| {c['name']} | {c['mode']} | {c['base']} | {'yes' if c['passed'] else 'NO'} | "
                 f"{c.get('seconds_per_step', '-')} | {c.get('answer_tokens', '-')} | "
                 f"{c.get('answer_tokens_per_second', '-')} | {speed} | "
                 f"{c.get('logp_gap_worst', '-')} | {c.get('reward', '-')} | "
                 f"{c.get('peak_gpu_mib', '-')} |")
    problems = [f"- {c['name']}: {p}" for c in report["checks"] for p in c.get("problems", [])]
    L += ["", "Problems:" if problems else "No problems.", *problems, ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One TRL + vLLM check: a few GRPO steps.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    result = run_check(a.name, a.mode, a.base, a.data, a.out, a.steps, a.adapter, a.port)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
