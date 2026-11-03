"""The TRL + vLLM check: GRPO settings with vLLM on and off, the pass rules, the windows it
picks, its report, and one real check with vLLM off on a tiny model. vLLM itself needs a GPU,
so the colocate and server checks run only in notebooks/07_check_trl_vllm.ipynb."""
import json
import sys
from pathlib import Path

import pytest

from src import check_vllm, rlvr
from tests.test_small_loop import small_sets, tiny, tiny_cfg  # noqa: F401


@pytest.mark.parametrize("mode", ["off", "colocate", "server"])
def test_grpo_args_switch_sampling_to_vllm(mode, tmp_path):
    cfg = check_vllm.check_cfg(mode, steps=4, port=8123)
    args = rlvr.grpo_args(cfg, tmp_path, n_rows=4, max_steps=4)
    assert args["logging_steps"] == 1 and args["max_steps"] == 4
    if mode == "off":
        assert "use_vllm" not in args and args["generation_kwargs"] == {"use_cache": True}
        assert cfg["rlvr"].get("quantize") is None
    else:
        assert args["use_vllm"] and args["vllm_mode"] == mode and "generation_kwargs" not in args
        assert args["vllm_server_port"] == 8123 and args["vllm_max_model_length"] == 5120
        assert cfg["rlvr"]["quantize"] is False, "vLLM needs the merged weights unquantized"
    pytest.importorskip("trl")
    from trl import GRPOConfig
    GRPOConfig(**args)                     # TRL accepts every setting


def test_step_3_settings_keep_vllm_off():
    from src import real_rlvr
    cfg = real_rlvr.load_cfg()
    assert cfg["rlvr"]["use_vllm"] is False
    assert "use_vllm" not in rlvr.grpo_args(cfg, Path("x"), n_rows=4)


def stats(steps=4, gap=0.02, reward=0.6, with_gap=True):
    row = {"reward": reward, "completions/mean_length": 250.0, "loss": 0.1}
    if with_gap:
        row["sampling/sampling_logp_difference/mean"] = gap
    return {"steps": steps, "seconds_per_step": 10.0, "peak_gpu_mib": 9000,
            "log": [dict(row) for _ in range(steps)] + [{"train_loss": 0.1}]}


def test_a_check_passes_only_with_its_steps_reward_and_close_log_probs():
    ok = check_vllm.summarize("server", "server", "m", stats(), 4)
    assert ok["passed"] and ok["problems"] == [] and ok["logp_gap_worst"] == 0.02
    assert ok["answer_tokens_per_second"] == 4 * 250 / 10
    far = check_vllm.summarize("server", "server", "m", stats(gap=0.4), 4)
    assert not far["passed"] and "may not have reached vLLM" in far["problems"][0]
    none = check_vllm.summarize("server", "server", "m", stats(with_gap=False), 4)
    assert not none["passed"] and "no vLLM log-prob gap" in none["problems"][0]
    short = check_vllm.summarize("c", "colocate", "m", stats(steps=2), 4)
    assert "stopped after 2 of 4 steps" in short["problems"]
    zero = check_vllm.summarize("c", "colocate", "m", stats(reward=0.0), 4)
    assert "no answer earned any reward" in zero["problems"]
    fresh = check_vllm.summarize("c", "colocate", "m", stats(reward=0.0), 4, trained=False)
    assert fresh["passed"], "a fresh adapter on a small base is not expected to earn reward"
    base = check_vllm.summarize("baseline", "off", "m", stats(with_gap=False), 4)
    assert base["passed"], "the baseline samples with transformers, so it has no gap"


def test_windows_are_the_ones_nearest_the_median_answer():
    rows = [{"id": f"w{i}"} for i in range(7)]
    tokens = {"w0": 50, "w1": 900, "w2": 210, "w3": 230, "w4": 222, "w5": 1000, "w6": 300}
    got = check_vllm.pick_windows(rows, tokens, 3)        # median 230
    assert [x["id"] for x in got] == ["w3", "w4", "w2"]


def test_report_compares_each_check_with_the_baseline():
    checks = [check_vllm.summarize("baseline", "off", "A", dict(stats(with_gap=False),
                                                                 seconds_per_step=40.0), 4),
              check_vllm.summarize("colocate", "colocate", "small", stats(), 4),
              {"name": "server", "mode": "server", "base": "A", "passed": False,
               "problems": ["vllm-serve exited with 1; see log"]}]
    md = check_vllm.report_md({"versions": {"vllm": "0.32.0", "trl": "1.17.1"},
                               "gpus": ["Tesla T4", "Tesla T4"], "steps": 4,
                               "windows": ["a", "b", "c", "d"], "checks": checks})
    assert "| colocate | colocate | small | yes | 10.0 |" in md and "| 4.0x |" in md
    assert "| server | server | A | NO |" in md and "- server: vllm-serve exited with 1" in md


def test_fp16_copy_rewrites_the_dtype(tmp_path, monkeypatch):
    def fake_download(model_id, local_dir):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text(json.dumps(
            {"torch_dtype": "bfloat16", "hidden_size": 8}))
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    out = check_vllm.fp16_copy("org/model", tmp_path / "copy")
    config = json.loads((out / "config.json").read_text())
    assert config == {"torch_dtype": "float16", "hidden_size": 8}


def test_a_check_that_ran_but_failed_keeps_its_result(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "result.json").write_text(json.dumps({"name": "x", "passed": False}))
    got = check_vllm.launch_check(sys.executable, ["-c", "raise SystemExit(1)", "--out", out],
                                  "0", tmp_path / "x.log", timeout=120)
    assert got == {"name": "x", "passed": False}


def test_a_failed_check_is_reported_not_raised(tmp_path, capsys):
    log = tmp_path / "bad.log"
    got = check_vllm.launch_check(sys.executable, ["--name", "x", "--mode", "nope", "--base", "m",
                                                   "--data", tmp_path / "d", "--out", tmp_path / "o"],
                                  "0", log, timeout=120)
    assert got is None and "invalid choice" in log.read_text()
    assert "check failed" in capsys.readouterr().out


def test_a_check_with_vllm_off_runs_on_a_tiny_model(tiny, tmp_path, monkeypatch):
    small_sets(tmp_path / "data")
    real = check_vllm.check_cfg

    def small(mode, steps, port=8000):
        cfg = real(mode, steps, port)
        cfg["rlvr"].update(num_generations=2, max_completion_length=8)
        return cfg
    monkeypatch.setattr(check_vllm, "check_cfg", small)
    result = check_vllm.run_check("baseline", "off", tiny, tmp_path / "data" / "rlvr.jsonl",
                                  tmp_path / "out", steps=2)
    assert result["steps"] == 2 and result["mode"] == "off"
    assert (tmp_path / "out" / "fresh_adapter" / "adapter_config.json").exists()
    assert json.loads((tmp_path / "out" / "result.json").read_text()) == result
