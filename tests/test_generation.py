import pytest

torch = pytest.importorskip("torch")

from src.generation import RepetitionStop, apply_windowed_penalty, WindowedRepeatPenalty


def test_windowed_penalty_only_touches_last_n_tokens():
    scores = torch.tensor([[2.0, -2.0, 4.0, 1.0, 3.0]])
    ids = torch.tensor([[4, 0, 1, 1]])          # token 4 is outside a window of 3
    out = apply_windowed_penalty(scores.clone(), ids, penalty=2.0, last_n=3)
    assert out.tolist() == [[1.0, -4.0, 4.0, 1.0, 3.0]]


def test_penalty_one_is_identity_and_processor_wraps():
    scores = torch.randn(2, 10)
    ids = torch.randint(0, 10, (2, 70))
    assert torch.equal(apply_windowed_penalty(scores.clone(), ids, 1.0, 64), scores)
    p = WindowedRepeatPenalty(1.05)
    full = apply_windowed_penalty(scores.clone(), ids, 1.05, 64)
    assert torch.allclose(p(ids, scores.clone()), full)


def test_repetition_stop():
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "Same looping sentence in the contract text. " * int(ids.shape[0] // 8)

    stop = RepetitionStop(Tok(), prompt_len=10, every=32)
    assert not stop(torch.zeros(1, 10 + 31, dtype=torch.long), None).item()
    assert stop(torch.zeros(1, 10 + 64, dtype=torch.long), None).item()
    assert stop.fired


def test_prefill_gives_the_same_output_as_a_single_pass():
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
    except (ImportError, RuntimeError):
        pytest.skip("transformers cannot import Qwen3 in this environment")
    from src.generation import prefill_cache
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=97, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=64, head_dim=16,
        layer_types=["full_attention"] * 2, max_position_embeddings=512,
        attn_implementation="sdpa")).eval()
    ids = torch.randint(0, 97, (1, 137))
    mask = torch.ones_like(ids)
    kw = dict(attention_mask=mask, max_new_tokens=25, min_new_tokens=25, do_sample=False,
              temperature=None, top_p=None, top_k=None, pad_token_id=0)
    with torch.inference_mode():
        plain = model.generate(input_ids=ids, **kw)
        for size in (10, 40, 136):
            cache = prefill_cache(model, ids, size)
            assert cache.get_seq_length() == 136
            sliced = model.generate(input_ids=ids, past_key_values=cache, **kw)
            assert torch.equal(plain, sliced), size
