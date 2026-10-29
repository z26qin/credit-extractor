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
