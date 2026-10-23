"""Per-request sampling through the existing uncached generation loop."""

import math
from collections import Counter

import attrs
import pytest
import torch
from torch import nn

from cs336_basics.config.inference import SamplingParams
from cs336_basics.generation import (
    FinishReason,
    GenerationResult,
    Req,
    Sampler,
    batched_generate,
)


WeightedStep = tuple[tuple[tuple[int, ...], dict[int, float]], ...]


class WeightedBatchLM(nn.Module):
    """Check shared forwards and give known weights at each valid last position."""

    def __init__(self, steps: tuple[WeightedStep, ...]) -> None:
        super().__init__()
        self.steps = steps
        self.seen_batches: list[tuple[tuple[int, ...], ...]] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        index = len(self.seen_batches)
        assert index < len(self.steps), "Forward called after completion"
        step = self.steps[index]
        width = max(len(prefix) for prefix, _ in step)
        assert token_ids.shape == (len(step), width)
        assert token_ids.dtype == torch.int64
        assert token_ids.device.type == "cpu"
        assert not self.training
        assert not torch.is_grad_enabled()

        rows = tuple(tuple(row) for row in token_ids.tolist())
        by_padded_row = {
            prefix + (0,) * (width - len(prefix)): (prefix, weights)
            for prefix, weights in step
        }
        assert Counter(rows) == Counter(by_padded_row.keys())
        self.seen_batches.append(rows)

        logits = torch.full((len(step), width, 32), -math.inf)
        # A wrong sequence position produces token 31, including at padding.
        logits[:, :, 31] = 1000.0
        for row_index, row in enumerate(rows):
            prefix, weights = by_padded_row[row]
            position = len(prefix) - 1
            logits[row_index, position, :] = -math.inf
            for token_id, weight in weights.items():
                logits[row_index, position, token_id] = math.log(weight)
        return logits

    def assert_complete(self) -> None:
        assert len(self.seen_batches) == len(self.steps), "Generation stopped early"


def observe_low_probability_draws(
    monkeypatch: pytest.MonkeyPatch, model: WeightedBatchLM
) -> list[tuple[int, torch.Tensor]]:
    """Choose valid non-argmax draws without fixing row or vocabulary order."""
    observed: list[tuple[int, torch.Tensor]] = []

    def draw(
        probs: torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert num_samples == 1
        assert probs.ndim in (1, 2)
        assert probs.device.type == "cpu"
        rows = probs.reshape(-1, probs.shape[-1])
        assert torch.isfinite(rows).all()
        assert (rows >= 0).all()
        support = rows > 0
        assert support.any(dim=-1).all()
        normalized = rows / rows.sum(dim=-1, keepdim=True)
        observed.extend(
            (len(model.seen_batches), row.detach().clone())
            for row in normalized.unbind(0)
        )

        minimum = rows.masked_fill(~support, math.inf).amin(dim=-1, keepdim=True)
        candidates = support & (rows == minimum)
        columns = torch.arange(rows.shape[-1], device=probs.device)
        selected = torch.where(candidates, columns, -1).amax(dim=-1, keepdim=True)
        return selected[0] if probs.ndim == 1 else selected

    monkeypatch.setattr(torch, "multinomial", draw)
    return observed


def assert_saw_distribution(
    observed: list[tuple[int, torch.Tensor]],
    step: int,
    probabilities: tuple[float, ...],
) -> None:
    expected = torch.tensor(probabilities).sort().values
    for actual_step, row in observed:
        if actual_step != step:
            continue
        # Sampling may use a compact retained vocabulary rather than zero columns.
        actual = row[row > 0].sort().values
        if actual.shape == expected.shape and torch.allclose(
            actual, expected, rtol=1e-6, atol=1e-7
        ):
            return
    pytest.fail(f"Missing the request's categorical distribution at forward {step}")


def assert_outcomes(
    requests: list[Req],
    results: list[GenerationResult],
    expected: tuple[tuple[tuple[int, ...], FinishReason], ...],
) -> None:
    assert isinstance(results, list)
    assert len(results) == len(requests) == len(expected)
    for request, result, (tokens, reason) in zip(
        requests, results, expected, strict=True
    ):
        assert isinstance(result, GenerationResult)
        assert isinstance(result.token_ids, tuple)
        assert result.token_ids == tokens
        assert result.finish_reason is reason
        assert result.error is None
        assert request.finished
        assert request.finish_reason is reason
        assert tuple(request.token_ids) == tokens


def test_mixed_sampling_uses_each_valid_last_position_in_one_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = [
        Req(token_ids=[11, 24, 7]),
        Req(token_ids=[4], sampling_params=SamplingParams(temperature=0.5)),
    ]
    model = WeightedBatchLM(
        (
            (
                ((11, 24, 7), {8: 1.0, 9: 1.0}),
                ((4,), {5: 1.0, 9: 4.0, 12: 2.0}),
            ),
        )
    )
    observed = observe_low_probability_draws(monkeypatch, model)

    for request in requests:
        request.sampling_params = attrs.evolve(
            request.sampling_params, max_new_tokens=1, eos_token_id=None
        )
    results = batched_generate(
        model=model,
        sampler=Sampler(vocab_size=32, device="cpu"),
        requests=requests,
        context_capacity=16,
        device="cpu",
        pad_id=0,
    )

    assert_outcomes(
        requests,
        results,
        (
            ((11, 24, 7, 8), FinishReason.MAX_NEW_TOKENS),
            ((4, 5), FinishReason.MAX_NEW_TOKENS),
        ),
    )
    assert_saw_distribution(observed, 1, (1 / 21, 16 / 21, 4 / 21))
    model.assert_complete()


def test_different_request_filters_drive_non_argmax_tokens_in_one_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = [
        Req(
            token_ids=[4],
            sampling_params=SamplingParams(temperature=0.5, top_k=2),
        ),
        Req(
            token_ids=[11, 24, 7],
            sampling_params=SamplingParams(temperature=2.0, top_p=0.7),
        ),
    ]
    model = WeightedBatchLM(
        (
            (
                ((4,), {6: 1.0, 9: 4.0, 5: 2.0, 7: 3.0}),
                ((11, 24, 7), {10: 1.0, 12: 9.0, 13: 4.0}),
            ),
        )
    )
    observed = observe_low_probability_draws(monkeypatch, model)

    for request in requests:
        request.sampling_params = attrs.evolve(
            request.sampling_params, max_new_tokens=1, eos_token_id=None
        )
    results = batched_generate(
        model=model,
        sampler=Sampler(vocab_size=32, device="cpu"),
        requests=requests,
        context_capacity=16,
        device="cpu",
        pad_id=0,
    )

    assert_outcomes(
        requests,
        results,
        (
            ((4, 7), FinishReason.MAX_NEW_TOKENS),
            ((11, 24, 7, 13), FinishReason.MAX_NEW_TOKENS),
        ),
    )
    assert_saw_distribution(observed, 1, (16 / 25, 9 / 25))
    assert_saw_distribution(observed, 1, (3 / 5, 2 / 5))
    model.assert_complete()


def test_sampling_settings_follow_survivors_after_two_eos_removals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = [
        Req(token_ids=[11, 24, 7]),
        Req(
            token_ids=[4],
            sampling_params=SamplingParams(temperature=0.5, top_k=2),
        ),
        Req(
            token_ids=[8, 6],
            sampling_params=SamplingParams(temperature=2.0, top_p=0.7),
        ),
    ]
    model = WeightedBatchLM(
        (
            (
                ((11, 24, 7), {2: 1.0}),
                ((4,), {6: 1.0, 9: 4.0, 5: 2.0, 7: 3.0}),
                ((8, 6), {10: 1.0, 12: 9.0, 13: 4.0}),
            ),
            (
                ((4, 7), {9: 4.0, 2: 3.0}),
                ((8, 6, 13), {10: 1.0, 12: 9.0, 14: 4.0}),
            ),
            (
                ((8, 6, 13, 14), {10: 1.0, 12: 9.0, 15: 4.0}),
            ),
        )
    )
    observed = observe_low_probability_draws(monkeypatch, model)

    # Each request owns its budget; EOS still wins on its last allowed token.
    for budget, request in enumerate(requests, start=1):
        request.sampling_params = attrs.evolve(
            request.sampling_params, max_new_tokens=budget, eos_token_id=2
        )
    results = batched_generate(
        model=model,
        sampler=Sampler(vocab_size=32, device="cpu"),
        requests=requests,
        context_capacity=16,
        device="cpu",
        pad_id=0,
    )

    assert_outcomes(
        requests,
        results,
        (
            ((11, 24, 7, 2), FinishReason.EOS),
            ((4, 7, 2), FinishReason.EOS),
            ((8, 6, 13, 14, 15), FinishReason.MAX_NEW_TOKENS),
        ),
    )
    for step in (1, 2):
        assert_saw_distribution(observed, step, (16 / 25, 9 / 25))
    for step in (1, 2, 3):
        assert_saw_distribution(observed, step, (3 / 5, 2 / 5))
    model.assert_complete()


def test_disabled_top_k_preserves_the_whole_model_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab_size = 32768

    class UniformLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert self.calls == 1, "Forward called after completion"
            assert token_ids.shape == (1, 1)
            assert token_ids.dtype == torch.int64
            assert token_ids.device.type == "cpu"
            assert not self.training
            assert not torch.is_grad_enabled()
            return torch.zeros((1, 1, vocab_size))

    draws: list[torch.Tensor] = []

    def draw(
        probs: torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert num_samples == 1
        assert probs.ndim in (1, 2)
        assert probs.device.type == "cpu"
        rows = probs.reshape(-1, probs.shape[-1])
        assert rows.shape == (1, vocab_size)
        assert torch.isfinite(rows).all()
        assert (rows > 0).all(), "Disabled top-k must retain every vocabulary token"
        normalized = rows / rows.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(
            normalized,
            torch.full_like(normalized, 1 / vocab_size),
            rtol=1e-6,
            atol=1e-9,
        )
        draws.append(normalized.detach().clone())
        # Every column is supported; the implementation may remap sorted ranks.
        return torch.zeros(
            (*probs.shape[:-1], 1), dtype=torch.int64, device=probs.device
        )

    monkeypatch.setattr(torch, "multinomial", draw)
    model = UniformLM()
    requests = [
        Req(
            token_ids=[11],
            sampling_params=SamplingParams(
                temperature=1.0, top_k=None, top_p=1.0, max_new_tokens=1
            ),
        )
    ]

    results = batched_generate(
        model=model,
        sampler=Sampler(vocab_size=vocab_size, device="cpu"),
        requests=requests,
        context_capacity=16,
        device="cpu",
        pad_id=0,
    )

    assert draws, "A positive-temperature request must use categorical sampling"
    assert model.calls == 1
    assert len(results) == 1
    assert len(results[0].token_ids) == 2
    token_id = results[0].token_ids[-1]
    assert 0 <= token_id < vocab_size
    assert_outcomes(
        requests,
        results,
        (((11, token_id), FinishReason.MAX_NEW_TOKENS),),
    )
