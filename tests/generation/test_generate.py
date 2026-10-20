"""Behavioral contract for one uncached greedy CPU request."""

import pytest
import torch
from torch import nn

from cs336_basics.generation import (
    FinishReason,
    GenerationResult,
    generate,
)


class ScriptedLM(nn.Module):
    """Emit known next-token scores while checking the model boundary."""

    def __init__(
        self,
        expected_prefixes: tuple[tuple[int, ...], ...],
        selected_tokens: tuple[int, ...],
    ) -> None:
        super().__init__()
        assert len(expected_prefixes) == len(selected_tokens)
        self.expected_prefixes = expected_prefixes
        self.selected_tokens = selected_tokens
        self.seen_prefixes: list[tuple[int, ...]] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        step = len(self.seen_prefixes)
        assert step < len(self.expected_prefixes), "Forward called after completion"
        expected = self.expected_prefixes[step]
        assert token_ids.shape == (1, len(expected))
        assert token_ids.dtype == torch.int64
        assert token_ids.device.type == "cpu"
        assert token_ids.tolist() == [list(expected)]
        assert not self.training, "Generation must use evaluation behavior"
        assert not torch.is_grad_enabled(), "Generation must disable autograd"
        self.seen_prefixes.append(tuple(token_ids[0].tolist()))

        logits = torch.full(
            (1, len(expected), 32), -20.0, dtype=torch.float32, device="cpu"
        )
        # Earlier positions deliberately outrank every last-position score.
        logits[:, :-1, 31] = 1000.0
        logits[0, -1, self.selected_tokens[step]] = 10.0
        return logits


def assert_generation_case(
    *,
    selected_tokens: tuple[int, ...],
    expected_prefixes: tuple[tuple[int, ...], ...],
    eos_id: int | None,
    max_new_tokens: int,
    finish_reason: FinishReason,
    context_capacity: int = 16,
) -> None:
    prompt = [11, 24, 7]
    original_prompt = prompt.copy()
    model = ScriptedLM(expected_prefixes, selected_tokens)

    result = generate(
        model=model,
        token_ids=prompt,
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        context_capacity=context_capacity,
        device="cpu",
    )

    assert isinstance(result, GenerationResult)
    assert result.token_ids == (*original_prompt, *selected_tokens)
    assert result.finish_reason is finish_reason
    assert model.seen_prefixes == list(expected_prefixes)
    assert prompt == original_prompt, "Generation must not mutate the caller's prompt"


def test_greedy_uses_last_position_and_recomputes_prefix_until_token_budget():
    assert_generation_case(
        selected_tokens=(9, 5, 6),
        expected_prefixes=(
            (11, 24, 7),
            (11, 24, 7, 9),
            (11, 24, 7, 9, 5),
        ),
        eos_id=2,
        max_new_tokens=3,
        finish_reason=FinishReason.MAX_NEW_TOKENS,
    )


@pytest.mark.parametrize(
    ("eos_id", "selected_tokens", "expected_prefixes"),
    [
        pytest.param(
            0,
            (0,),
            ((11, 24, 7),),
            id="zero_eos_on_first_token",
        ),
        pytest.param(
            2,
            (9, 2),
            ((11, 24, 7), (11, 24, 7, 9)),
            id="eos_after_one_token",
        ),
    ],
)
def test_selected_eos_is_retained_and_prevents_another_forward(
    eos_id, selected_tokens, expected_prefixes
):
    assert_generation_case(
        selected_tokens=selected_tokens,
        expected_prefixes=expected_prefixes,
        eos_id=eos_id,
        max_new_tokens=3,
        finish_reason=FinishReason.EOS,
    )


@pytest.mark.parametrize("eos_id", [0, 2])
def test_prompt_ending_in_eos_returns_unchanged_without_a_forward(eos_id):
    prompt = [11, eos_id]
    original_prompt = prompt.copy()
    model = ScriptedLM(expected_prefixes=(), selected_tokens=())

    result = generate(
        model=model,
        token_ids=prompt,
        eos_id=eos_id,
        max_new_tokens=3,
        context_capacity=16,
        device="cpu",
    )

    assert isinstance(result, GenerationResult)
    assert result.token_ids == tuple(original_prompt)
    assert result.finish_reason is FinishReason.EOS
    assert result.error is None
    assert model.seen_prefixes == []
    assert prompt == original_prompt


def test_none_disables_eos_stopping_and_uses_the_full_new_token_budget():
    assert_generation_case(
        selected_tokens=(0, 2, 5),
        expected_prefixes=(
            (11, 24, 7),
            (11, 24, 7, 0),
            (11, 24, 7, 0, 2),
        ),
        eos_id=None,
        max_new_tokens=3,
        finish_reason=FinishReason.MAX_NEW_TOKENS,
    )


def test_eos_wins_when_selected_on_the_last_allowed_token():
    assert_generation_case(
        selected_tokens=(9, 2),
        expected_prefixes=((11, 24, 7), (11, 24, 7, 9)),
        eos_id=2,
        max_new_tokens=2,
        finish_reason=FinishReason.EOS,
    )


def test_capacity_returns_shorter_continuation_instead_of_rejecting_request():
    assert_generation_case(
        selected_tokens=(9, 5),
        expected_prefixes=((11, 24, 7), (11, 24, 7, 9)),
        eos_id=None,
        max_new_tokens=4,
        context_capacity=5,
        finish_reason=FinishReason.MAX_CONTEXT_CAPACITY,
    )


def test_one_remaining_capacity_slot_allows_exactly_one_new_token():
    assert_generation_case(
        selected_tokens=(9,),
        expected_prefixes=((11, 24, 7),),
        eos_id=None,
        max_new_tokens=4,
        context_capacity=4,
        finish_reason=FinishReason.MAX_CONTEXT_CAPACITY,
    )


def test_smaller_new_token_budget_stops_before_context_capacity_is_full():
    assert_generation_case(
        selected_tokens=(9, 5),
        expected_prefixes=((11, 24, 7), (11, 24, 7, 9)),
        eos_id=None,
        max_new_tokens=2,
        context_capacity=6,
        finish_reason=FinishReason.MAX_NEW_TOKENS,
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "context_capacity", "max_new_tokens"),
    [
        pytest.param([], 5, 4, id="empty_prompt"),
        pytest.param([11, 24, 7, 9, 5], 5, 4, id="prompt_at_capacity"),
        pytest.param([11, 24, 7, 9, 5, 6], 5, 4, id="prompt_exceeds_capacity"),
        pytest.param([11, 24, 7], 5, 0, id="zero_new_token_budget"),
        pytest.param([11, 24, 7], 5, -1, id="negative_new_token_budget"),
    ],
)
def test_invalid_request_raises_value_error_without_a_forward(
    prompt_tokens, context_capacity, max_new_tokens
):
    original_prompt = prompt_tokens.copy()
    model = ScriptedLM(expected_prefixes=(), selected_tokens=())

    with pytest.raises(ValueError):
        generate(
            model=model,
            token_ids=prompt_tokens,
            eos_id=None,
            max_new_tokens=max_new_tokens,
            context_capacity=context_capacity,
            device="cpu",
        )

    assert model.seen_prefixes == []
    assert prompt_tokens == original_prompt, "Rejection must not mutate the prompt"
