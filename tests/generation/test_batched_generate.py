"""Valid static batches, independent completion, and uncached equivalence."""

from collections import Counter

import pytest
import torch
from torch import nn

from cs336_basics.generation import (
    FinishReason,
    GenerationResult,
    Req,
    batched_generate,
    generate,
)
from cs336_basics.transformers.transformers import TransformerLM


BatchStep = tuple[tuple[tuple[int, ...], int], ...]


class ScriptedBatchLM(nn.Module):
    """Give valid-prefix logits and put distractors at all other positions."""

    def __init__(self, steps: tuple[BatchStep, ...]) -> None:
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
        assert not self.training, "Generation must use evaluation behavior"
        assert not torch.is_grad_enabled(), "Generation must disable autograd"

        rows = tuple(tuple(row) for row in token_ids.tolist())
        expected_rows = [
            prefix + (0,) * (width - len(prefix)) for prefix, _ in step
        ]
        # Internal row order may vary; request/result association must not.
        assert Counter(rows) == Counter(expected_rows)
        action_by_row = {
            padded: (prefix, selected)
            for padded, (prefix, selected) in zip(expected_rows, step, strict=True)
        }
        self.seen_batches.append(rows)

        logits = torch.full((len(step), width, 32), -20.0)
        logits[:, :, 31] = 1000.0
        for row_index, row in enumerate(rows):
            prefix, selected = action_by_row[row]
            position = len(prefix) - 1
            logits[row_index, position, :] = -20.0
            logits[row_index, position, selected] = 10.0
        return logits

    def assert_complete(self) -> None:
        assert len(self.seen_batches) == len(self.steps), "Generation stopped early"


def run_batch(
    model: nn.Module,
    requests: list[Req],
    *,
    eos_id: int | None = None,
    max_new_tokens: int = 3,
    context_capacity: int = 16,
) -> list[GenerationResult]:
    results = batched_generate(
        model=model,
        requests=requests,
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        context_capacity=context_capacity,
        device="cpu",
        pad_id=0,
    )
    assert isinstance(results, list)
    assert len(results) == len(requests)
    return results


def assert_finished_results(
    requests: list[Req],
    results: list[GenerationResult],
    expected: tuple[tuple[tuple[int, ...], FinishReason], ...],
) -> None:
    for request, result, (token_ids, reason) in zip(
        requests, results, expected, strict=True
    ):
        assert isinstance(result, GenerationResult)
        assert isinstance(result.token_ids, tuple)
        assert result.token_ids == token_ids
        assert result.finish_reason is reason
        assert result.error is None
        assert request.finished
        assert request.finish_reason is reason
        assert tuple(request.token_ids) == token_ids


def test_unequal_prefixes_and_duplicates_keep_per_row_selection_and_output_order():
    requests = [
        Req(token_ids=[11, 24, 7]),
        Req(token_ids=[4]),
        Req(token_ids=[11, 24, 7]),
    ]
    model = ScriptedBatchLM(
        (
            (((11, 24, 7), 2), ((4,), 0), ((11, 24, 7), 2)),
            (((11, 24, 7, 2), 6), ((4, 0), 8), ((11, 24, 7, 2), 6)),
        )
    )

    results = run_batch(model, requests, max_new_tokens=2)

    assert_finished_results(
        requests,
        results,
        (
            ((11, 24, 7, 2, 6), FinishReason.MAX_NEW_TOKENS),
            ((4, 0, 8), FinishReason.MAX_NEW_TOKENS),
            ((11, 24, 7, 2, 6), FinishReason.MAX_NEW_TOKENS),
        ),
    )
    model.assert_complete()


def test_generated_eos_stops_only_its_request_while_the_other_reaches_its_budget():
    requests = [Req(token_ids=[11, 24, 7]), Req(token_ids=[4])]
    model = ScriptedBatchLM(
        (
            (((11, 24, 7), 2), ((4,), 9)),
            (((4, 9), 5),),
            (((4, 9, 5), 6),),
        )
    )

    results = run_batch(model, requests, eos_id=2)

    assert_finished_results(
        requests,
        results,
        (
            ((11, 24, 7, 2), FinishReason.EOS),
            ((4, 9, 5, 6), FinishReason.MAX_NEW_TOKENS),
        ),
    )
    model.assert_complete()


def test_prompt_ending_in_zero_eos_stays_unchanged_and_new_eos_wins_over_limits():
    requests = [Req(token_ids=[11, 0]), Req(token_ids=[4])]
    model = ScriptedBatchLM(
        (
            (((4,), 9),),
            (((4, 9), 0),),
        )
    )

    results = run_batch(
        model, requests, eos_id=0, max_new_tokens=2, context_capacity=3
    )

    assert_finished_results(
        requests,
        results,
        (
            ((11, 0), FinishReason.EOS),
            ((4, 9, 0), FinishReason.EOS),
        ),
    )
    model.assert_complete()


def test_capacity_and_new_token_budget_stop_different_requests_independently():
    requests = [Req(token_ids=[11, 24, 7]), Req(token_ids=[4])]
    model = ScriptedBatchLM(
        (
            (((11, 24, 7), 9), ((4,), 8)),
            (((11, 24, 7, 9), 5), ((4, 8), 10)),
            (((4, 8, 10), 12),),
        )
    )

    results = run_batch(model, requests, max_new_tokens=3, context_capacity=5)

    assert_finished_results(
        requests,
        results,
        (
            ((11, 24, 7, 9, 5), FinishReason.MAX_CONTEXT_CAPACITY),
            ((4, 8, 10, 12), FinishReason.MAX_NEW_TOKENS),
        ),
    )
    model.assert_complete()


def test_tiny_causal_model_matches_independent_runs_after_reordering_and_duplicates():
    """Compare a multi-request batch with separate batch-size-one calls."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(336)
        model = TransformerLM(
            vocab_size=32,
            context_length=16,
            d_model=16,
            num_layers=2,
            num_heads=2,
            d_ff=32,
            rope_theta=10000.0,
        ).cpu()

    prompts = ([11, 24, 7], [4], [8, 6])
    reference = [
        generate(
            model=model,
            token_ids=prompt.copy(),
            eos_id=None,
            max_new_tokens=3,
            context_capacity=16,
            device="cpu",
        )
        for prompt in prompts
    ]

    for order in ((0, 1, 2, 0), (2, 0, 1, 0)):
        requests = [Req(token_ids=prompts[index].copy()) for index in order]
        results = run_batch(model, requests)
        assert_finished_results(
            requests,
            results,
            tuple(
                (reference[index].token_ids, reference[index].finish_reason)
                for index in order
            ),
        )


def test_shared_forward_failure_propagates_to_the_caller():
    failure = RuntimeError("Injected shared forward failure")

    class FailingLM(nn.Module):
        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            raise failure

    with pytest.raises(RuntimeError, match="Injected shared forward failure") as caught:
        run_batch(
            FailingLM(),
            [Req(token_ids=[11, 24, 7]), Req(token_ids=[4])],
        )

    assert caught.value is failure
