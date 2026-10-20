"""Sequential request outcomes and failure isolation with a shared CPU model."""

import pytest
import torch
from torch import nn

from cs336_basics.generation import FinishReason, GenerationResult
from cs336_basics.generation.offline_runner import runner


PROMPT_IDS = {
    "A": (11, 24, 7),
    "B": (21, 22),
    "C": (4, 8),
}


class RecordingTokenizer:
    def __init__(self, *, failing_prompt: str | None = None) -> None:
        self.failing_prompt = failing_prompt
        self.calls: list[str] = []

    def encode(self, prompt: str) -> list[int]:
        self.calls.append(prompt)
        if prompt == self.failing_prompt:
            raise ValueError("Injected tokenization failure")
        return list(PROMPT_IDS[prompt])


class ScriptedLM(nn.Module):
    """Check request prefixes and emit a token or raise at a known forward."""

    def __init__(
        self,
        steps: tuple[tuple[tuple[int, ...], int | BaseException], ...],
    ) -> None:
        super().__init__()
        self.steps = steps
        self.seen_prefixes: list[tuple[int, ...]] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        step = len(self.seen_prefixes)
        assert step < len(self.steps), "Unexpected extra forward"
        expected_prefix, action = self.steps[step]
        assert token_ids.shape == (1, len(expected_prefix))
        assert token_ids.dtype == torch.int64
        assert token_ids.device.type == "cpu"
        prefix = tuple(token_ids[0].tolist())
        assert prefix == expected_prefix
        self.seen_prefixes.append(prefix)

        if isinstance(action, BaseException):
            raise action

        logits = torch.full(
            (1, len(prefix), 32), -20.0, dtype=torch.float32, device="cpu"
        )
        logits[0, -1, action] = 10.0
        return logits


def run_prompts(
    model: ScriptedLM, tokenizer: RecordingTokenizer, prompts: list[str]
) -> list[object]:
    outcomes = runner(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        eos_id=None,
        max_new_tokens=2,
        context_capacity=16,
        device="cpu",
    )
    assert isinstance(outcomes, list)
    return outcomes


def assert_success(outcome: object, expected_ids: tuple[int, ...]) -> None:
    assert isinstance(outcome, GenerationResult)
    assert outcome.token_ids == expected_ids
    assert outcome.finish_reason is FinishReason.MAX_NEW_TOKENS
    assert outcome.error is None


def assert_failure(
    outcome: object,
    expected_ids: tuple[int, ...],
    *,
    error_type: type[Exception],
    error_message: str,
) -> None:
    assert isinstance(outcome, GenerationResult)
    assert outcome.finish_reason is FinishReason.ERROR
    assert isinstance(outcome.token_ids, tuple), "Error IDs must preserve the tuple contract"
    assert outcome.token_ids == expected_ids
    assert isinstance(outcome.error, str)
    assert error_type.__name__ in outcome.error
    assert error_message in outcome.error


def test_successes_preserve_input_order_and_duplicate_prompts():
    tokenizer = RecordingTokenizer()
    model = ScriptedLM(
        (
            ((11, 24, 7), 9),
            ((11, 24, 7, 9), 5),
            ((4, 8), 6),
            ((4, 8, 6), 10),
            ((11, 24, 7), 12),
            ((11, 24, 7, 12), 13),
        )
    )

    outcomes = run_prompts(model, tokenizer, ["A", "C", "A"])

    assert len(outcomes) == 3
    assert_success(outcomes[0], (11, 24, 7, 9, 5))
    assert_success(outcomes[1], (4, 8, 6, 10))
    assert_success(outcomes[2], (11, 24, 7, 12, 13))
    assert tokenizer.calls == ["A", "C", "A"]
    assert model.seen_prefixes == [prefix for prefix, _ in model.steps]


def test_model_failure_after_one_token_keeps_an_outcome_and_c_starts_fresh():
    tokenizer = RecordingTokenizer()
    model = ScriptedLM(
        (
            ((11, 24, 7), 9),
            ((11, 24, 7, 9), 5),
            ((21, 22), 15),
            ((21, 22, 15), RuntimeError("Injected failure after B's first token")),
            ((4, 8), 6),
            ((4, 8, 6), 10),
        )
    )

    outcomes = run_prompts(model, tokenizer, ["A", "B", "C"])

    assert tokenizer.calls == ["A", "B", "C"]
    assert model.seen_prefixes == [prefix for prefix, _ in model.steps]
    assert len(outcomes) == 3, "Do not drop the failed request's outcome"
    assert_success(outcomes[0], (11, 24, 7, 9, 5))
    # The runner knows B's prompt, but generate did not return its partial continuation.
    assert_failure(
        outcomes[1],
        (21, 22),
        error_type=RuntimeError,
        error_message="Injected failure after B's first token",
    )
    assert_success(outcomes[2], (4, 8, 6, 10))


def test_tokenization_failure_keeps_an_outcome_and_c_still_runs():
    tokenizer = RecordingTokenizer(failing_prompt="B")
    model = ScriptedLM(
        (
            ((11, 24, 7), 9),
            ((11, 24, 7, 9), 5),
            ((4, 8), 6),
            ((4, 8, 6), 10),
        )
    )

    outcomes = run_prompts(model, tokenizer, ["A", "B", "C"])

    assert tokenizer.calls == ["A", "B", "C"]
    assert model.seen_prefixes == [prefix for prefix, _ in model.steps]
    assert len(outcomes) == 3, "Tokenization failure must retain its result position"
    assert_success(outcomes[0], (11, 24, 7, 9, 5))
    # No encoded IDs exist when tokenization fails; do not reuse A's IDs.
    assert_failure(
        outcomes[1],
        (),
        error_type=ValueError,
        error_message="Injected tokenization failure",
    )
    assert_success(outcomes[2], (4, 8, 6, 10))


def test_first_prompt_tokenization_failure_does_not_abort_the_next_request():
    tokenizer = RecordingTokenizer(failing_prompt="B")
    model = ScriptedLM(
        (
            ((4, 8), 6),
            ((4, 8, 6), 10),
        )
    )

    outcomes = run_prompts(model, tokenizer, ["B", "C"])

    assert tokenizer.calls == ["B", "C"]
    assert model.seen_prefixes == [prefix for prefix, _ in model.steps]
    assert len(outcomes) == 2
    assert_failure(
        outcomes[0],
        (),
        error_type=ValueError,
        error_message="Injected tokenization failure",
    )
    assert_success(outcomes[1], (4, 8, 6, 10))


@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "preceding_success", [False, True], ids=["first_request", "after_success"]
)
def test_process_control_exceptions_propagate(signal_type, preceding_success):
    tokenizer = RecordingTokenizer()
    signal = signal_type("Injected process interruption")
    steps = ()
    if preceding_success:
        steps = (
            ((11, 24, 7), 9),
            ((11, 24, 7, 9), 5),
        )
    model = ScriptedLM((*steps, ((21, 22), signal)))
    prompts = ["A", "B", "C"] if preceding_success else ["B", "C"]

    with pytest.raises(signal_type) as caught:
        run_prompts(model, tokenizer, prompts)

    assert caught.value is signal
    assert model.seen_prefixes == [prefix for prefix, _ in model.steps]
