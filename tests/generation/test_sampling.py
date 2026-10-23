"""CPU/CUDA batch preparation and sampling; request RNG isolation is a later step."""

import math

import pytest
import torch

from cs336_basics.config.inference import SamplingParams
from cs336_basics.generation.requests import Batch, Req
from cs336_basics.generation.sampling import BatchSamplingArgs, Sampler


def make_batch(*params: SamplingParams) -> Batch:
    return Batch(
        reqs=[
            Req(token_ids=[index + 1], sampling_params=param)
            for index, param in enumerate(params)
        ]
    )


def assert_token_vector(
    tokens: torch.Tensor, expected: list[int], *, device: torch.device
) -> None:
    assert tokens.shape == (len(expected),)
    assert tokens.dtype == torch.int64
    assert tokens.device == device
    assert tokens.tolist() == expected


def script_draw(
    monkeypatch: pytest.MonkeyPatch,
    selected: list[int],
    *,
    execution_device: torch.device,
) -> list[torch.Tensor]:
    """Observe the categorical distribution without a statistical assertion."""
    observed: list[torch.Tensor] = []

    def draw(
        probs: torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert num_samples == 1
        assert probs.ndim == 2
        assert probs.shape[0] == len(selected)
        assert probs.device == execution_device
        observed.append(probs.detach().clone())
        return torch.tensor(selected, dtype=torch.int64, device=probs.device)[:, None]

    monkeypatch.setattr(torch, "multinomial", draw)
    return observed


def test_all_greedy_preparation_uses_argmax_without_a_random_draw(
    monkeypatch: pytest.MonkeyPatch,
    execution_device,
) -> None:
    sampler = Sampler(vocab_size=5, device=execution_device)
    args = sampler.prepare_batch_sampling_args(
        make_batch(SamplingParams(), SamplingParams())
    )
    assert args.temperatures is None, "All None temperatures mean greedy"

    def forbid_draw(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("An entirely greedy batch must not draw random samples")

    monkeypatch.setattr(torch, "multinomial", forbid_draw)
    # A tie also exercises exact argmax's first-maximum behavior.
    logits = torch.tensor(
        [[1.0, 4.0, 4.0, 0.0, -1.0], [5.0, 0.0, 4.0, 1.0, -2.0]],
        device=execution_device,
    )
    assert_token_vector(sampler.sample(logits, args), [1, 0], device=execution_device)


def test_stochastic_preparation_preserves_per_request_settings_as_columns(
    execution_device,
) -> None:
    sampler = Sampler(vocab_size=5, device=execution_device)
    args = sampler.prepare_batch_sampling_args(
        make_batch(
            SamplingParams(temperature=0.5, top_k=1, top_p=0.8),
            SamplingParams(temperature=2.0, top_k=None, top_p=1.0),
        )
    )
    assert args.temperatures is not None, "Positive temperatures are stochastic"
    assert args.top_k is not None
    assert args.top_p is not None
    for values in (args.temperatures, args.top_k, args.top_p):
        assert isinstance(values, torch.Tensor)
        assert values.shape == (2, 1), "Each parameter must broadcast across its row"
        assert values.device == execution_device
    assert args.temperatures.is_floating_point()
    assert args.top_p.is_floating_point()
    assert args.top_k.dtype in (torch.int32, torch.int64)
    torch.testing.assert_close(
        args.temperatures,
        torch.tensor([[0.5], [2.0]], device=execution_device),
        check_dtype=False,
    )
    torch.testing.assert_close(
        args.top_k, torch.tensor([[1], [5]], device=execution_device), check_dtype=False
    )
    torch.testing.assert_close(
        args.top_p,
        torch.tensor([[0.8], [1.0]], device=execution_device),
        check_dtype=False,
    )


def test_mixed_preparation_keeps_columns_and_stochastic_request_settings(
    execution_device,
) -> None:
    """Check preparation shape; exact mixed-greedy selection is not certified here."""
    sampler = Sampler(vocab_size=5, device=execution_device)
    args = sampler.prepare_batch_sampling_args(
        make_batch(
            SamplingParams(),
            SamplingParams(temperature=0.7, top_k=2, top_p=0.9),
        )
    )
    assert args.temperatures is not None
    assert args.top_k is not None
    assert args.top_p is not None
    for values in (args.temperatures, args.top_k, args.top_p):
        assert isinstance(values, torch.Tensor)
        assert values.shape == (2, 1)
        assert values.device == execution_device
    # Do not prescribe an epsilon or any other representation for the greedy row.
    torch.testing.assert_close(
        args.temperatures[1],
        torch.tensor([0.7], device=execution_device),
        check_dtype=False,
    )
    torch.testing.assert_close(
        args.top_k, torch.tensor([[5], [2]], device=execution_device), check_dtype=False
    )
    torch.testing.assert_close(
        args.top_p,
        torch.tensor([[1.0], [0.9]], device=execution_device),
        check_dtype=False,
    )


def test_each_temperature_changes_its_own_categorical_distribution(
    monkeypatch: pytest.MonkeyPatch,
    execution_device,
) -> None:
    sampler = Sampler(vocab_size=3, device=execution_device)
    args = BatchSamplingArgs(
        temperatures=torch.tensor([[0.5], [2.0]], device=execution_device)
    )
    logits = torch.tensor(
        [[math.log(4.0), 0.0, -math.inf]], device=execution_device
    ).repeat(2, 1)
    observed = script_draw(monkeypatch, [0, 1], execution_device=execution_device)

    assert_token_vector(sampler.sample(logits, args), [0, 1], device=execution_device)
    assert len(observed) == 1
    # Weights 4:1 become 16:1 at T=0.5 and 2:1 at T=2.
    expected = torch.tensor(
        [[16 / 17, 1 / 17, 0.0], [2 / 3, 1 / 3, 0.0]], device=execution_device
    )
    torch.testing.assert_close(observed[0], expected, rtol=1e-6, atol=1e-7)


def test_different_top_k_values_keep_the_correct_vocabulary_entries(
    monkeypatch: pytest.MonkeyPatch,
    execution_device,
) -> None:
    sampler = Sampler(vocab_size=5, device=execution_device)
    args = BatchSamplingArgs(
        temperatures=torch.ones((2, 1), device=execution_device),
        top_k=torch.tensor([[1], [3]], device=execution_device),
    )
    logits = torch.tensor(
        [[0.0, math.log(4.0), math.log(2.0), math.log(3.0), -math.inf]],
        device=execution_device,
    ).repeat(2, 1)
    observed = script_draw(monkeypatch, [1, 3], execution_device=execution_device)

    assert_token_vector(sampler.sample(logits, args), [1, 3], device=execution_device)
    assert len(observed) == 1
    expected = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 4 / 9, 2 / 9, 3 / 9, 0.0]],
        device=execution_device,
    )
    torch.testing.assert_close(observed[0], expected, rtol=1e-6, atol=1e-7)


def test_single_stochastic_request_retains_its_batch_dimension(
    execution_device,
) -> None:
    sampler = Sampler(vocab_size=3, device=execution_device)
    args = BatchSamplingArgs(
        temperatures=torch.ones((1, 1), device=execution_device),
        top_k=torch.tensor([[1]], device=execution_device),
    )
    # Only token 1 survives, so this has no statistical failure probability.
    with torch.random.fork_rng(
        devices=[execution_device] if execution_device.type == "cuda" else []
    ):
        assert_token_vector(
            sampler.sample(
                torch.tensor([[0.0, 5.0, 1.0]], device=execution_device), args
            ),
            [1],
            device=execution_device,
        )


def assert_exact_mixed_selection(
    monkeypatch: pytest.MonkeyPatch,
    params: list[SamplingParams],
    logits: torch.Tensor,
    greedy_tokens: dict[int, int],
    expected_random_probs: list[list[float]],
) -> None:
    """Exercise possible draws without fixing row grouping or vocabulary order."""
    execution_device = logits.device
    sampler = Sampler(vocab_size=logits.shape[-1], device=execution_device)

    # Trying both support endpoints catches the epsilon shortcut even if a
    # backend samples sorted columns and maps their ranks back to vocabulary IDs.
    for pick_last in (False, True):
        observed: list[torch.Tensor] = []

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
            assert probs.device == execution_device
            rows = probs.reshape(-1, probs.shape[-1])
            assert torch.isfinite(rows).all()
            assert (rows >= 0).all()
            support = rows > 0
            assert support.any(dim=-1).all()
            normalized = rows / rows.sum(dim=-1, keepdim=True)
            observed.extend(normalized.detach().clone().unbind(0))
            columns = torch.arange(rows.shape[-1], device=probs.device)
            if pick_last:
                selected = torch.where(support, columns, -1).amax(dim=-1, keepdim=True)
            else:
                selected = torch.where(support, columns, rows.shape[-1]).amin(
                    dim=-1, keepdim=True
                )
            return selected[0] if probs.ndim == 1 else selected

        monkeypatch.setattr(torch, "multinomial", draw)
        args = sampler.prepare_batch_sampling_args(make_batch(*params))
        tokens = sampler.sample(logits.clone(), args)
        assert tokens.shape == (len(params),)
        assert tokens.dtype == torch.int64
        assert tokens.device == execution_device
        for row, param in enumerate(params):
            token = int(tokens[row])
            if row in greedy_tokens:
                assert (
                    token == greedy_tokens[row]
                ), "A stochastic peer must not change exact greedy selection"
            else:
                assert param.temperature is not None
                assert 0 <= token < logits.shape[-1]
                assert math.isfinite(float(logits[row, token]))

        # Sampling can partition/reorder rows or operate in sorted vocabulary
        # space. Compare probability multisets, without prescribing those choices.
        for expected in expected_random_probs:
            expected_sorted = (
                torch.tensor(expected, device=execution_device).sort().values
            )
            assert any(
                row.shape == expected_sorted.shape
                and torch.allclose(
                    row.sort().values, expected_sorted, rtol=1e-6, atol=1e-7
                )
                for row in observed
            ), "The stochastic request must retain its categorical distribution"


@pytest.mark.parametrize("greedy_first", [True, False], ids=["first", "last"])
def test_greedy_ties_use_exact_argmax_with_a_stochastic_peer(
    monkeypatch: pytest.MonkeyPatch, greedy_first: bool, execution_device
) -> None:
    greedy = SamplingParams()
    random = SamplingParams(temperature=1.0)
    greedy_logits = [0.0, 0.0, -math.inf]
    random_logits = [0.0, math.log(4.0), math.log(2.0)]
    params = [greedy, random] if greedy_first else [random, greedy]
    rows = (
        [greedy_logits, random_logits]
        if greedy_first
        else [random_logits, greedy_logits]
    )
    assert_exact_mixed_selection(
        monkeypatch,
        params,
        torch.tensor(rows, device=execution_device),
        greedy_tokens={0 if greedy_first else 1: 0},
        expected_random_probs=[[1 / 7, 4 / 7, 2 / 7]],
    )


def test_greedy_middle_row_preserves_both_stochastic_peers(
    monkeypatch: pytest.MonkeyPatch,
    execution_device,
) -> None:
    denominator = 2.0 + math.sqrt(2.0)
    assert_exact_mixed_selection(
        monkeypatch,
        [
            SamplingParams(temperature=0.5),
            SamplingParams(),
            SamplingParams(temperature=2.0),
        ],
        torch.tensor(
            [
                [0.0, math.log(4.0), math.log(2.0)],
                [0.0, -1e-7, -math.inf],
                [math.log(4.0), math.log(2.0), -math.inf],
            ],
            device=execution_device,
        ),
        greedy_tokens={1: 0},
        expected_random_probs=[
            [1 / 21, 16 / 21, 4 / 21],
            [2 / denominator, math.sqrt(2.0) / denominator, 0.0],
        ],
    )


def test_each_top_p_keeps_its_own_nucleus_in_vocabulary_order(
    monkeypatch: pytest.MonkeyPatch,
    execution_device,
) -> None:
    sampler = Sampler(vocab_size=4, device=execution_device)
    args = sampler.prepare_batch_sampling_args(
        make_batch(
            SamplingParams(temperature=1.0, top_p=0.2),
            SamplingParams(temperature=1.0, top_p=0.6),
            SamplingParams(temperature=1.0, top_p=1.0),
        )
    )
    # Unsorted vocabulary weights 1:4:2:3 keep mapping errors observable.
    logits = torch.tensor(
        [[0.0, math.log(4.0), math.log(2.0), math.log(3.0)]],
        device=execution_device,
    ).repeat(3, 1)
    observed = script_draw(monkeypatch, [1, 3, 0], execution_device=execution_device)

    assert_token_vector(
        sampler.sample(logits, args), [1, 3, 0], device=execution_device
    )
    assert len(observed) == 1
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 4 / 7, 0.0, 3 / 7],
            [1 / 10, 4 / 10, 2 / 10, 3 / 10],
        ],
        device=execution_device,
    )
    torch.testing.assert_close(observed[0], expected, rtol=1e-6, atol=1e-7)


def test_combined_top_k_then_top_p_uses_renormalized_retained_mass(
    monkeypatch: pytest.MonkeyPatch,
    execution_device,
) -> None:
    sampler = Sampler(vocab_size=4, device=execution_device)
    args = sampler.prepare_batch_sampling_args(
        make_batch(SamplingParams(temperature=1.0, top_k=3, top_p=0.75))
    )
    observed = script_draw(monkeypatch, [3], execution_device=execution_device)

    # Top-k leaves weights 4,3,2. Their first two normalized masses sum to
    # 7/9 > 0.75, whereas their original masses sum to only 7/10 < 0.75.
    logits = torch.tensor(
        [[0.0, math.log(4.0), math.log(2.0), math.log(3.0)]],
        device=execution_device,
    )
    assert_token_vector(sampler.sample(logits, args), [3], device=execution_device)
    assert len(observed) == 1
    expected = torch.tensor([[0.0, 4 / 7, 0.0, 3 / 7]], device=execution_device)
    torch.testing.assert_close(observed[0], expected, rtol=1e-6, atol=1e-7)


def test_each_temperature_changes_its_own_top_p_nucleus(
    monkeypatch: pytest.MonkeyPatch,
    execution_device,
) -> None:
    sampler = Sampler(vocab_size=3, device=execution_device)
    args = sampler.prepare_batch_sampling_args(
        make_batch(
            SamplingParams(temperature=0.5, top_p=0.7),
            SamplingParams(temperature=2.0, top_p=0.7),
        )
    )
    logits = torch.tensor(
        [[0.0, math.log(4.0), math.log(2.0)]], device=execution_device
    ).repeat(2, 1)
    observed = script_draw(monkeypatch, [1, 2], execution_device=execution_device)

    assert_token_vector(sampler.sample(logits, args), [1, 2], device=execution_device)
    assert len(observed) == 1
    # At T=0.5, token 1 alone exceeds p. At T=2, tokens 1 and 2 are needed.
    denominator = 2.0 + math.sqrt(2.0)
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2 / denominator, math.sqrt(2.0) / denominator],
        ],
        device=execution_device,
    )
    torch.testing.assert_close(observed[0], expected, rtol=1e-6, atol=1e-7)


def test_real_multinomial_preserves_mixed_batch_shape_device_and_support(
    execution_device,
):
    """Exercise the actual categorical kernel with deterministic retained support."""
    sampler = Sampler(vocab_size=4, device=execution_device)
    args = sampler.prepare_batch_sampling_args(
        make_batch(
            SamplingParams(),
            SamplingParams(temperature=0.7, top_k=1),
            SamplingParams(),
        )
    )
    logits = torch.tensor(
        [[0.0, 5.0, 5.0, -1.0], [0.0, 2.0, 6.0, 1.0], [7.0, 1.0, 0.0, 2.0]],
        device=execution_device,
    )
    devices = [execution_device] if execution_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        selected = sampler.sample(logits, args)
    assert_token_vector(selected, [1, 2, 0], device=execution_device)
