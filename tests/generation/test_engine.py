"""Numerical Engine execution with CPU/CUDA fakes and a tiny causal model."""

import importlib
import math
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch import nn

from cs336_basics.config import EngineConfig, ModelConfig, RuntimeConfig, SamplingParams
from cs336_basics.generation.requests import Batch, Req
from cs336_basics.generation.results import ModelOutput
from cs336_basics.generation.sampling import Sampler
from cs336_basics.transformers.transformers import TransformerLM

engine_module = importlib.import_module("cs336_basics.generation.engine")
WeightedStep = tuple[tuple[tuple[int, ...], dict[int, float]], ...]


def make_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: nn.Module,
    *,
    device: torch.device,
):
    """Use the real constructor and sampler without opening a checkpoint."""
    model_config = ModelConfig(
        vocab_size=32,
        context_length=16,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
    )
    config = EngineConfig(
        model=model_config,
        checkpoint=tmp_path / "unused.pt",
        runtime=RuntimeConfig(device=device.type),
    )
    built: list[ModelConfig] = []
    loaded: list[tuple[Path, nn.Module]] = []

    def build(config: ModelConfig) -> nn.Module:
        built.append(config)
        return model

    def load(src: Path, model: nn.Module) -> None:
        loaded.append((src, model))

    monkeypatch.setattr(engine_module, "build_model", build)
    monkeypatch.setattr(engine_module, "load_model_weights", load)
    engine = engine_module.Engine(config)
    assert engine.model is model
    assert isinstance(engine.sampler, Sampler)
    assert engine.sampler.vocab_size == config.model.vocab_size
    assert torch.device(engine.sampler.device).type == device.type
    assert all(parameter.device == device for parameter in model.parameters())
    assert all(buffer.device == device for buffer in model.buffers())
    return engine, config, built, loaded


def request_snapshot(batch: Batch) -> tuple:
    return tuple(
        (
            id(req),
            tuple(req.token_ids),
            req.prompt_tokens_count,
            req.status,
            req.finish_reason,
        )
        for req in batch.reqs
    )


def assert_output(
    output: ModelOutput, expected: list[int], *, device: torch.device
) -> None:
    assert isinstance(output, ModelOutput)
    tokens = output.next_tokens_gpu
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (len(expected),)
    assert tokens.dtype == torch.int64
    assert tokens.device == device
    assert tokens.tolist() == expected


class WeightedLM(nn.Module):
    """Check shared forwards and give known logits at valid last positions."""

    def __init__(self, steps: tuple[WeightedStep, ...]) -> None:
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.steps = steps
        self.seen: list[tuple[tuple[int, ...], ...]] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert not self.training, "Engine must configure evaluation behavior"
        assert not torch.is_grad_enabled(), "Engine must disable autograd"
        index = len(self.seen)
        assert index < len(self.steps), "Unexpected extra model forward"
        step = self.steps[index]
        width = max(len(prefix) for prefix, _ in step)
        assert token_ids.shape == (len(step), width)
        assert token_ids.dtype == torch.int64
        assert token_ids.device == self._device_anchor.device
        rows = tuple(tuple(row) for row in token_ids.tolist())
        expected_rows = [prefix + (0,) * (width - len(prefix)) for prefix, _ in step]
        assert Counter(rows) == Counter(expected_rows)
        by_row = {
            padded: (prefix, weights)
            for padded, (prefix, weights) in zip(expected_rows, step, strict=True)
        }
        self.seen.append(rows)

        logits = torch.full((len(step), width, 32), -math.inf, device=token_ids.device)
        logits[:, :, 31] = 1000.0
        for row_index, row in enumerate(rows):
            prefix, weights = by_row[row]
            position = len(prefix) - 1
            logits[row_index, position, :] = -math.inf
            for token_id, weight in weights.items():
                logits[row_index, position, token_id] = math.log(weight)
        return logits


def test_engine_initializes_once_and_reuses_resources_without_updating_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execution_device
) -> None:
    first = Req(token_ids=[11, 24, 7])
    first.start()
    second = Req(token_ids=[4])
    first_action = ((11, 24, 7), {5: 1.0, 9: 4.0})
    second_action = ((4,), {8: 4.0, 2: 1.0})
    model = WeightedLM(
        ((first_action, second_action), (second_action, first_action, second_action))
    )
    engine, config, built, loaded = make_engine(
        monkeypatch, tmp_path, model, device=execution_device
    )
    sampler = engine.sampler
    assert not model.training, "The retained model must be ready for inference"

    for batch, expected in (
        (Batch(reqs=[first, second]), [9, 8]),
        (Batch(reqs=[second, first, second]), [8, 9, 8]),
    ):
        before = request_snapshot(batch)
        args = sampler.prepare_batch_sampling_args(batch)
        assert_output(
            engine.forward_batch(batch, args), expected, device=execution_device
        )
        assert request_snapshot(batch) == before
        assert engine.model is model
        assert engine.sampler is sampler

    assert built == [config.model]
    assert loaded == [(config.checkpoint, model)]
    assert len(model.seen) == 2


def test_engine_uses_supplied_mixed_sampling_args_and_valid_last_positions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execution_device
) -> None:
    batch = Batch(
        reqs=[
            Req(token_ids=[11, 24, 7]),
            Req(
                token_ids=[4],
                sampling_params=SamplingParams(temperature=0.5, top_k=2),
            ),
        ]
    )
    model = WeightedLM(
        (
            (
                ((11, 24, 7), {8: 1.0, 9: 1.0}),
                ((4,), {6: 1.0, 9: 4.0, 5: 2.0, 7: 3.0}),
            ),
        )
    )
    engine, _, _, _ = make_engine(monkeypatch, tmp_path, model, device=execution_device)
    observed: list[torch.Tensor] = []

    def draw(probs, num_samples, replacement=False, *, generator=None, out=None):
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
        minimum = rows.masked_fill(~support, math.inf).amin(dim=-1, keepdim=True)
        columns = torch.arange(rows.shape[-1], device=rows.device)
        selected = torch.where(support & (rows == minimum), columns, -1).amax(
            dim=-1, keepdim=True
        )
        return selected[0] if probs.ndim == 1 else selected

    monkeypatch.setattr(torch, "multinomial", draw)
    before = request_snapshot(batch)
    args = engine.sampler.prepare_batch_sampling_args(batch)
    assert_output(engine.forward_batch(batch, args), [8, 7], device=execution_device)
    expected_probs = torch.tensor([9 / 25, 16 / 25], device=execution_device)
    assert any(
        row[row > 0].shape == expected_probs.shape
        and torch.allclose(
            row[row > 0].sort().values, expected_probs, rtol=1e-6, atol=1e-7
        )
        for row in observed
    ), "The stochastic row must retain its temperature/top-k distribution"
    assert len(model.seen) == 1
    assert request_snapshot(batch) == before


class RecordingTransformerLM(TransformerLM):
    def __init__(self) -> None:
        super().__init__(
            vocab_size=32,
            context_length=16,
            d_model=16,
            num_layers=1,
            num_heads=2,
            d_ff=32,
            rope_theta=10_000.0,
        )
        self.recorded: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = super().forward(token_ids)
        self.recorded.append((token_ids.detach().clone(), logits.detach().clone()))
        return logits


def test_engine_right_padding_preserves_real_valid_logits_with_different_fillers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execution_device
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(336)
        model = RecordingTransformerLM()
    engine, _, _, _ = make_engine(monkeypatch, tmp_path, model, device=execution_device)
    prefixes = ((11, 24, 7), (4,), (8, 6))
    batch = Batch(reqs=[Req(token_ids=list(prefix)) for prefix in prefixes])
    before = request_snapshot(batch)
    with torch.inference_mode():
        reference = torch.stack(
            [
                model(
                    torch.tensor([prefix], dtype=torch.int64, device=execution_device)
                )[0, -1]
                for prefix in prefixes
            ]
        )
    expected_tokens = reference.argmax(dim=-1).tolist()
    model.recorded.clear()

    for pad_id in (0, 31):
        args = engine.sampler.prepare_batch_sampling_args(batch)
        assert_output(
            engine.forward_batch(batch, args, pad_id=pad_id),
            expected_tokens,
            device=execution_device,
        )
        padded, logits = model.recorded[-1]
        expected_padded = [
            list(prefix) + [pad_id] * (3 - len(prefix)) for prefix in prefixes
        ]
        assert padded.tolist() == expected_padded
        selected_logits = logits[
            torch.arange(len(prefixes), device=execution_device),
            torch.tensor(
                [len(prefix) - 1 for prefix in prefixes], device=execution_device
            ),
            :,
        ]
        torch.testing.assert_close(selected_logits, reference, rtol=1e-5, atol=1e-6)
        assert request_snapshot(batch) == before
    assert len(model.recorded) == 2, "Each batch step must share one forward"


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(RuntimeError("Injected execution failure"), id="runtime_error"),
        pytest.param(
            KeyboardInterrupt("Injected interruption"), id="keyboard_interrupt"
        ),
        pytest.param(SystemExit("Injected shutdown"), id="system_exit"),
    ],
)
def test_engine_propagates_shared_forward_failures_without_updating_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
    execution_device,
) -> None:
    class FailingLM(nn.Module):
        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            assert not torch.is_grad_enabled()
            raise failure

    engine, _, _, _ = make_engine(
        monkeypatch, tmp_path, FailingLM(), device=execution_device
    )
    batch = Batch(reqs=[Req(token_ids=[11, 24, 7]), Req(token_ids=[4])])
    batch.reqs[0].start()
    before = request_snapshot(batch)
    args = engine.sampler.prepare_batch_sampling_args(batch)
    with pytest.raises(type(failure)) as caught:
        engine.forward_batch(batch, args)
    assert caught.value is failure
    assert request_snapshot(batch) == before
