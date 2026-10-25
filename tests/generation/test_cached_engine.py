"""Singleton cached Engine execution, with unchanged uncached batching.

The caller retains one LayerKVCache per model layer and passes it using
``engine.forward_batch(batch=batch, batch_sampling_args=args, kv_caches=caches)``.
``cache.valid_kv_lengths`` is None when empty, then an integer tensor [1] on the
KV device containing this request's valid prefix length. Cached execution consumes
only the uncomputed suffix; Engine supplies its absolute positions without
requiring them from the caller. Engine never changes request tokens or lifecycle.
"""

import importlib
from pathlib import Path

import pytest
import torch

from cs336_basics.config import EngineConfig, ModelConfig, RuntimeConfig
from cs336_basics.generation.requests import Batch, Req
from cs336_basics.generation.results import ModelOutput
from cs336_basics.transformers.transformers import LayerKVCache, TransformerLM

_MODEL_ARGS = dict(
    vocab_size=32,
    context_length=16,
    d_model=8,
    num_layers=3,
    num_heads=2,
    d_ff=16,
    rope_theta=10_000.0,
)


class RecordingCachedLM(TransformerLM):
    def __init__(self) -> None:
        super().__init__(**_MODEL_ARGS)
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        self.expected_cached_input: tuple[list[int], list[int]] | None = None
        self.projection_lengths: list[list[int]] = [[] for _ in self.layers]
        for layer, history in zip(self.layers, self.projection_lengths, strict=True):
            layer.attn.w_qkv.register_forward_pre_hook(
                lambda _module, args, history=history: history.append(args[0].shape[-2])
            )

    def forward(self, token_ids, *, token_positions=None, kv_caches=None):
        assert not self.training, "Engine must retain evaluation mode"
        assert not torch.is_grad_enabled(), "Engine must disable autograd itself"
        device = self.token_embeddings.weight.device
        assert token_ids.dtype == torch.int64
        assert token_ids.device == device
        if self.expected_cached_input is not None:
            ids, positions = self.expected_cached_input
            assert kv_caches is not None, "Engine must forward the supplied KV caches"
            assert token_ids.shape == (1, len(ids)), (
                f"Cached input must contain only {len(ids)} uncomputed token(s), "
                f"with shape [1, {len(ids)}]; got {tuple(token_ids.shape)}"
            )
            assert token_ids.tolist() == [ids], (
                f"Cached model input must be the uncomputed suffix {[ids]}, "
                f"not the complete request; got {token_ids.tolist()}"
            )
            assert (
                token_positions is not None
            ), f"Engine must supply absolute token_positions={[positions]} for this cached suffix"
            assert token_positions.shape == token_ids.shape, (
                "token_positions must match the suffix shape [1, Q], "
                f"got {tuple(token_positions.shape)} for {tuple(token_ids.shape)}"
            )
            assert token_positions.tolist() == [positions], (
                f"Expected absolute token_positions={[positions]}; "
                f"got {token_positions.tolist()}. Each input token needs its "
                "own position: cached prefix length + its offset in the suffix."
            )
        if token_positions is not None:
            assert token_positions.dtype in (torch.int32, torch.int64)
            assert token_positions.device == device
        self.calls.append(
            (
                token_ids.detach().clone(),
                None if token_positions is None else token_positions.detach().clone(),
            )
        )
        return super().forward(
            token_ids, token_positions=token_positions, kv_caches=kv_caches
        )


@pytest.fixture
def cached_engine_setup(monkeypatch, tmp_path: Path, execution_device):
    engine_module = importlib.import_module("cs336_basics.generation.engine")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(51336)
        model = RecordingCachedLM()
        reference = TransformerLM(**_MODEL_ARGS)
    reference.load_state_dict(model.state_dict())
    reference = reference.to(execution_device).eval()
    config = EngineConfig(
        model=ModelConfig(**_MODEL_ARGS),
        checkpoint=tmp_path / "unused.pt",
        runtime=RuntimeConfig(device=execution_device.type),
    )
    monkeypatch.setattr(engine_module, "build_model", lambda _config: model)
    monkeypatch.setattr(engine_module, "load_model_weights", lambda **_kwargs: None)
    engine = engine_module.Engine(config)
    sampled_logits: list[torch.Tensor] = []
    sampler_type = type(engine.sampler)
    original_sample = sampler_type.sample

    def record_sample(sampler, logits, args):
        assert sampler is engine.sampler
        assert logits.device == execution_device
        assert not logits.requires_grad
        sampled_logits.append(logits.detach().clone())
        return original_sample(sampler, logits, args)

    monkeypatch.setattr(sampler_type, "sample", record_sample)
    assert engine.model is model
    assert all(p.device == execution_device for p in model.parameters())
    assert all(b.device == execution_device for b in model.buffers())
    return engine, model, reference, sampled_logits


def _new_caches() -> list[LayerKVCache]:
    return [LayerKVCache() for _ in range(_MODEL_ARGS["num_layers"])]


def _request_snapshot(batch: Batch) -> tuple:
    return tuple(
        (
            id(req),
            req._id,
            tuple(req.token_ids),
            req.prompt_tokens_count,
            req.status,
            req.finish_reason,
            req.sampling_params,
        )
        for req in batch.reqs
    )


def _cache_snapshot(caches) -> list:
    return [
        (
            None if cache.valid_kv_lengths is None else cache.valid_kv_lengths.clone(),
            None if cache.valid_kv_lengths is None else cache.k_cache().clone(),
            None if cache.valid_kv_lengths is None else cache.v_cache().clone(),
        )
        for cache in caches
    ]


def _assert_cache_unchanged(caches, snapshot) -> None:
    for cache, (lengths, keys, values) in zip(caches, snapshot, strict=True):
        if lengths is None:
            assert cache.valid_kv_lengths is None
        else:
            assert cache.valid_kv_lengths is not None
            torch.testing.assert_close(cache.valid_kv_lengths, lengths, rtol=0, atol=0)
            torch.testing.assert_close(cache.k_cache(), keys, rtol=0, atol=0)
            torch.testing.assert_close(cache.v_cache(), values, rtol=0, atol=0)


def _assert_cache_length(caches, length: int, device) -> None:
    for cache in caches:
        lengths = cache.valid_kv_lengths
        assert isinstance(lengths, torch.Tensor)
        assert lengths.shape == (1,)
        assert lengths.dtype in (torch.int32, torch.int64)
        assert lengths.device == device
        assert lengths.tolist() == [length]
        assert cache.k_cache().shape == cache.v_cache().shape == (2, 1, length, 4)
        assert cache.k_cache().device == cache.v_cache().device == device
        assert cache.k_cache().dtype == cache.v_cache().dtype == torch.float32


def _reference_last_logits(reference, requests, device) -> torch.Tensor:
    with torch.inference_mode():
        return torch.stack(
            [
                reference(
                    torch.tensor([req.token_ids], dtype=torch.int64, device=device)
                )[0, -1]
                for req in requests
            ]
        )


def _step(setup, requests, device, *, caches=None, expected_input=None):
    engine, model, reference, sampled_logits = setup
    batch = Batch(reqs=requests)
    before = _request_snapshot(batch)
    expected = _reference_last_logits(reference, requests, device)
    args = engine.sampler.prepare_batch_sampling_args(batch)
    call_count, sample_count = len(model.calls), len(sampled_logits)
    cache_kwargs = {} if caches is None else {"kv_caches": caches}
    if caches is not None:
        assert expected_input is not None, "The test must state its cached input"
    model.expected_cached_input = expected_input
    try:
        output = engine.forward_batch(
            batch=batch, batch_sampling_args=args, **cache_kwargs
        )
    finally:
        model.expected_cached_input = None
    assert len(model.calls) == call_count + 1
    assert len(sampled_logits) == sample_count + 1
    torch.testing.assert_close(sampled_logits[-1], expected, rtol=1e-5, atol=1e-6)
    assert isinstance(output, ModelOutput)
    assert output.next_tokens_gpu.shape == (len(requests),)
    assert output.next_tokens_gpu.dtype == torch.int64
    assert output.next_tokens_gpu.device == device
    assert output.next_tokens_gpu.tolist() == expected.argmax(dim=-1).tolist()
    assert _request_snapshot(batch) == before
    return output


def _assert_last_input(model, ids, positions, device) -> None:
    observed_ids, observed_positions = model.calls[-1]
    assert observed_ids.tolist() == [ids], "Engine must forward only uncomputed IDs"
    assert observed_ids.device == device
    assert observed_positions is not None, "Cached execution needs absolute positions"
    assert observed_positions.tolist() == [
        positions
    ], "Positions must continue after the cached prefix"
    assert observed_positions.device == device


def test_cached_engine_prefill_supplies_absolute_positions(
    cached_engine_setup, execution_device
):
    _, model, _, _ = cached_engine_setup
    req = Req(token_ids=[11, 24, 7])
    caches = _new_caches()
    assert all(cache.valid_kv_lengths is None for cache in caches)

    _step(
        cached_engine_setup,
        [req],
        execution_device,
        caches=caches,
        expected_input=([11, 24, 7], [0, 1, 2]),
    )

    _assert_last_input(model, [11, 24, 7], [0, 1, 2], execution_device)
    _assert_cache_length(caches, 3, execution_device)
    assert model.projection_lengths == [[3]] * 3


def test_cached_engine_decode_uses_suffix_with_a_primed_cache(
    cached_engine_setup, execution_device
):
    _, model, _, sampled_logits = cached_engine_setup
    req = Req(token_ids=[11, 24, 7])
    caches = _new_caches()

    # Prepare real KV independently of Engine prefill, so this test isolates decode.
    with torch.inference_mode():
        model(
            torch.tensor([[11, 24, 7]], dtype=torch.int64, device=execution_device),
            token_positions=torch.tensor(
                [[0, 1, 2]], dtype=torch.int64, device=execution_device
            ),
            kv_caches=caches,
        )
    _assert_cache_length(caches, 3, execution_device)
    model.calls.clear()
    for history in model.projection_lengths:
        history.clear()
    assert sampled_logits == [], "Priming the model must not invoke the sampler"

    req.append_token(9)
    _step(
        cached_engine_setup,
        [req],
        execution_device,
        caches=caches,
        expected_input=([9], [3]),
    )

    _assert_last_input(model, [9], [3], execution_device)
    _assert_cache_length(caches, 4, execution_device)
    assert model.projection_lengths == [[1]] * 3


def test_cached_engine_prefill_and_decode_match_full_prefix_without_reprojection(
    cached_engine_setup, execution_device
):
    _, model, _, _ = cached_engine_setup
    req = Req(token_ids=[11, 24, 7])
    req.start()
    caches = _new_caches()
    assert all(cache.valid_kv_lengths is None for cache in caches)

    for ids, positions, length in (
        ([11, 24, 7], [0, 1, 2], 3),
        ([9], [3], 4),
        ([5], [4], 5),
    ):
        if length > 3:
            req.append_token(ids[0])
        _step(
            cached_engine_setup,
            [req],
            execution_device,
            caches=caches,
            expected_input=(ids, positions),
        )
        _assert_last_input(model, ids, positions, execution_device)
        _assert_cache_length(caches, length, execution_device)

    assert model.projection_lengths == [[3, 1, 1]] * 3


def test_cached_engine_interleaves_histories_and_uncached_batches_without_request_mutation(
    cached_engine_setup, execution_device
):
    engine, model, _, _ = cached_engine_setup
    sampler = engine.sampler
    a, b = Req(token_ids=[11, 24, 7]), Req(token_ids=[8, 6])
    a_caches, b_caches = _new_caches(), _new_caches()
    _step(
        cached_engine_setup,
        [a],
        execution_device,
        caches=a_caches,
        expected_input=([11, 24, 7], [0, 1, 2]),
    )
    a_before = _cache_snapshot(a_caches)
    _step(
        cached_engine_setup,
        [b],
        execution_device,
        caches=b_caches,
        expected_input=([8, 6], [0, 1]),
    )
    _assert_cache_unchanged(a_caches, a_before)
    b_before = _cache_snapshot(b_caches)

    _step(
        cached_engine_setup,
        [Req(token_ids=[4]), Req(token_ids=[5, 6, 7, 8])],
        execution_device,
    )
    _assert_cache_unchanged(a_caches, a_before)
    _assert_cache_unchanged(b_caches, b_before)

    a.append_token(9)
    _step(
        cached_engine_setup,
        [a],
        execution_device,
        caches=a_caches,
        expected_input=([9], [3]),
    )
    _assert_last_input(model, [9], [3], execution_device)
    _assert_cache_length(a_caches, 4, execution_device)
    _assert_cache_unchanged(b_caches, b_before)
    assert engine.model is model
    assert engine.sampler is sampler


def test_cached_engine_rejects_multiple_requests_before_forward_or_cache_updates(
    cached_engine_setup, execution_device
):
    engine, model, _, sampled_logits = cached_engine_setup
    batch = Batch(reqs=[Req(token_ids=[11, 24, 7]), Req(token_ids=[8])])
    caches = _new_caches()
    before, cache_before = _request_snapshot(batch), _cache_snapshot(caches)
    args = engine.sampler.prepare_batch_sampling_args(batch)

    with pytest.raises(ValueError):
        engine.forward_batch(batch=batch, batch_sampling_args=args, kv_caches=caches)

    assert model.calls == []
    assert sampled_logits == []
    assert _request_snapshot(batch) == before
    _assert_cache_unchanged(caches, cache_before)


@pytest.mark.parametrize("invalid", ["wrong_count", "inconsistent_layer_lengths"])
def test_cached_engine_rejects_invalid_layer_caches_before_mutation(
    cached_engine_setup, execution_device, invalid
):
    engine, model, _, sampled_logits = cached_engine_setup
    batch = Batch(reqs=[Req(token_ids=[11, 24, 7])])
    caches = _new_caches()
    if invalid == "wrong_count":
        caches.pop()
    else:
        keys = torch.ones((2, 1, 1, 4), device=execution_device)
        caches[1].update_kv(keys, keys.clone())
    before, cache_before = _request_snapshot(batch), _cache_snapshot(caches)
    args = engine.sampler.prepare_batch_sampling_args(batch)

    with pytest.raises(ValueError):
        engine.forward_batch(batch=batch, batch_sampling_args=args, kv_caches=caches)

    assert model.calls == []
    assert sampled_logits == []
    assert _request_snapshot(batch) == before
    _assert_cache_unchanged(caches, cache_before)


def test_cached_engine_rejects_a_fully_computed_request_without_another_forward(
    cached_engine_setup, execution_device
):
    engine, model, _, sampled_logits = cached_engine_setup
    req = Req(token_ids=[11, 24, 7])
    caches = _new_caches()
    _step(
        cached_engine_setup,
        [req],
        execution_device,
        caches=caches,
        expected_input=([11, 24, 7], [0, 1, 2]),
    )
    batch = Batch(reqs=[req])
    before, cache_before = _request_snapshot(batch), _cache_snapshot(caches)
    args = engine.sampler.prepare_batch_sampling_args(batch)

    with pytest.raises(ValueError):
        engine.forward_batch(batch=batch, batch_sampling_args=args, kv_caches=caches)

    assert len(model.calls) == len(sampled_logits) == 1
    assert _request_snapshot(batch) == before
    _assert_cache_unchanged(caches, cache_before)
