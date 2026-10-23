"""Caller-owned layer caches propagated through a block and the full model.

Block calls accept ``token_positions=None, *, kv_cache=None``; model calls accept
``token_positions=None, *, kv_caches=None``. A model cache list has one distinct
LayerKVCache per layer. Both APIs return only the output tensor. These CPU/CUDA cases
pass absolute positions explicitly and leave Engine and request handling aside.
"""

import pytest
import torch

from cs336_basics.transformers.transformers import (
    LayerKVCache,
    TransformerBlock,
    TransformerLM,
)


def _make_block(device) -> TransformerBlock:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        return TransformerBlock(
            d_model=8,
            num_heads=2,
            d_ff=16,
            max_seq_len=16,
            theta=10000.0,
        ).to(device)


def _make_model(device) -> TransformerLM:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(5678)
        return TransformerLM(
            vocab_size=32,
            context_length=16,
            d_model=8,
            num_layers=3,
            num_heads=2,
            d_ff=16,
            rope_theta=10000.0,
        ).to(device)


def _positions(start: int, stop: int, device) -> torch.Tensor:
    return torch.arange(start, stop, dtype=torch.int64, device=device).unsqueeze(0)


def _new_model_caches() -> list[LayerKVCache]:
    return [LayerKVCache() for _ in range(3)]


def _reference_logits(model: TransformerLM, token_ids: torch.Tensor) -> torch.Tensor:
    reference = _make_model(token_ids.device).eval()
    reference.load_state_dict(model.state_dict())
    return reference(token_ids)


def _assert_cache_length(cache: LayerKVCache, length: int, device) -> None:
    assert cache.k_cache().device == cache.v_cache().device == device
    assert cache.k_cache().shape == (2, 1, length, 4)
    assert cache.v_cache().shape == (2, 1, length, 4)


@torch.inference_mode()
def test_transformer_block_cached_prefill_and_decode_match_full_prefix(
    execution_device,
):
    block = _make_block(execution_device).eval()
    inputs = (
        torch.arange(40, dtype=torch.float32, device=execution_device).reshape(1, 5, 8)
        / 13
        - 1
    )
    cache = LayerKVCache()

    start = 0
    for stop in (3, 4, 5):
        reference = _make_block(execution_device).eval()
        reference.load_state_dict(block.state_dict())
        expected = reference(inputs[:, :stop])[:, start:stop]
        actual = block(
            inputs[:, start:stop],
            token_positions=_positions(start, stop, execution_device),
            kv_cache=cache,
        )

        assert isinstance(actual, torch.Tensor)
        assert actual.shape == (1, stop - start, 8)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        _assert_cache_length(cache, stop, execution_device)
        start = stop


@torch.inference_mode()
def test_three_layer_cached_logits_match_full_prefix_without_reprojecting_history(
    execution_device,
):
    model = _make_model(execution_device).eval()
    token_ids = torch.tensor(
        [[11, 24, 7, 9, 5]], dtype=torch.int64, device=execution_device
    )
    caches = _new_model_caches()
    projection_lengths = [[] for _ in model.layers]
    hooks = [
        layer.attn.w_qkv.register_forward_pre_hook(
            lambda _module, args, history=history: history.append(args[0].shape[-2])
        )
        for layer, history in zip(model.layers, projection_lengths, strict=True)
    ]

    try:
        start = 0
        for stop in (3, 4, 5):
            expected = _reference_logits(model, token_ids[:, :stop])[:, start:stop]
            actual = model(
                token_ids[:, start:stop],
                token_positions=_positions(start, stop, execution_device),
                kv_caches=caches,
            )

            assert isinstance(actual, torch.Tensor)
            assert actual.shape == (1, stop - start, 32)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            for cache in caches:
                _assert_cache_length(cache, stop, execution_device)
            start = stop
    finally:
        for hook in hooks:
            hook.remove()

    assert projection_lengths == [[3, 1, 1], [3, 1, 1], [3, 1, 1]]


@torch.inference_mode()
def test_one_model_interleaves_two_independent_request_cache_lists(execution_device):
    model = _make_model(execution_device).eval()
    a = torch.tensor([[11, 24, 7, 9, 5]], dtype=torch.int64, device=execution_device)
    b = torch.tensor([[9, 7, 8, 6]], dtype=torch.int64, device=execution_device)
    a_caches, b_caches = _new_model_caches(), _new_model_caches()

    for token_ids, caches, start, stop in (
        (a, a_caches, 0, 3),
        (b, b_caches, 0, 2),
        (a, a_caches, 3, 4),
        (b, b_caches, 2, 3),
    ):
        expected = _reference_logits(model, token_ids[:, :stop])[:, start:stop]
        actual = model(
            token_ids[:, start:stop],
            token_positions=_positions(start, stop, execution_device),
            kv_caches=caches,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    for cache in a_caches:
        _assert_cache_length(cache, 4, execution_device)
    for cache in b_caches:
        _assert_cache_length(cache, 3, execution_device)


def test_uncached_training_iterations_leave_caller_caches_unchanged(execution_device):
    # Create normal parameters before entering inference mode for the cache call.
    model = _make_model(execution_device).train()
    cached_ids = torch.tensor(
        [[11, 24, 7, 9]], dtype=torch.int64, device=execution_device
    )
    caches = _new_model_caches()

    with torch.inference_mode():
        model(
            cached_ids[:, :3],
            token_positions=_positions(0, 3, execution_device),
            kv_caches=caches,
        )
        snapshots = [
            (cache.k_cache().clone(), cache.v_cache().clone()) for cache in caches
        ]

    for token_ids, kwargs in (
        (torch.tensor([[12, 4, 7, 6]], dtype=torch.int64, device=execution_device), {}),
        (
            torch.tensor([[2, 15]], dtype=torch.int64, device=execution_device),
            {"kv_caches": None},
        ),
    ):
        reference = _make_model(execution_device).train()
        reference.load_state_dict(model.state_dict())
        model.zero_grad(set_to_none=True)
        expected = reference(token_ids)
        actual = model(token_ids, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        actual.square().mean().backward()
        expected.square().mean().backward()
        for parameter, reference_parameter in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            assert parameter.grad is not None
            assert reference_parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, reference_parameter.grad, rtol=1e-5, atol=1e-6
            )

        for cache, (stored_k, stored_v) in zip(caches, snapshots, strict=True):
            torch.testing.assert_close(cache.k_cache(), stored_k)
            torch.testing.assert_close(cache.v_cache(), stored_v)

    with torch.inference_mode():
        expected = _reference_logits(model, cached_ids)[:, 3:4]
        actual = model(
            cached_ids[:, 3:4],
            token_positions=_positions(3, 4, execution_device),
            kv_caches=caches,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for cache in caches:
            _assert_cache_length(cache, 4, execution_device)


@pytest.mark.parametrize("cache_count", [2, 4], ids=["too_few", "too_many"])
@torch.inference_mode()
def test_wrong_layer_cache_count_is_rejected_before_any_cache_update(
    cache_count, execution_device
):
    model = _make_model(execution_device).eval()
    token_ids = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=execution_device)
    caches = [LayerKVCache() for _ in range(cache_count)]

    with pytest.raises(ValueError, match=r"(?i)(cache|layer)"):
        model(
            token_ids,
            token_positions=_positions(0, 3, execution_device),
            kv_caches=caches,
        )

    for cache in caches:
        with pytest.raises(RuntimeError):
            cache.k_cache()
        with pytest.raises(RuntimeError):
            cache.v_cache()
