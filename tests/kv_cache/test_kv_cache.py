"""Layer-cache contract, including the optional attention.forward keyword.

attention(x, token_positions=None, *, kv_cache=None) returns only the output
tensor. A supplied LayerKVCache is updated and retained by the caller; omitted
or explicit None means uncached execution. RoPE cases supply absolute positions.
"""

import pytest
import torch

from cs336_basics.transformers.transformers import LayerKVCache, MultiHeadSelfAttention


def _make_attention(*, use_rope: bool = False, device="cpu") -> MultiHeadSelfAttention:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        return MultiHeadSelfAttention(
            d_model=4,
            num_heads=2,
            max_seq_len=16,
            theta=10000.0 if use_rope else None,
        ).to(device)


def _hidden_states(device) -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.2, -0.6, 1.3, 0.5],
                [1.1, 0.4, -0.8, 0.3],
                [-0.7, 1.5, 0.2, -1.0],
                [0.9, -0.2, 0.6, 1.4],
                [-1.2, 0.8, -0.4, 0.7],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )


def _full_prefix_reference(
    attention: MultiHeadSelfAttention,
    inputs: torch.Tensor,
    *,
    use_rope: bool,
) -> torch.Tensor:
    reference = _make_attention(use_rope=use_rope, device=inputs.device).eval()
    reference.load_state_dict(attention.state_dict())
    positions = torch.arange(
        inputs.shape[1], dtype=torch.int64, device=inputs.device
    ).unsqueeze(0)
    return reference(inputs, token_positions=positions)


@torch.inference_mode()
def test_layer_cache_preserves_kv_through_prefill_and_two_decode_appends(
    execution_device,
):
    expected_k = torch.arange(40, dtype=torch.float32, device=execution_device).reshape(
        2, 1, 5, 4
    )
    expected_v = expected_k.square() + 3
    cache = LayerKVCache()

    start = 0
    for stop in (3, 4, 5):
        cache.update_kv(
            expected_k[..., start:stop, :].clone(),
            expected_v[..., start:stop, :].clone(),
        )
        assert cache.k_cache().shape == (2, 1, stop, 4)
        assert cache.k_cache().device == cache.v_cache().device == execution_device
        assert cache.v_cache().shape == (2, 1, stop, 4)
        torch.testing.assert_close(cache.k_cache(), expected_k[..., :stop, :])
        torch.testing.assert_close(cache.v_cache(), expected_v[..., :stop, :])
        start = stop


@torch.inference_mode()
def test_layer_caches_keep_independent_sequence_histories(execution_device):
    a_k = torch.arange(16, dtype=torch.float32, device=execution_device).reshape(
        1, 1, 4, 4
    )
    a_v = -a_k - 1
    b_k = (
        torch.arange(8, dtype=torch.float32, device=execution_device).reshape(
            1, 1, 2, 4
        )
        + 100
    )
    b_v = b_k + 200
    a, b = LayerKVCache(), LayerKVCache()

    a.update_kv(a_k[..., :3, :].clone(), a_v[..., :3, :].clone())
    b.update_kv(b_k.clone(), b_v.clone())
    a.update_kv(a_k[..., 3:, :].clone(), a_v[..., 3:, :].clone())

    torch.testing.assert_close(a.k_cache(), a_k)
    torch.testing.assert_close(a.v_cache(), a_v)
    torch.testing.assert_close(b.k_cache(), b_k)
    torch.testing.assert_close(b.v_cache(), b_v)


@pytest.mark.parametrize("use_rope", [False, True], ids=["without_rope", "with_rope"])
@torch.inference_mode()
def test_default_attention_calls_do_not_reuse_another_sequence(
    use_rope, execution_device
):
    attention = _make_attention(use_rope=use_rope, device=execution_device).eval()
    reference = _make_attention(use_rope=use_rope, device=execution_device).eval()
    reference.load_state_dict(attention.state_dict())
    first = (
        torch.arange(12, dtype=torch.float32, device=execution_device).reshape(1, 3, 4)
        / 7
    )
    second = (
        torch.arange(8, dtype=torch.float32, device=execution_device).reshape(1, 2, 4)
        / 10
        - 2
    )

    expected = reference(second)
    attention(first)
    actual = attention(second)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_default_attention_supports_independent_training_iterations(execution_device):
    attention = _make_attention(device=execution_device).train()
    reference = _make_attention(device=execution_device).train()
    reference.load_state_dict(attention.state_dict())
    first = (
        torch.arange(12, dtype=torch.float32, device=execution_device).reshape(1, 3, 4)
        / 7
    ).requires_grad_()
    second = (
        torch.arange(8, dtype=torch.float32, device=execution_device).reshape(1, 2, 4)
        / 10
        - 2
    ).requires_grad_()
    reference_input = second.detach().clone().requires_grad_()

    expected = reference(reference_input)
    attention(first).square().mean().backward()
    assert first.grad is not None
    attention.zero_grad(set_to_none=True)

    actual = attention(second)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()

    assert second.grad is not None
    assert reference_input.grad is not None
    torch.testing.assert_close(second.grad, reference_input.grad, rtol=1e-5, atol=1e-6)
    for parameter, reference_parameter in zip(
        attention.parameters(), reference.parameters(), strict=True
    ):
        assert parameter.grad is not None
        assert reference_parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("use_rope", [False, True], ids=["without_rope", "with_rope"])
@torch.inference_mode()
def test_caller_owned_cache_matches_full_prefix_prefill_and_decode(
    use_rope, execution_device
):
    attention = _make_attention(use_rope=use_rope, device=execution_device).eval()
    cache = LayerKVCache()
    inputs = _hidden_states(execution_device)
    projection_lengths = []
    hook = attention.w_qkv.register_forward_pre_hook(
        lambda _module, args: projection_lengths.append(args[0].shape[-2])
    )

    try:
        start = 0
        for stop in (3, 4, 5):
            positions = torch.arange(
                start, stop, dtype=torch.int64, device=execution_device
            ).unsqueeze(0)
            actual = attention(
                inputs[:, start:stop], token_positions=positions, kv_cache=cache
            )
            expected = _full_prefix_reference(
                attention, inputs[:, :stop], use_rope=use_rope
            )[:, start:stop]

            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            assert cache.k_cache().shape == (2, 1, stop, 2)
            assert cache.k_cache().device == cache.v_cache().device == execution_device
            assert cache.v_cache().shape == (2, 1, stop, 2)
            start = stop
    finally:
        hook.remove()

    assert projection_lengths == [3, 1, 1]


@torch.inference_mode()
def test_two_caller_owned_caches_can_interleave_on_one_attention_layer(
    execution_device,
):
    attention = _make_attention(use_rope=True, device=execution_device).eval()
    a = _hidden_states(execution_device)
    b = a.flip(-1) * 0.7 + 0.2
    a_cache, b_cache = LayerKVCache(), LayerKVCache()

    for inputs, cache, start, stop in (
        (a, a_cache, 0, 3),
        (b, b_cache, 0, 2),
        (a, a_cache, 3, 4),
        (b, b_cache, 2, 3),
    ):
        positions = torch.arange(
            start, stop, dtype=torch.int64, device=execution_device
        ).unsqueeze(0)
        actual = attention(
            inputs[:, start:stop], token_positions=positions, kv_cache=cache
        )
        expected = _full_prefix_reference(attention, inputs[:, :stop], use_rope=True)[
            :, start:stop
        ]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    assert a_cache.k_cache().shape == a_cache.v_cache().shape == (2, 1, 4, 2)
    assert a_cache.k_cache().device == a_cache.v_cache().device == execution_device
    assert b_cache.k_cache().shape == b_cache.v_cache().shape == (2, 1, 3, 2)
    assert b_cache.k_cache().device == b_cache.v_cache().device == execution_device


@torch.inference_mode()
def test_uncached_calls_do_not_consume_or_change_an_existing_caller_cache(
    execution_device,
):
    attention = _make_attention(use_rope=True, device=execution_device).eval()
    cache = LayerKVCache()
    inputs = _hidden_states(execution_device)
    attention(
        inputs[:, :3],
        token_positions=torch.tensor([[0, 1, 2]], device=execution_device),
        kv_cache=cache,
    )
    stored_k = cache.k_cache().clone()
    stored_v = cache.v_cache().clone()
    unrelated = inputs[:, :2].flip(-1) + 0.3
    expected = _full_prefix_reference(attention, unrelated, use_rope=True)

    for kwargs in ({}, {"kv_cache": None}):
        actual = attention(unrelated, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(cache.k_cache(), stored_k)
        torch.testing.assert_close(cache.v_cache(), stored_v)

    actual = attention(
        inputs[:, 3:4],
        token_positions=torch.tensor([[3]], device=execution_device),
        kv_cache=cache,
    )
    expected = _full_prefix_reference(attention, inputs[:, :4], use_rope=True)[:, 3:4]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert cache.k_cache().shape == cache.v_cache().shape == (2, 1, 4, 2)
    assert cache.k_cache().device == cache.v_cache().device == execution_device
