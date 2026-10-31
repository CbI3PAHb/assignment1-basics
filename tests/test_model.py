import numpy
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from cs336_basics.transformers.transformers import MultiHeadSelfAttention, TransformerLM

from .adapters import (
    run_embedding,
    run_linear,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_rope,
    run_scaled_dot_product_attention,
    run_silu,
    run_swiglu,
    run_transformer_block,
    run_transformer_lm,
)


def _state_dict_on_device(state_and_config, device):
    state_dict, config = state_and_config
    return {name: value.to(device) for name, value in state_dict.items()}, config


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff, execution_device):
    in_embeddings = in_embeddings.to(execution_device)
    ts_state_dict = _state_dict_on_device(ts_state_dict, execution_device)
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    assert output.device == execution_device
    numpy_snapshot.assert_match(output)


def test_embedding(
    numpy_snapshot,
    ts_state_dict,
    in_indices,
    vocab_size,
    d_model,
    execution_device,
):
    in_indices = in_indices.to(execution_device)
    ts_state_dict = _state_dict_on_device(ts_state_dict, execution_device)
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=embedding_weight,
        token_ids=in_indices,
    )
    assert output.device == execution_device
    numpy_snapshot.assert_match(output)


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff, execution_device):
    in_embeddings = in_embeddings.to(execution_device)
    ts_state_dict = _state_dict_on_device(ts_state_dict, execution_device)
    w1_weight, w2_weight, w3_weight = [ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_embeddings,
    )
    assert actual_output.device == execution_device
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask, execution_device):
    q = q.to(execution_device)
    k = k.to(execution_device)
    v = v.to(execution_device)
    mask = mask.to(execution_device)
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    assert actual_output.device == execution_device
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask, execution_device):
    q = q.to(execution_device)
    k = k.to(execution_device)
    v = v.to(execution_device)
    mask = mask.to(execution_device)
    # Shape: (batch_size, num_heads, seq_len, qk_head_dim)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    assert actual_output.device == execution_device
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_multihead_self_attention(
    numpy_snapshot,
    in_embeddings,
    d_model,
    n_heads,
    ts_state_dict,
    execution_device,
):
    in_embeddings = in_embeddings.to(execution_device)
    ts_state_dict = _state_dict_on_device(ts_state_dict, execution_device)
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    assert actual_output.device == execution_device
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_multihead_self_attention_with_rope(
    numpy_snapshot,
    in_embeddings,
    d_model,
    n_heads,
    ts_state_dict,
    n_keys,
    theta,
    pos_ids,
    execution_device,
):
    in_embeddings = in_embeddings.to(execution_device)
    pos_ids = pos_ids.to(execution_device)
    ts_state_dict = _state_dict_on_device(ts_state_dict, execution_device)
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    actual_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )
    assert actual_output.device == execution_device
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_transformer_lm(
    numpy_snapshot,
    vocab_size,
    n_keys,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    ts_state_dict,
    in_indices,
    execution_device,
):
    in_indices = in_indices.to(execution_device)
    ts_state_dict = _state_dict_on_device(ts_state_dict, execution_device)
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )
    assert actual_output.device == execution_device
    numpy_snapshot.assert_match(actual_output, atol=1e-4, rtol=1e-2)


def test_transformer_lm_truncated_input(
    numpy_snapshot,
    vocab_size,
    n_keys,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    ts_state_dict,
    in_indices,
    execution_device,
):
    in_indices = in_indices.to(execution_device)
    ts_state_dict = _state_dict_on_device(ts_state_dict, execution_device)
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=ts_state_dict[0],
        in_indices=in_indices_truncated,
    )

    assert truncated_actual_output.device == execution_device
    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )


def test_transformer_block(
    numpy_snapshot,
    ts_state_dict,
    in_embeddings,
    d_model,
    n_heads,
    d_ff,
    n_keys,
    theta,
    execution_device,
):
    in_embeddings = in_embeddings.to(execution_device)
    ts_state_dict = _state_dict_on_device(ts_state_dict, execution_device)
    block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )
    assert actual_output.device == execution_device
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings, execution_device):
    in_embeddings = in_embeddings.to(execution_device)
    ts_state_dict = _state_dict_on_device(ts_state_dict, execution_device)
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]

    actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_embeddings)

    assert actual_output.device == execution_device
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_rope(
    numpy_snapshot,
    in_embeddings,
    d_model,
    theta,
    n_queries,
    pos_ids,
    execution_device,
):
    in_embeddings = in_embeddings.to(execution_device)
    pos_ids = pos_ids.to(execution_device)
    output = run_rope(
        d_model,
        theta=theta,
        max_seq_len=n_queries,
        in_query_or_key=in_embeddings,
        token_positions=pos_ids,
    )
    assert output.device == execution_device
    numpy_snapshot.assert_match(output, atol=1e-6)


def test_silu_matches_pytorch(execution_device):
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ],
        device=execution_device,
    )
    expected_output = F.silu(x)
    actual_output = run_silu(x)
    assert actual_output.device == execution_device
    numpy.testing.assert_allclose(
        actual_output.detach().cpu().numpy(),
        expected_output.detach().cpu().numpy(),
        atol=1e-6,
    )


def _make_uncached_attention(*, use_rope: bool = False, device="cpu") -> MultiHeadSelfAttention:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        return MultiHeadSelfAttention(
            d_model=4,
            num_heads=2,
            max_seq_len=16,
            theta=10000.0 if use_rope else None,
        ).to(device)


@pytest.mark.parametrize("use_rope", [False, True], ids=["without_rope", "with_rope"])
@torch.inference_mode()
def test_default_attention_calls_do_not_reuse_another_sequence(use_rope, execution_device):
    attention = _make_uncached_attention(use_rope=use_rope, device=execution_device).eval()
    reference = _make_uncached_attention(use_rope=use_rope, device=execution_device).eval()
    reference.load_state_dict(attention.state_dict())
    first = torch.arange(12, dtype=torch.float32, device=execution_device).reshape(1, 3, 4) / 7
    second = torch.arange(8, dtype=torch.float32, device=execution_device).reshape(1, 2, 4) / 10 - 2

    expected = reference(second)
    attention(first)
    actual = attention(second)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_default_attention_supports_independent_training_iterations(execution_device):
    attention = _make_uncached_attention(device=execution_device).train()
    reference = _make_uncached_attention(device=execution_device).train()
    reference.load_state_dict(attention.state_dict())
    first = (torch.arange(12, dtype=torch.float32, device=execution_device).reshape(1, 3, 4) / 7).requires_grad_()
    second = (torch.arange(8, dtype=torch.float32, device=execution_device).reshape(1, 2, 4) / 10 - 2).requires_grad_()
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
    for parameter, reference_parameter in zip(attention.parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None
        assert reference_parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-5, atol=1e-6)


def _make_uncached_model(device) -> TransformerLM:
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


@torch.inference_mode()
def test_uncached_model_right_padding_preserves_valid_logits(execution_device):
    """Right-padding filler tokens cannot affect any earlier valid position."""
    model = _make_uncached_model(execution_device).eval()
    prefixes = ((11, 24, 7), (4,), (8, 6))
    reference = torch.stack(
        [model(torch.tensor([prefix], dtype=torch.int64, device=execution_device))[0, -1] for prefix in prefixes]
    )
    last_positions = torch.tensor([len(prefix) - 1 for prefix in prefixes], device=execution_device)
    for pad_id in (0, 31):
        inputs = torch.tensor(
            [list(prefix) + [pad_id] * (3 - len(prefix)) for prefix in prefixes],
            dtype=torch.int64,
            device=execution_device,
        )
        logits = model(inputs)
        selected = logits[torch.arange(len(prefixes), device=execution_device), last_positions]
        torch.testing.assert_close(selected, reference, rtol=1e-5, atol=1e-6)


def test_default_model_supports_independent_training_iterations(execution_device):
    model = _make_uncached_model(execution_device).train()
    with torch.inference_mode():
        model(torch.tensor([[11, 24, 7]], dtype=torch.int64, device=execution_device))

    for token_ids in (
        torch.tensor([[12, 4, 7, 6]], dtype=torch.int64, device=execution_device),
        torch.tensor([[2, 15]], dtype=torch.int64, device=execution_device),
    ):
        reference = _make_uncached_model(execution_device).train()
        reference.load_state_dict(model.state_dict())
        model.zero_grad(set_to_none=True)
        expected = reference(token_ids)
        actual = model(token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        actual.square().mean().backward()
        expected.square().mean().backward()
        for parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
            assert parameter.grad is not None
            assert reference_parameter.grad is not None
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-5, atol=1e-6)
