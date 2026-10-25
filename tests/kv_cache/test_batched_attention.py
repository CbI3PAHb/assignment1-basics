"""Batched cached model execution: ragged, mixed, and chunked inputs.

TransformerLM, TransformerBlock, and MultiHeadSelfAttention accept optional
input_lengths: integer Tensor[B]. Each count is NEW valid tokens in a right-padded
row; None preserves the existing all-Q-valid behavior. The caller supplies
absolute token_positions[B,Q], using safe zero positions for padded queries.
Valid logits must match independent uncached histories; padded logits are unspecified.
One batched projection per layer is required. Unused storage is poisoned before
a subsequent cache update; valid outputs must remain unaffected. Cleanup inside
LayerKVCache.update_kv satisfies this check; its placement is not prescribed.
"""

import torch

from cs336_basics.transformers.transformers import LayerKVCache, TransformerLM

_MODEL_ARGS = dict(
    vocab_size=32,
    context_length=32,
    d_model=12,
    num_layers=2,
    num_heads=3,  # Deliberately different from batch size 2.
    d_ff=24,
    rope_theta=10_000.0,
)


def _make_models(device):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(25336)
        model = TransformerLM(**_MODEL_ARGS)
        reference = TransformerLM(**_MODEL_ARGS)
    reference.load_state_dict(model.state_dict())
    return model.to(device).eval(), reference.to(device).eval()


def _poison_unused_storage(caches):
    """Unused capacity is not valid history and must not influence attention."""
    for cache in caches:
        lengths = cache.valid_kv_lengths
        if lengths is None:
            continue
        for row, length in enumerate(lengths.tolist()):
            cache.k_cache()[:, row, length:, :].fill_(float("nan"))
            cache.v_cache()[:, row, length:, :].fill_(float("nan"))


@torch.inference_mode()
def _assert_trace(device, steps, *, poison_padding=False):
    model, reference = _make_models(device)
    caches = [LayerKVCache() for _ in model.layers]
    histories = [[], []]
    projected_shapes = [[] for _ in model.layers]
    hooks = [
        layer.attn.w_qkv.register_forward_pre_hook(
            lambda _module, args, history=history: history.append(tuple(args[0].shape))
        )
        for layer, history in zip(model.layers, projected_shapes, strict=True)
    ]
    expected_projection_shapes = []

    try:
        for new_rows in steps:
            if poison_padding:
                _poison_unused_storage(caches)
            old_lengths = [len(history) for history in histories]
            counts = [len(row) for row in new_rows]
            width = max(counts)
            token_ids = torch.tensor(
                [list(row) + [0] * (width - len(row)) for row in new_rows],
                dtype=torch.int64,
                device=device,
            )
            positions = torch.tensor(
                [
                    list(range(old, old + count)) + [0] * (width - count)
                    for old, count in zip(old_lengths, counts, strict=True)
                ],
                dtype=torch.int64,
                device=device,
            )
            input_lengths = torch.tensor(counts, dtype=torch.int64, device=device)
            input_before = token_ids.clone(), positions.clone(), input_lengths.clone()

            actual = model(
                token_ids,
                token_positions=positions,
                kv_caches=caches,
                input_lengths=input_lengths,
            )

            assert actual.shape == (2, width, _MODEL_ARGS["vocab_size"])
            assert actual.device == device
            expected_projection_shapes.append((2, width, _MODEL_ARGS["d_model"]))
            for shapes in projected_shapes:
                assert shapes == expected_projection_shapes, (
                    "Each layer must project one [B,Q,D] batch per forward, "
                    "without recomputing history or executing separate rows"
                )
            for tensor, before in zip(
                (token_ids, positions, input_lengths), input_before, strict=True
            ):
                torch.testing.assert_close(tensor, before, rtol=0, atol=0)

            for row, new_tokens in enumerate(new_rows):
                histories[row].extend(new_tokens)
                count = counts[row]
                if count == 0:
                    continue  # No valid queries; padded outputs have no contract.
                reference_ids = torch.tensor(
                    [histories[row]], dtype=torch.int64, device=device
                )
                expected = reference(reference_ids)[0, old_lengths[row] :]
                valid_logits = actual[row, :count]
                assert torch.isfinite(
                    valid_logits
                ).all(), "Invalid KV slots must not contaminate valid query logits"
                torch.testing.assert_close(
                    valid_logits,
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                    msg=f"Row {row}: compare every new valid query with full history",
                )

            totals = [len(history) for history in histories]
            for cache in caches:
                lengths = cache.valid_kv_lengths
                assert isinstance(lengths, torch.Tensor)
                assert lengths.shape == (2,)
                assert lengths.dtype in (torch.int32, torch.int64)
                assert lengths.device == device
                assert lengths.tolist() == totals
                keys, values = cache.k_cache(), cache.v_cache()
                assert keys.shape == values.shape
                assert keys.shape[:2] == (3, 2)
                assert keys.shape[2] >= max(totals)
                assert keys.shape[3] == 4
                assert keys.device == values.device == device
                assert keys.dtype == values.dtype == actual.dtype
    finally:
        for hook in hooks:
            hook.remove()


def test_ragged_prefill_and_decode_ignore_invalid_kv_storage(execution_device):
    _assert_trace(
        execution_device,
        steps=(
            ([11, 24, 7], [4]),  # Valid histories [3,1], physical input width 3.
            ([9], [8]),
            ([5], [10]),
        ),
        poison_padding=True,
    )


def test_one_batch_mixes_decode_with_a_fresh_prefill(execution_device):
    _assert_trace(
        execution_device,
        steps=(
            ([11, 24, 7], []),  # Prime A; B owns zero valid cached tokens.
            ([9], [4, 8, 6]),  # A decodes at position 3; B prefills positions 0..2.
            ([5], [10]),  # Both now decode with different absolute positions.
        ),
    )


def test_chunked_prefill_matches_every_valid_full_prefix_logit(execution_device):
    _assert_trace(
        execution_device,
        steps=(
            ([11, 24], [4]),
            ([7, 9], [8, 6, 10]),  # Multiple queries with nonzero cached prefixes.
            ([5], [12]),
        ),
    )
