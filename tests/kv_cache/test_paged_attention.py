"""Packed reference attention through the public paged context API.

PagedBatchMetadata receives device int64 q_offsets, kv_lengths, block_tables and
write_slots, plus matching q_offsets_cpu/kv_lengths_cpu tuples. Construct
PagedForwardContext(kv_store=..., metadata=...), obtain for_layer(layer_idx=...),
then attend(q=..., k=..., v=...) with already-rotated Q/K and projected V [H,T,D].
Outputs retain packed query order and shape. Each layer writes packed K/V once;
each query sees only its own request's causal history, including cached tokens.
All examples use valid metadata. No model, scheduler or specific attention kernel
is required; the oracle explicitly lists each query's visible K/V prefix.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def api():
    targets = (
        "cs336_basics.transformers.paged_kv",
        "cs336_basics.transformers.paged_attention",
    )
    modules = []
    for target in targets:
        try:
            modules.append(importlib.import_module(target))
        except ModuleNotFoundError as exc:
            if exc.name not in targets:
                raise
            pytest.fail(f"Implement the paged attention task: missing {exc.name}", pytrace=False)
    storage, attention = modules
    for name in ("PagedBatchMetadata", "PagedForwardContext"):
        if not hasattr(attention, name):
            pytest.fail(f"Implement {targets[1]}.{name} for this task", pytrace=False)
    return SimpleNamespace(
        Store=storage.PagedKVStore,
        Metadata=attention.PagedBatchMetadata,
        Context=attention.PagedForwardContext,
    )


@pytest.fixture
def store(api, execution_device):
    result = api.Store(
        num_layers=2,
        num_blocks=8,
        block_size=2,
        num_heads=3,
        head_dim=4,
        device=execution_device,
        dtype=torch.float32,
    )
    result.k_pages.fill_(float("nan"))
    result.v_pages.fill_(float("nan"))
    return result


def _indices(values, device):
    return torch.tensor(values, device=device, dtype=torch.int64)


def _context(api, store, device, *, offsets, lengths, tables, slots):
    metadata = api.Metadata(
        q_offsets=_indices(offsets, device),
        kv_lengths=_indices(lengths, device),
        block_tables=_indices(tables, device),
        write_slots=_indices(slots, device),
        q_offsets_cpu=tuple(offsets),
        kv_lengths_cpu=tuple(lengths),
    )
    return api.Context(kv_store=store, metadata=metadata)


def _qkv(tokens, device, *, offset=0):
    values = torch.arange(3 * tokens * 4, device=device, dtype=torch.float32)
    values = values.reshape(3, tokens, 4) + offset
    return torch.sin(values / 7), torch.cos(values / 5), values / 13 - 1


def _reference(q, visible_k, visible_v):
    scores = (q @ visible_k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    return torch.softmax(scores, dim=-1) @ visible_v


def _check_attention(layer, q, k, v, expected):
    before = [values.clone() for values in (q, k, v)]
    actual = layer.attend(q=q, k=k, v=v)
    assert actual.shape == q.shape
    assert torch.isfinite(actual).all(), "Unused NaN pages must never enter attention"
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for values, original in zip((q, k, v), before, strict=True):
        torch.testing.assert_close(values, original, rtol=0, atol=0)


def test_one_packed_call_mixes_cached_decode_and_causal_fresh_prefill(api, store, execution_device):
    device = execution_device
    # A owns blocks [5,1,4]: four old tokens, then one decode input at slot 8.
    _, old_k, old_v = _qkv(4, device, offset=30)
    store.write(layer_idx=1, slots=_indices([10, 11, 2, 3], device), k=old_k, v=old_v)
    # The unselected layer is independent despite using the same physical IDs.
    store.k_pages[0].fill_(-25)
    store.v_pages[0].fill_(-75)
    other_layer = (store.k_pages[0].clone(), store.v_pages[0].clone())
    context = _context(
        api,
        store,
        device,
        offsets=(0, 1, 4),
        lengths=(5, 3),
        tables=((5, 1, 4), (3, 7, -1)),
        slots=(8, 6, 7, 14),
    )
    q, k, v = _qkv(4, device, offset=7)
    a_k, a_v = torch.cat((old_k, k[:, :1]), dim=1), torch.cat((old_v, v[:, :1]), dim=1)
    expected = torch.cat(
        (
            _reference(q[:, :1], a_k, a_v),
            _reference(q[:, 1:2], k[:, 1:2], v[:, 1:2]),
            _reference(q[:, 2:3], k[:, 1:3], v[:, 1:3]),
            _reference(q[:, 3:4], k[:, 1:4], v[:, 1:4]),
        ),
        dim=1,
    )

    _check_attention(context.for_layer(layer_idx=1), q, k, v, expected)
    torch.testing.assert_close(store.k_pages[0], other_layer[0], rtol=0, atol=0)
    torch.testing.assert_close(store.v_pages[0], other_layer[1], rtol=0, atol=0)


def test_later_prefill_chunk_uses_absolute_causal_offsets_across_pages(api, store, execution_device):
    device = execution_device
    first = _context(
        api,
        store,
        device,
        offsets=(0, 2),
        lengths=(2,),
        tables=((6, -1),),
        slots=(12, 13),
    )
    q0, k0, v0 = _qkv(2, device, offset=3)
    expected0 = torch.cat(
        (
            _reference(q0[:, :1], k0[:, :1], v0[:, :1]),
            _reference(q0[:, 1:2], k0, v0),
        ),
        dim=1,
    )
    _check_attention(first.for_layer(layer_idx=0), q0, k0, v0, expected0)

    second = _context(
        api,
        store,
        device,
        offsets=(0, 3),
        lengths=(5,),
        tables=((6, 2, 0),),
        slots=(4, 5, 0),
    )
    q1, k1, v1 = _qkv(3, device, offset=50)
    full_k, full_v = torch.cat((k0, k1), dim=1), torch.cat((v0, v1), dim=1)
    # Queries at absolute positions 2,3,4 see 3,4,5 keys, respectively.
    expected1 = torch.cat(
        (
            _reference(q1[:, :1], full_k[:, :3], full_v[:, :3]),
            _reference(q1[:, 1:2], full_k[:, :4], full_v[:, :4]),
            _reference(q1[:, 2:3], full_k, full_v),
        ),
        dim=1,
    )
    _check_attention(second.for_layer(layer_idx=0), q1, k1, v1, expected1)


def test_layer_contexts_keep_separate_histories_and_packed_request_order(api, store, execution_device):
    device = execution_device
    histories = []
    for layer_idx, offset in ((0, 10), (1, 100)):
        _, old_k, old_v = _qkv(3, device, offset=offset)
        # A has one cached token in block 4; B has two in block 1.
        store.write(layer_idx=layer_idx, slots=_indices([8, 2, 3], device), k=old_k, v=old_v)
        histories.append((old_k, old_v))
    # Packed order is B then A; both layers use these same request addresses.
    context = _context(
        api,
        store,
        device,
        offsets=(0, 1, 2),
        lengths=(3, 2),
        tables=((1, 6), (4, -1)),
        slots=(12, 9),
    )
    layers = [context.for_layer(layer_idx=0), context.for_layer(layer_idx=1)]
    for layer_idx, offset in ((0, 20), (1, 200)):
        q, k, v = _qkv(2, device, offset=offset)
        old_k, old_v = histories[layer_idx]
        b_k, b_v = torch.cat((old_k[:, 1:], k[:, :1]), dim=1), torch.cat((old_v[:, 1:], v[:, :1]), dim=1)
        a_k, a_v = torch.cat((old_k[:, :1], k[:, 1:]), dim=1), torch.cat((old_v[:, :1], v[:, 1:]), dim=1)
        expected = torch.cat(
            (
                _reference(q[:, :1], b_k, b_v),
                _reference(q[:, 1:2], a_k, a_v),
            ),
            dim=1,
        )
        _check_attention(layers[layer_idx], q, k, v, expected)
