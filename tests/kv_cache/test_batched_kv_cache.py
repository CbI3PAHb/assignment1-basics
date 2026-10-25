"""Task 2A: per-row LayerKVCache storage; no attention or Engine changes.

LayerKVCache().update_kv(k, v, *, input_lengths=None) receives [H, B, Q, D].
input_lengths is an integer tensor [B] counting NEW valid tokens per row (default
Q); append each input prefix at that row's old valid end. valid_kv_lengths is
None before initialization, then an integer tensor [B] on the K/V device.
Invalid values/cache compatibility raise ValueError before mutation. Argument
shape mismatches may instead raise jaxtyping.TypeCheckError before method entry.
Row membership stays fixed. Physical capacity and padding values are unspecified.
These tests should fail until this storage contract is implemented.
"""

import pytest
import torch
from jaxtyping import TypeCheckError

from cs336_basics.transformers.transformers import LayerKVCache


def _kv(width, device, *, offset=0, heads=2, batch=2, dim=2):
    keys = (
        torch.arange(
            heads * batch * width * dim, dtype=torch.float64, device=device
        ).reshape(heads, batch, width, dim)
        + offset
    )
    return keys, 1000 - 3 * keys


def _assert_history(cache, expected_keys, expected_values):
    lengths = [row.shape[-2] for row in expected_keys]
    actual_lengths = cache.valid_kv_lengths
    assert isinstance(actual_lengths, torch.Tensor)
    assert actual_lengths.dtype in (torch.int32, torch.int64)
    assert actual_lengths.shape == (len(lengths),)
    assert actual_lengths.device == expected_keys[0].device
    assert actual_lengths.tolist() == lengths
    keys, values = cache.k_cache(), cache.v_cache()
    assert keys.shape == values.shape
    assert keys.ndim == 4
    assert keys.shape[:2] == (expected_keys[0].shape[0], len(lengths))
    assert keys.shape[2] >= max(lengths)
    assert keys.shape[3] == expected_keys[0].shape[-1]
    assert keys.dtype == values.dtype == expected_keys[0].dtype
    assert keys.device == values.device == expected_keys[0].device
    for row, length in enumerate(lengths):
        torch.testing.assert_close(
            keys[:, row, :length], expected_keys[row], rtol=0, atol=0
        )
        torch.testing.assert_close(
            values[:, row, :length], expected_values[row], rtol=0, atol=0
        )


def _snapshot(cache):
    return (
        cache.valid_kv_lengths.clone(),
        cache.k_cache().clone(),
        cache.v_cache().clone(),
    )


def _assert_unchanged(cache, snapshot):
    lengths, keys, values = snapshot
    torch.testing.assert_close(cache.valid_kv_lengths, lengths, rtol=0, atol=0)
    # Existing padding must remain unchanged, but its initial values are unspecified.
    torch.testing.assert_close(cache.k_cache(), keys, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(cache.v_cache(), values, rtol=0, atol=0, equal_nan=True)


def test_uniform_batches_infer_input_lengths_when_omitted(execution_device):
    cache = LayerKVCache()
    assert cache.valid_kv_lengths is None
    first_k, first_v = _kv(3, execution_device)
    next_k, next_v = _kv(1, execution_device, offset=100)
    expected_k = torch.cat((first_k, next_k), dim=2)
    expected_v = torch.cat((first_v, next_v), dim=2)

    cache.update_kv(first_k, first_v)
    _assert_history(cache, list(first_k.unbind(1)), list(first_v.unbind(1)))
    cache.update_kv(next_k, next_v)
    _assert_history(cache, list(expected_k.unbind(1)), list(expected_v.unbind(1)))


def test_ragged_prefill_and_two_decodes_append_at_each_rows_valid_end(execution_device):
    cache = LayerKVCache()
    keys, values = _kv(3, execution_device)
    lengths = torch.tensor([3, 1], dtype=torch.int64, device=execution_device)
    expected_k = [keys[:, row, :length].clone() for row, length in enumerate([3, 1])]
    expected_v = [values[:, row, :length].clone() for row, length in enumerate([3, 1])]
    cache.update_kv(keys, values, input_lengths=lengths)
    assert lengths.tolist() == [
        3,
        1,
    ], "Input counts must not become cumulative cache lengths"
    _assert_history(cache, expected_k, expected_v)

    for offset in (100, 200):
        next_k, next_v = _kv(1, execution_device, offset=offset)
        for row in range(2):
            expected_k[row] = torch.cat((expected_k[row], next_k[:, row]), dim=1)
            expected_v[row] = torch.cat((expected_v[row], next_v[:, row]), dim=1)
        counts = torch.tensor([1, 1], dtype=torch.int64, device=execution_device)
        cache.update_kv(next_k, next_v, input_lengths=counts)
        assert counts.tolist() == [1, 1]
        _assert_history(cache, expected_k, expected_v)
    assert lengths.tolist() == [
        3,
        1,
    ], "Later updates must not mutate earlier input counts"
    assert cache.valid_kv_lengths.tolist() == [5, 3]


def test_unequal_chunks_ignore_input_padding_and_can_restore_uniform_lengths(
    execution_device,
):
    cache = LayerKVCache()
    keys, values = _kv(3, execution_device)
    next_k, next_v = _kv(3, execution_device, offset=100)
    expected_k = [
        torch.cat((keys[:, 0], next_k[:, 0, :1]), dim=1),
        torch.cat((keys[:, 1, :1], next_k[:, 1]), dim=1),
    ]
    expected_v = [
        torch.cat((values[:, 0], next_v[:, 0, :1]), dim=1),
        torch.cat((values[:, 1, :1], next_v[:, 1]), dim=1),
    ]
    cache.update_kv(
        keys,
        values,
        input_lengths=torch.tensor([3, 1], dtype=torch.int64, device=execution_device),
    )
    input_before = next_k.clone(), next_v.clone()
    counts = torch.tensor([1, 3], dtype=torch.int64, device=execution_device)
    cache.update_kv(next_k, next_v, input_lengths=counts)
    assert counts.tolist() == [1, 3]
    torch.testing.assert_close(next_k, input_before[0], rtol=0, atol=0)
    torch.testing.assert_close(next_v, input_before[1], rtol=0, atol=0)
    _assert_history(cache, expected_k, expected_v)
    assert cache.valid_kv_lengths.tolist() == [4, 4]

    # A zero count leaves that row's valid history unchanged.
    extra_k, extra_v = _kv(1, execution_device, offset=200)
    expected_k[1] = torch.cat((expected_k[1], extra_k[:, 1]), dim=1)
    expected_v[1] = torch.cat((expected_v[1], extra_v[:, 1]), dim=1)
    cache.update_kv(
        extra_k,
        extra_v,
        input_lengths=torch.tensor([0, 1], dtype=torch.int64, device=execution_device),
    )
    _assert_history(cache, expected_k, expected_v)
    assert cache.valid_kv_lengths.tolist() == [4, 5]


def test_extra_padding_preserves_valid_history_on_the_next_update(execution_device):
    cache = LayerKVCache()
    keys, values = _kv(4, execution_device)
    expected_k = [keys[:, 0, :1].clone(), keys[:, 1, :2].clone()]
    expected_v = [values[:, 0, :1].clone(), values[:, 1, :2].clone()]
    counts = torch.tensor([1, 2], dtype=torch.int64, device=execution_device)

    # Neither row fills the input width; the extra columns are only padding.
    cache.update_kv(keys, values, input_lengths=counts)
    _assert_history(cache, expected_k, expected_v)

    next_k, next_v = _kv(1, execution_device, offset=100)
    for row in range(2):
        expected_k[row] = torch.cat((expected_k[row], next_k[:, row]), dim=1)
        expected_v[row] = torch.cat((expected_v[row], next_v[:, row]), dim=1)
    cache.update_kv(
        next_k,
        next_v,
        input_lengths=torch.tensor([1, 1], dtype=torch.int64, device=execution_device),
    )
    # Only valid history matters; keeping or trimming unused capacity is allowed.
    _assert_history(cache, expected_k, expected_v)
    assert cache.valid_kv_lengths.tolist() == [2, 3]


@pytest.mark.parametrize(
    "counts", [[1], [-1, 1], [3, 1]], ids=["wrong_count", "negative", "too_long"]
)
def test_invalid_input_counts_leave_existing_storage_unchanged(
    execution_device, counts
):
    cache = LayerKVCache()
    cache.update_kv(*_kv(2, execution_device))
    before = _snapshot(cache)
    keys, values = _kv(2, execution_device, offset=100)
    supplied_counts = torch.tensor(counts, dtype=torch.int64, device=execution_device)

    # A checked [batch_size] annotation can reject wrong cardinality at entry.
    expected_error = (
        (ValueError, TypeCheckError) if len(counts) != keys.shape[1] else ValueError
    )
    with pytest.raises(expected_error):
        cache.update_kv(keys, values, input_lengths=supplied_counts)

    assert supplied_counts.tolist() == counts
    _assert_unchanged(cache, before)


@pytest.mark.parametrize("invalid", ["kv_time", "heads", "batch", "head_dim"])
def test_invalid_shapes_leave_existing_storage_unchanged(execution_device, invalid):
    cache = LayerKVCache()
    cache.update_kv(*_kv(2, execution_device))
    before = _snapshot(cache)
    shape = {"heads": 2, "batch": 2, "dim": 2}
    if invalid == "heads":
        shape["heads"] = 1
    elif invalid == "batch":
        shape["batch"] = 3
    elif invalid == "head_dim":
        shape["dim"] = 3
    keys, values = _kv(1, execution_device, offset=100, **shape)
    if invalid == "kv_time":
        values = values[:, :, :0]

    # Shared K/V axis names check incoming arguments, not existing cache state.
    expected_error = (
        (ValueError, TypeCheckError) if invalid == "kv_time" else ValueError
    )
    with pytest.raises(expected_error):
        cache.update_kv(keys, values)

    _assert_unchanged(cache, before)
