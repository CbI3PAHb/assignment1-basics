"""PagedKVStore storage contract; every numerical case runs on CPU and CUDA.

PagedKVStore(*, num_layers, num_blocks, block_size, num_heads, head_dim, device,
             dtype) exposes separate stable k_pages/v_pages tensors [L,N,P,H,D].
write(layer_idx=..., slots=int64[T], k=..., v=...) stores [H,T,D] at the supplied
physical token slots and returns None. Slots are valid, unique and on the device.
gather(layer_idx=..., block_table=int64[M], kv_len=int) returns K/V [H,kv_len,D]
in logical history order. Unused table entries and page tails are not history.
Initial pool contents are arbitrary. Address examples below are explicit, with
P=3, H=2 and D=4; these tests do not supply a general addressing implementation.
"""

import importlib

import pytest
import torch


@pytest.fixture
def store_factory(execution_device):
    target = "cs336_basics.transformers.paged_kv"
    try:
        module = importlib.import_module(target)
    except ModuleNotFoundError as exc:
        if exc.name != target:
            raise
        pytest.fail(f"Implement PagedKVStore in {target} for this task", pytrace=False)
    store_type = getattr(module, "PagedKVStore", None)
    if store_type is None:
        pytest.fail(f"Implement {target}.PagedKVStore for this task", pytrace=False)

    def make(*, dtype=torch.float32):
        return store_type(
            num_layers=2,
            num_blocks=5,
            block_size=3,
            num_heads=2,
            head_dim=4,
            device=execution_device,
            dtype=dtype,
        )

    return make


def _kv(num_tokens, device, *, offset=0):
    values = torch.arange(2 * num_tokens * 4, dtype=torch.float32, device=device)
    values = values.reshape(2, num_tokens, 4)
    return values + offset, 1000 - 2 * values - offset


def _indices(values, device):
    return torch.tensor(values, dtype=torch.int64, device=device)


def _assert_kv(actual, expected):
    actual_k, actual_v = actual
    expected_k, expected_v = expected
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)


def test_pool_allocation_has_requested_axes_dtype_device_and_distinct_kv(store_factory, execution_device):
    store = store_factory(dtype=torch.float64)
    for pages in (store.k_pages, store.v_pages):
        assert isinstance(pages, torch.Tensor)
        assert pages.shape == (2, 5, 3, 2, 4)
        assert pages.dtype == torch.float64
        assert pages.device == execution_device
    assert store.k_pages.data_ptr() != store.v_pages.data_ptr()
    store.k_pages.fill_(7)
    store.v_pages.fill_(-11)
    torch.testing.assert_close(store.k_pages, torch.full_like(store.k_pages, 7), rtol=0, atol=0)
    torch.testing.assert_close(store.v_pages, torch.full_like(store.v_pages, -11), rtol=0, atol=0)


def test_unsorted_write_spans_pages_and_preserves_head_channel_and_layer_axes(store_factory, execution_device):
    store = store_factory()
    store.k_pages.fill_(-100)
    store.v_pages.fill_(-200)
    k, v = _kv(5, execution_device)
    original_k, original_v = k.clone(), v.clone()
    slots = _indices([10, 0, 8, 4, 12], execution_device)
    assert store.write(layer_idx=1, slots=slots, k=k, v=v) is None
    _assert_kv((k, v), (original_k, original_v))

    expected_k = torch.full_like(store.k_pages, -100)
    expected_v = torch.full_like(store.v_pages, -200)
    # With P=3 these five slots name the following physical (block, offset) pairs.
    for token, (block, offset) in enumerate(((3, 1), (0, 0), (2, 2), (1, 1), (4, 0))):
        expected_k[1, block, offset] = original_k[:, token, :]
        expected_v[1, block, offset] = original_v[:, token, :]
    _assert_kv((store.k_pages, store.v_pages), (expected_k, expected_v))


def test_append_keeps_pool_buffers_and_all_unwritten_history_unchanged(store_factory, execution_device):
    store = store_factory()
    store.k_pages.fill_(-100)
    store.v_pages.fill_(-200)
    original_k_ptr, original_v_ptr = store.k_pages.data_ptr(), store.v_pages.data_ptr()
    k, v = _kv(2, execution_device)
    store.write(layer_idx=0, slots=_indices([9, 10], execution_device), k=k, v=v)
    expected_k, expected_v = store.k_pages.clone(), store.v_pages.clone()
    new_k, new_v = _kv(2, execution_device, offset=100)
    store.write(layer_idx=0, slots=_indices([11, 3], execution_device), k=new_k, v=new_v)

    # Append finishes block 3 and starts block 1; every other location stays intact.
    expected_k[0, 3, 2], expected_v[0, 3, 2] = new_k[:, 0, :], new_v[:, 0, :]
    expected_k[0, 1, 0], expected_v[0, 1, 0] = new_k[:, 1, :], new_v[:, 1, :]
    _assert_kv((store.k_pages, store.v_pages), (expected_k, expected_v))
    assert store.k_pages.data_ptr() == original_k_ptr, "Appending must retain the K pool"
    assert store.v_pages.data_ptr() == original_v_ptr, "Appending must retain the V pool"


def test_gather_follows_table_order_and_excludes_partial_tail_and_unused_entries(store_factory, execution_device):
    store = store_factory()
    store.k_pages.fill_(float("nan"))
    store.v_pages.fill_(float("nan"))
    k, v = _kv(5, execution_device, offset=20)
    # Seed pages directly, so inverse mistakes in write and gather cannot cancel.
    for token, (block, offset) in enumerate(((3, 0), (3, 1), (3, 2), (0, 0), (0, 1))):
        store.k_pages[1, block, offset] = k[:, token, :]
        store.v_pages[1, block, offset] = v[:, token, :]
    table = _indices([3, 0, -1], execution_device)
    gathered = store.gather(layer_idx=1, block_table=table, kv_len=5)
    _assert_kv(gathered, (k, v))
    assert all(torch.isfinite(values).all() for values in gathered)


def test_reusing_a_page_overwrites_only_new_history_and_preserves_another_owner(store_factory, execution_device):
    store = store_factory()
    store.k_pages.fill_(-100)
    store.v_pages.fill_(-200)
    old_k, old_v = _kv(3, execution_device, offset=100)
    other_k, other_v = _kv(3, execution_device, offset=500)
    store.write(layer_idx=0, slots=_indices([12, 13, 14], execution_device), k=old_k, v=old_v)
    store.write(layer_idx=0, slots=_indices([3, 4, 5], execution_device), k=other_k, v=other_v)
    expected_k, expected_v = store.k_pages.clone(), store.v_pages.clone()

    # The CPU owner may recycle block 4 without clearing it; new history has length 1.
    new_k, new_v = _kv(1, execution_device, offset=2000)
    store.write(layer_idx=0, slots=_indices([12], execution_device), k=new_k, v=new_v)
    expected_k[0, 4, 0], expected_v[0, 4, 0] = new_k[:, 0, :], new_v[:, 0, :]
    _assert_kv((store.k_pages, store.v_pages), (expected_k, expected_v))
    _assert_kv(
        store.gather(layer_idx=0, block_table=_indices([4, -1], execution_device), kv_len=1),
        (new_k, new_v),
    )
    _assert_kv(
        store.gather(layer_idx=0, block_table=_indices([1, -1], execution_device), kv_len=3),
        (other_k, other_v),
    )
