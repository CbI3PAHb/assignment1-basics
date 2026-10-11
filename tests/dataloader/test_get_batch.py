import warnings

import numpy as np
import pytest
import torch

from cs336_basics.dataloader.dataloader import get_batch


@pytest.mark.parametrize("dtype", ["uint16", "uint32", "int64"])
def test_read_only_memmap_batch(tmp_path, dtype):
    path = tmp_path / "tokens.bin"
    np.arange(40, dtype=dtype).tofile(path)
    original_bytes = path.read_bytes()
    dataset = np.memmap(path, dtype=dtype, mode="r")

    previous_warn_always = torch.is_warn_always_enabled()
    try:
        torch.set_warn_always(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            inputs, targets = get_batch(dataset, batch_size=8, context_length=7, device="cpu")
    finally:
        torch.set_warn_always(previous_warn_always)

    assert not any("not writable" in str(warning.message) for warning in caught)
    assert inputs.shape == targets.shape == (8, 7)
    assert inputs.dtype == targets.dtype == torch.int64
    assert inputs.device.type == targets.device.type == "cpu"
    torch.testing.assert_close(targets, inputs + 1)
    assert inputs.min() >= 0
    assert targets.max() < len(dataset)

    inputs.fill_(-1)
    targets.fill_(-2)
    np.testing.assert_array_equal(dataset, np.arange(40, dtype=dtype))
    assert path.read_bytes() == original_bytes


def test_read_only_array_batch_owns_its_storage():
    dataset = np.arange(9, dtype=np.int64)
    dataset.flags.writeable = False

    inputs, targets = get_batch(dataset, batch_size=2, context_length=8, device="cpu")

    torch.testing.assert_close(inputs, torch.arange(8).repeat(2, 1))
    torch.testing.assert_close(targets, torch.arange(1, 9).repeat(2, 1))
    assert inputs.is_contiguous()
    assert targets.is_contiguous()
    original_targets = targets.clone()
    inputs.fill_(-1)
    torch.testing.assert_close(targets, original_targets)
    targets.fill_(-2)
    np.testing.assert_array_equal(dataset, np.arange(9, dtype=np.int64))
    assert not dataset.flags.writeable


def test_uint32_token_ids_are_not_truncated(tmp_path):
    tokens = np.array([0, 2**31 - 1, 2**31, 2**32 - 2, 2**32 - 1], dtype=np.uint32)
    path = tmp_path / "tokens.bin"
    tokens.tofile(path)
    dataset = np.memmap(path, dtype=np.uint32, mode="r")

    inputs, targets = get_batch(dataset, batch_size=3, context_length=4, device="cpu")

    torch.testing.assert_close(inputs, torch.tensor(tokens[:-1].astype(np.int64)).repeat(3, 1))
    torch.testing.assert_close(targets, torch.tensor(tokens[1:].astype(np.int64)).repeat(3, 1))


def test_batch_preserves_random_start_indices():
    dataset = np.arange(31, dtype=np.uint16)
    context_length = 6
    batch_size = 16

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
        expected_next_draw = torch.rand(4)

        torch.manual_seed(42)
        inputs, targets = get_batch(dataset, batch_size, context_length, device="cpu")
        actual_next_draw = torch.rand(4)

    expected_inputs = starts[:, None] + torch.arange(context_length)
    torch.testing.assert_close(inputs, expected_inputs)
    torch.testing.assert_close(targets, expected_inputs + 1)
    torch.testing.assert_close(actual_next_draw, expected_next_draw)
