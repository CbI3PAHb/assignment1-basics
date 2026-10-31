"""One packed model call must match independent, uncached request histories.

TransformerLM accepts token IDs/absolute positions [T] and paged_context=.
PagedForwardContext(kv_store=..., metadata=...) carries per-request CPU bounds
and matching device int64 metadata; it supplies per-layer attention contexts.
Existing dense calls remain the numerical reference. These tests use fixed page
ownership directly; scheduling, allocation policy and sampling are out of scope.
"""

import importlib

import pytest
import torch

from cs336_basics.transformers.paged_kv import PagedKVStore
from cs336_basics.transformers.transformers import RoPE, TransformerLM

_MODEL_ARGS = dict(vocab_size=32, d_model=12, num_layers=2, num_heads=3, d_ff=24, rope_theta=10_000.0)
# P=2; each request keeps its own disjoint, nonconsecutive physical pages.
_TABLES = {"A": (6, 1, 8, 4), "B": (3, 9, 0, 7)}


@pytest.fixture
def paged_api():
    target = "cs336_basics.transformers.paged_attention"
    try:
        module = importlib.import_module(target)
    except ModuleNotFoundError as exc:
        if exc.name != target:
            raise
        pytest.fail(f"Implement {target} for this task", pytrace=False)
    names = ("PagedBatchMetadata", "PagedForwardContext")
    for name in names:
        if not hasattr(module, name):
            pytest.fail(f"Implement {target}.{name} for this task", pytrace=False)
    return tuple(getattr(module, name) for name in names)


def _make_models(device, *, context_length=8):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(30336)
        model = TransformerLM(context_length=context_length, **_MODEL_ARGS)
        reference = TransformerLM(context_length=context_length, **_MODEL_ARGS)
    reference.load_state_dict(model.state_dict())
    return model.to(device).eval(), reference.to(device).eval()


def _make_store(device):
    store = PagedKVStore(
        num_layers=2,
        num_blocks=10,
        block_size=2,
        num_heads=3,
        head_dim=4,
        device=device,
        dtype=torch.float32,
    )
    store.k_pages.fill_(float("nan"))
    store.v_pages.fill_(float("nan"))
    return store


def _pack(paged_api, store, device, rows):
    """Rows are CPU (start_position, new_tokens, owned_block_table) tuples."""
    metadata_type, context_type = paged_api
    ids, positions, slots, tables, lengths, offsets = [], [], [], [], [], [0]
    for start, tokens, table in rows:
        end = start + len(tokens)
        ids.extend(tokens)
        positions.extend(range(start, end))
        offsets.append(offsets[-1] + len(tokens))
        lengths.append(end)
        tables.append(table[: (end + 1) // 2])
        slots.extend(table[pos // 2] * 2 + pos % 2 for pos in range(start, end))
    width = max(map(len, tables))
    padded_tables = [list(table) + [-1] * (width - len(table)) for table in tables]

    def tensor(values):
        return torch.tensor(values, dtype=torch.int64, device=device)

    metadata = metadata_type(
        q_offsets=tensor(offsets),
        kv_lengths=tensor(lengths),
        block_tables=tensor(padded_tables),
        write_slots=tensor(slots),
        q_offsets_cpu=tuple(offsets),
        kv_lengths_cpu=tuple(lengths),
    )
    return tensor(ids), tensor(positions), context_type(kv_store=store, metadata=metadata)


@torch.inference_mode()
def _assert_trace(paged_api, device, steps, *, context_length=8):
    model, reference = _make_models(device, context_length=context_length)
    store = _make_store(device)
    pool_pointers = store.k_pages.data_ptr(), store.v_pages.data_ptr()
    histories = {"A": [], "B": []}
    projections = [[] for _ in model.layers]
    hooks = [
        layer.attn.w_qkv.register_forward_pre_hook(lambda _module, args, seen=seen: seen.append(tuple(args[0].shape)))
        for layer, seen in zip(model.layers, projections, strict=True)
    ]
    expected_shapes = []
    try:
        for entries in steps:
            rows = [(len(histories[name]), tokens, _TABLES[name]) for name, tokens in entries]
            ids, positions, context = _pack(paged_api, store, device, rows)
            actual = model(ids, token_positions=positions, paged_context=context)
            assert actual.shape == (len(ids), _MODEL_ARGS["vocab_size"])
            assert actual.device == device
            assert torch.isfinite(actual).all(), "Unused page bytes must not enter attention"
            expected_shapes.append((len(ids), _MODEL_ARGS["d_model"]))
            for seen in projections:
                assert seen == expected_shapes, "Project all new packed tokens once per layer"
            assert (store.k_pages.data_ptr(), store.v_pages.data_ptr()) == pool_pointers

            offset = 0
            for name, tokens in entries:
                start = len(histories[name])
                histories[name].extend(tokens)
                full_ids = torch.tensor([histories[name]], dtype=torch.int64, device=device)
                expected = reference(full_ids)[0, start:]
                torch.testing.assert_close(
                    actual[offset : offset + len(tokens)],
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                    msg=f"Request {name}: every new logit must match its own full prefix",
                )
                offset += len(tokens)
    finally:
        for hook in hooks:
            hook.remove()


def test_one_packed_forward_mixes_decode_with_fresh_prefill(paged_api, execution_device):
    _assert_trace(
        paged_api,
        execution_device,
        steps=(
            (("A", [11, 24, 7]),),
            (("A", [9]), ("B", [4, 8, 6])),
            (("A", [5]), ("B", [10])),
        ),
    )


def test_chunked_prefill_uses_absolute_causality_after_nonzero_prefix(paged_api, execution_device):
    _assert_trace(
        paged_api,
        execution_device,
        steps=(
            (("A", [11, 24]), ("B", [4])),
            (("A", [7, 9, 5]), ("B", [8, 6])),
            (("A", [10]), ("B", [12, 13])),
        ),
    )


def test_request_pages_survive_omission_and_temporary_batch_reordering(paged_api, execution_device):
    _assert_trace(
        paged_api,
        execution_device,
        steps=(
            (("A", [11, 24]), ("B", [4, 8, 6])),
            (("B", [10]),),
            (("A", [7, 9]),),
            (("B", [12]), ("A", [5])),
        ),
    )


def test_packed_total_can_exceed_context_while_each_request_fits(paged_api, execution_device):
    _assert_trace(
        paged_api,
        execution_device,
        context_length=4,
        steps=(
            (("A", [11, 24, 7, 9]), ("B", [4, 8, 6])),  # T=7, positions at most 3.
            (("B", [10]),),
        ),
    )
    rope = RoPE(theta=10_000.0, qk_head_dim=4, max_seq_len=4, device=execution_device)
    with pytest.raises(ValueError):
        rope(torch.zeros((3, 5, 4), device=execution_device))
