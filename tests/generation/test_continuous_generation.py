"""Real paged generation admits a new prompt between decode steps."""

import importlib
from unittest.mock import Mock

import pytest
import torch

from cs336_basics.config import EngineConfig, ModelConfig, RuntimeConfig, SamplingParams
from cs336_basics.generation.generate import run_step
from cs336_basics.generation.kv_cache import BlockManager
from cs336_basics.generation.requests import Req
from cs336_basics.generation.results import FinishReason
from cs336_basics.generation.scheduler import Scheduler
from cs336_basics.transformers.paged_kv import PagedKVStore
from cs336_basics.transformers.transformers import TransformerLM


def test_real_arrival_shares_decode_forward_and_preserves_paged_history(monkeypatch, tmp_path, execution_device):
    device = execution_device
    dimensions = dict(
        vocab_size=32,
        context_length=16,
        d_model=12,
        num_layers=2,
        num_heads=3,
        d_ff=24,
        rope_theta=10_000.0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(60336)
        model, reference = TransformerLM(**dimensions), TransformerLM(**dimensions)
    reference.load_state_dict(model.state_dict())
    reference.to(device).eval()
    engine_module = importlib.import_module("cs336_basics.generation.engine")
    monkeypatch.setattr(engine_module, "build_model", lambda config: model)
    monkeypatch.setattr(engine_module, "load_model_weights", Mock())
    engine = engine_module.Engine(
        EngineConfig(
            model=ModelConfig(**dimensions),
            checkpoint=tmp_path / "unused.pt",
            runtime=RuntimeConfig(device=device.type),
        )
    )
    manager = BlockManager(num_blocks=4, block_size=4)
    scheduler = Scheduler(
        block_manager=manager,
        max_step_input_tokens=4,
        max_active_requests=2,
        max_prefill_chunk_tokens=3,
        max_context_tokens=16,
    )
    store = PagedKVStore(
        num_layers=2,
        num_blocks=manager.num_blocks,
        block_size=manager.block_size,
        num_heads=3,
        head_dim=4,
        device=device,
        dtype=torch.float32,
    )
    store.k_pages.fill_(float("nan"))
    store.v_pages.fill_(float("nan"))
    pointers = store.k_pages.data_ptr(), store.v_pages.data_ptr()
    a = Req(token_ids=[11, 24, 7], sampling_params=SamplingParams(max_new_tokens=3))
    b = Req(token_ids=[4, 6, 10, 12, 15], sampling_params=SamplingParams(max_new_tokens=1))
    expected = []
    with torch.inference_mode():
        for req in (a, b):
            history = list(req.token_ids)
            for _ in range(req.sampling_params.max_new_tokens):
                ids = torch.tensor([history], device=device, dtype=torch.int64)
                history.append(int(reference(ids)[0, -1].argmax().item()))
            expected.append(tuple(history))

    forwards = []

    def record_forward(_module, args, kwargs, output):
        assert torch.is_inference_mode_enabled() and not model.training
        assert torch.isfinite(output).all()
        metadata = kwargs["paged_context"].metadata
        forwards.append(
            (
                kwargs["token_ids"].clone(),
                kwargs["token_positions"].clone(),
                metadata.q_offsets_cpu,
                metadata.kv_lengths_cpu,
                metadata.block_tables.clone(),
            )
        )

    hook = model.register_forward_hook(record_forward, with_kwargs=True)
    try:
        scheduler.add_request(req=a)
        assert run_step(engine=engine, scheduler=scheduler, kv_store=store) is True
        assert a.num_computed_tokens == 3 and a.generated_tokens_count == 1
        a_table = manager.get_block_table(request_id=a.request_id)
        assert len(a_table) == 1
        a_prefix = [pages[:, a_table[0], :3].clone() for pages in (store.k_pages, store.v_pages)]

        # B arrives after A has sampled: the next forward must mix decode and prefill.
        scheduler.add_request(req=b)
        assert run_step(engine=engine, scheduler=scheduler, kv_store=store) is True
        assert a.num_computed_tokens == 4 and a.generated_tokens_count == 2
        assert b.num_computed_tokens == 3 and b.generated_tokens_count == 0
        assert manager.get_block_table(request_id=a.request_id) == a_table
        b_table = manager.get_block_table(request_id=b.request_id)
        assert len(b_table) == 1 and set(a_table).isdisjoint(b_table)
        assert manager.available_block_count == 0 and manager.free_block_count == 2
        for pages, before in zip((store.k_pages, store.v_pages), a_prefix):
            torch.testing.assert_close(pages[:, a_table[0], :3], before, rtol=0, atol=0)
        prefixes = [
            (pages[:, a_table[0], :4].clone(), pages[:, b_table[0], :3].clone())
            for pages in (store.k_pages, store.v_pages)
        ]

        assert run_step(engine=engine, scheduler=scheduler, kv_store=store) is True
        assert run_step(engine=engine, scheduler=scheduler, kv_store=store) is False
    finally:
        hook.remove()

    assert len(forwards) == 3, "Each step executes one shared model forward; idle executes none"
    assert [item[2] for item in forwards] == [(0, 3), (0, 1, 4), (0, 1, 3)]
    assert [item[3] for item in forwards] == [(3,), (4, 3), (5, 5)]
    expected_inputs = ([11, 24, 7], [expected[0][3], 4, 6, 10], [expected[0][4], 12, 15])
    expected_positions = ([0, 1, 2], [3, 0, 1, 2], [4, 3, 4])
    for (ids, positions, *_), tokens, absolute_positions in zip(forwards, expected_inputs, expected_positions):
        torch.testing.assert_close(ids, torch.tensor(tokens, device=device, dtype=torch.int64))
        torch.testing.assert_close(positions, torch.tensor(absolute_positions, device=device, dtype=torch.int64))
    final_tables = forwards[-1][4]
    assert final_tables.shape == (2, 2) and final_tables.unique().numel() == 4
    torch.testing.assert_close(final_tables[:, :1], forwards[1][4])
    for pages, (a_before, b_before) in zip((store.k_pages, store.v_pages), prefixes):
        torch.testing.assert_close(pages[:, a_table[0], :4], a_before, rtol=0, atol=0)
        torch.testing.assert_close(pages[:, b_table[0], :3], b_before, rtol=0, atol=0)
    assert [tuple(req.token_ids) for req in (a, b)] == expected
    assert all(req.finished and req.finish_reason is FinishReason.MAX_NEW_TOKENS for req in (a, b))
    assert a.num_computed_tokens == b.num_computed_tokens == 5
    assert (store.k_pages.data_ptr(), store.v_pages.data_ptr()) == pointers
    assert manager.free_block_count == manager.available_block_count == manager.num_blocks
    for req in (a, b):
        with pytest.raises(KeyError):
            manager.get_block_table(request_id=req.request_id)
