"""Offline generation submits once, drives run_step, and preserves input order.

The caller supplies a fresh Scheduler, a matching paged store and valid fresh
requests. Results contain full prompt-plus-output histories. Execution failures
propagate after run_step cleanup; this driver adds no retry or partial results.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cs336_basics.config import EngineConfig, ModelConfig, RuntimeConfig, SamplingParams
from cs336_basics.generation.kv_cache import BlockManager
from cs336_basics.generation.requests import Req
from cs336_basics.generation.results import FinishReason, GenerationResult, ModelOutput
from cs336_basics.generation.scheduler import Scheduler
from cs336_basics.transformers.paged_kv import PagedKVStore
from cs336_basics.transformers.transformers import TransformerLM

_MODEL = dict(
    vocab_size=32,
    context_length=16,
    d_model=12,
    num_layers=2,
    num_heads=3,
    d_ff=24,
    rope_theta=10_000.0,
)


@pytest.fixture
def offline_driver(monkeypatch):
    module = importlib.import_module("cs336_basics.generation.generate")
    function = getattr(module, "generate_offline", None)
    if not callable(function):
        pytest.fail(
            "Implement generate_offline(*, engine, scheduler, kv_store, requests) in generation/generate.py",
            pytrace=False,
        )
    step = Mock(wraps=module.run_step)
    monkeypatch.setattr(module, "run_step", step)
    return SimpleNamespace(generate=function, step=step)


def _resources(device, *, active_cap=2):
    manager = BlockManager(num_blocks=8, block_size=2)
    scheduler = Scheduler(
        block_manager=manager,
        max_step_input_tokens=3,
        max_active_requests=active_cap,
        max_prefill_chunk_tokens=2,
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
    return scheduler, manager, store


def _assert_released(scheduler, manager, requests):
    assert scheduler.schedule() is None
    assert manager.free_block_count == manager.available_block_count == manager.num_blocks
    for req in requests:
        with pytest.raises(KeyError):
            manager.get_block_table(request_id=req.request_id)


def test_real_offline_generation_matches_uncached_greedy_rollouts_in_input_order(
    offline_driver, monkeypatch, tmp_path, execution_device
):
    device = execution_device
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(60336)
        model, reference = TransformerLM(**_MODEL), TransformerLM(**_MODEL)
    reference.load_state_dict(model.state_dict())
    reference.to(device).eval()
    engine_module = importlib.import_module("cs336_basics.generation.engine")
    config = EngineConfig(
        model=ModelConfig(**_MODEL),
        checkpoint=tmp_path / "unused.pt",
        runtime=RuntimeConfig(device=device.type),
    )
    monkeypatch.setattr(engine_module, "build_model", lambda config: model)
    monkeypatch.setattr(engine_module, "load_model_weights", Mock())
    engine = engine_module.Engine(config)
    engine.execute = Mock(wraps=engine.execute)
    scheduler, manager, store = _resources(device)
    pointers = store.k_pages.data_ptr(), store.v_pages.data_ptr()
    requests = [
        Req(token_ids=prompt, sampling_params=SamplingParams(max_new_tokens=count, eos_token_id=None))
        for prompt, count in (([11, 24, 7, 4, 8], 3), ([9], 1), ([4, 6, 10, 12, 15], 1))
    ]
    expected = []
    with torch.inference_mode():
        for req in requests:
            history = list(req.token_ids)
            for _ in range(req.sampling_params.max_new_tokens):
                ids = torch.tensor([history], dtype=torch.int64, device=device)
                history.append(int(reference(ids)[0, -1].argmax().item()))
            expected.append(tuple(history))

    submitted, forwards = [], []
    original_add = Scheduler.add_request

    def add_request(self, *, req):
        original_add(self, req=req)
        submitted.append(req)

    def record_forward(_module, args, kwargs, output):
        assert [id(req) for req in submitted] == [id(req) for req in requests]
        assert torch.is_inference_mode_enabled() and not model.training
        assert torch.isfinite(output).all()
        forwards.append(kwargs["token_ids"].numel())

    monkeypatch.setattr(Scheduler, "add_request", add_request)
    hook = model.register_forward_hook(record_forward, with_kwargs=True)
    try:
        results = offline_driver.generate(engine=engine, scheduler=scheduler, kv_store=store, requests=requests)
    finally:
        hook.remove()

    assert isinstance(results, list)
    assert results == [
        GenerationResult(token_ids=tokens, finish_reason=FinishReason.MAX_NEW_TOKENS) for tokens in expected
    ]
    assert all(isinstance(result, GenerationResult) and isinstance(result.token_ids, tuple) for result in results)
    assert [tuple(req.token_ids) for req in requests] == expected
    assert all(req.finished and req.finish_reason is FinishReason.MAX_NEW_TOKENS for req in requests)
    assert [req.num_computed_tokens for req in requests] == [7, 1, 5]
    a, b, c = [req.request_id for req in requests]
    plans = [call.kwargs["plan"] for call in engine.execute.call_args_list]
    assert [[(row.request_id, row.num_input_tokens, row.start_pos) for row in plan.requests] for plan in plans] == [
        [(a, 2, 0), (b, 1, 0)],
        [(a, 2, 2), (c, 1, 0)],
        [(a, 1, 4), (c, 2, 1)],
        [(a, 1, 5), (c, 2, 3)],
        [(a, 1, 6)],
    ]
    assert [plan.sampling_request_ids for plan in plans] == [(b,), (), (a,), (a, c), (a,)]
    assert forwards == [3, 3, 3, 3, 1]
    assert all(call.kwargs["kv_store"] is store for call in engine.execute.call_args_list)
    assert offline_driver.step.call_count == 6, "Drive the final idle step through run_step too"
    for call in offline_driver.step.call_args_list:
        assert call.kwargs["engine"] is engine
        assert call.kwargs["scheduler"] is scheduler
        assert call.kwargs["kv_store"] is store
    assert (store.k_pages.data_ptr(), store.v_pages.data_ptr()) == pointers
    _assert_released(scheduler, manager, requests)


@pytest.mark.parametrize("prompts", [[], [[11, 31], [31]]], ids=["empty", "initial_eos"])
def test_empty_or_initial_eos_returns_without_execution(offline_driver, prompts):
    scheduler, manager, store = _resources(torch.device("cpu"))
    engine = SimpleNamespace(execute=Mock(side_effect=AssertionError("No model work is needed")))
    requests = [
        Req(token_ids=list(prompt), sampling_params=SamplingParams(max_new_tokens=3, eos_token_id=31))
        for prompt in prompts
    ]

    results = offline_driver.generate(engine=engine, scheduler=scheduler, kv_store=store, requests=tuple(requests))

    assert results == [GenerationResult(token_ids=tuple(prompt), finish_reason=FinishReason.EOS) for prompt in prompts]
    engine.execute.assert_not_called()
    assert all(req.finished and req.finish_reason is FinishReason.EOS for req in requests)
    assert all(req.num_computed_tokens == req.generated_tokens_count == 0 for req in requests)
    _assert_released(scheduler, manager, requests)


def test_execution_error_preserves_committed_progress_aborts_waiting_and_does_not_retry(offline_driver):
    scheduler, manager, store = _resources(torch.device("cpu"), active_cap=1)
    active = Req(token_ids=[1, 2], sampling_params=SamplingParams(max_new_tokens=3))
    waiting = Req(token_ids=[3], sampling_params=SamplingParams(max_new_tokens=1))
    failure = RuntimeError("second execution failed")
    engine = SimpleNamespace(
        execute=Mock(side_effect=[ModelOutput(next_tokens_gpu=torch.tensor([9], dtype=torch.int64)), failure])
    )

    with pytest.raises(RuntimeError) as caught:
        offline_driver.generate(engine=engine, scheduler=scheduler, kv_store=store, requests=[active, waiting])

    assert caught.value is failure
    assert engine.execute.call_count == offline_driver.step.call_count == 2
    assert active.token_ids == [1, 2, 9] and active.num_computed_tokens == 2
    assert waiting.token_ids == [3] and waiting.num_computed_tokens == 0
    assert all(req.finished and req.finish_reason is FinishReason.ERROR for req in (active, waiting))
    plans = [call.kwargs["plan"] for call in engine.execute.call_args_list]
    assert [plan.sampling_request_ids for plan in plans] == [(active.request_id,), (active.request_id,)]
    assert plans[1].requests[0].input_token_ids == (9,)
    assert plans[1].requests[0].start_pos == 2
    _assert_released(scheduler, manager, [active, waiting])
