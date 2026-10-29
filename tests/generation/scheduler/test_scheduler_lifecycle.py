"""CPU lifecycle checks using public submission, scheduling and commit APIs.

cancel_request(*, request_id) runs only between steps: finish CANCELLED, release
ownership, preserve progress; terminal requests are unchanged and unknown IDs raise
KeyError. While a plan is pending, every cancellation raises RuntimeError first.
abort_all_requests(*, plan) accepts only the exact pending object, commits nothing,
finishes every live owned request ERROR, releases all ownership and clears work.
Previously terminal requests keep their reasons. Invalid aborts preserve pending
work for a correct commit or abort. No private scheduler containers are inspected.
"""

import importlib
from types import SimpleNamespace

import pytest

from cs336_basics.config import SamplingParams
from cs336_basics.generation.requests import Req, ReqStatus
from cs336_basics.generation.results import FinishReason


@pytest.fixture
def lifecycle_factory():
    targets = (
        "cs336_basics.generation.kv_cache.block_manager",
        "cs336_basics.generation.scheduler",
    )
    modules = []
    for target in targets:
        try:
            modules.append(importlib.import_module(target))
        except ModuleNotFoundError as exc:
            if exc.name not in targets:
                raise
            pytest.fail(f"Lifecycle task is not implemented: missing {exc.name}", pytrace=False)
    manager_module, scheduler_module = modules

    def make(*, active_cap=2):
        manager = manager_module.BlockManager(num_blocks=8, block_size=2)
        scheduler = scheduler_module.Scheduler(
            block_manager=manager,
            max_step_input_tokens=4,
            max_active_requests=active_cap,
            max_prefill_chunk_tokens=2,
            max_context_tokens=16,
        )
        return SimpleNamespace(scheduler=scheduler, manager=manager, requests=[])

    return make


def _submit(context, tokens, *, max_new_tokens=3, eos_token_id=None):
    req = Req(
        token_ids=list(tokens),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens, eos_token_id=eos_token_id),
    )
    context.scheduler.add_request(req=req)
    context.requests.append(req)
    return req


def _progress(*requests):
    return tuple((tuple(req.token_ids), req.num_computed_tokens) for req in requests)


def _snapshot(context):
    tables = []
    for req in context.requests:
        try:
            tables.append(context.manager.get_block_table(request_id=req.request_id))
        except KeyError:
            tables.append(None)
    return (
        _progress(*context.requests),
        tuple((req.status, req.finish_reason) for req in context.requests),
        tuple(tables),
        context.manager.free_block_count,
        context.manager.available_block_count,
    )


def _released(context, req):
    with pytest.raises(KeyError):
        context.manager.get_block_table(request_id=req.request_id)


def test_cancellation_releases_only_its_ownership_and_makes_room_for_arrival(lifecycle_factory):
    context = lifecycle_factory()
    a, b = _submit(context, [1, 2]), _submit(context, [3, 4])
    waiting = _submit(context, [5])
    first = context.scheduler.schedule()
    assert first.sampling_request_ids == (a.request_id, b.request_id)
    context.scheduler.apply_result(plan=first, sampled_token_ids=[10, 20])
    assert waiting.status is ReqStatus.WAITING
    b_table = context.manager.get_block_table(request_id=b.request_id)
    progress = _progress(a, b, waiting)
    assert (context.manager.free_block_count, context.manager.available_block_count) == (6, 2)

    context.scheduler.cancel_request(request_id=waiting.request_id)
    assert waiting.finished and waiting.finish_reason is FinishReason.CANCELLED
    assert (context.manager.free_block_count, context.manager.available_block_count) == (6, 2)
    _released(context, waiting)
    context.scheduler.cancel_request(request_id=a.request_id)
    assert a.finished and a.finish_reason is FinishReason.CANCELLED
    assert _progress(a, b, waiting) == progress
    _released(context, a)
    assert b.status is ReqStatus.RUNNING
    assert context.manager.get_block_table(request_id=b.request_id) == b_table
    assert (context.manager.free_block_count, context.manager.available_block_count) == (7, 5)

    # breakpoint()
    arrival = _submit(context, [6], max_new_tokens=1)
    next_plan = context.scheduler.schedule()
    assert tuple(item.request_id for item in next_plan.requests) == (b.request_id, arrival.request_id)
    assert next_plan.requests[0].block_table[: len(b_table)] == b_table
    context.scheduler.apply_result(plan=next_plan, sampled_token_ids=[21, 30])
    assert b.token_ids == [3, 4, 20, 21] and b.num_computed_tokens == 3
    assert arrival.finished and not b.finished


def test_repeated_and_terminal_cancellation_are_noops_and_unknown_id_is_rejected(lifecycle_factory):
    context = lifecycle_factory()
    cancelled = _submit(context, [1])
    context.scheduler.cancel_request(request_id=cancelled.request_id)
    assert cancelled.finish_reason is FinishReason.CANCELLED
    eos = _submit(context, [31], eos_token_id=31)
    completed = _submit(context, [2], max_new_tokens=1)
    plan = context.scheduler.schedule()
    assert plan.sampling_request_ids == (completed.request_id,)
    context.scheduler.apply_result(plan=plan, sampled_token_ids=[9])
    assert eos.finish_reason is FinishReason.EOS
    assert completed.finish_reason is FinishReason.MAX_NEW_TOKENS
    before = _snapshot(context)
    for req in (cancelled, eos, completed):
        context.scheduler.cancel_request(request_id=req.request_id)
        assert _snapshot(context) == before
    with pytest.raises(KeyError):
        context.scheduler.cancel_request(request_id="never-submitted")
    assert _snapshot(context) == before
    assert context.manager.free_block_count == context.manager.available_block_count == 8
    assert context.scheduler.schedule() is None


def test_pending_cancellation_rejects_active_waiting_and_unknown_targets_without_mutation(lifecycle_factory):
    context = lifecycle_factory(active_cap=1)
    active, waiting = _submit(context, [1, 2]), _submit(context, [3])
    plan = context.scheduler.schedule()
    assert plan.sampling_request_ids == (active.request_id,)
    assert waiting.status is ReqStatus.WAITING
    before = _snapshot(context)
    for request_id in (active.request_id, waiting.request_id, "never-submitted"):
        with pytest.raises(RuntimeError):
            context.scheduler.cancel_request(request_id=request_id)
        assert _snapshot(context) == before
    context.scheduler.apply_result(plan=plan, sampled_token_ids=[10])
    assert active.token_ids == [1, 2, 10] and active.num_computed_tokens == 2
    assert waiting.status is ReqStatus.WAITING


def test_abort_cleans_all_live_requests_preserving_committed_progress_and_terminal_reason(lifecycle_factory):
    context = lifecycle_factory()
    terminal = _submit(context, [9], max_new_tokens=1)
    completed_plan = context.scheduler.schedule()
    context.scheduler.apply_result(plan=completed_plan, sampled_token_ids=[99])
    a = _submit(context, [1, 2])
    b = _submit(context, [3, 4, 5, 6, 7])
    waiting = _submit(context, [8])
    first = context.scheduler.schedule()
    assert first.sampling_request_ids == (a.request_id,)
    context.scheduler.apply_result(plan=first, sampled_token_ids=[10])
    failed = context.scheduler.schedule()
    assert tuple(item.request_id for item in failed.requests) == (a.request_id, b.request_id)
    assert waiting.status is ReqStatus.WAITING
    assert a.num_computed_tokens == b.num_computed_tokens == 2
    progress = _progress(*context.requests)
    # breakpoint()
    context.scheduler.abort_all_requests(plan=failed)
    assert _progress(*context.requests) == progress
    assert terminal.finished and terminal.finish_reason is FinishReason.MAX_NEW_TOKENS
    assert all(req.finished and req.finish_reason is FinishReason.ERROR for req in (a, b, waiting))
    for req in context.requests:
        _released(context, req)
    assert context.manager.free_block_count == context.manager.available_block_count == 8
    assert context.scheduler.schedule() is None
    before = _snapshot(context)
    with pytest.raises(RuntimeError):
        context.scheduler.apply_result(plan=failed, sampled_token_ids=[11])
    assert _snapshot(context) == before


def test_invalid_abort_plans_leave_current_work_usable_and_repeated_abort_is_rejected(lifecycle_factory):
    context = lifecycle_factory()
    req = _submit(context, [1])
    first = context.scheduler.schedule()
    forged = type(first)(step_id=first.step_id, requests=first.requests)
    before = _snapshot(context)
    with pytest.raises(RuntimeError):
        context.scheduler.abort_all_requests(plan=forged)
    assert _snapshot(context) == before
    context.scheduler.apply_result(plan=first, sampled_token_ids=[10])

    # No step is pending; an already applied plan cannot be aborted or reapplied.
    before = _snapshot(context)
    with pytest.raises(RuntimeError):
        context.scheduler.abort_all_requests(plan=first)
    with pytest.raises(RuntimeError):
        context.scheduler.apply_result(plan=first, sampled_token_ids=[10])
    assert _snapshot(context) == before
    second = context.scheduler.schedule()
    before = _snapshot(context)
    with pytest.raises(RuntimeError):
        context.scheduler.abort_all_requests(plan=first)
    assert _snapshot(context) == before
    context.scheduler.apply_result(plan=second, sampled_token_ids=[11])
    assert req.token_ids == [1, 10, 11] and req.num_computed_tokens == 2

    current = context.scheduler.schedule()
    progress = _progress(req)
    context.scheduler.abort_all_requests(plan=current)
    assert _progress(req) == progress
    assert req.finished and req.finish_reason is FinishReason.ERROR
    assert context.manager.free_block_count == context.manager.available_block_count == 8
    before = _snapshot(context)
    with pytest.raises(RuntimeError):
        context.scheduler.abort_all_requests(plan=current)
    assert _snapshot(context) == before
    assert context.scheduler.schedule() is None
