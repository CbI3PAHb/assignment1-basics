"""Waiting-admission subtask, independent of schedule() and result commits.

Scheduler._plan_waiting_requests(*, token_budget: int) -> PlanningResult returns
named scheduled_requests (list) and remaining_budget fields. Valid requests enter
through add_request. Admit FIFO only with positive work, an active slot and
successful lifetime-credit reservation; stop at the first blocked head.
Only newly admitted requests appear in the result. History and computed counts
stay unchanged; successful admission changes WAITING to RUNNING. Negative budget
raises before mutation, including with an empty queue. Other no-work states are
normal empty results with unchanged budget. No private containers are inspected.
Repeated helper calls below probe admission bookkeeping only; no execution or
completed generation step is simulated between them.
"""

import importlib
from types import SimpleNamespace

import pytest

from cs336_basics.config import SamplingParams
from cs336_basics.generation.requests import Req, ReqStatus


@pytest.fixture
def admission_factory():
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
            pytest.fail(f"Admission subtask is not implemented: missing {exc.name}", pytrace=False)
    manager_module, scheduler_module = modules
    if not callable(getattr(scheduler_module.Scheduler, "_plan_waiting_requests", None)):
        pytest.fail("Implement Scheduler._plan_waiting_requests(*, token_budget)", pytrace=False)

    def make(*, num_blocks=8, max_active_requests=3, max_context_tokens=16):
        manager = manager_module.BlockManager(num_blocks=num_blocks, block_size=2)
        scheduler = scheduler_module.Scheduler(
            block_manager=manager,
            max_step_input_tokens=8,
            max_active_requests=max_active_requests,
            max_prefill_chunk_tokens=2,
            max_context_tokens=max_context_tokens,
        )
        return SimpleNamespace(manager=manager, scheduler=scheduler, requests=[])

    return make


def _submit(context, tokens, *, max_new_tokens=3):
    req = Req(
        token_ids=list(tokens),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens, temperature=0.7),
    )
    context.scheduler.add_request(req=req)
    context.requests.append(req)
    return req


def _history(context):
    return tuple(
        (
            req.request_id,
            tuple(req.token_ids),
            req.num_computed_tokens,
            req.prompt_tokens_count,
            req.finish_reason,
            req.sampling_params,
        )
        for req in context.requests
    )


def _snapshot(context):
    tables = []
    for req in context.requests:
        try:
            tables.append(context.manager.get_block_table(request_id=req.request_id))
        except KeyError:
            tables.append(None)
    return (
        _history(context),
        tuple(req.status for req in context.requests),
        tuple(tables),
        context.manager.free_block_count,
        context.manager.available_block_count,
    )


def _plan(context, *, token_budget):
    history = _history(context)
    result = context.scheduler._plan_waiting_requests(token_budget=token_budget)
    assert isinstance(result, tuple)
    assert isinstance(result.scheduled_requests, list)
    assert isinstance(result.remaining_budget, int)
    assert _history(context) == history, "Admission must not commit input or output tokens"
    assert all(item.num_input_tokens > 0 for item in result.scheduled_requests)
    for item in result.scheduled_requests:
        assert item.block_table == context.manager.get_block_table(request_id=item.request_id)
    return result


def _unreserved(context, req):
    with pytest.raises(KeyError):
        context.manager.get_block_table(request_id=req.request_id)


def test_waiting_admission_spends_only_remaining_budget_and_reserves_selected_requests(admission_factory):
    context = admission_factory()
    a = _submit(context, [1, 2, 3, 4, 5])
    b = _submit(context, [6], max_new_tokens=1)
    c = _submit(context, [8, 9], max_new_tokens=1)
    result = _plan(context, token_budget=3)
    assert [item.request_id for item in result.scheduled_requests] == [a.request_id, b.request_id]
    assert [(item.start_pos, item.input_token_ids) for item in result.scheduled_requests] == [
        (0, (1, 2)),
        (0, (6,)),
    ]
    assert result.remaining_budget == 0
    assert result.scheduled_requests[0].sampling_params is None
    assert result.scheduled_requests[1].sampling_params is b.sampling_params
    assert a.status == b.status == ReqStatus.RUNNING
    assert c.status == ReqStatus.WAITING
    _unreserved(context, c)
    assert context.manager.available_block_count == 3  # A/B reserve 4/1 credits.
    assert context.manager.free_block_count == 6
    old_tables = [item.block_table for item in result.scheduled_requests]

    later = _plan(context, token_budget=1)
    assert [item.request_id for item in later.scheduled_requests] == [c.request_id]
    assert later.scheduled_requests[0].input_token_ids == (8,)
    assert later.scheduled_requests[0].sampling_params is None
    assert later.remaining_budget == 0 and c.status == ReqStatus.RUNNING
    assert [context.manager.get_block_table(request_id=req.request_id) for req in (a, b)] == old_tables
    assert context.manager.available_block_count == 1
    assert context.manager.free_block_count == 5


def test_full_active_slots_leave_waiting_requests_and_existing_ownership_unchanged(admission_factory):
    context = admission_factory(num_blocks=6, max_active_requests=1)
    a = _submit(context, [1, 2, 3], max_new_tokens=1)
    b = _submit(context, [4], max_new_tokens=1)
    first = _plan(context, token_budget=4)
    assert [item.request_id for item in first.scheduled_requests] == [a.request_id]
    assert first.remaining_budget == 2
    assert a.status == ReqStatus.RUNNING and b.status == ReqStatus.WAITING
    _unreserved(context, b)
    before = _snapshot(context)
    again = _plan(context, token_budget=4)
    assert again.scheduled_requests == [] and again.remaining_budget == 4
    assert _snapshot(context) == before


def test_fifo_credit_pressure_keeps_the_blocked_head_and_does_not_bypass_it(admission_factory):
    context = admission_factory(num_blocks=3, max_context_tokens=8)
    a = _submit(context, [1, 2], max_new_tokens=2)
    b = _submit(context, [3, 4], max_new_tokens=2)
    c = _submit(context, [5], max_new_tokens=1)
    first = _plan(context, token_budget=4)
    assert [item.request_id for item in first.scheduled_requests] == [a.request_id]
    assert first.remaining_budget == 2
    assert a.status == ReqStatus.RUNNING and b.status == c.status == ReqStatus.WAITING
    assert context.manager.available_block_count == 1
    assert context.manager.free_block_count == 2
    _unreserved(context, b)
    _unreserved(context, c)
    before = _snapshot(context)
    again = _plan(context, token_budget=4)
    assert again.scheduled_requests == [] and again.remaining_budget == 4
    assert _snapshot(context) == before


def test_context_limit_clips_reserved_lifetime_but_pages_remain_lazy(admission_factory):
    context = admission_factory(num_blocks=3, max_context_tokens=6)
    req = _submit(context, [1, 2, 3, 4], max_new_tokens=10)
    result = _plan(context, token_budget=4)
    assert [item.request_id for item in result.scheduled_requests] == [req.request_id]
    item = result.scheduled_requests[0]
    assert (item.start_pos, item.input_token_ids) == (0, (1, 2))
    assert item.sampling_params is None and result.remaining_budget == 2
    assert len(item.block_table) == 1
    assert context.manager.available_block_count == 0  # min(6, 4 + 10) / 2 = 3 credits.
    assert context.manager.free_block_count == 2
    assert req.status == ReqStatus.RUNNING and req.num_computed_tokens == 0


def test_empty_zero_and_negative_budgets_preserve_waiting_state(admission_factory):
    context = admission_factory()
    before = _snapshot(context)
    empty = _plan(context, token_budget=5)
    assert empty.scheduled_requests == [] and empty.remaining_budget == 5
    with pytest.raises(ValueError):
        context.scheduler._plan_waiting_requests(token_budget=-1)
    assert _snapshot(context) == before

    req = _submit(context, [1, 2])
    before = _snapshot(context)
    zero = _plan(context, token_budget=0)
    assert zero.scheduled_requests == [] and zero.remaining_budget == 0
    with pytest.raises(ValueError):
        context.scheduler._plan_waiting_requests(token_budget=-1)
    assert _snapshot(context) == before
    _unreserved(context, req)
    valid = _plan(context, token_budget=1)
    assert [item.request_id for item in valid.scheduled_requests] == [req.request_id]
    assert valid.scheduled_requests[0].input_token_ids == (1,)
