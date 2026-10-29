"""Schedule composition only; result commits and cancellation are not required.

schedule() plans existing active requests first, then admits waiting requests with
only the remaining input budget. Return one nonempty immutable StepPlan, or None
when idle. A second call with a pending plan raises before changing ownership or
request state. No scheduling call commits computed counts or sampled output.

Active histories are prepared through the tested waiting helper, then explicit
test-only progress updates. This is fixture setup, not a simulated serving loop;
no private scheduler containers or unfinished apply/cancel/abort methods are used.
"""

import importlib
from types import SimpleNamespace

import pytest

from cs336_basics.config import SamplingParams
from cs336_basics.generation.requests import Req, ReqStatus


@pytest.fixture
def scheduler_factory():
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
            pytest.fail(f"Schedule subtask is not implemented: missing {exc.name}", pytrace=False)
    manager_module, scheduler_module = modules

    def make(*, token_budget=4, active_cap=3, prefill_chunk=2):
        manager = manager_module.BlockManager(num_blocks=16, block_size=2)
        scheduler = scheduler_module.Scheduler(
            block_manager=manager,
            max_step_input_tokens=token_budget,
            max_active_requests=active_cap,
            max_prefill_chunk_tokens=prefill_chunk,
            max_context_tokens=16,
        )
        return SimpleNamespace(
            scheduler=scheduler,
            manager=manager,
            requests=[],
            plan_type=scheduler_module.StepPlan,
        )

    return make


def _submit(context, tokens):
    req = Req(token_ids=list(tokens), sampling_params=SamplingParams(max_new_tokens=3))
    context.scheduler.add_request(req=req)
    context.requests.append(req)
    return req


def _histories(context):
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
        _histories(context),
        tuple(req.status for req in context.requests),
        tuple(tables),
        context.manager.free_block_count,
        context.manager.available_block_count,
    )


def _schedule(context):
    before = _histories(context)
    plan = context.scheduler.schedule()
    assert _histories(context) == before, "Scheduling must not commit progress or output"
    assert isinstance(plan, context.plan_type)
    assert isinstance(plan.step_id, int) and plan.step_id >= 0
    assert isinstance(plan.requests, tuple) and plan.requests
    assert all(item.num_input_tokens > 0 for item in plan.requests)
    assert sum(item.num_input_tokens for item in plan.requests) <= context.scheduler.max_step_input_tokens
    for item in plan.requests:
        assert item.block_table == context.manager.get_block_table(request_id=item.request_id)
    return plan


def test_idle_calls_leave_scheduler_ready_for_one_fresh_plan(scheduler_factory):
    context = scheduler_factory()
    before = _snapshot(context)
    assert context.scheduler.schedule() is None
    assert context.scheduler.schedule() is None
    assert _snapshot(context) == before
    a, b = _submit(context, [1, 2, 3]), _submit(context, [4])
    plan = _schedule(context)
    assert [item.request_id for item in plan.requests] == [a.request_id, b.request_id]
    assert [(item.start_pos, item.input_token_ids) for item in plan.requests] == [
        (0, (1, 2)),
        (0, (4,)),
    ]
    assert plan.sampling_request_ids == (b.request_id,)
    assert plan.requests[0].sampling_params is None
    assert plan.requests[1].sampling_params is b.sampling_params
    assert a.status == b.status == ReqStatus.RUNNING
    assert context.manager.available_block_count == 11  # Lifetime credits: A3 + B2.
    assert context.manager.free_block_count == 14


def test_decode_and_new_prefill_share_only_the_remaining_step_budget(scheduler_factory):
    context = scheduler_factory(token_budget=3, active_cap=2, prefill_chunk=3)
    a = _submit(context, [1, 2])
    primed = context.scheduler._plan_waiting_requests(token_budget=3)
    old_table = primed.scheduled_requests[0].block_table
    a.num_computed_tokens = 2
    a.append_token(9)
    b = _submit(context, [11, 12, 13])
    plan = _schedule(context)
    assert [item.request_id for item in plan.requests] == [a.request_id, b.request_id]
    assert [(item.start_pos, item.input_token_ids) for item in plan.requests] == [
        (2, (9,)),
        (0, (11, 12)),
    ]
    assert plan.sampling_request_ids == (a.request_id,)
    assert plan.requests[0].sampling_params is a.sampling_params
    assert plan.requests[1].sampling_params is None
    assert plan.requests[0].block_table[: len(old_table)] == old_table
    assert set(plan.requests[0].block_table).isdisjoint(plan.requests[1].block_table)
    assert b.status == ReqStatus.RUNNING and b.num_computed_tokens == 0


def test_active_prefill_keeps_admission_order_and_exhausted_budget_leaves_waiting_unreserved(scheduler_factory):
    context = scheduler_factory(token_budget=3)
    a, b = _submit(context, [1, 2, 3, 4, 5]), _submit(context, [6])
    context.scheduler._plan_waiting_requests(token_budget=3)
    a.num_computed_tokens = 2
    b.num_computed_tokens = 1
    b.append_token(20)
    c = _submit(context, [7])
    credits = context.manager.available_block_count
    plan = _schedule(context)
    assert [item.request_id for item in plan.requests] == [a.request_id, b.request_id]
    assert [(item.start_pos, item.input_token_ids) for item in plan.requests] == [
        (2, (3, 4)),
        (1, (20,)),
    ]
    assert plan.sampling_request_ids == (b.request_id,)
    assert c.status == ReqStatus.WAITING and c.num_computed_tokens == 0
    with pytest.raises(KeyError):
        context.manager.get_block_table(request_id=c.request_id)
    assert context.manager.available_block_count == credits


def test_active_only_partial_prefill_produces_nonempty_plan_without_samples(scheduler_factory):
    context = scheduler_factory()
    req = _submit(context, [1, 2, 3, 4, 5])
    context.scheduler._plan_waiting_requests(token_budget=4)
    req.num_computed_tokens = 2
    plan = _schedule(context)
    assert len(plan.requests) == 1
    item = plan.requests[0]
    assert (item.request_id, item.start_pos, item.input_token_ids) == (req.request_id, 2, (3, 4))
    assert item.sampling_params is None and plan.sampling_request_ids == ()
    assert req.status == ReqStatus.RUNNING and req.num_computed_tokens == 2


def test_pending_schedule_rejects_before_request_or_ownership_mutation(scheduler_factory):
    context = scheduler_factory(token_budget=2, active_cap=2)
    a, b, c = _submit(context, [1, 2, 3, 4]), _submit(context, [5]), _submit(context, [6])
    plan = _schedule(context)
    assert [item.request_id for item in plan.requests] == [a.request_id]
    assert b.status == c.status == ReqStatus.WAITING
    before = _snapshot(context)
    with pytest.raises(RuntimeError):
        context.scheduler.schedule()
    assert _snapshot(context) == before
