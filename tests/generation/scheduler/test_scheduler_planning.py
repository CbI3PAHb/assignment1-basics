"""Focused Scheduler planning helpers, without admission or result commits.

Scheduler._plan_request(*, req, token_budget) -> ScheduledRequest | None chooses
min(uncomputed tokens, budget), additionally capped by max_prefill_chunk_tokens
for prefill. Positive work grows physical capacity to computed + chosen tokens.
The returned snapshot contains the exact input slice, absolute start position,
block table and original sampling policy only when the available suffix ends.

Zero budget or zero uncomputed tokens returns None without allocation. Negative
budget raises ValueError before mutation. Planning never changes request history,
computed count, lifecycle, admission credits or scheduler queue membership.
Tests call this helper directly; add_request/schedule/apply_result are not needed.

The next helper, _plan_active_requests(*, active_requests: Iterable[Req],
token_budget: int), returns a NamedTuple with scheduled_requests (list) and
remaining_budget (int) fields. It visits
already-reserved RUNNING requests once in their supplied order, using the shared
budget. It never admits requests or changes progress, lifecycle or queue membership.
"""

import importlib
from types import SimpleNamespace

import pytest

from cs336_basics.config import SamplingParams
from cs336_basics.generation.requests import Req


@pytest.fixture
def planning():
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
            pytest.fail(f"Planning subtask is not implemented: missing {exc.name}", pytrace=False)
    manager_module, scheduler_module = modules
    if not callable(getattr(scheduler_module.Scheduler, "_plan_request", None)):
        pytest.fail(
            "Implement Scheduler._plan_request(*, req, token_budget) for this subtask",
            pytrace=False,
        )
    manager = manager_module.BlockManager(num_blocks=8, block_size=2)
    scheduler = scheduler_module.Scheduler(
        block_manager=manager,
        max_step_input_tokens=8,
        max_active_requests=2,
        max_prefill_chunk_tokens=2,
        max_context_tokens=16,
    )
    return SimpleNamespace(
        manager=manager,
        scheduler=scheduler,
        ScheduledRequest=scheduler_module.ScheduledRequest,
    )


def _reserved_request(planning, tokens):
    req = Req(
        token_ids=list(tokens),
        sampling_params=SamplingParams(max_new_tokens=3, temperature=0.7, top_k=3, top_p=0.8),
    )
    assert planning.manager.reserve(request_id=req.request_id, max_tokens=8)
    return req


def _state(req):
    return (
        req.request_id,
        tuple(req.token_ids),
        req.prompt_tokens_count,
        req.num_computed_tokens,
        req.status,
        req.finish_reason,
        req.sampling_params,
    )


def _plan(planning, req, *, token_budget):
    before = _state(req)
    credits = planning.manager.available_block_count
    old_table = planning.manager.get_block_table(request_id=req.request_id)
    result = planning.scheduler._plan_request(req=req, token_budget=token_budget)
    assert _state(req) == before, "Planning must not commit tokens or change lifecycle"
    assert planning.manager.available_block_count == credits, "Reservation belongs to the caller"
    table = planning.manager.get_block_table(request_id=req.request_id)
    assert table[: len(old_table)] == old_table, "Existing physical addresses must stay stable"
    if result is not None:
        assert isinstance(result, planning.ScheduledRequest)
        assert result.request_id == req.request_id
        assert result.block_table == table
    return result


def test_partial_and_final_prompt_chunks_use_exact_suffixes(planning):
    req = _reserved_request(planning, [11, 12, 13, 14, 15])
    for start, expected in [(0, (11, 12)), (2, (13, 14)), (4, (15,))]:
        assert req.is_prefill and not req.is_decode
        result = _plan(planning, req, token_budget=8)
        assert (result.start_pos, result.input_token_ids) == (start, expected)
        assert result.num_input_tokens == len(expected)
        assert result.sampling_params is (req.sampling_params if start == 4 else None)
        assert len(result.block_table) == (start + len(expected) + 1) // 2
        # Simulate only the completed-input count between direct helper calls.
        req.num_computed_tokens += result.num_input_tokens
    assert req.is_decode and not req.is_prefill
    assert req.token_ids == [11, 12, 13, 14, 15]


def test_decode_grows_across_a_page_boundary_without_relocating_history(planning):
    req = _reserved_request(planning, [1, 2, 3, 4])
    previous = planning.manager.ensure_capacity(request_id=req.request_id, num_tokens=4)
    req.num_computed_tokens = 4
    req.append_token(9)
    req.start()
    assert req.is_decode and not req.is_prefill
    result = _plan(planning, req, token_budget=8)
    assert (result.start_pos, result.input_token_ids, result.num_input_tokens) == (4, (9,), 1)
    assert result.sampling_params is req.sampling_params
    assert len(result.block_table) == 3
    assert result.block_table[:2] == previous
    assert planning.manager.free_block_count == 5


def test_step_budget_caps_prefill_and_a_fresh_single_token_can_sample(planning):
    req = _reserved_request(planning, [1, 2, 3])
    result = _plan(planning, req, token_budget=1)
    assert (result.start_pos, result.input_token_ids, result.num_input_tokens) == (0, (1,), 1)
    assert result.sampling_params is None

    fresh = _reserved_request(planning, [7])
    assert fresh.is_prefill and not fresh.is_decode
    result = _plan(planning, fresh, token_budget=8)
    assert (result.start_pos, result.input_token_ids, result.num_input_tokens) == (0, (7,), 1)
    assert result.sampling_params is fresh.sampling_params
    assert fresh.num_computed_tokens == 0 and fresh.is_prefill


def test_zero_budget_or_no_uncomputed_tokens_does_not_allocate(planning):
    req = _reserved_request(planning, [1, 2, 3, 4])
    for computed, token_budget in [(0, 0), (2, 0), (4, 5)]:
        previous = planning.manager.ensure_capacity(request_id=req.request_id, num_tokens=computed)
        req.num_computed_tokens = computed
        free_blocks = planning.manager.free_block_count
        assert _plan(planning, req, token_budget=token_budget) is None
        assert planning.manager.get_block_table(request_id=req.request_id) == previous
        assert planning.manager.free_block_count == free_blocks


def test_negative_budget_is_rejected_without_partial_mutation(planning):
    req = _reserved_request(planning, [1, 2, 3])
    previous = planning.manager.ensure_capacity(request_id=req.request_id, num_tokens=2)
    req.num_computed_tokens = 2
    before = _state(req)
    counts = (planning.manager.free_block_count, planning.manager.available_block_count)
    with pytest.raises(ValueError):
        planning.scheduler._plan_request(req=req, token_budget=-1)
    assert _state(req) == before
    assert planning.manager.get_block_table(request_id=req.request_id) == previous
    assert (planning.manager.free_block_count, planning.manager.available_block_count) == counts


@pytest.fixture
def active_planning(planning):
    scheduler_type = type(planning.scheduler)
    if not callable(getattr(scheduler_type, "_plan_active_requests", None)):
        pytest.fail(
            "Implement Scheduler._plan_active_requests(*, active_requests, token_budget)",
            pytrace=False,
        )
    manager = type(planning.manager)(num_blocks=10, block_size=2)
    scheduler = scheduler_type(
        block_manager=manager,
        max_step_input_tokens=8,
        max_active_requests=3,
        max_prefill_chunk_tokens=2,
        max_context_tokens=16,
    )
    requests = tuple(
        Req(
            token_ids=tokens,
            sampling_params=SamplingParams(max_new_tokens=3, temperature=0.7, top_k=index + 1),
        )
        for index, tokens in enumerate(([1, 2, 3, 4, 5], [11, 12], [21, 22, 23]))
    )
    for req in requests:
        assert manager.reserve(
            request_id=req.request_id,
            max_tokens=req.prompt_tokens_count + req.sampling_params.max_new_tokens,
        )
        req.start()
    a, b, c = requests
    for req in (b, c):
        manager.ensure_capacity(request_id=req.request_id, num_tokens=2)
        req.num_computed_tokens = 2
    b.append_token(19)
    assert a.is_prefill and b.is_decode and c.is_prefill
    assert manager.available_block_count == 0  # A/B/C reserve 4/3/3 blocks.
    return SimpleNamespace(manager=manager, scheduler=scheduler, requests=requests)


def _active_snapshot(planning):
    return (
        tuple(_state(req) for req in planning.requests),
        tuple(planning.manager.get_block_table(request_id=req.request_id) for req in planning.requests),
        planning.manager.free_block_count,
        planning.manager.available_block_count,
    )


def test_active_pass_keeps_mixed_work_order_and_returns_remaining_budget(active_planning):
    context = active_planning
    before, tables, _, credits = _active_snapshot(context)
    result = context.scheduler._plan_active_requests(
        active_requests=context.requests,
        token_budget=5,
    )
    assert isinstance(result, tuple)
    planned = result.scheduled_requests
    a, b, c = context.requests
    assert isinstance(planned, list)
    assert [item.request_id for item in planned] == [a.request_id, b.request_id, c.request_id]
    assert [(item.start_pos, item.input_token_ids) for item in planned] == [
        (0, (1, 2)),
        (2, (19,)),
        (2, (23,)),
    ]
    assert planned[0].sampling_params is None
    assert planned[1].sampling_params is b.sampling_params
    assert planned[2].sampling_params is c.sampling_params
    assert result.remaining_budget == 1
    assert tuple(_state(req) for req in context.requests) == before
    assert context.manager.available_block_count == credits == 0
    for item, previous in zip(planned, tables, strict=True):
        assert item.block_table == context.manager.get_block_table(request_id=item.request_id)
        assert item.block_table[: len(previous)] == previous
        assert len(item.block_table) == (item.start_pos + item.num_input_tokens + 1) // 2


def test_active_pass_stops_at_the_supplied_budget_without_zero_work(active_planning):
    context = active_planning
    a, b, c = context.requests
    before, tables, _, credits = _active_snapshot(context)
    for budget, expected_ids, expected_tokens in (
        (3, [a.request_id, b.request_id], [(1, 2), (19,)]),
        (1, [a.request_id], [(1,)]),
    ):
        tables_before = {
            req.request_id: context.manager.get_block_table(request_id=req.request_id) for req in context.requests
        }
        result = context.scheduler._plan_active_requests(
            active_requests=context.requests,
            token_budget=budget,
        )
        planned = result.scheduled_requests
        assert [item.request_id for item in planned] == expected_ids
        assert [item.input_token_ids for item in planned] == expected_tokens
        assert all(item.num_input_tokens > 0 for item in planned)
        assert result.remaining_budget == 0
        assert tuple(_state(req) for req in context.requests) == before
        assert context.manager.get_block_table(request_id=c.request_id) == tables[2]
        assert context.manager.available_block_count == credits
        for req in context.requests:
            if req.request_id not in expected_ids:
                assert context.manager.get_block_table(request_id=req.request_id) == tables_before[req.request_id]


def test_active_pass_empty_input_or_zero_budget_allocates_nothing(active_planning):
    context = active_planning
    before = _active_snapshot(context)
    for requests, budget in (((), 7), (context.requests, 0)):
        result = context.scheduler._plan_active_requests(
            active_requests=requests,
            token_budget=budget,
        )
        assert isinstance(result, tuple)
        assert result.scheduled_requests == []
        assert result.remaining_budget == budget
        assert _active_snapshot(context) == before


def test_active_pass_rejects_negative_budget_before_any_allocation(active_planning):
    context = active_planning
    before = _active_snapshot(context)
    for requests in (context.requests, ()):
        with pytest.raises(ValueError):
            context.scheduler._plan_active_requests(active_requests=requests, token_budget=-1)
        assert _active_snapshot(context) == before
