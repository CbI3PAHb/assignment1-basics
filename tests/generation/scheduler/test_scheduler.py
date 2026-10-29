"""CPU control-plane contract; scripted token IDs stand in for an executor.

Scheduler(block_manager=..., max_step_input_tokens=..., max_active_requests=...,
          max_prefill_chunk_tokens=..., max_context_tokens=...) owns admission and progress.
add_request(req=...); schedule() -> immutable StepPlan | None;
apply_result(plan=..., sampled_token_ids=...) commits the step; cancel_request(request_id=...)
and abort_all_requests(plan=...) release ownership. There is at most one outstanding plan.

Each frozen ScheduledRequest contains request_id, this step's input_token_ids tuple,
start_pos, block_table tuple and optional sampling_params; num_input_tokens is its
token count. StepPlan has an increasing step_id, a requests tuple, and ordered
sampling_request_ids for rows that finish their available suffix. No zero-work
rows, model calls, or GPU metadata belong in this scheduler contract.
Plan existing RUNNING work in admission order before considering WAITING requests.
Admit a waiting request only when this step can immediately process a positive slice.
"""

import importlib
import inspect
from types import SimpleNamespace

import pytest

from cs336_basics.config import SamplingParams
from cs336_basics.generation.requests import Req, ReqStatus
from cs336_basics.generation.results import FinishReason


@pytest.fixture
def api():
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
            pytest.fail(f"Control-plane task is not implemented: missing {exc.name}")
    manager_module, scheduler_module = modules
    return SimpleNamespace(
        BlockManager=manager_module.BlockManager,
        Scheduler=scheduler_module.Scheduler,
        StepPlan=scheduler_module.StepPlan,
        ScheduledRequest=scheduler_module.ScheduledRequest,
    )


@pytest.fixture
def make_scheduler(api):
    def make(
        *,
        num_blocks=16,
        block_size=2,
        max_step_input_tokens=4,
        max_active_requests=2,
        max_prefill_chunk_tokens=2,
        max_context_tokens=16,
    ):
        manager = api.BlockManager(num_blocks=num_blocks, block_size=block_size)
        scheduler = api.Scheduler(
            block_manager=manager,
            max_step_input_tokens=max_step_input_tokens,
            max_active_requests=max_active_requests,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            max_context_tokens=max_context_tokens,
        )
        return scheduler, manager

    return make


def _req(tokens, *, max_new_tokens=3, **policy):
    return Req(
        token_ids=list(tokens),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens, **policy),
    )


def _ids(plan):
    assert isinstance(plan.requests, tuple)
    assert all(row.num_input_tokens == len(row.input_token_ids) > 0 for row in plan.requests)
    assert all(isinstance(row.input_token_ids, tuple) for row in plan.requests)
    assert all(isinstance(row.block_table, tuple) for row in plan.requests)
    return tuple(row.request_id for row in plan.requests)


def _state(*requests):
    return tuple((tuple(req.token_ids), req.num_computed_tokens, req.status, req.finish_reason) for req in requests)


def _released(manager, req):
    with pytest.raises(KeyError):
        manager.get_block_table(request_id=req.request_id)


def test_scheduled_request_requires_identity_and_explicit_sampling_policy(api):
    inputs = dict(input_token_ids=(11, 12), start_pos=3, block_table=(7, 2))
    partial = api.ScheduledRequest(request_id="A", sampling_params=None, **inputs)
    assert partial.request_id == "A"
    assert partial.input_token_ids == (11, 12)
    assert partial.num_input_tokens == 2
    assert partial.sampling_params is None
    with pytest.raises(AttributeError):
        partial.num_input_tokens = 99
    with pytest.raises(TypeError):
        api.ScheduledRequest(sampling_params=None, **inputs)
    with pytest.raises(TypeError):
        api.ScheduledRequest(request_id="A", **inputs)

    policy = SamplingParams(max_new_tokens=2, temperature=0.7, top_k=3, top_p=0.8)
    ready = api.ScheduledRequest(request_id="A", sampling_params=policy, **inputs)
    assert ready.request_id == "A"
    assert ready.num_input_tokens == 2
    assert ready.sampling_params is policy


def test_step_plan_derives_sampling_id_order_without_caller_override(api):
    policy = SamplingParams(max_new_tokens=2)
    requests = tuple(
        api.ScheduledRequest(
            request_id=request_id,
            input_token_ids=(11,),
            start_pos=0,
            block_table=(block,),
            sampling_params=sampling_params,
        )
        for request_id, block, sampling_params in (("B", 7, policy), ("A", 2, None), ("C", 5, policy))
    )
    plan = api.StepPlan(step_id=1, requests=requests)
    assert plan.requests == requests
    assert plan.sampling_request_ids == ("B", "C")
    with pytest.raises(AttributeError):
        plan.sampling_request_ids = ("A",)
    with pytest.raises(TypeError):
        api.StepPlan(step_id=1, requests=requests, sampling_request_ids=("A",))
    intermediate = api.StepPlan(step_id=2, requests=(requests[1],))
    assert intermediate.sampling_request_ids == ()
    with pytest.raises(ValueError):
        api.StepPlan(step_id=3, requests=())


def test_scheduler_requires_external_block_manager_and_consistent_limits(api):
    parameter = inspect.signature(api.Scheduler).parameters["block_manager"]
    assert parameter.default is inspect.Parameter.empty
    limits = dict(
        max_step_input_tokens=4,
        max_active_requests=2,
        max_prefill_chunk_tokens=2,
        max_context_tokens=16,
    )
    with pytest.raises(TypeError):
        api.Scheduler(**limits)
    manager = api.BlockManager(num_blocks=8, block_size=2)
    scheduler = api.Scheduler(block_manager=manager, **limits)
    assert scheduler.block_manager is manager
    with pytest.raises(ValueError):
        api.Scheduler(block_manager=manager, **{**limits, "max_active_requests": 5})


def test_request_identity_is_stable_and_read_only(api):
    req, other = _req([1]), _req([1])
    identity = req.request_id
    assert isinstance(identity, str) and identity == req._id
    assert identity != other.request_id
    assert req.num_computed_tokens == 0
    assert req.num_uncomputed_tokens == 1
    assert req.is_prefill and not req.is_decode, "A fresh one-token prompt is prefill"
    with pytest.raises(AttributeError):
        req.request_id = "replacement"
    req.append_token(2)
    assert req.request_id == identity


def test_idle_scheduler_returns_none(make_scheduler):
    scheduler, manager = make_scheduler()
    assert scheduler.schedule() is None
    assert scheduler.schedule() is None
    assert manager.free_block_count == manager.available_block_count == 16


def test_chunked_prefill_commits_only_executed_tokens(make_scheduler, api):
    scheduler, manager = make_scheduler()
    req = _req([1, 2, 3, 4, 5], max_new_tokens=1, temperature=0.7, top_k=3, top_p=0.8)
    scheduler.add_request(req=req)
    previous_step = None
    for start, tokens in [(0, (1, 2)), (2, (3, 4)), (4, (5,))]:
        plan = scheduler.schedule()
        assert isinstance(plan, api.StepPlan)
        assert isinstance(plan.step_id, int)
        assert previous_step is None or plan.step_id > previous_step
        previous_step = plan.step_id
        assert _ids(plan) == (req.request_id,)
        row = plan.requests[0]
        assert isinstance(row, api.ScheduledRequest)
        assert (row.start_pos, row.input_token_ids) == (start, tokens)
        assert req.status == ReqStatus.RUNNING
        assert req.num_computed_tokens == start
        assert req.is_prefill and not req.is_decode
        assert req.token_ids == [1, 2, 3, 4, 5]
        with pytest.raises(AttributeError):
            row.start_pos = 100
        with pytest.raises(AttributeError):
            plan.step_id = 100
        final_chunk = start == 4
        assert plan.sampling_request_ids == ((req.request_id,) if final_chunk else ())
        assert row.sampling_params == (req.sampling_params if final_chunk else None)
        scheduler.apply_result(plan=plan, sampled_token_ids=[9] if final_chunk else [])
        assert req.num_computed_tokens == start + len(tokens)
        assert req.is_prefill is (not final_chunk)
        assert req.is_decode is final_chunk
    assert req.token_ids == [1, 2, 3, 4, 5, 9]
    assert req.finish_reason == FinishReason.MAX_NEW_TOKENS
    assert scheduler.schedule() is None
    _released(manager, req)


def test_decode_and_new_prompt_share_a_step_without_moving_ownership(make_scheduler):
    scheduler, manager = make_scheduler()
    a, b = _req([1, 2]), _req([3, 4, 5], max_new_tokens=2)
    scheduler.add_request(req=a)
    first = scheduler.schedule()
    a_table = first.requests[0].block_table
    scheduler.apply_result(plan=first, sampled_token_ids=[10])
    assert a.is_decode and not a.is_prefill
    scheduler.add_request(req=b)
    waiting = _req([6])
    scheduler.add_request(req=waiting)

    mixed = scheduler.schedule()
    assert _ids(mixed) == (a.request_id, b.request_id)
    assert [(r.start_pos, r.input_token_ids) for r in mixed.requests] == [(2, (10,)), (0, (3, 4))]
    assert mixed.sampling_request_ids == (a.request_id,)
    assert b.is_prefill and not b.is_decode
    assert waiting.status == ReqStatus.WAITING, "Remaining budget does not override the active cap"
    _released(manager, waiting)
    assert mixed.requests[0].block_table[: len(a_table)] == a_table
    b_table = mixed.requests[1].block_table
    assert set(mixed.requests[0].block_table).isdisjoint(b_table)
    scheduler.apply_result(plan=mixed, sampled_token_ids=[11])
    assert (a.num_computed_tokens, b.num_computed_tokens) == (3, 2)
    assert b.token_ids == [3, 4, 5]
    scheduler.cancel_request(request_id=waiting.request_id)

    final_prompt = scheduler.schedule()
    assert _ids(final_prompt) == (a.request_id, b.request_id)
    assert final_prompt.sampling_request_ids == (a.request_id, b.request_id)
    assert [(r.start_pos, r.input_token_ids) for r in final_prompt.requests] == [(3, (11,)), (2, (5,))]
    assert b.is_prefill and not b.is_decode, "The final prompt token remains prefill until commit"
    scheduler.apply_result(plan=final_prompt, sampled_token_ids=[12, 20])
    assert a.token_ids == [1, 2, 10, 11, 12]
    assert b.token_ids == [3, 4, 5, 20]
    assert a.finished and not b.finished
    assert a.is_decode and b.is_decode, "Phase depends on progress, not terminal status"
    _released(manager, a)

    b_only = scheduler.schedule()
    assert _ids(b_only) == (b.request_id,)
    assert (b_only.requests[0].start_pos, b_only.requests[0].input_token_ids) == (3, (20,))
    assert b_only.requests[0].block_table[: len(b_table)] == b_table
    scheduler.apply_result(plan=b_only, sampled_token_ids=[21])
    assert b.finished and b.num_computed_tokens == 4


def test_token_budget_omits_requests_without_zero_work_rows(make_scheduler):
    scheduler, manager = make_scheduler(max_step_input_tokens=3, max_active_requests=3)
    a, b, c = _req([1, 2, 3, 4, 5]), _req([6]), _req([7])
    waiting = _req([8])
    for req in (a, b, c, waiting):
        scheduler.add_request(req=req)
    first = scheduler.schedule()
    assert _ids(first) == (a.request_id, b.request_id)
    assert [row.input_token_ids for row in first.requests] == [(1, 2), (6,)]
    assert first.sampling_request_ids == (b.request_id,)
    assert c.status == waiting.status == ReqStatus.WAITING
    _released(manager, c)
    _released(manager, waiting)
    assert manager.available_block_count == 10, "Only A/B reserve lifetime credits"
    assert c.num_computed_tokens == 0
    scheduler.apply_result(plan=first, sampled_token_ids=[20])

    second = scheduler.schedule()
    assert _ids(second) == (a.request_id, b.request_id), "Existing work keeps admission order"
    assert [row.input_token_ids for row in second.requests] == [(3, 4), (20,)]
    assert second.sampling_request_ids == (b.request_id,)
    assert c.num_computed_tokens == 0
    assert c.status == waiting.status == ReqStatus.WAITING
    _released(manager, c)
    assert manager.available_block_count == 10
    scheduler.apply_result(plan=second, sampled_token_ids=[21])
    third = scheduler.schedule()
    assert _ids(third) == (a.request_id, b.request_id, c.request_id)
    assert [row.input_token_ids for row in third.requests] == [(5,), (21,), (7,)]
    assert third.sampling_request_ids == (a.request_id, b.request_id, c.request_id)
    assert c.status == ReqStatus.RUNNING and waiting.status == ReqStatus.WAITING
    assert manager.available_block_count == 8
    _released(manager, waiting)
    scheduler.apply_result(plan=third, sampled_token_ids=[10, 22, 30])
    assert a.token_ids == [1, 2, 3, 4, 5, 10]
    assert b.token_ids == [6, 20, 21, 22] and c.token_ids == [7, 30]


def test_fifo_reservations_allow_growth_and_admit_after_release(make_scheduler):
    scheduler, manager = make_scheduler(
        num_blocks=3,
        max_step_input_tokens=3,
        max_active_requests=3,
        max_prefill_chunk_tokens=3,
        max_context_tokens=8,
    )
    a, b, c = (_req([1, 2], max_new_tokens=2), _req([3, 4], max_new_tokens=2), _req([5], max_new_tokens=1))
    for req in (a, b, c):
        scheduler.add_request(req=req)
    first = scheduler.schedule()
    assert _ids(first) == (a.request_id,)
    assert b.status == c.status == ReqStatus.WAITING
    assert manager.available_block_count == 1
    assert manager.free_block_count == 2
    first_table = first.requests[0].block_table
    scheduler.apply_result(plan=first, sampled_token_ids=[10])
    growth = scheduler.schedule()
    assert _ids(growth) == (a.request_id,)
    assert len(growth.requests[0].block_table) == 2
    assert growth.requests[0].block_table[: len(first_table)] == first_table
    assert manager.free_block_count == manager.available_block_count == 1
    scheduler.apply_result(plan=growth, sampled_token_ids=[11])
    assert manager.free_block_count == manager.available_block_count == 3

    admitted = scheduler.schedule()
    assert _ids(admitted) == (b.request_id, c.request_id)
    assert admitted.sampling_request_ids == (b.request_id, c.request_id)
    assert set(admitted.requests[0].block_table).isdisjoint(admitted.requests[1].block_table)
    assert manager.available_block_count == 0
    scheduler.apply_result(plan=admitted, sampled_token_ids=[20, 30])
    scheduler.cancel_request(request_id=b.request_id)
    assert manager.free_block_count == manager.available_block_count == 3


def test_prompt_eos_finishes_without_reserving(make_scheduler):
    scheduler, manager = make_scheduler(num_blocks=1)
    req = _req([1, 31], eos_token_id=31)
    scheduler.add_request(req=req)
    assert req.finished and req.finish_reason == FinishReason.EOS
    assert req.num_computed_tokens == 0 and req.token_ids == [1, 31]
    assert scheduler.schedule() is None
    assert manager.free_block_count == manager.available_block_count == 1
    _released(manager, req)
    before = _state(req)
    with pytest.raises(ValueError):
        scheduler.add_request(req=req)
    assert _state(req) == before
    assert manager.free_block_count == manager.available_block_count == 1
    _released(manager, req)
    assert scheduler.schedule() is None


def test_boundary_cancellation_releases_active_and_waiting_requests(make_scheduler):
    scheduler, manager = make_scheduler(num_blocks=2, max_context_tokens=8)
    a, b = _req([1, 2], max_new_tokens=2), _req([3, 4], max_new_tokens=2)
    scheduler.add_request(req=a)
    scheduler.add_request(req=b)
    first = scheduler.schedule()
    scheduler.apply_result(plan=first, sampled_token_ids=[10])
    assert a.status == ReqStatus.RUNNING and b.status == ReqStatus.WAITING
    before = (tuple(a.token_ids), a.num_computed_tokens)
    scheduler.cancel_request(request_id=b.request_id)
    assert b.finished and b.finish_reason == FinishReason.CANCELLED
    scheduler.cancel_request(request_id=a.request_id)
    assert a.finished and a.finish_reason == FinishReason.CANCELLED
    assert (tuple(a.token_ids), a.num_computed_tokens) == before
    assert manager.free_block_count == manager.available_block_count == 2
    assert scheduler.schedule() is None
    _released(manager, a)
    _released(manager, b)


def test_stopping_precedence_and_exact_context_boundary(make_scheduler):
    cases = [
        (5, None, FinishReason.MAX_CONTEXT_CAPACITY),
        (1, None, FinishReason.MAX_NEW_TOKENS),
        (1, 7, FinishReason.EOS),
    ]
    for max_new, eos, reason in cases:
        scheduler, manager = make_scheduler(
            num_blocks=2,
            max_prefill_chunk_tokens=3,
            max_context_tokens=4,
        )
        req = _req([1, 2, 3], max_new_tokens=max_new, eos_token_id=eos)
        scheduler.add_request(req=req)
        plan = scheduler.schedule()
        scheduler.apply_result(plan=plan, sampled_token_ids=[7])
        assert req.finished and req.finish_reason == reason
        assert req.token_ids == [1, 2, 3, 7] and req.num_computed_tokens == 3
        assert manager.free_block_count == manager.available_block_count == 2
        assert scheduler.schedule() is None


def test_pending_step_and_result_cardinality_are_transactional(make_scheduler):
    scheduler, manager = make_scheduler(num_blocks=4, max_step_input_tokens=2)
    a, b = _req([1], max_new_tokens=2), _req([2], max_new_tokens=2)
    scheduler.add_request(req=a)
    scheduler.add_request(req=b)
    plan = scheduler.schedule()
    assert plan.sampling_request_ids == (a.request_id, b.request_id)
    before = _state(a, b)
    counts = (manager.free_block_count, manager.available_block_count)
    with pytest.raises(ValueError):
        scheduler.apply_result(plan=plan, sampled_token_ids=[10])
    with pytest.raises(RuntimeError):
        scheduler.schedule()
    with pytest.raises(RuntimeError):
        scheduler.add_request(req=_req([3]))
    with pytest.raises(RuntimeError):
        scheduler.cancel_request(request_id=a.request_id)
    assert _state(a, b) == before
    assert (manager.free_block_count, manager.available_block_count) == counts
    scheduler.apply_result(plan=plan, sampled_token_ids=[10, 20])
    assert a.token_ids == [1, 10] and b.token_ids == [2, 20]
    assert a.num_computed_tokens == b.num_computed_tokens == 1
    with pytest.raises(RuntimeError):
        scheduler.apply_result(plan=plan, sampled_token_ids=[10, 20])
    next_plan = scheduler.schedule()
    assert next_plan.step_id > plan.step_id
    before = _state(a, b)
    with pytest.raises(RuntimeError):
        scheduler.apply_result(plan=plan, sampled_token_ids=[10, 20])
    assert _state(a, b) == before
    scheduler.apply_result(plan=next_plan, sampled_token_ids=[11, 21])
    assert a.finished and b.finished


def test_abort_releases_all_owned_requests_without_committing(make_scheduler):
    scheduler, manager = make_scheduler(num_blocks=2, max_context_tokens=8)
    a, b = _req([1, 2], max_new_tokens=2), _req([3, 4], max_new_tokens=2)
    scheduler.add_request(req=a)
    scheduler.add_request(req=b)
    first = scheduler.schedule()
    scheduler.apply_result(plan=first, sampled_token_ids=[10])
    failed = scheduler.schedule()
    before = [(tuple(req.token_ids), req.num_computed_tokens) for req in (a, b)]
    scheduler.abort_all_requests(plan=failed)
    assert [(tuple(req.token_ids), req.num_computed_tokens) for req in (a, b)] == before
    assert all(req.finished and req.finish_reason == FinishReason.ERROR for req in (a, b))
    assert manager.free_block_count == manager.available_block_count == 2
    _released(manager, a)
    _released(manager, b)
    assert scheduler.schedule() is None
    with pytest.raises(RuntimeError):
        scheduler.apply_result(plan=failed, sampled_token_ids=[11])


@pytest.mark.parametrize("eos_token_id", [None, 31], ids=["ordinary-prompt", "eos-prompt"])
def test_prompt_at_context_limit_is_rejected_even_with_sufficient_blocks(make_scheduler, eos_token_id):
    scheduler, manager = make_scheduler(num_blocks=2, max_context_tokens=4)
    req = _req([1, 2, 3, 31], max_new_tokens=1, eos_token_id=eos_token_id)
    before = _state(req)
    with pytest.raises(ValueError):
        scheduler.add_request(req=req)
    assert _state(req) == before
    assert manager.free_block_count == manager.available_block_count == 2
    _released(manager, req)
    assert scheduler.schedule() is None


def test_invalid_initial_admission_retains_nothing(make_scheduler):
    scheduler, manager = make_scheduler(num_blocks=1, max_context_tokens=4)
    invalid = [_req([1, 2, 3, 4], max_new_tokens=1), _req([1], max_new_tokens=3)]
    for req in invalid:
        before = _state(req)
        with pytest.raises(ValueError):
            scheduler.add_request(req=req)
        assert _state(req) == before
        _released(manager, req)
    assert scheduler.schedule() is None
    assert manager.free_block_count == manager.available_block_count == 1
    # Rejected submissions must not retain their ID. Repair the rejected policy
    # and submit the same request, rather than hiding leaked identity with a new ID.
    valid = invalid[1]
    valid.sampling_params = SamplingParams(max_new_tokens=1)
    scheduler.add_request(req=valid)
    plan = scheduler.schedule()
    assert _ids(plan) == (valid.request_id,)
    scheduler.apply_result(plan=plan, sampled_token_ids=[2])
    assert valid.finished
