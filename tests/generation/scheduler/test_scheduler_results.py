"""Commit Scheduler results through public submission, scheduling and apply APIs.

apply_result(*, plan, sampled_token_ids) requires the exact pending plan object.
Reject absent/stale/replayed/forged plans with RuntimeError and wrong sample counts
with ValueError, before any mutation. Corrected results may retry the pending plan.
Commit every planned input count, append only eligible outputs in plan order, and
stop with precedence EOS > output-token limit > context limit. Final sampled tokens
remain uncomputed; completed requests release pages/credits and active capacity.
Successful commits allow another schedule with a strictly larger execution ID.
No private containers, unfinished cancellation or abort APIs are exercised here.
"""

import importlib
from types import SimpleNamespace

import pytest

from cs336_basics.config import SamplingParams
from cs336_basics.generation.requests import Req, ReqStatus
from cs336_basics.generation.results import FinishReason


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
            pytest.fail(f"Result-commit subtask is not implemented: missing {exc.name}", pytrace=False)
    manager_module, scheduler_module = modules

    def make(*, num_blocks=12, token_budget=4, active_cap=3, prefill_chunk=2, context_limit=16):
        manager = manager_module.BlockManager(num_blocks=num_blocks, block_size=2)
        scheduler = scheduler_module.Scheduler(
            block_manager=manager,
            max_step_input_tokens=token_budget,
            max_active_requests=active_cap,
            max_prefill_chunk_tokens=prefill_chunk,
            max_context_tokens=context_limit,
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


def _snapshot(context):
    tables = []
    for req in context.requests:
        try:
            tables.append(context.manager.get_block_table(request_id=req.request_id))
        except KeyError:
            tables.append(None)
    return (
        tuple(
            (
                req.request_id,
                tuple(req.token_ids),
                req.num_computed_tokens,
                req.status,
                req.finish_reason,
                req.prompt_tokens_count,
            )
            for req in context.requests
        ),
        tuple(tables),
        context.manager.free_block_count,
        context.manager.available_block_count,
    )


def _released(context, req):
    with pytest.raises(KeyError):
        context.manager.get_block_table(request_id=req.request_id)


def test_chunked_commits_advance_inputs_without_computing_the_new_sample(scheduler_factory):
    context = scheduler_factory()
    req = _submit(context, [1, 2, 3, 4, 5], max_new_tokens=2)
    previous_id = None
    for inputs, samples, computed, history in (
        ((1, 2), [], 2, [1, 2, 3, 4, 5]),
        ((3, 4), [], 4, [1, 2, 3, 4, 5]),
        ((5,), [9], 5, [1, 2, 3, 4, 5, 9]),
        ((9,), [10], 6, [1, 2, 3, 4, 5, 9, 10]),
    ):
        # breakpoint()
        plan = context.scheduler.schedule()
        assert plan is not None
        assert previous_id is None or plan.step_id > previous_id
        previous_id = plan.step_id
        assert [item.input_token_ids for item in plan.requests] == [inputs]
        assert plan.sampling_request_ids == ((req.request_id,) if samples else ())
        context.scheduler.apply_result(plan=plan, sampled_token_ids=samples)
        assert req.num_computed_tokens == computed
        assert req.token_ids == history
    assert req.finished and req.finish_reason is FinishReason.MAX_NEW_TOKENS
    assert req.num_uncomputed_tokens == 1
    _released(context, req)
    assert context.manager.free_block_count == context.manager.available_block_count == 12
    assert context.scheduler.schedule() is None


def test_sparse_samples_map_by_plan_order_and_completion_makes_room_for_arrival(scheduler_factory):
    context = scheduler_factory()
    a = _submit(context, [1, 2, 3, 4, 5])
    b = _submit(context, [6], max_new_tokens=1)
    c = _submit(context, [7])
    first = context.scheduler.schedule()
    assert [item.request_id for item in first.requests] == [a.request_id, b.request_id, c.request_id]
    assert first.sampling_request_ids == (b.request_id, c.request_id)
    c_table = context.manager.get_block_table(request_id=c.request_id)
    context.scheduler.apply_result(plan=first, sampled_token_ids=[20, 30])
    assert (a.num_computed_tokens, b.num_computed_tokens, c.num_computed_tokens) == (2, 1, 1)
    assert a.token_ids == [1, 2, 3, 4, 5]
    assert b.token_ids == [6, 20] and c.token_ids == [7, 30]
    assert b.finished and b.finish_reason is FinishReason.MAX_NEW_TOKENS
    assert a.status == c.status == ReqStatus.RUNNING
    _released(context, b)
    assert context.manager.free_block_count == 10
    assert context.manager.available_block_count == 6

    d = _submit(context, [8], max_new_tokens=1)
    second = context.scheduler.schedule()
    assert second.step_id > first.step_id
    assert [item.request_id for item in second.requests] == [a.request_id, c.request_id, d.request_id]
    assert [item.input_token_ids for item in second.requests] == [(3, 4), (30,), (8,)]
    assert second.sampling_request_ids == (c.request_id, d.request_id)
    assert second.requests[1].block_table[: len(c_table)] == c_table
    context.scheduler.apply_result(plan=second, sampled_token_ids=[31, 40])
    assert a.num_computed_tokens == 4 and a.token_ids == [1, 2, 3, 4, 5]
    assert c.num_computed_tokens == 2 and c.token_ids == [7, 30, 31]
    assert d.num_computed_tokens == 1 and d.token_ids == [8, 40]
    assert d.finished and not c.finished
    _released(context, d)
    assert context.manager.free_block_count == 9
    assert context.manager.available_block_count == 6


@pytest.mark.parametrize(
    "eos_token_id,max_new_tokens,reason",
    [(7, 1, FinishReason.EOS), (None, 1, FinishReason.MAX_NEW_TOKENS), (None, 5, FinishReason.MAX_CONTEXT_CAPACITY)],
    ids=["eos-over-both-limits", "output-limit-over-context", "context-only"],
)
def test_stop_precedence_keeps_final_output_and_releases_all_ownership(
    scheduler_factory,
    eos_token_id,
    max_new_tokens,
    reason,
):
    context = scheduler_factory(num_blocks=2, prefill_chunk=3, context_limit=4)
    req = _submit(context, [1, 2, 3], max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)
    plan = context.scheduler.schedule()
    assert plan.sampling_request_ids == (req.request_id,)
    context.scheduler.apply_result(plan=plan, sampled_token_ids=[7])
    assert req.finished and req.finish_reason is reason
    assert req.token_ids == [1, 2, 3, 7]
    assert req.num_computed_tokens == 3 and req.num_uncomputed_tokens == 1
    _released(context, req)
    assert context.manager.free_block_count == context.manager.available_block_count == 2
    assert context.scheduler.schedule() is None


def test_wrong_sample_counts_preserve_partial_and_eligible_requests_for_retry(scheduler_factory):
    context = scheduler_factory(token_budget=3, active_cap=2)
    a, b = _submit(context, [1, 2, 3]), _submit(context, [4])
    plan = context.scheduler.schedule()
    assert plan.sampling_request_ids == (b.request_id,)
    before = _snapshot(context)
    for bad_samples in ([], [20, 21]):
        with pytest.raises(ValueError):
            context.scheduler.apply_result(plan=plan, sampled_token_ids=bad_samples)
        assert _snapshot(context) == before
    context.scheduler.apply_result(plan=plan, sampled_token_ids=[20])
    assert a.num_computed_tokens == 2 and a.token_ids == [1, 2, 3]
    assert b.num_computed_tokens == 1 and b.token_ids == [4, 20]
    assert context.scheduler.schedule().step_id > plan.step_id


def test_forged_replayed_and_stale_plans_cannot_commit_or_discard_pending_work(scheduler_factory):
    context = scheduler_factory()
    req = _submit(context, [1])
    first = context.scheduler.schedule()
    forged = type(first)(step_id=first.step_id, requests=first.requests)
    assert forged is not first
    before = _snapshot(context)
    with pytest.raises(RuntimeError):
        context.scheduler.apply_result(plan=forged, sampled_token_ids=[7])
    assert _snapshot(context) == before
    context.scheduler.apply_result(plan=first, sampled_token_ids=[7])
    assert req.token_ids == [1, 7] and req.num_computed_tokens == 1

    # No plan is pending now: replay cannot append or free anything twice.
    before = _snapshot(context)
    with pytest.raises(RuntimeError):
        context.scheduler.apply_result(plan=first, sampled_token_ids=[7])
    assert _snapshot(context) == before
    second = context.scheduler.schedule()
    assert second.step_id > first.step_id
    before = _snapshot(context)
    with pytest.raises(RuntimeError):
        context.scheduler.apply_result(plan=first, sampled_token_ids=[99])
    assert _snapshot(context) == before
    context.scheduler.apply_result(plan=second, sampled_token_ids=[8])
    assert req.token_ids == [1, 7, 8] and req.num_computed_tokens == 2
