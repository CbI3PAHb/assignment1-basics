"""One driver step owns schedule -> execute -> sampled-ID readback -> apply.

Only an idle schedule returns False. A nonempty plan returns True after commit,
including the last step that finishes all requests. Once a plan is acquired,
execution/readback/apply failures abort that exact plan and propagate unchanged.
The driver does not own request stopping rules or retry failed execution.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cs336_basics.config import SamplingParams
from cs336_basics.generation.kv_cache import BlockManager
from cs336_basics.generation.requests import Req
from cs336_basics.generation.results import FinishReason, ModelOutput
from cs336_basics.generation.scheduler import ScheduledRequest, Scheduler, StepPlan


@pytest.fixture
def run_step():
    target = "cs336_basics.generation.generate"
    try:
        module = importlib.import_module(target)
    except ModuleNotFoundError as exc:
        if exc.name != target:
            raise
        pytest.fail("Implement run_step(*, engine, scheduler, kv_store) in generation/generate.py", pytrace=False)
    function = getattr(module, "run_step", None)
    if not callable(function):
        pytest.fail("Implement run_step(*, engine, scheduler, kv_store)", pytrace=False)
    return function


def _scheduler(*, active_cap=2):
    manager = BlockManager(num_blocks=8, block_size=2)
    scheduler = Scheduler(
        block_manager=manager,
        max_step_input_tokens=3,
        max_active_requests=active_cap,
        max_prefill_chunk_tokens=2,
        max_context_tokens=16,
    )
    observed = SimpleNamespace(
        schedule=Mock(wraps=scheduler.schedule),
        apply_result=Mock(wraps=scheduler.apply_result),
        abort_all_requests=Mock(wraps=scheduler.abort_all_requests),
    )
    return scheduler, manager, observed


def _submit(scheduler, tokens, *, max_new_tokens):
    req = Req(token_ids=list(tokens), sampling_params=SamplingParams(max_new_tokens=max_new_tokens))
    scheduler.add_request(req=req)
    return req


class UnreadableSamples:
    """Boundary fault injection; accept the common readback spellings equally."""

    def __init__(self, *, count, failure):
        self.shape = (count,)
        self.failure = failure

    def numel(self):
        return self.shape[0]

    def __len__(self):
        return self.shape[0]

    def detach(self):
        return self

    def _fail(self, *args, **kwargs):
        raise self.failure

    cpu = to = tolist = numpy = item = __iter__ = __getitem__ = _fail


def test_idle_schedules_once_and_returns_false_without_other_work(run_step):
    scheduler = SimpleNamespace(schedule=Mock(return_value=None), apply_result=Mock(), abort_all_requests=Mock())
    engine = SimpleNamespace(execute=Mock())
    assert run_step(engine=engine, scheduler=scheduler, kv_store=object()) is False
    scheduler.schedule.assert_called_once_with()
    engine.execute.assert_not_called()
    scheduler.apply_result.assert_not_called()
    scheduler.abort_all_requests.assert_not_called()


def test_device_samples_reach_apply_once_as_cpu_integers_in_plan_order(run_step, execution_device):
    plan = StepPlan(
        step_id=7,
        requests=tuple(
            ScheduledRequest(
                request_id=name,
                input_token_ids=(token,),
                start_pos=0,
                block_table=(index,),
                sampling_params=policy,
            )
            for index, (name, token, policy) in enumerate(
                (("partial", 1, None), ("z", 2, SamplingParams()), ("a", 3, SamplingParams()))
            )
        ),
    )
    output = ModelOutput(next_tokens_gpu=torch.tensor([29, 17], dtype=torch.int64, device=execution_device))
    engine = SimpleNamespace(execute=Mock(return_value=output))
    scheduler = SimpleNamespace(schedule=Mock(return_value=plan), apply_result=Mock(), abort_all_requests=Mock())
    store = object()

    assert run_step(engine=engine, scheduler=scheduler, kv_store=store) is True
    scheduler.schedule.assert_called_once_with()
    engine.execute.assert_called_once_with(plan=plan, kv_store=store)
    assert engine.execute.call_args.kwargs["plan"] is plan
    scheduler.apply_result.assert_called_once()
    applied = scheduler.apply_result.call_args.kwargs
    assert applied["plan"] is plan
    assert all(type(token) is int for token in applied["sampled_token_ids"])
    assert list(applied["sampled_token_ids"]) == [29, 17]
    assert plan.sampling_request_ids == ("z", "a")
    scheduler.abort_all_requests.assert_not_called()


def test_real_scheduler_chunk_arrival_mixed_work_completion_and_idle(run_step):
    scheduler, manager, observed = _scheduler()
    a = _submit(scheduler, [11, 12, 13], max_new_tokens=3)
    store = object()
    expected = []

    def execute(*, plan, kv_store):
        assert kv_store is store
        rows, samples = expected.pop(0)
        assert [(row.request_id, row.input_token_ids, row.start_pos) for row in plan.requests] == rows
        assert len(plan.sampling_request_ids) == len(samples)
        for row in plan.requests:
            assert manager.get_block_table(request_id=row.request_id) == row.block_table
        if not samples:
            return SimpleNamespace(
                next_tokens_gpu=UnreadableSamples(count=0, failure=AssertionError("S=0 must skip readback"))
            )
        return ModelOutput(next_tokens_gpu=torch.tensor(samples, dtype=torch.int64))

    engine = SimpleNamespace(execute=Mock(side_effect=execute))

    def step(rows, samples):
        expected.append((rows, samples))
        assert run_step(engine=engine, scheduler=observed, kv_store=store) is True
        assert not expected
        assert observed.schedule.call_count == engine.execute.call_count == observed.apply_result.call_count
        plan = engine.execute.call_args.kwargs["plan"]
        applied = observed.apply_result.call_args.kwargs
        assert applied["plan"] is plan
        tokens = applied["sampled_token_ids"]
        assert all(type(token) is int for token in tokens)
        assert list(tokens) == samples
        observed.abort_all_requests.assert_not_called()

    step([(a.request_id, (11, 12), 0)], [])
    assert a.token_ids == [11, 12, 13] and a.num_computed_tokens == 2
    first_table = manager.get_block_table(request_id=a.request_id)
    assert first_table and manager.available_block_count < manager.num_blocks
    step([(a.request_id, (13,), 2)], [30])

    b = _submit(scheduler, [21, 22, 23], max_new_tokens=1)
    step([(a.request_id, (30,), 3), (b.request_id, (21, 22), 0)], [31])
    a_table = manager.get_block_table(request_id=a.request_id)
    b_table = manager.get_block_table(request_id=b.request_id)
    assert a_table[: len(first_table)] == first_table
    assert set(a_table).isdisjoint(b_table)
    assert b.token_ids == [21, 22, 23] and b.num_computed_tokens == 2

    step([(a.request_id, (31,), 4), (b.request_id, (23,), 2)], [32, 40])
    assert a.token_ids == [11, 12, 13, 30, 31, 32] and a.num_computed_tokens == 5
    assert b.token_ids == [21, 22, 23, 40] and b.num_computed_tokens == 3
    for req in (a, b):
        assert req.finished and req.finish_reason is FinishReason.MAX_NEW_TOKENS
        with pytest.raises(KeyError):
            manager.get_block_table(request_id=req.request_id)
    assert manager.free_block_count == manager.available_block_count == manager.num_blocks
    assert run_step(engine=engine, scheduler=observed, kv_store=store) is False
    assert observed.schedule.call_count == 5
    assert engine.execute.call_count == observed.apply_result.call_count == 4
    observed.abort_all_requests.assert_not_called()


@pytest.mark.parametrize("stage", ["execute", "readback", "apply"])
def test_failed_step_aborts_same_plan_and_preserves_original_exception(run_step, stage):
    scheduler, manager, observed = _scheduler(active_cap=1)
    active = _submit(scheduler, [1, 2], max_new_tokens=2)
    waiting = _submit(scheduler, [3], max_new_tokens=1)
    failure = RuntimeError(f"{stage} failed")
    samples = (
        UnreadableSamples(count=1, failure=failure) if stage == "readback" else torch.tensor([9], dtype=torch.int64)
    )
    engine = SimpleNamespace(execute=Mock(return_value=SimpleNamespace(next_tokens_gpu=samples)))
    if stage == "execute":
        engine.execute.side_effect = failure
    if stage == "apply":
        observed.apply_result.side_effect = failure
    store = object()

    with pytest.raises(RuntimeError) as caught:
        run_step(engine=engine, scheduler=observed, kv_store=store)
    assert caught.value is failure
    observed.schedule.assert_called_once_with()
    engine.execute.assert_called_once()
    plan = engine.execute.call_args.kwargs["plan"]
    assert engine.execute.call_args.kwargs["kv_store"] is store
    observed.abort_all_requests.assert_called_once_with(plan=plan)
    assert observed.abort_all_requests.call_args.kwargs["plan"] is plan
    assert observed.apply_result.call_count == (1 if stage == "apply" else 0)
    if stage == "apply":
        assert observed.apply_result.call_args.kwargs["plan"] is plan
    assert active.token_ids == [1, 2] and waiting.token_ids == [3]
    for req in (active, waiting):
        assert req.finished and req.finish_reason is FinishReason.ERROR
        assert req.num_computed_tokens == 0
        with pytest.raises(KeyError):
            manager.get_block_table(request_id=req.request_id)
    assert manager.free_block_count == manager.available_block_count == manager.num_blocks
    assert scheduler.schedule() is None


def test_schedule_failure_propagates_without_execution_apply_or_abort(run_step):
    failure = RuntimeError("schedule did not return a plan")
    scheduler = SimpleNamespace(schedule=Mock(side_effect=failure), apply_result=Mock(), abort_all_requests=Mock())
    engine = SimpleNamespace(execute=Mock())
    with pytest.raises(RuntimeError) as caught:
        run_step(engine=engine, scheduler=scheduler, kv_store=object())
    assert caught.value is failure
    scheduler.schedule.assert_called_once_with()
    engine.execute.assert_not_called()
    scheduler.apply_result.assert_not_called()
    scheduler.abort_all_requests.assert_not_called()
