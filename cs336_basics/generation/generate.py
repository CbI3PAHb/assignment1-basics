from cs336_basics.transformers.paged_kv import PagedKVStore

from .engine import Engine
from .results import ModelOutput
from .scheduler import Scheduler, StepPlan


def run_step(*, engine: Engine, scheduler: Scheduler, kv_store: PagedKVStore) -> bool:
    step_plan: StepPlan | None = scheduler.schedule()

    if step_plan is None:
        return False
    try:
        model_output: ModelOutput = engine.execute(plan=step_plan, kv_store=kv_store)
        sampled_token_ids = model_output.next_tokens_gpu.tolist() if step_plan.sampling_request_ids else []
        scheduler.apply_result(plan=step_plan, sampled_token_ids=sampled_token_ids)
    except Exception:
        scheduler.abort_all_requests(plan=step_plan)
        raise
    return True
