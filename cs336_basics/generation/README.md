# Generation

- [sampling.py](sampling.py): greedy, temperature, top-k/top-p. `prepare_sampling_args(policies=...)` accepts policies for selected logit rows.
- [requests.py](requests.py), [results.py](results.py): request histories/progress, lifecycle and result types.
- [BlockManager](kv_cache/block_manager.py): request block tables, reservation credits and lazy page allocation/release.
- [Scheduler](scheduler.py): one-step planning, result commits, stopping, cancellation and abort; CPU gate **55 passed**.
- [Paged storage](../transformers/paged_kv.py), [attention](../transformers/paged_attention.py) and model integration: **10 + 6 + 8 CPU/CUDA cases passed**.
- [Engine](engine.py): checkpoint/model initialization retained. `execute(plan=..., kv_store=...)` is the next implementation task.
- Old serving loop, request `Batch`, demos and mapper/server CLI were removed. There is no end-to-end generation entrypoint yet.

Usage:

- Submit requests with `Scheduler.add_request(req=...)` and obtain one `StepPlan` with `schedule()`.
- Completed samples must follow `plan.sampling_request_ids`; commit them with `apply_result(plan=..., sampled_token_ids=...)`.
- The remaining driver will connect schedule → execute → sample transfer → apply. [Contract](../../docs/learning/PLAN.md).

TODO:

- Implement `Engine.execute` with packed inputs and sampling of eligible final-query logits.
- Add `run_step`, an offline driver and a packed demo; test continuous arrivals and failure cleanup.
