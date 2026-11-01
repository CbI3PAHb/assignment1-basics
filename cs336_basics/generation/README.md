# Generation

- [sampling.py](sampling.py): greedy, temperature, top-k/top-p. `prepare_sampling_args(policies=...)` accepts policies for selected logit rows.
- [requests.py](requests.py), [results.py](results.py): request histories/progress, lifecycle and result types.
- [BlockManager](kv_cache/block_manager.py): request block tables, reservation credits and lazy page allocation/release.
- [Scheduler](scheduler.py): one-step planning, result commits, stopping, cancellation and abort; CPU gate **55 passed**.
- [Paged storage](../transformers/paged_kv.py), [attention](../transformers/paged_attention.py) and model integration: **10 + 6 + 8 CPU/CUDA cases passed**.
- [Engine](engine.py): packed execution and eligible sampling; latest related regression **76 passed**.
- [run_step](generate.py): one schedule/execute/readback/apply cycle with failure cleanup; **8 passed**.
- Old serving loop, request `Batch`, demos and mapper/server CLI were removed. There is no end-to-end generation entrypoint yet.

Usage:

- Submit requests with `Scheduler.add_request(req=...)` and obtain one `StepPlan` with `schedule()`.
- Completed samples must follow `plan.sampling_request_ids`; commit them with `apply_result(plan=..., sampled_token_ids=...)`.
- Drive one step with `run_step(engine=..., scheduler=..., kv_store=...)`; it returns False when idle. [Contract](../../docs/learning/PLAN.md).

TODO:

- Implement `generate_offline`: submit requests, drive steps and return results in submission order.
- Add a packed checkpoint/tokenizer demo; verify integrated continuous arrivals.
