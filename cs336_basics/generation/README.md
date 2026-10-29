# Generation

- [sampling.py](sampling.py): greedy, temperature, top-k/top-p; preparation still accepts `Batch`.
- [requests.py](requests.py), [results.py](results.py): histories, computed-token progress and lifecycle.
- [BlockManager](kv_cache/block_manager.py): CPU credits, request block tables, lazy block allocation/release; no tensors.
- [Scheduler](scheduler.py): running-first/FIFO admission, token/chunk limits, mixed prefill/decode plans,
  result commits, cancellation and abort. GPU driver is unfinished.
- [engine.py](engine.py), [generate.py](generate.py): obsolete dense serving path awaiting replacement.
  Known failure: `batched_generate` calls the now-property `Req.last_token()`.

CPU API:

- Import `BlockManager` and `Scheduler` from their modules; submit with `add_request(req=...)`.
- Call `schedule()`, then `apply_result(plan=..., sampled_token_ids=...)`.
- One pending plan; samples follow `plan.sampling_request_ids`. [Contract](../../docs/learning/PLAN.md).
- Learner-host CPU gate: **55 passed in 1.24s** (2026-10-29). Run from the repository root:

```sh
uv run pytest tests/generation/scheduler tests/kv_cache/test_block_manager.py -q
```

- Both `one_request_generate_demo.py` and `one_batch_generate_demo.py` use the broken old path;
  their tokenizer/checkpoint inputs live under `artifacts/`.
- Dense model/KV code remains a numerical reference; some generation tests still target legacy interfaces.

TODO:

- Add packed GPU pages and request-isolated attention, with CPU/CUDA tests first.
- Implement `Engine.execute` and `run_step`; migrate demos and retire the old loop.
- Verify continuous arrivals, mixed execution and chunked prefill end to end.
