# Generation

- [sampling.py](sampling.py): greedy, temperature, top-k/top-p. `prepare_sampling_args(policies=...)` accepts policies for selected logit rows.
- [requests.py](requests.py), [results.py](results.py): request histories/progress, lifecycle and result types.
- [BlockManager](kv_cache/block_manager.py): request block tables, reservation credits and lazy page allocation/release.
- [Scheduler](scheduler.py): one-step planning, result commits, stopping, cancellation and abort
- [Paged storage](../transformers/paged_kv.py), [attention](../transformers/paged_attention.py) and model integration
- [Engine](engine.py): packed execution and eligible sampling
- [run_step](generate.py): one schedule/execute/readback/apply cycle with failure cleanup
- [generate_offline](generate.py): complete requests and return full token histories in submission order
- [demo](demo.py): checkpoint/tokenizer CLI with JSONL results; learner host **4 passed in 4.58s**, including real tiny checkpoint/tokenizer CPU/CUDA equivalence, initial EOS without a forward and JSONL policy/whitespace parsing.

Usage:

- For a fixed request list, call `generate_offline(engine=..., scheduler=..., kv_store=..., requests=...)` with a fresh scheduler and matching store.
- Submit requests with `Scheduler.add_request(req=...)` and obtain one `StepPlan` with `schedule()`.
- Completed samples must follow `plan.sampling_request_ids`; commit them with `apply_result(plan=..., sampled_token_ids=...)`.
- Drive one step with `run_step(engine=..., scheduler=..., kv_store=...)`; it returns False when idle. [Contract](../../docs/learning/PLAN.md).

Run the demo from the repository root with matching checkpoint configuration and tokenizer:

```sh
uv run python -m cs336_basics.generation.demo \
  --prompts-file-path prompts.jsonl \
  --tokenizer-pickle-path tokenizer.pkl \
  --engine-config-path engine_config.yaml \
  --num-blocks 128 --block-size 8 \
  --max-step-input-tokens 32 --max-active-requests 4 --max-prefill-chunk-tokens 16
```

- Each nonblank input line is JSON: `{"prompt": "Hello", "max_new_tokens": 16}`. Flat per-request fields use `SamplingParams` defaults when omitted; prompt whitespace is preserved.
- Output is one JSON object per input in order: `prompt`, `text` (generated suffix only), `finish_reason`. Debug prints are disabled; initial EOS produces empty `text`.

TODO:

- Measure inference performance. Attention uses a Python segmented correctness reference, not an optimized paged kernel; no speedup is established.
