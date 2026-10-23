# Generation

Uncached autoregressive generation with a reusable model and sampler.

- Static batches with different prompt lengths and independent stopping.
- Per-request greedy, temperature, top-k and top-p sampling.
- EOS, output-budget and context-capacity limits; results preserve input order.
- Results contain the full token sequence and a finish reason.

| File | Purpose |
| --- | --- |
| `engine.py` | Model setup, batch padding, forward pass and sampling. |
| `sampling.py` | Batch sampling arguments and token selection. |
| `requests.py` | Request tokens, sampling policy, lifecycle and active batches. |
| `results.py` | Numerical model output, generation results and finish reasons. |
| `generate.py` | Request lifecycle and generation through Engine batches. |
| `one_request_generate_demo.py` | Single-prompt example. |
| `one_batch_generate_demo.py` | Multiple prompts with different sampling settings. |
| `__init__.py` | Public exports. |

Example entry points, from the repository root:

```bash
python -m cs336_basics.generation.one_request_generate_demo
python -m cs336_basics.generation.one_batch_generate_demo
```

The batch demo still needs migration to the Engine API. Both demos use these
default local artifacts:

- `artifacts/tokenizers/owt_train/tokenizer.pkl`
- `artifacts/training/train_small_adamw/config.yaml`
- `artifacts/training/train_small_adamw/checkpoint_0000000100.pt`

Run the CPU tests:

```bash
python -m pytest tests/generation -q
```
