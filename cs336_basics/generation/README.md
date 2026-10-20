# Greedy generation

The package provides uncached generation with full-prefix recomputation:

- `generate()` handles one prompt through the batched execution core and preserves
  the caller's token list.
- `batched_generate()` handles a static group of mutable `Req` objects, with
  right padding, independent stopping, and results in input order.
- The sequential runner reuses loaded model/tokenizer objects and returns an
  outcome for every prompt, including request errors.

Public token-level API:

```python
from cs336_basics.generation import Req, batched_generate, generate
```

The modules follow these responsibilities:

| Module | Responsibility |
| --- | --- |
| `requests.py` | Mutable request tokens, progress, and lifecycle state. |
| `results.py` | Immutable result snapshots and finish reasons. |
| `generate.py` | The static-batch execution loop and single-request adapter. |
| `offline_runner.py` | Sequential text requests and request error outcomes. |
| `demo.py` | Local model/tokenizer loading and an executable example. |
| `__init__.py` | Public imports. |

Run the focused CPU tests from the repository root:

```bash
python -m pytest tests/generation -q
```

The real-model demo uses these local artifacts, resolved from the repository root:

- `artifacts/tokenizers/owt_train/tokenizer.pkl`
- `artifacts/training/train_small_adamw/config.yaml`
- `artifacts/training/train_small_adamw/checkpoint_0000000100.pt`

With matching artifacts available:

```bash
python -m cs336_basics.generation.demo
```

Artifact files are ignored by Git. Adjust the demo's settings for a different
local checkpoint. The retired mapper/server execution commands are unavailable;
the existing YAML configuration inspection commands remain supported.
