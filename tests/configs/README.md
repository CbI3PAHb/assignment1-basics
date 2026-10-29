# Configuration tests

- CPU tests for [configuration classes and YAML loading](../../cs336_basics/config/README.md).
- Run from the repository root:

```bash
uv run pytest tests/configs -q
```

- Field/default validation, frozen objects and training/inference compatibility.
- Strict cattrs conversion, YAML key handling and configuration round trips.
- Checkpoint policies, preserved paths and loading without runtime side effects.
- No dataset, checkpoint or CUDA device needed; model/CLI execution has separate suites.
