# BPE trainer scaffold

This package is the replacement for the monolithic
`cs336_basics/tokenization/train_bpe.py`. The old module remains active until
this scaffold is implemented and the imports are switched deliberately.

## Module boundaries

```text
python -m ...tokenization.bpe
          |
          v
        cli.py              argparse, logging, artifact output
          |
          v
     trainer.train_bpe()    public assignment-compatible facade
          |
          +--> pretokenization.count_pretokens()
          |
          +--> BPETrainer.train()
```

Do not introduce a generic `utils.py`: put functionality in the module that
owns it.

## Suggested implementation order

1. `_validate_training_arguments()`
2. `calculate_num_chunks()` and empty-file behavior
3. `_count_pretokens_in_chunk()` without multiprocessing
4. `count_pretokens()` serial path
5. `BPETrainer` using brute-force pair selection
6. make the regression tests pass in brute-force mode
7. add safe chunk boundaries and multiprocessing
8. add the heap optimization and prove it matches brute force
9. implement atomic artifact saving and CLI logging

During development, keep the existing adapter pointed at the old module. Test
the scaffold directly in focused tests. Switch the adapter and shell scripts
only after the Stanford tests and all regression tests pass.

## Test command after switching

```bash
uv run pytest -q \
  tests/tokenization/test_train_bpe.py \
  tests/tokenization/test_train_bpe_regressions.py
```
