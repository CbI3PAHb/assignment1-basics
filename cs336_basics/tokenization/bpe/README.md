# BPE training

- Active implementation used by [assignment adapters](../../../tests/adapters.py).
- [trainer.py](trainer.py): pair counts/merges; brute-force selection, then heap selection.
- [pretokenization.py](pretokenization.py): special-token boundaries, chunks, serial/multiprocess counts.
- [cli.py](cli.py): arguments, stream logging, atomic pickle output.
- [__main__.py](__main__.py): `uv run python -m cs336_basics.tokenization.bpe`.

API:

- `train_bpe(input_path, vocab_size, special_tokens, *, chunk_size=1048576, n_process=4, heap_after=300)`
  returns `(vocabulary, merges)`.
- Requires at least one nonempty unique special token, positive chunk/process counts,
  and vocabulary capacity for 256 bytes plus special tokens.
- `heap_after=0`: heap selection from the first merge.

See [tokenization usage](../README.md) for CLI examples and stale shell helpers.
Focused assignment/regression checks, from the repository root:

```sh
uv run pytest tests/tokenization/test_train_bpe.py tests/tokenization/test_train_bpe_regressions.py -q
```
