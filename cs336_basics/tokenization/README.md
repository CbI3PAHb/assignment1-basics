# Tokenization

- [bpe/](bpe/README.md): vocabulary and merge training.
- [tokenizer.py](tokenizer.py): `Tokenizer.from_pickle(path)`, `encode`, `encode_iterable`, `decode`.
- [encode_dataset.py](encode_dataset.py): token files for model training.

From the repository root, after downloading the data described in the
[assignment README](../../README.md), train a tokenizer:

```sh
uv run python -m cs336_basics.tokenization.bpe \
  --dataset-file-path data/TinyStoriesV2-GPT4-train.txt \
  --save-file-path artifacts/tokenizers/tinystories_train/tokenizer.pkl \
  --vocab-size 10000 --n-process 8
```

- Pickle: vocabulary, merges, special tokens; existing artifacts are atomically replaced.
- Default special token: `<|endoftext|>`; repeat `--special-token` to override.
- Logs: stdout/stderr; no JSONL output.

Encode a dataset with the trained tokenizer:

```sh
uv run python -m cs336_basics.tokenization.encode_dataset \
  --input-path data/TinyStoriesV2-GPT4-train.txt \
  --output-path tokenized_datasets/tinystories_train.bin \
  --tokenizer-file-path artifacts/tokenizers/tinystories_train/tokenizer.pkl
```

- Output: raw `uint16` IDs; requires an unused output path and IDs that fit in `uint16`.
- `--document-separator`: must be a special token; defaults to `<|endoftext|>`.
- Use the same tokenizer for training and validation data.

```sh
uv run pytest tests/tokenization -q
```

Known stale helpers / TODO:

- Fix [BPE scripts](bpe/train_bpe_scripts/): all four pass removed `--json-log-file` / `--log-file`
  options and fail parsing. Use the direct CLI above.
- [plot_bpe_stats.py](plot_bpe_stats.py) needs historical JSONL absent from current training:
  restore statistics export or retire the helper.
