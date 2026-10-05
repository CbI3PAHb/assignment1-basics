# Tokenization

This directory contains the byte-pair encoding (BPE) implementation used by
Assignment 1:

- `train_bpe.py` trains a BPE vocabulary and merge table;
- `tokenizer.py` encodes and decodes text with a trained tokenizer;
- `plot_bpe_stats.py` plots statistics recorded during training;
- the dataset-specific `.sh` files explicitly define their input, output, and
  training parameters.

## Data layout

The scripts expect the assignment datasets in the repository-level `data/`
directory:

```text
assignment1-basics/
├── data/
│   ├── TinyStoriesV2-GPT4-train.txt
│   ├── TinyStoriesV2-GPT4-valid.txt
│   ├── owt_train.txt
│   └── owt_valid.txt
└── cs336_basics/
    └── tokenization/
```

`/data/` is ignored by Git because the datasets and generated tokenizers are
large local artifacts.

## Ready-to-run commands

The scripts can be launched from any working directory. They locate the
repository from the script file itself, so activating the virtual environment
or `cd`-ing into `cs336_basics/tokenization` is not required. Each script runs
the Python module through `uv run`.

From the repository root:

```bash
./cs336_basics/tokenization/bpe/train_bpe_scripts/tinystories_valid.sh
./cs336_basics/tokenization/bpe/train_bpe_scripts/tinystories_train.sh
./cs336_basics/tokenization/bpe/train_bpe_scripts/owt_valid.sh
./cs336_basics/tokenization/bpe/train_bpe_scripts/owt_train.sh
```

For a one-off custom run, call the Python module directly:

```bash
uv run python -m cs336_basics.tokenization.bpe \
  --dataset-file-path data/my_corpus.txt \
  --save-file-path data/tokenizers/my_corpus/tokenizer.pkl \
  --json-log-file data/tokenizers/my_corpus/training.jsonl \
  --log-file data/tokenizers/my_corpus/training.log \
  --vocab-size 32000 \
  --n-process 8 \
  --chunk-size 1048576
```

Each dataset-specific script writes the following files under its chosen
`data/tokenizers/<run-name>/` directory:

```text
tokenizer.pkl   # vocabulary and merge table
training.log    # human-readable debug log
training.jsonl  # per-iteration statistics
```

Running the same script again replaces these three files, so every log belongs
to one training run.

OpenWebText training is expensive. Start with `tinystories_valid` when checking
that the pipeline works.

## Adding a dataset

Copy one of the dataset-specific scripts and edit its explicit values:

```bash
cp cs336_basics/tokenization/tinystories_valid.sh \
  cs336_basics/tokenization/my_corpus_train.sh
```

The new script is self-contained; no registry or shared launcher needs to be
updated. Extra arguments are forwarded to Python, so a later repeated option
can be used for a one-off override:

```bash
./cs336_basics/tokenization/tinystories_valid.sh --vocab-size 512
```

See all Python options without starting training:

```bash
uv run python -m cs336_basics.tokenization.bpe --help
```

## Why `dirname "$0"` printed different values

`$0` preserves the spelling used to invoke a script. Consequently:

```text
bash owt_valid.sh              -> dirname "$0" is .
bash tokenization/owt_valid.sh -> dirname "$0" is tokenization
```

That does not identify an absolute directory. These scripts instead resolve
`${BASH_SOURCE[0]}` and then run `cd ... && pwd`, producing the same absolute
path regardless of the caller's current directory.

## Tests without access to OpenAI Blob Storage

The GPT-2 compatibility tests use `tiktoken` as an oracle. `tiktoken` normally
downloads two GPT-2 files, which fails in environments where that host is not
allowlisted. The repository fixtures can be used as a verified local cache:

```bash
mkdir -p .cache/tiktoken
cp -- tests/fixtures/gpt2_vocab.json \
  .cache/tiktoken/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6
{
  printf '#version: 0.2\n'
  cat tests/fixtures/gpt2_merges.txt
} > .cache/tiktoken/6d1cbeee0f20b3d9449abfede4726ed8212e3aee

TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" \
  uv run pytest tests/tokenization/test_tokenizer.py
```

This preserves TLS verification and tests the implementation against the same
GPT-2 data expected by `tiktoken`.
