# CS336 Spring 2025 Assignment 1: Basics

- Assignment: [handout](./cs336_spring2025_assignment1_basics.pdf).
- Report handout/code issues through GitHub issues or pull requests.
- Implemented: tokenizer, model, training and CPU Scheduler.
- Unfinished: paged GPU serving integration.
- Links: [usage](cs336_basics/README.md), [generation status](cs336_basics/generation/README.md),
  [learning plan](docs/learning/PLAN.md).

## Setup

### Environment

- Install [uv](https://github.com/astral-sh/uv); [project guide](https://docs.astral.sh/uv/guides/projects/#managing-dependencies).
- Run from the repository root:

```sh
uv sync
uv run cs336-basics --help
```

- Logs: DEBUG/INFO to stdout; WARNING and above to stderr. Default level: INFO.
- Override with `CS336_LOG_LEVEL=DEBUG`; silence through YAML `logging: {silent: true}`.
- Capture streams yourself; no log files are created.

### Format staged files before committing

Enable once per clone:

```sh
git config --local core.hooksPath .githooks
```

- Staged Python files: isort, then Black. Requires `uv` on PATH.
- Changes/errors block the commit; review and manually restage before retrying.
- No automatic `git add`. Details: [pre-commit](https://pre-commit.com/#pre-commit).

### Run unit tests

```sh
uv run pytest
```

- Assignment integration: [tests/adapters.py](tests/adapters.py), already implemented.
- Full suite: not reverified after current inference changes.
- Completed CPU Scheduler gate: [generation status](cs336_basics/generation/README.md).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## TODO

- Implement packed paged KV/attention, then connect Engine and the serving loop.
- Validate integrated CPU/CUDA behavior and measure inference performance.

