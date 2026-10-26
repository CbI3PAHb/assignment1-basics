# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## TODO
- GPU tests !!!!!!
- logging everywhere !!!!!!
- [x] write application logs to stdout/stderr
- [x] create some random-temp-dir if output_dir has not been received explicitly
- [x] add validataion dataset
- add validataion metrics (perplexity?)
- compute and log train/eval metrics each n-steps
- [x] save checkpoint after train (by option mb?)
- do smth with `.../transformers.py`, its too big
- do smth with `.../training/config/...`, folder consists of too many files. It can be refactored

## Setup

### Logging

Application logs go to stdout (INFO and DEBUG) and stderr (WARNING and above).
The default level is INFO; set `CS336_LOG_LEVEL=DEBUG` to include debug messages.
Training YAML supports `logging: {silent: true}` to disable the application's
stream handlers. Handlers installed by a host application are preserved.

The application does not create log files. Capture streams using the shell or
your job runner; choose paths outside the new training output directory:

```sh
uv run cs336-basics train run --config experiment.yaml >train.stdout.log 2>train.stderr.log
```

These streams contain formatted application messages. They are not a dedicated
JSON metrics stream; structured metric export is a separate concern.

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Format staged files before committing

Enable the versioned hook once per clone (already enabled in this checkout):

```sh
git config --local core.hooksPath .githooks
```

On `git commit`, `.githooks/pre-commit` runs pre-commit 4.7.0 through uv's isolated
tool environment. The first run downloads that small tool environment. Keep `uv`
on PATH, including when committing from an editor.

The hooks in `.pre-commit-config.yaml` run `uv run --no-sync isort` followed by
`uv run --no-sync black` on staged `.py` and `.pyi` files. Both use line length
120, and isort uses its Black-compatible profile. `--no-sync` uses the existing
project environment; install project dependencies with `uv sync` during setup.

If either formatter changes a file or fails, the commit is blocked. Hooks never
stage their changes. Review and stage the fixes yourself, then retry:

```sh
git add path/to/file.py
git commit -m "Implement block manager"
# If formatting changed the file, the commit above is blocked.
git diff -- path/to/file.py
git add path/to/file.py
git commit -m "Implement block manager"
```

Partially staged files are checked using their staged contents: pre-commit
temporarily saves unstaged changes and restores them afterward. If restoring
those changes conflicts with formatting, it rolls back the formatter changes
and blocks the commit; review the file and resolve staging manually.

Git does not copy local configuration to new clones, so repeat the activation
command there. See the [pre-commit documentation](https://pre-commit.com/#pre-commit)
for staged-file handling.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

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

