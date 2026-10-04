#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

dataset_path="${repo_root}/data/TinyStoriesV2-GPT4-valid.txt"
output_dir="${repo_root}/artifacts/tokenizers/tinystories_valid"

cd -- "$repo_root"

exec uv run python -m cs336_basics.tokenization.bpe \
    --dataset-file-path "$dataset_path" \
    --save-file-path "${output_dir}/tokenizer.pkl" \
    --json-log-file "${output_dir}/training.jsonl" \
    --log-file "${output_dir}/training.log" \
    --vocab-size 32000 \
    --n-process 8 \
    --chunk-size $((1 * 1024 * 1024)) \
    "$@"
