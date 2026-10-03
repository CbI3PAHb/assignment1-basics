#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

dataset_path="${repo_root}/data/owt_valid.txt"
output_dir="${repo_root}/data/tokenizers/owt_valid"

cd -- "$repo_root"

exec uv run python -m cs336_basics.tokenization.train_bpe \
    --dataset-file-path "$dataset_path" \
    --save-file-path "${output_dir}/tokenizer.pkl" \
    --json-log-file "${output_dir}/training.jsonl" \
    --log-file "${output_dir}/training.log" \
    --vocab-size 32000 \
    --n-process 8 \
    --chunk-size $((16 * 1024 * 1024)) \
    "$@"
