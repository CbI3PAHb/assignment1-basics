#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

dataset_path="${repo_root}/data/owt_train.txt"
output_dir="${OUTPUT_DIR:-${repo_root}/artifacts/tokenizers/owt_train}"
n_process="${N_PROCESS:-16}"
vocab_size="${VOCAB_SIZE:-32000}"
heap_after="${HEAP_AFTER:-500}"

cd -- "$repo_root"

exec uv run python -m cs336_basics.tokenization.bpe \
    --dataset-file-path "$dataset_path" \
    --save-file-path "${output_dir}/tokenizer.pkl" \
    --json-log-file "${output_dir}/training.jsonl" \
    --log-file "${output_dir}/training.log" \
    --vocab-size "${vocab_size}" \
    --n-process "${n_process}" \
    --chunk-size $((16 * 1024 * 1024)) \
    --heap-after "${heap_after}" \
    "$@"
