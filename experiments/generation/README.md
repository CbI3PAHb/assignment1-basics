# Generation experiment

- [engine_config.yaml](engine_config.yaml): small model, context 128, checkpoint from [train_small_adamw.yaml](../train/train_small_adamw.yaml); CUDA by default.
- [requests.jsonl](requests.jsonl): five prompts with per-request sampling settings; omitted fields use `SamplingParams` defaults.
- Supply the matching checkpoint and tokenizer under `artifacts/`; these files are not included in Git.
- Run from the repository root:

```sh
uv run python -m cs336_basics.generation.demo \
  --engine-config-path experiments/generation/engine_config.yaml \
  --prompts-file-path experiments/generation/requests.jsonl \
  --tokenizer-pickle-path artifacts/tokenizers/owt_train/tokenizer.pkl
```

- Output: one JSON object per request containing `prompt`, generated-only `text`, and `finish_reason`.
- Page capacity and scheduler limits remain CLI flags; see `--help`.
- To use CPU, add `runtime: {device: cpu}` to the engine config.
