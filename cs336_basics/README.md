# Package usage

- CLI: `cs336-basics`, defined in [cli.py](cli.py). Run from the repository root.
- Tokenization: [BPE training and dataset encoding](tokenization/README.md).
- Model: [Transformer components](transformers/transformers.py), [builder](transformers/build.py).
- Training: [run.py](training/run.py), [Trainer](training/trainer.py), [presets](training/training_presets/README.md).
  Includes evaluation loss/perplexity and configurable checkpoints.
- Configuration: [typed YAML](config/README.md).
- Generation: [sampling, CPU Scheduler, unfinished GPU integration](generation/README.md).

Validate the example training YAML without starting training:

```sh
uv run cs336-basics train config --config cs336_basics/training/training_presets/train_small_adamw.yaml
```

Train after setting dataset, model and output paths in that YAML:

```sh
uv run cs336-basics train run --config cs336_basics/training/training_presets/train_small_adamw.yaml
```

- Exactly one `--config` is required.
- Inference `mapper config` / `server config`: legacy schema validation only; `run` disabled.
- Generation demos: migration pending; see [status](generation/README.md).
