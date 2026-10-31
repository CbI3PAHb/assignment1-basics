# Configuration

- Public API: frozen, keyword-only classes and `load_config` / `dump_config` from `cs336_basics.config`.
- Training YAML: `load_config(Path("experiment.yaml"), ExperimentConfig)`.
- Serialization: `dump_config(config)` returns YAML including defaults.
- Fields: [training.py](training.py), [model.py](model.py), [execution.py](execution.py), [tokenizer.py](tokenizer.py), [inference.py](inference.py).
- Validation: rejects duplicate keys, unknown fields, invalid scalar types and incompatible settings.
- Loading opens no datasets/checkpoints and creates no run directories; relative paths use the working directory.
- Inference configuration: `EngineConfig` and `SamplingParams`; the CPU Scheduler takes explicit limits.
- TODO: paged-storage and packed-execution settings during integration.
- See [training presets](../training/training_presets/README.md) and [configuration tests](../../tests/configs/README.md).
