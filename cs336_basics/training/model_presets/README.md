# Model presets

- YAML fields: [ModelConfig](../../config/model.py).
- Shared values: `vocab_size: 32000`, `context_length: 128`, `rope_theta: 10000.0`.

| File | `d_model` | `num_layers` | `num_heads` | `d_ff` |
| --- | --- | --- | --- | --- |
| [nano.yaml](nano.yaml) | 128 | 4 | 4 | 256 |
| [small.yaml](small.yaml) | 768 | 12 | 12 | 3072 |
| [medium.yaml](medium.yaml) | 1024 | 24 | 16 | 4096 |
| [large.yaml](large.yaml) | 1280 | 36 | 20 | 5120 |
| [xl.yaml](xl.yaml) | 1600 | 48 | 25 | 6400 |
| [2.7b.yaml](2.7b.yaml) | 2560 | 32 | 32 | 10240 |

- Copy the mapping into `model:` in a full [training configuration](../training_presets/README.md).
- CLI: one complete experiment YAML; no preset composition or selection by name.
- Context length must cover `data.batching.sequence_length`; vocabulary size must cover dataset token IDs.
- Names are labels, not parameter-count or hardware-fit guarantees.
