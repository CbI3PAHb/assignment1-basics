# Training presets

- Load as [ExperimentConfig](../../cs336_basics/config/training.py); omitted fields use defaults.
- Shared settings: AdamW, cosine learning rates, CUDA, repository-relative OWT train/validation paths.

| File | Actual settings |
| --- | --- |
| [train_small_adamw.yaml](train_small_adamw.yaml) | 100 updates, sequence 128, accumulation 1; evaluation every 10 updates; final checkpoint |
| [test_launch.yaml](test_launch.yaml) | 100 updates, sequence 128, accumulation 4; evaluation disabled; checkpoint every 20 updates and at the end; default timestamped output directory |
| [train_2000_steps__2048_seq_len__xl_size.yaml](train_2000_steps__2048_seq_len__xl_size.yaml) | 2000 updates, sequence 2048, accumulation 1; evaluation every 100 updates; checkpoint saving disabled |

- `xl_size` actually uses the **small** architecture: `d_model=768`, 12 layers, 12 heads, `d_ff=3072`; output directory: `train_1000_adamw`.

- From the repository root, inspect settings before launching:

```bash
uv run python -m cs336_basics train config --config experiments/train/train_small_adamw.yaml
uv run python -m cs336_basics train run --config experiments/train/train_small_adamw.yaml
```

- Paths resolve from the repository root; tokenized datasets live under `tokenized_datasets/`, outputs under `artifacts/training/`.
- Edit paths as needed; training requires the files and requested device.
- Output directory must not already exist.
- `config` validates and prints settings without starting training.
- [CheckpointConfig](../../cs336_basics/config/training.py) validates saving/resume settings; intervals count optimizer updates.
