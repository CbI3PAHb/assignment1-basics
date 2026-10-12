Presets contain one complete experiment configuration per YAML file.

The default `CheckpointConfig` disables saving. Enable it explicitly and choose
at least one trigger:

| Mode | `save_checkpoint` | `save_checkpoint_interval` | `save_checkpoint_at_the_end` |
| --- | --- | --- | --- |
| Disabled | `false` | `null` | `false` |
| Periodic | `true` | Positive integer | `false` |
| Final only | `true` | `null` | `true` |
| Periodic and final | `true` | Positive integer | `true` |

The interval counts completed optimizer updates, including gradient accumulation.

`train_small_adamw.yaml` saves a checkpoint after successful completion.
If the last update was already saved periodically, final saving does not write
the same checkpoint twice. Restoring `resume_from` or `init_weights_from` is
independent of whether saving is enabled.
