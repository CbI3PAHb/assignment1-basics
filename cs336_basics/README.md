```
cs336_basics/
├── __init__.py
├── cli.py                  # parse_args() и main()
└── training/
    ├── __init__.py
    ├── config/
    │   ├── __init__.py
    │   ├── model.py        # существующие классы
    │   ├── optimizer.py
    │   ├── experiment.py
    │   ├── ...
    │   └── io.py           # load_experiment(), dump_experiment()
    ├── build.py            # построение рабочих компонентов
    ├── run.py              # run_training(config)
    └── trainer.py          # цикл обучения
```

## For train:
```bash
uv run cs336-basics train run --config experiment.yaml
uv run cs336-basics train run --config /home/parii-artem/wip/EFDL/assignment1-basics/cs336_basics/training/training_presets/train_small_adamw.yaml
uv run cs336-basics train run \
    --config /home/parii-artem/wip/EFDL/assignment1-basics/cs336_basics/training/training_presets/train_small_adamw.yaml \
    > owt_train.stdout.log \
    2> owt_train.stderr.log
```
```bash
uv run cs336-basics train run \
    --config /home/parii-artem/wip/EFDL/assignment1-basics/cs336_basics/training/training_presets/train_small_adamw.yaml 
rm -r /home/parii-artem/wip/EFDL/assignment1-basics/artifacts/training/train_small_adamw
```

#### For dry-run - only check training config:
```bash
uv run cs336-basics train config --config experiment.yaml
```

## For inference

### Mapper

#### For dry-run - only check mapper config:

```bash
uv run cs336-basics inference mapper config \
    --config cs336_basics/generation/presets/mapper.yaml
```

#### run

```bash
uv run cs336-basics inference mapper run \
    --config cs336_basics/generation/presets/mapper.yaml
```
```bash
uv run cs336-basics inference mapper run \
    --config cs336_basics/generation/presets/mapper.yaml \
    > inference_mapper.stdout.log \
    2> inference_mapper.stderr.log
```

### Server

#### For dry-run - only check server config:

```bash
uv run cs336-basics inference server config \
    --config cs336_basics/generation/presets/server.yaml
```

#### run

```bash
uv run cs336-basics inference server run \
    --config cs336_basics/generation/presets/server.yaml
```
```bash
uv run cs336-basics inference server run \
    --config cs336_basics/generation/presets/server.yaml \
    > inference_server.stdout.log \
    2> inference_server.stderr.log
```