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

For train:
```bash
uv run cs336-basics train run --config experiment.yaml
uv run cs336-basics train run --config /home/parii-artem/wip/EFDL/assignment1-basics/cs336_basics/training/training_presets/train_small_adamw.yaml
```

For dry-run - only check training config:
```bash
uv run cs336-basics train config --config experiment.yaml
```