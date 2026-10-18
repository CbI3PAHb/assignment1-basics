# Config classes

```
cs336_basics/config/
├── __init__.py       # public API
│
├── model.py          # ModelConfig
├── execution.py      # RuntimeConfig, LoggingConfig
├── tokenizer.py      # TokenizerConfig
│
├── training.py       # everything about train
├── inference.py      # everything about inference 
│
├── io.py             # load_config / dump_config, YAML and cattrs
├── _validation.py    # fields validation
└── _errors.py        # ConfigError
```