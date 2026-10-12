import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from cattrs.errors import ClassValidationError, ForbiddenExtraKeysError
from yaml.constructor import ConstructorError

from cs336_basics import cli
from cs336_basics.training.config import ConfigError, ExperimentConfig
from cs336_basics.training.config.io import load_experiment


@pytest.mark.parametrize("action", ["run", "config"])
def test_parse_args_accepts_one_yaml_config(action):
    args = cli.parse_args(["train", action, "--config", "experiment.yaml"])

    assert args.command == "train"
    assert args.action == action
    assert args.config == Path("experiment.yaml")


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["train"],
        ["train", "--config", "experiment.yaml"],
        ["train", "run"],
        ["train", "config"],
        ["train", "run", "--config"],
        ["train", "config", "--config", "one.yaml", "--config", "two.yaml"],
        ["train", "run", "--config", "one.yaml", "--config", "one.yaml"],
        ["train", "inspect", "--config", "experiment.yaml"],
        ["train", "run", "--config", "experiment.yaml", "--max-steps", "10"],
        ["train", "config", "--config", "experiment.yaml", "--device", "cpu"],
        ["train", "run", "--config", "experiment.yaml", "--unknown"],
        ["train", "config", "--config", "one.yaml", "two.yaml"],
    ],
    ids=[
        "missing-command",
        "missing-action-and-config",
        "missing-action",
        "run-missing-config",
        "config-missing-config",
        "config-missing-value",
        "two-configs",
        "repeated-same-config",
        "bad-action",
        "no-loop-overrides",
        "no-runtime-overrides",
        "unknown-option",
        "extra-yaml-path",
    ],
)
def test_parse_args_rejects_invalid_cli(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args(argv)

    assert excinfo.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--help"], ["train", "--help"]])
def test_main_help_exits_successfully(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert "usage:" in captured.out
    assert captured.err == ""


def test_main_config_prints_complete_yaml_without_importing_training(
    experiment_path, minimal_experiment, monkeypatch, capsys
):
    module_name = "cs336_basics.training.run"
    monkeypatch.setitem(sys.modules, module_name, None)
    # Explicit argv must win over any arguments supplied by the caller's process.
    monkeypatch.setattr(sys, "argv", ["cs336-basics", "invalid-command"])

    assert cli.main(["train", "config", "--config", str(experiment_path)]) is None

    captured = capsys.readouterr()
    actual = yaml.safe_load(captured.out)
    expected = {
        "output_dir": minimal_experiment["output_dir"],
        "model": {**minimal_experiment["model"], "rope_theta": 10000.0},
        "data": {
            "train": {**minimal_experiment["data"]["train"], "dtype": "uint16"},
            "batching": {**minimal_experiment["data"]["batching"], "type": "random_sequences"},
        },
        "optimizer": {
            "lr": 1e-3,
            "type": "adamw",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
        "loop": {"max_steps": 4, "gradient_accumulation_steps": 1, "max_grad_norm": 1.0},
        "seed": 42,
        "scheduler": {"type": "cosine", "warmup_steps": 0, "min_lr": 1e-6},
        "runtime": {"device": "cuda"},
        "checkpoint": {
            "resume_from": None,
            "init_weights_from": None,
            "save_checkpoint": False,
            "save_checkpoint_interval": None,
            "save_checkpoint_at_the_end": False,
        },
    }
    assert actual == expected
    assert captured.err == ""
    assert sys.modules[module_name] is None
    assert not Path(actual["output_dir"]).exists()
    assert not Path(actual["data"]["train"]["path"]).exists()
    assert list(experiment_path.parent.iterdir()) == [experiment_path]


def test_main_run_dispatches_the_loaded_experiment(experiment_path, monkeypatch, capsys):
    expected = load_experiment(experiment_path)
    loaded = []
    dispatched = []

    def record_load(path):
        config = load_experiment(path)
        loaded.append(config)
        return config

    training_module = ModuleType("cs336_basics.training.run")
    training_module.run_training = dispatched.append
    monkeypatch.setitem(sys.modules, training_module.__name__, training_module)
    monkeypatch.setattr(cli, "load_experiment", record_load)

    assert cli.main(["train", "run", "--config", str(experiment_path)]) is None

    assert len(loaded) == len(dispatched) == 1
    assert isinstance(dispatched[0], ExperimentConfig)
    assert dispatched[0] is loaded[0]
    assert dispatched[0] == expected
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("action", ["config", "run"])
@pytest.mark.parametrize("duplicate", ["root", "nested"])
def test_main_rejects_duplicate_yaml_keys(experiment_path, monkeypatch, action, duplicate):
    monkeypatch.setitem(sys.modules, "cs336_basics.training.run", None)
    source = experiment_path.read_text(encoding="utf-8")
    if duplicate == "root":
        source += "seed: 1\nseed: 2\n"
    else:
        source = source.replace("  vocab_size: 32\n", "  vocab_size: 32\n  vocab_size: 64\n")
    experiment_path.write_text(source, encoding="utf-8")

    with pytest.raises(ConstructorError, match="duplicate key"):
        cli.main(["train", action, "--config", str(experiment_path)])


def _exception_leaves(error):
    nested = getattr(error, "exceptions", None)
    if nested is None:
        yield error
    else:
        for child in nested:
            yield from _exception_leaves(child)


@pytest.mark.parametrize("action", ["config", "run"])
@pytest.mark.parametrize("scope", ["root", "model"])
def test_main_rejects_unknown_yaml_fields(minimal_experiment, write_experiment, monkeypatch, action, scope):
    monkeypatch.setitem(sys.modules, "cs336_basics.training.run", None)
    section = minimal_experiment if scope == "root" else minimal_experiment[scope]
    section["unexpected_setting"] = 1
    path = write_experiment(minimal_experiment)

    with pytest.raises(ClassValidationError) as excinfo:
        cli.main(["train", action, "--config", str(path)])

    errors = list(_exception_leaves(excinfo.value))
    assert any(
        isinstance(error, ForbiddenExtraKeysError) and "unexpected_setting" in error.extra_fields
        for error in errors
    )


@pytest.mark.parametrize("action", ["config", "run"])
@pytest.mark.parametrize("field", ["max_steps", "micro_batch_size"])
def test_main_rejects_float_values_for_integer_fields(
    minimal_experiment, write_experiment, monkeypatch, action, field
):
    monkeypatch.setitem(sys.modules, "cs336_basics.training.run", None)
    section = minimal_experiment["loop"] if field == "max_steps" else minimal_experiment["data"]["batching"]
    section[field] = float(section[field])
    path = write_experiment(minimal_experiment)

    with pytest.raises(ClassValidationError) as excinfo:
        cli.main(["train", action, "--config", str(path)])

    errors = list(_exception_leaves(excinfo.value))
    assert any(isinstance(error, TypeError) and "integer" in str(error) for error in errors)


@pytest.mark.parametrize("action", ["config", "run"])
@pytest.mark.parametrize(
    "checkpoint",
    [
        {"save_checkpoint": False, "save_checkpoint_interval": 2},
        {"save_checkpoint": False, "save_checkpoint_at_the_end": True},
        {"save_checkpoint": True},
    ],
    ids=["disabled-with-periodic", "disabled-with-final", "enabled-without-trigger"],
)
def test_main_rejects_invalid_checkpoint_policy_before_importing_training(
    minimal_experiment, write_experiment, monkeypatch, action, checkpoint
):
    monkeypatch.setitem(sys.modules, "cs336_basics.training.run", None)
    minimal_experiment["checkpoint"] = checkpoint
    path = write_experiment(minimal_experiment)

    with pytest.raises((ClassValidationError, ConfigError)) as caught:
        cli.main(["train", action, "--config", str(path)])

    assert any(isinstance(error, ConfigError) for error in _exception_leaves(caught.value))
    assert not Path(minimal_experiment["output_dir"]).exists()
