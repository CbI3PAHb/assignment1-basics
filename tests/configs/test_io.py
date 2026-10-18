from datetime import datetime
from pathlib import Path

import pytest
import yaml
from attrs import asdict
from cattrs.errors import ClassValidationError, ForbiddenExtraKeysError
from yaml.constructor import ConstructorError

from cs336_basics.config import (
    DEFAULT_ROPE_THETA,
    BatchingConfig,
    CheckpointConfig,
    ConfigError,
    DataConfig,
    DatasetConfig,
    ExperimentConfig,
    LoopConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    LRSchedulerConfig,
)
from cs336_basics.config.io import dump_experiment, load_experiment

from ._helpers import exception_leaves


def _write_yaml(tmp_path, raw):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def _set_value(raw, dotted_path, value):
    *parents, name = dotted_path.split(".")
    for parent in parents:
        raw = raw.setdefault(parent, {})
    raw[name] = value


def test_load_experiment_builds_nested_configs_and_applies_defaults(tmp_path, raw_experiment):
    raw_experiment["output_dir"] = "результаты/эксперимент"
    raw_experiment["optimizer"] = {}
    raw_experiment["model"].pop("rope_theta", None)
    raw_experiment["data"]["train"].pop("dtype", None)
    raw_experiment["data"]["batching"].pop("type", None)
    raw_experiment["loop"].pop("gradient_accumulation_steps", None)
    raw_experiment["loop"].pop("max_grad_norm", None)
    for field in ("seed", "lr_scheduler", "runtime", "checkpoint"):
        raw_experiment.pop(field, None)

    path = _write_yaml(tmp_path, raw_experiment)
    config = load_experiment(path)

    assert "результаты" in path.read_text(encoding="utf-8")
    assert isinstance(config, ExperimentConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.data.train, DatasetConfig)
    assert isinstance(config.data.batching, BatchingConfig)
    assert isinstance(config.loop, LoopConfig)
    assert isinstance(config.optimizer, OptimizerConfig)
    assert isinstance(config.lr_scheduler, LRSchedulerConfig)
    assert isinstance(config.runtime, RuntimeConfig)
    assert isinstance(config.checkpoint, CheckpointConfig)
    assert config.output_dir == Path("результаты/эксперимент")
    assert config.model.context_length == raw_experiment["model"]["context_length"]
    assert config.data.batching.sequence_length == raw_experiment["data"]["batching"]["sequence_length"]
    assert config.model.rope_theta == DEFAULT_ROPE_THETA
    assert config.data.train.dtype == "uint16"
    assert config.data.batching.type == "random_sequences"
    assert config.loop.gradient_accumulation_steps == 1
    assert config.loop.max_grad_norm == 1.0
    assert config.optimizer == OptimizerConfig()
    assert config.lr_scheduler == LRSchedulerConfig()
    assert config.runtime == RuntimeConfig()
    assert config.checkpoint == CheckpointConfig()
    assert config.seed == 42


def test_load_experiment_does_not_open_datasets_or_create_run_artifacts(tmp_path, raw_experiment):
    output_dir = tmp_path / "run"
    train_dataset_path = tmp_path / "missing-train-tokens.bin"
    eval_dataset_path = tmp_path / "missing-eval-tokens.bin"
    resume_path = tmp_path / "missing-checkpoint.pt"
    raw_experiment["output_dir"] = str(output_dir)
    raw_experiment["data"]["train"]["path"] = str(train_dataset_path)
    raw_experiment["data"]["eval"]["path"] = str(eval_dataset_path)
    raw_experiment["runtime"] = {"device": "cuda"}
    raw_experiment["checkpoint"] = {"resume_from": str(resume_path)}
    path = _write_yaml(tmp_path, raw_experiment)

    config = load_experiment(path)

    assert config.runtime.device == "cuda"
    assert config.data.train.path == train_dataset_path
    assert config.data.eval.path == eval_dataset_path
    assert config.checkpoint.resume_from == resume_path
    assert not output_dir.exists()
    assert not train_dataset_path.exists()
    assert not eval_dataset_path.exists()
    assert not resume_path.exists()
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_load_experiment_preserves_path_values(tmp_path, raw_experiment, absolute):
    paths = {
        "output_dir": Path("runs/example"),
        "data.train.path": Path("datasets/tokens.bin"),
        "checkpoint.resume_from": Path("checkpoints/state.pt"),
    }
    if absolute:
        paths = {field: tmp_path / value for field, value in paths.items()}
    raw_experiment["checkpoint"] = {}
    for field, value in paths.items():
        _set_value(raw_experiment, field, str(value))

    config = load_experiment(_write_yaml(tmp_path, raw_experiment))

    assert config.output_dir == paths["output_dir"]
    assert config.data.train.path == paths["data.train.path"]
    assert config.checkpoint.resume_from == paths["checkpoint.resume_from"]
    assert config.output_dir.is_absolute() is absolute


@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested"])
def test_load_experiment_rejects_duplicate_keys(tmp_path, raw_experiment, nested):
    raw_experiment.pop("optimizer" if nested else "output_dir")
    duplicate = "optimizer:\n  lr: 0.001\n  lr: 0.002\n" if nested else "output_dir: first\noutput_dir: second\n"
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw_experiment) + duplicate, encoding="utf-8")

    with pytest.raises(ConstructorError, match="duplicate key") as caught:
        load_experiment(path)

    assert caught.value.context_mark is not None
    assert caught.value.problem_mark is not None
    assert caught.value.context_mark.line < caught.value.problem_mark.line


@pytest.mark.parametrize("field", ["unknown_setting", "model.unknown_setting", "data.train.unknown_setting"])
def test_load_experiment_rejects_unknown_fields(tmp_path, raw_experiment, field):
    _set_value(raw_experiment, field, 1)

    with pytest.raises(ClassValidationError) as caught:
        load_experiment(_write_yaml(tmp_path, raw_experiment))

    assert any(
        isinstance(error, ForbiddenExtraKeysError) and "unknown_setting" in error.extra_fields
        for error in exception_leaves(caught.value)
    )


def test_load_experiment_generates_output_dir_and_preserves_it_in_dump(tmp_path, raw_experiment, monkeypatch):
    monkeypatch.chdir(tmp_path)
    del raw_experiment["output_dir"]
    path = _write_yaml(tmp_path, raw_experiment)

    config = load_experiment(path)

    assert config.output_dir.parent == Path("artifacts/training")
    datetime.strptime(config.output_dir.name, "%Y%m%d_%H%M%S")
    assert list(tmp_path.iterdir()) == [path]

    path.write_text(dump_experiment(config), encoding="utf-8")
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["output_dir"] == str(config.output_dir)
    assert load_experiment(path) == config
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("value", [None, "", False])
def test_load_experiment_rejects_invalid_explicit_output_dir(tmp_path, raw_experiment, value):
    raw_experiment["output_dir"] = value

    with pytest.raises(ClassValidationError) as caught:
        load_experiment(_write_yaml(tmp_path, raw_experiment))

    assert any(isinstance(error, ConfigError) for error in exception_leaves(caught.value))


@pytest.mark.parametrize("field", ["model.d_model", "data.batching.micro_batch_size"])
def test_load_experiment_rejects_missing_required_fields(tmp_path, raw_experiment, field):
    *parents, name = field.split(".")
    raw = raw_experiment
    for parent in parents:
        raw = raw[parent]
    del raw[name]

    with pytest.raises(ClassValidationError) as caught:
        load_experiment(_write_yaml(tmp_path, raw_experiment))

    assert any(isinstance(error, KeyError) and error.args == (name,) for error in exception_leaves(caught.value))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model.num_layers", 1.5),
        ("model.num_layers", True),
        ("model.num_layers", "2"),
        ("optimizer.lr", True),
        ("optimizer.lr", "0.001"),
        ("seed", 1.5),
    ],
)
def test_load_experiment_rejects_invalid_numeric_types(tmp_path, raw_experiment, field, value):
    _set_value(raw_experiment, field, value)

    with pytest.raises(ClassValidationError) as caught:
        load_experiment(_write_yaml(tmp_path, raw_experiment))

    assert any(isinstance(error, TypeError) and str(value) in str(error) for error in exception_leaves(caught.value))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("optimizer.lr", 0),
        ("optimizer.eps", -1),
        ("optimizer.weight_decay", -1),
        ("loop.max_steps", 0),
        ("optimizer.betas", [0.9, 1]),
        ("optimizer.betas", [True, 0.9]),
    ],
)
def test_load_experiment_rejects_invalid_numeric_values(tmp_path, raw_experiment, field, value):
    _set_value(raw_experiment, field, value)

    with pytest.raises((ClassValidationError, ValueError)) as caught:
        load_experiment(_write_yaml(tmp_path, raw_experiment))

    assert any(
        isinstance(error, ValueError) and field.split(".")[-1] in str(error)
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("optimizer.lr", float("nan")), ("model.rope_theta", float("inf")), ("loop.max_grad_norm", float("-inf"))],
)
def test_load_experiment_rejects_nonfinite_numbers(tmp_path, raw_experiment, field, value):
    _set_value(raw_experiment, field, value)

    with pytest.raises((ClassValidationError, ConfigError)) as caught:
        load_experiment(_write_yaml(tmp_path, raw_experiment))

    assert any(
        isinstance(error, ConfigError) and "finite" in str(error) and field.split(".")[-1] in str(error)
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime.device", "mps"),
        ("optimizer.type", "sgd"),
        ("lr_scheduler.type", "linear"),
        ("data.train.dtype", "float32"),
        ("data.batching.type", "sequential"),
    ],
)
def test_load_experiment_rejects_unsupported_literals(tmp_path, raw_experiment, field, value):
    _set_value(raw_experiment, field, value)

    with pytest.raises(ClassValidationError) as caught:
        load_experiment(_write_yaml(tmp_path, raw_experiment))

    attribute_name = field.split(".")[-1]
    assert any(
        any(getattr(note, "name", None) == attribute_name for note in getattr(error, "__notes__", ()))
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize(
    "constraint", ["sequence_length", "min_lr", "warmup_steps", "divisible", "even", "mutually exclusive"]
)
def test_load_experiment_checks_field_relationships(tmp_path, raw_experiment, constraint):
    if constraint == "sequence_length":
        _set_value(raw_experiment, "data.batching.sequence_length", raw_experiment["model"]["context_length"] + 1)
    elif constraint == "min_lr":
        _set_value(raw_experiment, "optimizer.lr", 0.001)
        _set_value(raw_experiment, "lr_scheduler.min_lr", 0.002)
    elif constraint == "warmup_steps":
        _set_value(raw_experiment, "lr_scheduler.warmup_steps", raw_experiment["loop"]["max_steps"])
    elif constraint == "divisible":
        raw_experiment["model"].update(d_model=10, num_heads=4)
    elif constraint == "even":
        raw_experiment["model"].update(d_model=12, num_heads=4)
    else:
        raw_experiment["checkpoint"] = {"resume_from": "resume.pt", "init_weights_from": "initial.pt"}

    with pytest.raises((ClassValidationError, ConfigError)) as caught:
        load_experiment(_write_yaml(tmp_path, raw_experiment))

    assert any(
        isinstance(error, ConfigError) and constraint in str(error) for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"save_checkpoint": False, "save_checkpoint_interval": 2},
        {"save_checkpoint": False, "save_checkpoint_at_the_end": True},
        {"save_checkpoint": True},
    ],
    ids=["disabled-with-periodic", "disabled-with-final", "enabled-without-trigger"],
)
def test_load_experiment_rejects_invalid_checkpoint_policy_before_side_effects(tmp_path, raw_experiment, checkpoint):
    raw_experiment["checkpoint"] = checkpoint
    path = _write_yaml(tmp_path, raw_experiment)

    with pytest.raises((ClassValidationError, ConfigError)) as caught:
        load_experiment(path)

    assert any(isinstance(error, ConfigError) for error in exception_leaves(caught.value))
    assert list(tmp_path.iterdir()) == [path]
    assert not Path(raw_experiment["output_dir"]).exists()
    assert not Path(raw_experiment["data"]["train"]["path"]).exists()


@pytest.mark.parametrize("content", ["", "null\n", "hello\n", "3\n", "false\n", "[]\n", "- first\n- second\n"])
def test_load_experiment_requires_a_mapping_root(tmp_path, content):
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="must contain a mapping"):
        load_experiment(path)


@pytest.mark.parametrize(
    "content", ["model: [1,\n", "---\nseed: 1\n---\nseed: 2\n"], ids=["syntax", "multiple-documents"]
)
def test_load_experiment_rejects_invalid_yaml(tmp_path, content):
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_experiment(path)


def test_load_experiment_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment(tmp_path / "missing.yaml")


def test_dump_experiment_includes_defaults_in_safe_yaml(tmp_path, raw_experiment):
    config = load_experiment(_write_yaml(tmp_path, raw_experiment))

    raw = yaml.safe_load(dump_experiment(config))

    assert set(raw) == {
        "output_dir",
        "model",
        "data",
        "optimizer",
        "loop",
        "logging",
        "seed",
        "lr_scheduler",
        "runtime",
        "checkpoint",
        "evaluation",
    }
    assert raw["seed"] == 42
    assert raw["model"]["rope_theta"] == DEFAULT_ROPE_THETA
    assert raw["data"]["train"]["dtype"] == "uint16"
    assert raw["data"]["eval"]["dtype"] == "uint16"
    assert raw["data"]["batching"]["type"] == "random_sequences"
    assert raw["optimizer"] == {
        "lr": 0.001,
        "type": "adamw",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
    }
    assert raw["loop"] == {"max_steps": 100, "gradient_accumulation_steps": 1, "max_grad_norm": 1.0}
    assert raw["lr_scheduler"] == {"type": "cosine", "warmup_steps": 0, "min_lr": 1e-6}
    assert raw["runtime"] == {"device": "cuda"}
    assert raw["checkpoint"] == {
        "resume_from": None,
        "init_weights_from": None,
        "save_checkpoint": False,
        "save_checkpoint_interval": None,
        "save_checkpoint_at_the_end": False,
    }
    assert raw["evaluation"] == {
        "do_eval": False,
        "eval_interval": None,
        "num_batches": None,
    }


def test_dump_experiment_serializes_paths_and_betas_without_mutating_config(tmp_path, raw_experiment):
    raw_experiment["output_dir"] = "результаты/эксперимент"
    raw_experiment["data"]["train"]["path"] = "данные/tokens.bin"
    raw_experiment["optimizer"]["betas"] = [0.8, 0.95]
    weights_path = tmp_path / "веса" / "initial.pt"
    raw_experiment["checkpoint"] = {"init_weights_from": str(weights_path)}
    config = load_experiment(_write_yaml(tmp_path, raw_experiment))
    before = asdict(config)

    text = dump_experiment(config)
    raw = yaml.safe_load(text)

    assert "результаты/эксперимент" in text
    assert "данные/tokens.bin" in text
    assert "веса" in text
    assert raw["output_dir"] == "результаты/эксперимент"
    assert raw["data"]["train"]["path"] == "данные/tokens.bin"
    assert raw["checkpoint"]["init_weights_from"] == str(weights_path)
    assert raw["optimizer"]["betas"] == [0.8, 0.95]
    assert asdict(config) == before
    assert isinstance(config.output_dir, Path)
    assert isinstance(config.checkpoint.init_weights_from, Path)
    assert isinstance(config.optimizer.betas, tuple)


@pytest.mark.parametrize("checkpoint_mode", ["none", "resume", "weights"])
@pytest.mark.parametrize(
    "save_policy",
    [
        {},
        {"save_checkpoint": True, "save_checkpoint_interval": 2},
        {"save_checkpoint": True, "save_checkpoint_at_the_end": True},
        {"save_checkpoint": True, "save_checkpoint_interval": 2, "save_checkpoint_at_the_end": True},
    ],
    ids=["disabled", "periodic", "final", "periodic-and-final"],
)
def test_dump_experiment_round_trips_through_load(tmp_path, raw_experiment, checkpoint_mode, save_policy):
    raw_experiment["output_dir"] = "результаты/эксперимент"
    raw_experiment["seed"] = 0
    raw_experiment["model"]["rope_theta"] = 5_000
    raw_experiment["data"]["train"]["dtype"] = "uint32"
    raw_experiment["optimizer"].update(lr=0.0005, betas=[0.8, 0.95], weight_decay=0)
    raw_experiment["loop"]["gradient_accumulation_steps"] = 2
    raw_experiment["lr_scheduler"] = {"min_lr": 0, "warmup_steps": 1}
    raw_experiment["runtime"] = {"device": "cpu"}
    raw_experiment["checkpoint"] = dict(save_policy)
    if checkpoint_mode == "resume":
        raw_experiment["checkpoint"]["resume_from"] = str(tmp_path / "state.pt")
    elif checkpoint_mode == "weights":
        raw_experiment["checkpoint"]["init_weights_from"] = "weights/initial.pt"
    config = load_experiment(_write_yaml(tmp_path, raw_experiment))
    dumped_path = tmp_path / "dumped.yaml"

    dumped_path.write_text(dump_experiment(config), encoding="utf-8")
    restored = load_experiment(dumped_path)

    assert restored == config
