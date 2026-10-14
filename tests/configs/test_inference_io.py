"""Shared YAML I/O and CLI dispatch for inference configuration."""

import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from cattrs.errors import ClassValidationError, ForbiddenExtraKeysError
from yaml.constructor import ConstructorError

from cs336_basics import cli
from cs336_basics.training.config.inference import (
    GenerationConfig,
    InferenceEngineConfig,
    InferenceMapperConfig,
    InferenceServerConfig,
)
from cs336_basics.training.config.io import dump_config, load_config
from cs336_basics.training.config.tokenizer import TokenizerConfig

from ._helpers import exception_leaves


@pytest.fixture
def inference_yaml(tmp_path, raw_experiment):
    raw = {
        "model": raw_experiment["model"],
        "tokenizer": {},
        "output_dir": str(tmp_path / "inference-output"),
        "runtime": {"device": "cpu"},
    }
    path = tmp_path / "inference.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_load_inference_config_and_round_trip_have_no_resource_side_effects(inference_yaml):
    config = load_config(inference_yaml, InferenceConfig)

    assert isinstance(config, InferenceConfig)
    assert isinstance(config.tokenizer, TokenizerConfig)
    assert config.runtime.device == "cpu"
    assert config.seed == 42
    assert config.checkpoint.init_weights_from is None
    assert isinstance(config.output_dir, Path)
    assert not config.output_dir.exists()
    raw = yaml.safe_load(dump_config(config))
    assert raw["output_dir"] == str(config.output_dir)
    assert "optimizer" not in raw
    assert "data" not in raw
    inference_yaml.write_text(dump_config(config), encoding="utf-8")
    assert load_config(inference_yaml, InferenceConfig) == config
    assert list(inference_yaml.parent.iterdir()) == [inference_yaml]


@pytest.mark.parametrize("field", ["model", "tokenizer"])
def test_inference_config_rejects_missing_required_fields(inference_yaml, field):
    raw = yaml.safe_load(inference_yaml.read_text(encoding="utf-8"))
    del raw[field]
    inference_yaml.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ClassValidationError) as caught:
        load_config(inference_yaml, InferenceConfig)

    assert any(isinstance(error, KeyError) and error.args == (field,) for error in exception_leaves(caught.value))


def test_inference_config_rejects_unknown_fields(inference_yaml):
    raw = yaml.safe_load(inference_yaml.read_text(encoding="utf-8"))
    raw["optimizer"] = {}
    inference_yaml.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ClassValidationError) as caught:
        load_config(inference_yaml, InferenceConfig)

    assert any(
        isinstance(error, ForbiddenExtraKeysError) and "optimizer" in error.extra_fields
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize("value", [1.0, 1.5, True, "1"])
def test_inference_config_rejects_coercion_for_integer_fields(inference_yaml, value):
    raw = yaml.safe_load(inference_yaml.read_text(encoding="utf-8"))
    raw["model"]["num_layers"] = value
    inference_yaml.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ClassValidationError) as caught:
        load_config(inference_yaml, InferenceConfig)

    assert any(isinstance(error, TypeError) for error in exception_leaves(caught.value))


def test_inference_config_rejects_duplicate_keys(inference_yaml):
    with inference_yaml.open("a", encoding="utf-8") as output:
        output.write("seed: 1\nseed: 2\n")

    with pytest.raises(ConstructorError, match="duplicate key"):
        load_config(inference_yaml, InferenceConfig)


def test_cli_inference_config_does_not_import_runtime(inference_yaml, monkeypatch, capsys):
    for module in ("cs336_basics.training.run", "cs336_basics.generation.run"):
        monkeypatch.setitem(sys.modules, module, None)
    expected = load_config(inference_yaml, InferenceConfig)

    cli.main(["inference", "config", "--config", str(inference_yaml)])

    output = capsys.readouterr()
    assert yaml.safe_load(output.out) == yaml.safe_load(dump_config(expected))
    assert output.err == ""
    assert not expected.output_dir.exists()


def test_cli_inference_run_receives_validated_config(inference_yaml, monkeypatch, capsys):
    received = []
    module = ModuleType("cs336_basics.generation.run")
    module.run_inference = received.append
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "cs336_basics.training.run", None)

    cli.main(["inference", "run", "--config", str(inference_yaml)])

    assert len(received) == 1
    assert isinstance(received[0], InferenceConfig)
    assert received[0] == load_config(inference_yaml, InferenceConfig)
    assert capsys.readouterr().out == ""
    assert not received[0].output_dir.exists()
