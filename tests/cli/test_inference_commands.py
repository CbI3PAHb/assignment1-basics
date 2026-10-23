"""CLI behavior after retiring the legacy inference execution commands."""

import sys

import pytest
import yaml

from cs336_basics import cli
from cs336_basics.config import InferenceMapperConfig
from cs336_basics.config.io import dump_config, load_config


@pytest.fixture
def mapper_config_path(tmp_path):
    raw = {
        "llm": {
            "model": {
                "vocab_size": 32,
                "context_length": 8,
                "d_model": 8,
                "num_layers": 1,
                "num_heads": 2,
                "d_ff": 16,
            },
            "checkpoint": str(tmp_path / "missing-checkpoint.pt"),
            "tokenizer": {"path": str(tmp_path / "missing-tokenizer.pkl")},
            "runtime": {"device": "cpu"},
        },
        "prompt": "One offline prompt",
        "output_dir": str(tmp_path / "inference-output"),
        "logging": {"silent": True},
    }
    path = tmp_path / "mapper.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["mapper", "server"])
def test_inference_run_is_rejected_before_config_loading(
    mode, tmp_path, monkeypatch, capsys
):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Retired inference execution must not load a configuration")

    monkeypatch.setattr(cli, "load_config", unexpected_load)

    with pytest.raises(SystemExit) as caught:
        cli.main(["inference", mode, "run", "--config", str(tmp_path / "missing.yaml")])

    captured = capsys.readouterr()
    assert caught.value.code == 2
    assert captured.out == ""
    assert (
        "legacy inference execution was removed; use python -m cs336_basics.generation.one_request_generate_demo"
    ) in captured.err
    assert list(tmp_path.iterdir()) == []


def test_mapper_config_still_prints_yaml_without_loading_generation_assets(
    mapper_config_path, monkeypatch, capsys
):
    monkeypatch.setitem(sys.modules, "cs336_basics.generation", None)
    expected = load_config(mapper_config_path, InferenceMapperConfig)

    assert (
        cli.main(["inference", "mapper", "config", "--config", str(mapper_config_path)])
        is None
    )

    captured = capsys.readouterr()
    assert yaml.safe_load(captured.out) == yaml.safe_load(dump_config(expected))
    assert captured.err == ""
    assert sys.modules["cs336_basics.generation"] is None
    assert not expected.output_dir.exists()
    assert not expected.llm.checkpoint.exists()
    assert not expected.llm.tokenizer.path.exists()
    assert list(mapper_config_path.parent.iterdir()) == [mapper_config_path]
