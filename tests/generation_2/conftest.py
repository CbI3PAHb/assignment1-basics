"""Small configuration fixtures; no checkpoint or tokenizer assets are needed."""

from pathlib import Path

import pytest
import yaml

from cs336_basics.config.inference import InferenceMapperConfig
from cs336_basics.config.io import load_config


@pytest.fixture
def mapper_raw(tmp_path: Path) -> dict:
    """Describe a complete mapper run with paths that deliberately do not exist.

    Configuration loading must validate the document without opening model
    weights or tokenizer assets. Runtime tests replace those components.
    """
    return {
        "engine": {
            "model": {
                "vocab_size": 32,
                "context_length": 32,
                "d_model": 8,
                "num_layers": 1,
                "num_heads": 2,
                "d_ff": 16,
            },
            "checkpoint": str(tmp_path / "weights.pt"),
            "tokenizer": {"path": str(tmp_path / "tokenizer.pkl")},
            "runtime": {"device": "cpu"},
        },
        "prompt": "A short prompt",
        "generation": {
            "max_new_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "eos_token_id": 0,
        },
        "output_dir": str(tmp_path / "output"),
    }


@pytest.fixture
def write_mapper_config(tmp_path: Path):
    """Write caller-supplied YAML data and return its path for loader/CLI tests."""

    def write(raw: dict) -> Path:
        """Serialize one document while leaving referenced runtime assets absent."""
        path = tmp_path / "mapper.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return path

    return write


@pytest.fixture
def mapper_config(mapper_raw: dict, write_mapper_config) -> InferenceMapperConfig:
    """Produce the same validated nested object that the real CLI would load."""
    return load_config(write_mapper_config(mapper_raw), InferenceMapperConfig)
