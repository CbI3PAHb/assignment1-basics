"""EOS configuration regressions independent of the generation implementation."""

from pathlib import Path

import pytest
import yaml
from cattrs.errors import ClassValidationError

from cs336_basics.config.inference import SamplingParams
from cs336_basics.config.io import dump_config, load_config

from ._helpers import exception_leaves


@pytest.fixture
def sampling_raw() -> dict:
    return {
        "max_new_tokens": 3,
        "temperature": None,
        "top_p": 1.0,
        "eos_token_id": 0,
    }


def _write_sampling_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "sampling.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _identifies_eos(error: BaseException) -> bool:
    return "eos_token_id" in str(error) or any(
        getattr(note, "name", None) == "eos_token_id" for note in getattr(error, "__notes__", ())
    )


@pytest.mark.parametrize("eos_token_id", [None, 0, 7])
def test_eos_survives_yaml_round_trip_without_runtime_side_effects(sampling_raw, tmp_path, eos_token_id):
    sampling_raw["eos_token_id"] = eos_token_id
    path = _write_sampling_config(tmp_path, sampling_raw)

    config = load_config(path, SamplingParams)
    dumped = yaml.safe_load(dump_config(config))

    assert config.eos_token_id == eos_token_id
    assert type(config.eos_token_id) is type(eos_token_id)
    assert dumped["eos_token_id"] == eos_token_id
    assert type(dumped["eos_token_id"]) is type(eos_token_id)
    path.write_text(dump_config(config), encoding="utf-8")
    assert load_config(path, SamplingParams) == config
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("eos_token_id", [True, False, -1, 0.0, 1.5, "0"])
def test_yaml_rejects_invalid_eos_for_the_intended_field(sampling_raw, tmp_path, eos_token_id):
    path = _write_sampling_config(tmp_path, sampling_raw)
    # Establish that the unchanged fields form a valid configuration first.
    load_config(path, SamplingParams)
    sampling_raw["eos_token_id"] = eos_token_id
    _write_sampling_config(tmp_path, sampling_raw)

    with pytest.raises(ClassValidationError) as caught:
        load_config(path, SamplingParams)

    assert any(
        isinstance(error, (TypeError, ValueError)) and _identifies_eos(error)
        for error in exception_leaves(caught.value)
    )


@pytest.mark.parametrize("eos_token_id", [True, -1, 0.0])
def test_python_api_also_validates_eos(eos_token_id):
    with pytest.raises((TypeError, ValueError)) as caught:
        SamplingParams(eos_token_id=eos_token_id)

    assert _identifies_eos(caught.value)
