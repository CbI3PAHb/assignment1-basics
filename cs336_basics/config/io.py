"""Load one strict YAML config and serialize validated configuration objects."""

from pathlib import Path
from typing import TypeVar

import cattrs
import yaml

from ._cattrs import _make_converter
from ._errors import ConfigError
from ._yaml import _UniqueKeyLoader
from .experiment import ExperimentConfig
from .inference import InferenceMapperConfig, InferenceServerConfig

ConfigT = TypeVar(
    "ConfigT", ExperimentConfig, InferenceServerConfig, InferenceMapperConfig
)
_CONVERTER: cattrs.Converter = _make_converter()


def load_config(path: Path, config_cls: type[ConfigT]) -> ConfigT:
    """Read YAML and validate it: reject duplicate keys, unknown fields and invalid scalar types"""

    with path.open(encoding="utf-8") as config_file:
        raw = yaml.load(config_file, Loader=_UniqueKeyLoader)

    if not isinstance(raw, dict):
        raise ConfigError(
            f"YAML config must contain a mapping, got {type(raw).__name__}"
        )

    return _CONVERTER.structure(raw, config_cls)


def dump_config(
    config: ExperimentConfig | InferenceServerConfig | InferenceMapperConfig,
) -> str:
    raw = _CONVERTER.unstructure(config)
    return yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)


def load_experiment(path: Path) -> ExperimentConfig:
    return load_config(path, ExperimentConfig)


def dump_experiment(config: ExperimentConfig) -> str:
    return dump_config(config)
