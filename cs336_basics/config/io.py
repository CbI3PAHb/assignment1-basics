"""load_config, dump_config, YAML and cattrs"""

from collections.abc import Hashable
from pathlib import Path
from typing import TypeVar

import cattrs
import yaml
from yaml.constructor import ConstructorError
from yaml.error import Mark
from yaml.nodes import MappingNode

from ._errors import ConfigError
from .inference import EngineConfig, SamplingParams
from .training import ExperimentConfig

ConfigT = TypeVar("ConfigT", ExperimentConfig, EngineConfig, SamplingParams)


def _structure_integer(value: object, _type: type[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an integer, got {value!r} ({type(value).__name__})")
    return value


def _structure_float(value: object, _type: type[float]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected int or float, got {value!r} ({type(value).__name__})")
    return value


def _structure_boolean(value: object, _type: type[bool]) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"expected a boolean, got {value!r} ({type(value).__name__})")
    return value


def _make_converter() -> cattrs.Converter:
    converter = cattrs.Converter(
        forbid_extra_keys=True,
        prefer_attrib_converters=True,
        omit_if_default=False,
    )
    converter.register_structure_hook(int, _structure_integer)
    converter.register_structure_hook(float, _structure_float)
    converter.register_structure_hook(bool, _structure_boolean)
    return converter


_CONVERTER: cattrs.Converter = _make_converter()


def load_config(path: Path, config_cls: type[ConfigT]) -> ConfigT:
    """Read YAML and validate it: reject duplicate keys, unknown fields and invalid scalar types"""

    with path.open(encoding="utf-8") as config_file:
        raw = yaml.load(config_file, Loader=_UniqueKeyLoader)

    if not isinstance(raw, dict):
        raise ConfigError(f"YAML config must contain a mapping, got {type(raw).__name__}")

    return _CONVERTER.structure(raw, config_cls)


def dump_config(config: ExperimentConfig | EngineConfig | SamplingParams) -> str:
    raw = _CONVERTER.unstructure(config)
    return yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)


def load_experiment(path: Path) -> ExperimentConfig:
    return load_config(path, ExperimentConfig)


def dump_experiment(config: ExperimentConfig) -> str:
    return dump_config(config)


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[object, object]:
        # Check the resulting mapping, including keys brought in by YAML merges
        self.flatten_mapping(node)
        seen: dict[Hashable, Mark] = {}
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            if key in seen:
                raise ConstructorError(
                    "first occurrence of this key",
                    seen[key],
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen[key] = key_node.start_mark
        return super().construct_mapping(node, deep=deep)
