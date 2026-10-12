"""Read one experiment YAML and serialize validated configuration objects."""

from pathlib import Path

import cattrs
import yaml

from ._cattrs import _make_converter
from ._errors import ConfigError
from ._yaml import _UniqueKeyLoader
from .experiment import ExperimentConfig

_CONVERTER: cattrs.Converter = _make_converter()


def load_experiment(path: Path) -> ExperimentConfig:
    """Read one YAML file and construct its validated ExperimentConfig.

    Reject duplicate keys, unknown fields and invalid scalar types before training
    """

    with path.open(encoding="utf-8") as config_file:
        raw = yaml.load(config_file, Loader=_UniqueKeyLoader)

    if not isinstance(raw, dict):
        raise ConfigError(
            f"experiment YAML must contain a mapping, got {type(raw).__name__}"
        )

    return _CONVERTER.structure(raw, ExperimentConfig)


def dump_experiment(config: ExperimentConfig) -> str:
    """Return YAML containing the complete config, including field defaults"""

    raw = _CONVERTER.unstructure(config)
    return yaml.safe_dump(
        raw,
        allow_unicode=True,
        sort_keys=False,
    )
