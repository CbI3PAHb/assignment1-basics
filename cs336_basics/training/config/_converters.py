"""Conversions for configuration values loaded from serialized settings."""

from pathlib import Path

from ._errors import ConfigError


def _to_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or value == "":
        raise ConfigError(f"path must be a Path or a nonempty string, got {value!r}")
    return Path(value)


def _to_betas(value: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise ConfigError(f"betas must be a tuple or list, got {value!r}")
    return tuple(value)
