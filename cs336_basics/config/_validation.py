"""Shared attrs validators and converters for configuration fields."""

import math
from pathlib import Path

import attrs

from ._errors import ConfigError


def _validate_finite(
    _instance: object,
    attribute: attrs.Attribute,
    value: int | float,
) -> None:
    # Python integers are always finite; checking only floats avoids int-to-float overflow.
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{attribute.name} must be finite, got {value!r}")


# bool is an int subclass, but is not a numeric configuration value.
_NOT_BOOL = attrs.validators.not_(attrs.validators.instance_of(bool))
_INTEGER = attrs.validators.and_(attrs.validators.instance_of(int), _NOT_BOOL)
_FINITE_NUMBER = attrs.validators.and_(
    attrs.validators.instance_of((int, float)),
    _NOT_BOOL,
    _validate_finite,
)

_POSITIVE_INTEGER = attrs.validators.and_(_INTEGER, attrs.validators.gt(0))
_NONNEGATIVE_INTEGER = attrs.validators.and_(_INTEGER, attrs.validators.ge(0))
_POSITIVE_NUMBER = attrs.validators.and_(_FINITE_NUMBER, attrs.validators.gt(0))
_NONNEGATIVE_NUMBER = attrs.validators.and_(_FINITE_NUMBER, attrs.validators.ge(0))


def _to_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or value == "":
        raise ConfigError(f"path must be a Path or a nonempty string, got {value!r}")
    return Path(value)
