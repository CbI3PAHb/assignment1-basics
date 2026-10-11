"""Execution environment configuration."""

from typing import Literal, get_args

import attrs

DeviceType = Literal["cpu", "cuda"]
SUPPORTED_DEVICE_TYPES = get_args(DeviceType)


@attrs.frozen(kw_only=True)
class RuntimeConfig:
    """CPU or CUDA execution; availability belongs to startup validation."""

    device: DeviceType = attrs.field(default="cuda", validator=attrs.validators.in_(SUPPORTED_DEVICE_TYPES))
