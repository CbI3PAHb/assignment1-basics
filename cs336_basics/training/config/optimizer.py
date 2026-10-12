"""Optimizer configuration."""

from typing import Literal, get_args

import attrs

from ._converters import _to_betas
from ._validators import _BETAS, _NONNEGATIVE_NUMBER, _POSITIVE_NUMBER

OptimizerType = Literal["adamw"]
SUPPORTED_OPTIMIZER_TYPES = get_args(OptimizerType)


@attrs.frozen(kw_only=True)
class OptimizerConfig:
    """Optimizer settings; lr is the peak learning rate used by the scheduler"""

    lr: float = attrs.field(default=1e-3, validator=_POSITIVE_NUMBER)
    type: OptimizerType = attrs.field(
        default="adamw", validator=attrs.validators.in_(SUPPORTED_OPTIMIZER_TYPES)
    )
    betas: tuple[float, float] = attrs.field(
        default=(0.9, 0.999), converter=_to_betas, validator=_BETAS
    )
    eps: float = attrs.field(default=1e-8, validator=_POSITIVE_NUMBER)
    weight_decay: float = attrs.field(default=0.01, validator=_NONNEGATIVE_NUMBER)
