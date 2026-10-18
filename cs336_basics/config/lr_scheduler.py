"""Learning rate scheduler configuration."""

from typing import Literal, get_args

import attrs

from ._validators import _NONNEGATIVE_INTEGER, _NONNEGATIVE_NUMBER

LRSchedulerType = Literal["cosine"]
SUPPORTED_SCHEDULER_TYPES = get_args(LRSchedulerType)


@attrs.frozen(kw_only=True)
class LRSchedulerConfig:
    type: LRSchedulerType = attrs.field(
        default="cosine", validator=attrs.validators.in_(SUPPORTED_SCHEDULER_TYPES)
    )
    warmup_steps: int = attrs.field(default=0, validator=_NONNEGATIVE_INTEGER)
    min_lr: float = attrs.field(default=1e-6, validator=_NONNEGATIVE_NUMBER)
