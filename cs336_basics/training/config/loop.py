"""Training duration, accumulation, and gradient clipping configuration."""

import attrs

from ._validators import _POSITIVE_INTEGER, _POSITIVE_NUMBER


@attrs.frozen(kw_only=True)
class LoopConfig:
    """max_steps counts total optimizer updates, including updates completed before a resume"""

    max_steps: int = attrs.field(validator=_POSITIVE_INTEGER)
    gradient_accumulation_steps: int = attrs.field(default=1, validator=_POSITIVE_INTEGER)
    max_grad_norm: float = attrs.field(default=1.0, validator=_POSITIVE_NUMBER)
