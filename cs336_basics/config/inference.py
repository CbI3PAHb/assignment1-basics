from pathlib import Path

import attrs

from ._validation import (
    _NONNEGATIVE_INTEGER,
    _POSITIVE_INTEGER,
    _POSITIVE_NUMBER,
    _to_path,
)
from .execution import RuntimeConfig
from .model import ModelConfig


@attrs.frozen(kw_only=True)
class EngineConfig:
    model: ModelConfig = attrs.field(validator=attrs.validators.instance_of(ModelConfig))
    checkpoint: Path = attrs.field(converter=_to_path)
    runtime: RuntimeConfig = attrs.field(factory=RuntimeConfig, validator=attrs.validators.instance_of(RuntimeConfig))


@attrs.frozen(kw_only=True)
class SamplingParams:
    """Per-request sampling settings and stopping conditions"""

    max_new_tokens: int = attrs.field(default=16, validator=_POSITIVE_INTEGER)
    temperature: float | None = attrs.field(default=None, validator=attrs.validators.optional(_POSITIVE_NUMBER))
    top_p: float = attrs.field(
        default=1.0,
        validator=[
            attrs.validators.optional(_POSITIVE_NUMBER),
            attrs.validators.le(1.0),
        ],
    )
    top_k: int | None = attrs.field(default=None, validator=attrs.validators.optional(_POSITIVE_INTEGER))
    eos_token_id: int | None = attrs.field(default=None, validator=attrs.validators.optional(_NONNEGATIVE_INTEGER))
