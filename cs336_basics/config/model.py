"""Transformer architecture configuration."""

import attrs

from ._errors import ConfigError
from ._validation import _POSITIVE_INTEGER, _POSITIVE_NUMBER

DEFAULT_ROPE_THETA = 10_000.0


@attrs.frozen(kw_only=True)
class ModelConfig:
    """Architecture settings; context_length is the maximum supported sequence length"""

    vocab_size: int = attrs.field(validator=_POSITIVE_INTEGER)
    context_length: int = attrs.field(validator=_POSITIVE_INTEGER)
    d_model: int = attrs.field(validator=_POSITIVE_INTEGER)
    num_layers: int = attrs.field(validator=_POSITIVE_INTEGER)
    num_heads: int = attrs.field(validator=_POSITIVE_INTEGER)
    d_ff: int = attrs.field(validator=_POSITIVE_INTEGER)
    rope_theta: float = attrs.field(
        default=DEFAULT_ROPE_THETA, validator=_POSITIVE_NUMBER
    )

    def __attrs_post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ConfigError(
                f"d_model must be divisible by num_heads, got d_model={self.d_model} and num_heads={self.num_heads}"
            )
        head_dimension = self.d_model // self.num_heads
        if head_dimension % 2 != 0:
            raise ConfigError(
                f"attention head dimension must be even for RoPE, got {head_dimension}"
            )
