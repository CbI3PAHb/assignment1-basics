"""Checkpoint initialization, resuming, and saving configuration."""

from pathlib import Path

import attrs

from ._converters import _to_path
from ._errors import ConfigError
from ._validators import _POSITIVE_INTEGER


@attrs.frozen(kw_only=True)
class CheckpointConfig:
    """Resume full state or initialize weights; save_every_steps=None disables periodic saving"""

    resume_from: Path | None = attrs.field(default=None, converter=attrs.converters.optional(_to_path))
    init_weights_from: Path | None = attrs.field(default=None, converter=attrs.converters.optional(_to_path))
    save_every_steps: int | None = attrs.field(default=None, validator=attrs.validators.optional(_POSITIVE_INTEGER))

    def __attrs_post_init__(self) -> None:
        if self.resume_from is not None and self.init_weights_from is not None:
            raise ConfigError("resume_from and init_weights_from are mutually exclusive")
