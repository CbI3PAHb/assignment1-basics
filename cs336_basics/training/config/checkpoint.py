"""Checkpoint initialization, resuming, and saving configuration."""

from pathlib import Path

import attrs

from ._converters import _to_path
from ._errors import ConfigError
from ._validators import _POSITIVE_INTEGER


@attrs.frozen(kw_only=True)
class CheckpointConfig:
    """Restore optional state and configure saving; intervals count completed optimizer updates."""

    resume_from: Path | None = attrs.field(default=None, converter=attrs.converters.optional(_to_path))
    init_weights_from: Path | None = attrs.field(default=None, converter=attrs.converters.optional(_to_path))
    save_checkpoint: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))
    save_checkpoint_interval: int | None = attrs.field(
        default=None, validator=attrs.validators.optional(_POSITIVE_INTEGER)
    )
    save_checkpoint_at_the_end: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))

    def __attrs_post_init__(self) -> None:
        if self.resume_from is not None and self.init_weights_from is not None:
            raise ConfigError("resume_from and init_weights_from are mutually exclusive")

        if not self.save_checkpoint:
            if self.save_checkpoint_interval is not None:
                raise ConfigError("save_checkpoint_interval must be None when save_checkpoint is False")
            if self.save_checkpoint_at_the_end:
                raise ConfigError("save_checkpoint_at_the_end must be False when save_checkpoint is False")
        elif self.save_checkpoint_interval is None and not self.save_checkpoint_at_the_end:
            raise ConfigError(
                "save_checkpoint=True requires save_checkpoint_interval or save_checkpoint_at_the_end=True"
            )
