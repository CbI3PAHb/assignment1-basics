"""Logging"""

import attrs


@attrs.frozen(kw_only=True)
class LoggingConfig:
    silent: bool = attrs.field(
        default=False,
        validator=attrs.validators.instance_of(bool),
    )
