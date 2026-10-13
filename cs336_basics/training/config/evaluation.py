"""Logging"""

import attrs
from ._validators import _POSITIVE_INTEGER
from ._errors import ConfigError


class EvaluationConfigError(ConfigError):
    """lol kek"""

@attrs.frozen(kw_only=True)
class EvaluationConfig:
    do_eval: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    eval_interval: int | None = attrs.field(
        default=None, validator=attrs.validators.optional(_POSITIVE_INTEGER)
    )
    num_batches: int | None = attrs.field(
        default=None, validator=attrs.validators.optional(_POSITIVE_INTEGER)
    )

    def __attrs_post_init__(self) -> None:
        if not self.do_eval:
            if self.eval_interval is not None:
                raise EvaluationConfigError(
                    "eval_interval must be None when do_eval is False"
                )
            if self.num_batches is not None:
                raise EvaluationConfigError(
                    "num_batches must be None when do_eval is False"
                )
        else:
            if self.eval_interval is None:
                raise EvaluationConfigError(
                    "eval_interval must not be None when do_eval is True"
                )
            if self.num_batches is None:
                raise EvaluationConfigError(
                    "num_batches must not be None when do_eval is True"
                )
