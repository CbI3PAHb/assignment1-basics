"""Settings used by Trainer independently of model and dataset construction."""

import attrs

from .checkpoint import CheckpointConfig
from .evaluation import EvaluationConfig
from .logging import LoggingConfig
from .loop import LoopConfig


@attrs.frozen(kw_only=True)
class TrainerConfig:
    loop: LoopConfig = attrs.field(validator=attrs.validators.instance_of(LoopConfig))
    checkpoint: CheckpointConfig = attrs.field(
        factory=CheckpointConfig,
        validator=attrs.validators.instance_of(CheckpointConfig),
    )
    evaluation: EvaluationConfig = attrs.field(
        factory=EvaluationConfig,
        validator=attrs.validators.instance_of(EvaluationConfig),
    )
