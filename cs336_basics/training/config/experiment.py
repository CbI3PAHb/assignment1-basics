"""Experiment composition and validation between components."""

from datetime import datetime
from pathlib import Path

import attrs

from ._converters import _to_path
from ._errors import ConfigError
from ._validators import _NONNEGATIVE_INTEGER
from .checkpoint import CheckpointConfig
from .data import DataConfig
from .evaluation import EvaluationConfig
from .logging import LoggingConfig
from .loop import LoopConfig
from .model import ModelConfig
from .optimizer import OptimizerConfig
from .runtime import RuntimeConfig
from .scheduler import SchedulerConfig


def _default_output_dir() -> Path:
    """Choose a run path without creating any files or directories."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts/training") / timestamp


@attrs.frozen(kw_only=True)
class ExperimentConfig:
    """One experiment, composed from architecture and training settings"""

    output_dir: Path = attrs.field(factory=_default_output_dir, converter=_to_path)
    model: ModelConfig = attrs.field(
        validator=attrs.validators.instance_of(ModelConfig)
    )
    data: DataConfig = attrs.field(validator=attrs.validators.instance_of(DataConfig))
    optimizer: OptimizerConfig = attrs.field(
        validator=attrs.validators.instance_of(OptimizerConfig)
    )
    loop: LoopConfig = attrs.field(validator=attrs.validators.instance_of(LoopConfig))
    seed: int = attrs.field(
        default=42, validator=[_NONNEGATIVE_INTEGER, attrs.validators.le(2**32 - 1)]
    )
    scheduler: SchedulerConfig = attrs.field(
        factory=SchedulerConfig, validator=attrs.validators.instance_of(SchedulerConfig)
    )
    runtime: RuntimeConfig = attrs.field(
        factory=RuntimeConfig, validator=attrs.validators.instance_of(RuntimeConfig)
    )
    checkpoint: CheckpointConfig = attrs.field(
        factory=CheckpointConfig,
        validator=attrs.validators.instance_of(CheckpointConfig),
    )
    logging: LoggingConfig = attrs.field(
        factory=LoggingConfig,
        validator=attrs.validators.instance_of(LoggingConfig),
    )
    evaluation: EvaluationConfig = attrs.field(
        factory=EvaluationConfig,
        validator=attrs.validators.instance_of(EvaluationConfig),
    )

    def __attrs_post_init__(self) -> None:
        if self.data.batching.sequence_length > self.model.context_length:
            raise ConfigError(
                "sequence_length must not exceed model.context_length, "
                f"got {self.data.batching.sequence_length} > {self.model.context_length}"
            )
        if self.scheduler.min_lr > self.optimizer.lr:
            raise ConfigError(
                f"scheduler.min_lr must not exceed optimizer.lr, got {self.scheduler.min_lr} > {self.optimizer.lr}"
            )
        if self.scheduler.warmup_steps >= self.loop.max_steps:
            raise ConfigError(
                "scheduler.warmup_steps must be less than loop.max_steps, "
                f"got {self.scheduler.warmup_steps} >= {self.loop.max_steps}"
            )
