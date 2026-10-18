from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

import attrs

from ._errors import ConfigError
from ._validation import (
    _NONNEGATIVE_INTEGER,
    _NONNEGATIVE_NUMBER,
    _POSITIVE_INTEGER,
    _POSITIVE_NUMBER,
    _to_path,
)
from .execution import LoggingConfig, RuntimeConfig
from .model import ModelConfig

LRSchedulerType = Literal["cosine"]
OptimizerType = Literal["adamw"]
BatchingType = Literal["random_sequences"]
DatasetDType = Literal["uint16", "uint32"]

SUPPORTED_SCHEDULER_TYPES = get_args(LRSchedulerType)
SUPPORTED_OPTIMIZER_TYPES = get_args(OptimizerType)
SUPPORTED_BATCHING_TYPES = get_args(BatchingType)
SUPPORTED_DATASET_DTYPES = get_args(DatasetDType)


@attrs.frozen(kw_only=True)
class BatchingConfig:
    """Random contiguous token sequences per local pass, before gradient accumulation."""

    sequence_length: int = attrs.field(validator=_POSITIVE_INTEGER)
    micro_batch_size: int = attrs.field(validator=_POSITIVE_INTEGER)
    type: BatchingType = attrs.field(
        default="random_sequences",
        validator=attrs.validators.in_(SUPPORTED_BATCHING_TYPES),
    )


@attrs.frozen(kw_only=True)
class DatasetConfig:
    """A flat, already tokenized binary file; dtype describes its stored token IDs"""

    path: Path = attrs.field(converter=_to_path)
    dtype: DatasetDType = attrs.field(
        default="uint16", validator=attrs.validators.in_(SUPPORTED_DATASET_DTYPES)
    )


@attrs.frozen(kw_only=True)
class DataConfig:
    train: DatasetConfig = attrs.field(
        validator=attrs.validators.instance_of(DatasetConfig)
    )
    eval: DatasetConfig = attrs.field(
        validator=attrs.validators.instance_of(DatasetConfig)
    )
    batching: BatchingConfig = attrs.field(
        validator=attrs.validators.instance_of(BatchingConfig)
    )


def _to_betas(value: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise ConfigError(f"betas must be a tuple or list, got {value!r}")
    return tuple(value)


_BETAS = attrs.validators.and_(
    attrs.validators.instance_of(tuple),
    attrs.validators.min_len(2),
    attrs.validators.max_len(2),
    attrs.validators.deep_iterable(
        member_validator=attrs.validators.and_(
            _NONNEGATIVE_NUMBER, attrs.validators.lt(1)
        ),
    ),
)


@attrs.frozen(kw_only=True)
class OptimizerConfig:
    """Optimizer settings; lr is the peak learning rate used by the lr_scheduler"""

    lr: float = attrs.field(default=1e-3, validator=_POSITIVE_NUMBER)
    type: OptimizerType = attrs.field(
        default="adamw", validator=attrs.validators.in_(SUPPORTED_OPTIMIZER_TYPES)
    )
    betas: tuple[float, float] = attrs.field(
        default=(0.9, 0.999), converter=_to_betas, validator=_BETAS
    )
    eps: float = attrs.field(default=1e-8, validator=_POSITIVE_NUMBER)
    weight_decay: float = attrs.field(default=0.01, validator=_NONNEGATIVE_NUMBER)


@attrs.frozen(kw_only=True)
class LRSchedulerConfig:
    type: LRSchedulerType = attrs.field(
        default="cosine", validator=attrs.validators.in_(SUPPORTED_SCHEDULER_TYPES)
    )
    warmup_steps: int = attrs.field(default=0, validator=_NONNEGATIVE_INTEGER)
    min_lr: float = attrs.field(default=1e-6, validator=_NONNEGATIVE_NUMBER)


@attrs.frozen(kw_only=True)
class LoopConfig:
    """max_steps counts total optimizer updates, including updates completed before a resume"""

    max_steps: int = attrs.field(validator=_POSITIVE_INTEGER)
    gradient_accumulation_steps: int = attrs.field(
        default=1, validator=_POSITIVE_INTEGER
    )
    max_grad_norm: float = attrs.field(default=1.0, validator=_POSITIVE_NUMBER)


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


@attrs.frozen(kw_only=True)
class CheckpointConfig:
    """Restore optional state and configure saving; intervals count completed optimizer updates."""

    resume_from: Path | None = attrs.field(
        default=None, converter=attrs.converters.optional(_to_path)
    )
    init_weights_from: Path | None = attrs.field(
        default=None, converter=attrs.converters.optional(_to_path)
    )
    save_checkpoint: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    save_checkpoint_interval: int | None = attrs.field(
        default=None, validator=attrs.validators.optional(_POSITIVE_INTEGER)
    )
    save_checkpoint_at_the_end: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )

    def __attrs_post_init__(self) -> None:
        if self.resume_from is not None and self.init_weights_from is not None:
            raise ConfigError(
                "resume_from and init_weights_from are mutually exclusive"
            )

        if not self.save_checkpoint:
            if self.save_checkpoint_interval is not None:
                raise ConfigError(
                    "save_checkpoint_interval must be None when save_checkpoint is False"
                )
            if self.save_checkpoint_at_the_end:
                raise ConfigError(
                    "save_checkpoint_at_the_end must be False when save_checkpoint is False"
                )
        elif (
            self.save_checkpoint_interval is None
            and not self.save_checkpoint_at_the_end
        ):
            raise ConfigError(
                "save_checkpoint=True requires save_checkpoint_interval or save_checkpoint_at_the_end=True"
            )


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
    lr_scheduler: LRSchedulerConfig = attrs.field(
        factory=LRSchedulerConfig,
        validator=attrs.validators.instance_of(LRSchedulerConfig),
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
        if self.lr_scheduler.min_lr > self.optimizer.lr:
            raise ConfigError(
                f"lr_scheduler.min_lr must not exceed optimizer.lr, got {self.lr_scheduler.min_lr} > {self.optimizer.lr}"
            )
        if self.lr_scheduler.warmup_steps >= self.loop.max_steps:
            raise ConfigError(
                "lr_scheduler.warmup_steps must be less than loop.max_steps, "
                f"got {self.lr_scheduler.warmup_steps} >= {self.loop.max_steps}"
            )
