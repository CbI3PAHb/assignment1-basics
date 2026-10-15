"""Construct training components from their individual configuration objects."""

from collections.abc import Callable, Iterable

import attrs
import numpy as np
from torch.nn import Parameter
from torch.optim import Optimizer

from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.optimizer.learning_rate_scheduler import cosine_lr_scheduler

from ..config.data import DatasetConfig
from ..config.optimizer import OptimizerConfig
from ..config.scheduler import SchedulerConfig


def build_optimizer(
    config: OptimizerConfig, parameters: Iterable[Parameter]
) -> Optimizer:
    """Construct an optimizer for the supplied model parameters."""
    kwargs = attrs.asdict(config, recurse=False)
    optimizer_type = kwargs.pop("type")
    match optimizer_type:
        case "adamw":
            return AdamW(parameters, **kwargs)
        case _:
            raise NotImplementedError(f"Unsupported optimizer type: {optimizer_type!r}")


def load_dataset(config: DatasetConfig) -> np.memmap:
    """Open a flat token dataset as a read-only memory map with its configured dtype."""
    if not config.path.exists():
        raise RuntimeError(f"Dataset does not exist: {config.path}")

    if config.path.stat().st_size == 0:
        raise RuntimeError(f"File {config.path} is empty")

    return np.memmap(
        config.path,
        dtype=np.dtype(config.dtype),
        mode="r",
    )


def build_lr_schedule(
    config: SchedulerConfig,
    *,
    peak_lr: float,
    max_steps: int,
) -> Callable[[int], float]:
    """Return a function mapping the optimizer update index to a learning rate."""
    match config.type:
        case "cosine":

            def schedule(it: int) -> float:
                return cosine_lr_scheduler(
                    it=it,
                    max_learning_rate=peak_lr,
                    min_learning_rate=config.min_lr,
                    warmup_iters=config.warmup_steps,
                    cosine_cycle_iters=max_steps,
                )

            return schedule
        case _:
            raise NotImplementedError(f"Unsupported scheduler type: {config.type!r}")
