from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from cs336_basics.optimizer import AdamW
from cs336_basics.transformers.gradient_clipping import gradient_clipping
from cs336_basics.transformers.transformers import TransformerLM, cross_entropy_loss

from .config import (
    BatchingConfig,
    CheckpointConfig,
    ConfigError,
    DataConfig,
    DatasetConfig,
    ExperimentConfig,
    LoopConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SchedulerConfig,
)


class Trainer:
    def __init__(
        self,
        loop: LoopConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        next_batch: Callable,
        lr_schedule: Callable[[int], float],
        completed_steps: int,
    ) -> None:
        self.loop = loop
        self.model = model
        self.optimizer = optimizer
        self.next_batch = next_batch
        self.lr_schedule: Callable[[int], float] = lr_schedule
        self.completed_steps: int = completed_steps

    def train(self) -> None:
        while self.completed_steps < self.loop.max_steps:
            self.completed_steps += 1
            self.train_step()

    def train_step(self) -> torch.Tensor:
        lr = self.lr_schedule(self.completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        # self.optimizer.zero_grad()
        self.optimizer.zero_grad(set_to_none=True)  # ???

        for _ in range(self.loop.gradient_accumulation_steps):
            inputs, targets = self.next_batch()
            logits = self.model(inputs)
            loss = cross_entropy_loss(inputs=logits, targets=targets)

            (loss / self.loop.gradient_accumulation_steps).backward()

        gradient_clipping(self.model.parameters(), max_l2_norm=self.loop.max_grad_norm)
        self.optimizer.step()

        return loss.detach()
