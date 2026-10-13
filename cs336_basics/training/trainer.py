from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from cs336_basics.checkpoint.checkpoint import save_checkpoint
from cs336_basics.transformers.gradient_clipping import gradient_clipping
from cs336_basics.transformers.transformers import cross_entropy_loss

from .config import CheckpointConfig, LoopConfig


class Trainer:
    def __init__(
        self,
        /,
        *,
        loop: LoopConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        next_train_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        next_eval_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        lr_schedule: Callable[[int], float],
        completed_steps: int,
        checkpoint: CheckpointConfig,
        output_dir: Path,
    ) -> None:
        self.loop = loop
        self.model = model
        self.optimizer = optimizer
        self.next_train_batch = next_train_batch
        self.next_eval_batch = next_eval_batch
        self.lr_schedule: Callable[[int], float] = lr_schedule
        self.completed_steps: int = completed_steps
        self.checkpoint = checkpoint
        self.output_dir = output_dir

    def train(self) -> None:
        last_saved_step = None
        while self.completed_steps < self.loop.max_steps:
            self.train_step()
            self.completed_steps += 1

            interval = self.checkpoint.save_checkpoint_interval
            if (
                self.checkpoint.save_checkpoint
                and interval is not None
                and self.completed_steps % interval == 0
            ):
                self._save_checkpoint()
                last_saved_step = self.completed_steps

        if (
            self.checkpoint.save_checkpoint
            and self.checkpoint.save_checkpoint_at_the_end
            and last_saved_step != self.completed_steps
        ):
            self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        output_file = self.output_dir / f"checkpoint_{self.completed_steps:010d}.pt"
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=self.completed_steps,
            out=output_file,
        )

    def train_step(self) -> torch.Tensor:
        lr = self.lr_schedule(self.completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        losses = []
        for _ in range(self.loop.gradient_accumulation_steps):
            inputs, targets = self.next_train_batch()
            logits = self.model(inputs)
            loss = cross_entropy_loss(logits=logits, targets=targets)

            (loss / self.loop.gradient_accumulation_steps).backward()
            losses.append(loss.detach())

        gradient_clipping(self.model.parameters(), max_l2_norm=self.loop.max_grad_norm)
        self.optimizer.step()

        return torch.stack(losses).mean()
