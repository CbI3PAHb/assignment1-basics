import logging
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from cs336_basics.checkpoint.checkpoint import save_checkpoint
from cs336_basics.transformers.gradient_clipping import gradient_clipping
from cs336_basics.transformers.transformers import cross_entropy_loss

from .config import TrainerConfig

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        /,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        next_train_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        next_eval_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        lr_schedule: Callable[[int], float],
        completed_steps: int,
        config: TrainerConfig,
        output_dir: Path,
    ) -> None:
        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.next_train_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]] = (
            next_train_batch
        )
        self.next_eval_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]] = (
            next_eval_batch
        )
        self.lr_schedule: Callable[[int], float] = lr_schedule
        self.completed_steps: int = completed_steps
        self.config = config
        self.output_dir = output_dir

    def _log_train_metrics(self, loss: torch.Tensor):
        if self.completed_steps % 10 == 0:
            logger.info(
                f"[train] step = {self.completed_steps:05d} | "
                f"lr = {self.optimizer.param_groups[0]['lr']:>12.6e} | "
                f"loss = {loss.item():12f} |"
            )

    def _log_eval_metrics(self, eval_metrics: dict[str, torch.Tensor]):
        logger.info(
            f"[eval]  step = {self.completed_steps:05d} | "
            f"lr = {self.optimizer.param_groups[0]['lr']:>12.6e} | "
            f"loss = {eval_metrics['evaluation_loss'].item():12f} | "
            f"perplexity = {eval_metrics['evaluation_perplexity'].item():12f} |"
        )

    def train(self) -> None:
        logger.info(
            f"Training started: {self.completed_steps} completed steps / {self.config.loop.max_steps} max steps"
        )
        last_saved_step = None

        while self.completed_steps < self.config.loop.max_steps:
            loss = self.train_step()
            self.completed_steps += 1

            self._log_train_metrics(loss)

            if (
                self.config.evaluation.do_eval
                and self.completed_steps % self.config.evaluation.eval_interval == 0
            ):
                eval_metrics: dict[str, torch.Tensor] = self.evaluate()
                self._log_eval_metrics(eval_metrics)

            if (
                self.config.checkpoint.save_checkpoint
                and self.config.checkpoint.save_checkpoint_interval is not None
                and self.completed_steps
                % self.config.checkpoint.save_checkpoint_interval
                == 0
            ):
                self._save_checkpoint()
                last_saved_step = self.completed_steps

        if (
            self.config.checkpoint.save_checkpoint
            and self.config.checkpoint.save_checkpoint_at_the_end
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
        logger.info(f"Checkpoint saved in {output_file}")

    def train_step(self) -> torch.Tensor:
        lr = self.lr_schedule(self.completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        losses = []
        for _ in range(self.config.loop.gradient_accumulation_steps):
            inputs, targets = self.next_train_batch()
            logits = self.model(inputs)
            loss = cross_entropy_loss(logits=logits, targets=targets)

            (loss / self.config.loop.gradient_accumulation_steps).backward()
            losses.append(loss.detach())

        gradient_clipping(
            self.model.parameters(), max_l2_norm=self.config.loop.max_grad_norm
        )
        self.optimizer.step()

        return torch.stack(losses).mean()

    def evaluate(self) -> dict[str, torch.Tensor]:
        if not self.config.evaluation.do_eval:
            raise ValueError("Evaluation is disabled")
        was_training = self.model.training
        self.model.eval()
        try:
            losses = []
            total_tokens = 0
            with torch.inference_mode():
                for _ in range(self.config.evaluation.num_batches):
                    inputs, targets = self.next_eval_batch()
                    logits = self.model(inputs)
                    loss = cross_entropy_loss(logits=logits, targets=targets)
                    losses.append(loss * targets.numel())
                    total_tokens += targets.numel()
                loss = torch.stack(losses).sum() / total_tokens
                return {
                    "evaluation_loss": loss,
                    "evaluation_perplexity": loss.exp(),
                }
        finally:
            self.model.train(was_training)
