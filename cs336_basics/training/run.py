"""Assemble an experiment and run its training loop."""

import logging
from functools import partial
from pathlib import Path

import torch

from cs336_basics.checkpoint.checkpoint import load_checkpoint, load_model_weights
from cs336_basics.dataloader.dataloader import get_batch
from cs336_basics.logging_setup import setup_logging
from cs336_basics.training.reproducibility import seed_everything

from .build import build_lr_schedule, build_model, build_optimizer, load_dataset
from .config.experiment import ExperimentConfig
from .config.io import dump_experiment
from .config.trainer import TrainerConfig
from .trainer import Trainer

logger = logging.getLogger(__name__)


def _prepare_output(output_dir: Path, config_yaml: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config.yaml"
    config_created = False
    try:
        with config_path.open("x", encoding="utf-8") as config_file:
            config_created = True
            config_file.write(config_yaml)
    except BaseException:
        # Remove only our new config and directory; preserve unrelated files.
        try:
            if config_created:
                config_path.unlink(missing_ok=True)
            output_dir.rmdir()
        except OSError:
            pass
        raise


def run_training(config: ExperimentConfig) -> None:
    """Prepare a new run directory, restore optional state, and execute training."""

    setup_logging(silent=config.logging.silent)
    logger.info(f"Experiment config:\n{dump_experiment(config)}")
    logger.info(f"Starting training run in {config.output_dir}")

    if config.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")
    if config.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")

    _prepare_output(config.output_dir, dump_experiment(config))

    train_dataset = load_dataset(config.data.train)
    eval_dataset = load_dataset(config.data.eval)
    batching = config.data.batching
    for split, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        if dataset.size <= batching.sequence_length:
            raise ValueError(
                f"{split} dataset must contain at least {batching.sequence_length + 1} tokens, got {dataset.size}"
            )

    seed_everything(config.seed)
    model = build_model(config.model)
    model = model.to(config.runtime.device)

    optimizer = build_optimizer(config.optimizer, model.parameters())
    lr_schedule = build_lr_schedule(
        config.scheduler, peak_lr=config.optimizer.lr, max_steps=config.loop.max_steps
    )

    completed_steps = 0
    checkpoint = config.checkpoint

    if checkpoint.resume_from is not None:
        completed_steps = load_checkpoint(
            src=checkpoint.resume_from,
            model=model,
            optimizer=optimizer,
        )

    elif checkpoint.init_weights_from is not None:
        load_model_weights(src=checkpoint.init_weights_from, model=model)

    next_train_batch = partial(
        get_batch,
        dataset=train_dataset,
        batch_size=batching.micro_batch_size,
        context_length=batching.sequence_length,
        device=config.runtime.device,
    )
    next_eval_batch = partial(
        get_batch,
        dataset=eval_dataset,
        batch_size=batching.micro_batch_size,
        context_length=batching.sequence_length,
        device=config.runtime.device,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        next_train_batch=next_train_batch,
        next_eval_batch=next_eval_batch,
        lr_schedule=lr_schedule,
        completed_steps=completed_steps,
        config=TrainerConfig(
            loop=config.loop,
            checkpoint=config.checkpoint,
            evaluation=config.evaluation,
            logging=config.logging,
        ),
        output_dir=config.output_dir,
    )
    trainer.train()
    logger.info(f"Training finished after {trainer.completed_steps} updates")
