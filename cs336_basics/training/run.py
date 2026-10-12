"""Assemble an experiment and run its training loop."""

from functools import partial

import torch

from cs336_basics.checkpoint.checkpoint import load_checkpoint
from cs336_basics.dataloader.dataloader import get_batch
from cs336_basics.training.reproducibility import seed_everything

from .build import build_lr_schedule, build_model, build_optimizer, load_dataset
from .config.experiment import ExperimentConfig
from .trainer import Trainer


def run_training(config: ExperimentConfig) -> None:
    """Construct training components and execute training.

    TODO: validate runtime resources, seed random generators, prepare output,
    construct and place the model, then build its optimizer, dataset and LR
    schedule. Apply checkpoint initialization or resume, execute Trainer, and
    close resources. Save the resolved experiment config with run artifacts.
    """
    # 1. seed
    seed_everything(config.seed)

    # 2. prepare output
    if config.output_dir.exists():
        raise RuntimeError(f"working dir ({config.output_dir!r})")

    # there is important not to get race condition
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 3. construct model and place to device
    model = build_model(config.model)
    model = model.to(config.runtime.device)

    # 4. construct optimizer and lr scheduler
    optimizer = build_optimizer(config.optimizer, model.parameters())
    lr_schedule = build_lr_schedule(
        config.scheduler, peak_lr=config.optimizer.lr, max_steps=config.loop.max_steps
    )

    # 5. train dataset (who the fuck need validation dataset, lol)
    train_dataset = load_dataset(config.data.train)

    # ========================================================
    # 6. resume from and init weights from
    completed_steps = 0
    checkpoint = config.checkpoint

    if checkpoint.resume_from is not None:
        completed_steps = load_checkpoint(
            src=checkpoint.resume_from,
            model=model,
            optimizer=optimizer,
        )

    # this is not the place for this function
    elif checkpoint.init_weights_from is not None:
        artifact = torch.load(
            checkpoint.init_weights_from,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(artifact["model_state_dict"])
    # ========================================================

    batching = config.data.batching
    next_batch = partial(
        get_batch,
        dataset=train_dataset,
        batch_size=batching.micro_batch_size,
        context_length=batching.sequence_length,
        device=config.runtime.device,
    )

    trainer = Trainer(
        loop=config.loop,
        model=model,
        optimizer=optimizer,
        next_batch=next_batch,
        lr_schedule=lr_schedule,
        completed_steps=completed_steps,
    )

    trainer.train()
