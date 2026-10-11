"""Assemble an experiment and run its training loop."""

from .config.experiment import ExperimentConfig
from build import build_model, build_optimizer
from training.trainer import Trainer

def run_training(config: ExperimentConfig) -> None:
    """Construct training components and execute training.

    TODO: validate runtime resources, seed random generators, prepare output,
    construct and place the model, then build its optimizer, dataset and LR
    schedule. Apply checkpoint initialization or resume, execute Trainer, and
    close resources. Save the resolved experiment config with run artifacts.
    """
    model = build_model(config.model)
    model = model.to(config.runtime.device)

    optimizer = build_optimizer(config.optimizer, model.parameters())
    
    trainer = Trainer(
        loop=config.loop,
        model=model,
        optimizer=optimizer,
        next_batch=next_batch,
        lr_schedule=lr_schedule,
        completed_steps=completed_steps,
    )
