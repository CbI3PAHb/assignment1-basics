import logging
from functools import partial
from pathlib import Path

import torch

from cs336_basics.checkpoint.checkpoint import load_checkpoint, load_model_weights
from cs336_basics.config.inference import InferenceMapperConfig
from cs336_basics.dataloader.dataloader import get_batch
from cs336_basics.logging_setup import setup_logging
from cs336_basics.prepare_output_directory import _prepare_output_directory
from cs336_basics.training.reproducibility import seed_everything

from ..config.io import dump_config
from .engine import InferenceEngine, build_inference_engine

logger = logging.getLogger(__name__)


def read_prompt(config: InferenceMapperConfig) -> str:
    """Read either literal prompt text or the contents of a UTF-8 input file"""

    raise NotImplementedError("Mapper input reading is not implemented yet")


def run_inference_mapper(config: InferenceMapperConfig) -> None:
    """Load one engine, read the input, generate and save output, then exit"""

    setup_logging(silent=config.logging.silent)
    seed_everything(config.seed)
    logger.info(f"Mapper config:\n{dump_config(config)}")

    if config.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")
    if config.engine.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")

    _prepare_output_directory(config.output_dir, dump_config(config))

    engine: InferenceEngine = build_inference_engine(config.engine)

    text = engine.generate(
        prompt="Once upon a time",
        generation=config.generation,
    )

    raise NotImplementedError("Mapper inference is not implemented yet")
