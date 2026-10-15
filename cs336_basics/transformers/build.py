"""Construct Transformer models from configuration for training or inference."""

import attrs

from cs336_basics.config.model import ModelConfig

from .transformers import TransformerLM


def build_model(config: ModelConfig) -> TransformerLM:
    return TransformerLM(**attrs.asdict(config, recurse=False))
