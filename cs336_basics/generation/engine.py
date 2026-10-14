import torch.nn as nn

from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.training.config.inference import (
    GenerationConfig,
    InferenceEngineConfig,
)


class InferenceEngine:
    """Hold a model and tokenizer independently of files or HTTP requests."""

    def __init__(self, *, model: nn.Module, tokenizer: Tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Encode a prompt, generate token IDs, and decode the resulting text."""
        raise NotImplementedError(
            "Autoregressive text generation is not implemented yet"
        )


def build_inference_engine(config: InferenceEngineConfig) -> InferenceEngine:
    """Build the model, load its weights and tokenizer, and set evaluation mode."""

    raise NotImplementedError("Inference engine initialization is not implemented yet")
