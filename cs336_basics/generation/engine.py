import torch.nn as nn

from cs336_basics.config.inference import GenerationConfig, InferenceEngineConfig
from cs336_basics.tokenization.build import build_tokenizer
from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.transformers.build import build_model


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
    model = build_model(config.model)
    tokenizer = build_tokenizer(config.tokenizer)

    return InferenceEngine(
        model=model,
        tokenizer=tokenizer,
    )
