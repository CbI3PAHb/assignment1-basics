from pathlib import Path

from cs336_basics.checkpoint.checkpoint import load_model_weights
from cs336_basics.config import EngineConfig, ModelConfig
from cs336_basics.transformers.build import build_model

from ..transformers.transformers import TransformerLM
from .sampling import Sampler


class Engine:
    def __init__(self, config: EngineConfig):
        self.config: EngineConfig = config

        checkpoint_path: Path = config.checkpoint
        model_config: ModelConfig = config.model

        self.model: TransformerLM = build_model(model_config)

        load_model_weights(src=checkpoint_path, model=self.model)
        self.model = self.model.to(self.config.runtime.device)
        self.model.eval()

        self.sampler = Sampler(device=self.config.runtime.device, vocab_size=config.model.vocab_size)
