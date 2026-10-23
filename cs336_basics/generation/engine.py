import time
from pathlib import Path
from typing import Optional

import torch

from cs336_basics.checkpoint.checkpoint import load_model_weights
from cs336_basics.config import EngineConfig
from cs336_basics.config.training import ModelConfig
from cs336_basics.transformers.build import build_model

from ..transformers.transformers import TransformerLM
from .requests import Batch
from .results import ModelOutput
from .sampling import BatchSamplingArgs, Sampler


class Engine:
    def __init__(self, config: EngineConfig):
        self.config: EngineConfig = config

        checkpoint_path: Path = config.checkpoint
        model_config: ModelConfig = config.model

        start_time = time.time()
        # print(f"Creating CPU model ...")
        self.model: TransformerLM = build_model(model_config)

        # print(f"Loading model weights...")

        load_model_weights(src=checkpoint_path, model=self.model)
        self.model = self.model.to(self.config.runtime.device)
        self.model.eval()

        self.sampler = Sampler(
            device=self.config.runtime.device, vocab_size=config.model.vocab_size
        )

        # print("Model loadeded")
        # print(f"Time taken to create model: {(time.time() - start_time):.3f} sec")

    def forward_batch(
        self, batch: Batch, batch_sampling_args: BatchSamplingArgs, pad_id: int = 0
    ) -> ModelOutput:

        max_seq_len = max(req.total_tokens_count for req in batch.reqs)
        padded_batch = torch.stack(
            [
                torch.tensor(
                    req.token_ids + [pad_id] * (max_seq_len - req.total_tokens_count)
                )
                for req in batch.reqs
            ]
        )

        padded_batch = padded_batch.to(self.config.runtime.device)
        with torch.inference_mode():
            logits = self.model.forward(
                padded_batch
            )  # [batch_size, seq_len, vocab_size]

        batch_indices = torch.arange(
            len(batch.reqs),
            device=self.config.runtime.device,
        )

        last_token_positions = torch.tensor(
            [req.total_tokens_count - 1 for req in batch.reqs],
            dtype=torch.int32,
            device=self.config.runtime.device,
        )

        last_token_logits = logits[
            batch_indices, last_token_positions, :
        ]  # [batch, vocab]

        next_tokens_gpu = self.sampler.sample(last_token_logits[:], batch_sampling_args)
        # next_tokens_cpu = next_tokens_gpu.to("cpu")
        return ModelOutput(
            next_tokens_gpu=next_tokens_gpu,
            # next_tokens_cpu=next_tokens_cpu,
        )
