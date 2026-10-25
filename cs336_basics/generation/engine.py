import time
from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from cs336_basics.checkpoint.checkpoint import load_model_weights
from cs336_basics.config import EngineConfig, ModelConfig
from cs336_basics.transformers.build import build_model
from cs336_basics.transformers.transformers import LayerKVCache

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
        # TODO: logging here
        # print(f"Creating CPU model ...")
        self.model: TransformerLM = build_model(model_config)

        # TODO: logging here
        # print(f"Loading model weights...")

        load_model_weights(src=checkpoint_path, model=self.model)
        self.model = self.model.to(self.config.runtime.device)
        self.model.eval()

        self.sampler = Sampler(
            device=self.config.runtime.device, vocab_size=config.model.vocab_size
        )
        # TODO: logging here
        # print("Model loadeded")
        # print(f"Time taken to create model: {(time.time() - start_time):.3f} sec")

    def forward_batch(
        self,
        *,
        batch: Batch,
        batch_sampling_args: BatchSamplingArgs,
        pad_id: int = 0,
        kv_caches: list[LayerKVCache] | None = None,
    ) -> ModelOutput:
        max_seq_len = max(req.total_tokens_count for req in batch.reqs)

        if kv_caches is None:
            # stays the same
            padded_batch = torch.stack(
                [
                    torch.tensor(
                        req.token_ids
                        + [pad_id] * (max_seq_len - req.total_tokens_count)
                    )
                    for req in batch.reqs
                ]
            )
            # token_positions = torch.arange(max_seq_len).expand(size=padded_batch.shape)
            token_positions = None

        else:
            if len(batch.reqs) != 1:
                raise ValueError("Cached execution currently supports one request")
            if len(kv_caches) != len(self.model.layers):
                raise ValueError(
                    f"Expected {len(self.model.layers)} layer caches, got {len(kv_caches)}"
                )

            lengths = [
                (
                    0
                    if cache.valid_kv_lengths is None
                    else int(cache.valid_kv_lengths[0].item())
                )
                for cache in kv_caches
            ]
            if len(set(lengths)) != 1:
                raise ValueError(f"Inconsistent layer cache lengths: {lengths}")

            L = lengths[0]
            # call the full request length T
            T = max_seq_len
            if L >= T:
                raise ValueError(
                    f"No uncomputed input tokens: cached_length={L}, request_length={T}"
                )
            padded_batch = torch.stack(
                [
                    torch.tensor(req.token_ids[L:T], dtype=torch.int64)
                    for req in batch.reqs
                ]
            )
            token_positions = torch.arange(
                L,
                T,
                dtype=torch.int64,
                device=self.config.runtime.device,
            ).unsqueeze(0)

        padded_batch = padded_batch.to(self.config.runtime.device)

        with torch.inference_mode():
            logits = self.model.forward(
                padded_batch,
                token_positions=token_positions,
                kv_caches=kv_caches,
            )  # [batch_size, seq_len, vocab_size]

            batch_indices = torch.arange(
                len(batch.reqs),
                device=self.config.runtime.device,
            )
        if kv_caches is None:
            last_token_positions = torch.tensor(
                [req.total_tokens_count - 1 for req in batch.reqs],
                dtype=torch.int64,
                device=self.config.runtime.device,
            )
            last_token_logits = logits[
                batch_indices, last_token_positions, :
            ]  # [batch, vocab]
        else:
            last_token_logits = logits[:, -1, :]

        next_tokens_gpu = self.sampler.sample(last_token_logits[:], batch_sampling_args)

        # TODO: deal smth with next_tokens_cpu
        # next_tokens_cpu = next_tokens_gpu.to("cpu")
        return ModelOutput(
            next_tokens_gpu=next_tokens_gpu,
            # next_tokens_cpu=next_tokens_cpu,
        )
