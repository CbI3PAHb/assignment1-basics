import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from cs336_basics.checkpoint.checkpoint import load_model_weights
from cs336_basics.config import EngineConfig
from cs336_basics.config.inference import SamplingParams
from cs336_basics.config.io import load_config
from cs336_basics.config.tokenizer import TokenizerConfig
from cs336_basics.config.training import ExperimentConfig
from cs336_basics.generation import GenerationResult, Req, Sampler, batched_generate
from cs336_basics.tokenization.build import build_tokenizer
from cs336_basics.transformers.build import build_model

from .requests import Batch, Req
from .results import FinishReason, GenerationResult
from .sampling import BatchSamplingArgs, Sampler


class Engine:
    def __init__(self, config: EngineConfig):
        self.config: EngineConfig = config

        tokenizer_path = ...
        model_path = ...
        checkpoint_path = model_path / "checkpoint_0000000100.pt"
        checkpoint_config_path = model_path / "config.yaml"

        tokenizer_config = TokenizerConfig(path=tokenizer_path)
        tokenizer = build_tokenizer(tokenizer_config)

        # print(f"Loading model...")
        experiment_config = load_config(checkpoint_config_path, ExperimentConfig)
        self.model = build_model(conifg.model)
        load_model_weights(src=checkpoint_path, model=model)
        model = model.to(device)
        model.eval()

        requests = []

        for prompt, sampling_param in zip(prompts, sampling_params, strict=True):
            request = Req(
                token_ids=tokenizer.encode(prompt),
                sampling_params=sampling_param,
            )
            requests.append(request)

        sampler = Sampler(device=device, vocab_size=experiment_config.model.vocab_size)

        print("generation process starts")
        start_time = time.time()

        with torch.inference_mode():
            generation_results: list[GenerationResult] = batched_generate(
                model=model,
                requests=requests,
                sampler=sampler,
                context_capacity=context_capacity,
                device=device,
            )

    def step(self, batch: Batch) -> torch.Tensor: ...
