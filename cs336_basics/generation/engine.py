from itertools import accumulate
from pathlib import Path

import torch

from cs336_basics.checkpoint.checkpoint import load_model_weights
from cs336_basics.config import EngineConfig, ModelConfig
from cs336_basics.transformers.build import build_model
from cs336_basics.transformers.paged_attention import PagedBatchMetadata, PagedForwardContext
from cs336_basics.transformers.paged_kv import PagedKVStore

from ..transformers.transformers import TransformerLM
from .results import ModelOutput
from .sampling import Sampler
from .scheduler import StepPlan


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

    @torch.inference_mode()
    def execute(self, *, plan: StepPlan, kv_store: PagedKVStore) -> ModelOutput:
        device = self.config.runtime.device
        int_dtype = torch.int64

        # token_ids
        token_ids = torch.tensor(
            [token_id for req in plan.requests for token_id in req.input_token_ids],
            dtype=int_dtype,
            device=device,
        )

        # token_positions
        token_positions = torch.cat(
            [torch.arange(req.num_input_tokens, dtype=int_dtype) + req.start_pos for req in plan.requests]
        ).to(device)

        # q_offsets
        q_offsets_cpu: tuple = tuple(
            accumulate(
                iterable=[req.num_input_tokens for req in plan.requests],
                func=lambda x, y: x + y,
                initial=0,
            )
        )
        q_offsets = torch.tensor(
            q_offsets_cpu,
            dtype=int_dtype,
            device=device,
        )

        # kv_lengths
        kv_lengths_cpu = tuple(req.start_pos + req.num_input_tokens for req in plan.requests)

        kv_lengths = torch.tensor(
            kv_lengths_cpu,
            dtype=int_dtype,
            device=device,
        )

        # block_tables
        max_block_tables = max([len(req.block_table) for req in plan.requests])

        block_tables = torch.tensor(
            [list(req.block_table) + [-1] * (max_block_tables - len(req.block_table)) for req in plan.requests],
            dtype=int_dtype,
            device=device,
        )

        write_slots = []

        for req in plan.requests:
            start_pos = req.start_pos
            for relative_position in range(req.num_input_tokens):
                token_position = relative_position + start_pos
                offset = token_position % kv_store.block_size
                block_id = req.block_table[token_position // kv_store.block_size]
                write_slot = block_id * kv_store.block_size + offset
                write_slots.append(write_slot)

        write_slots = torch.tensor(
            write_slots,
            dtype=int_dtype,
            device=device,
        )

        metadata = PagedBatchMetadata(
            q_offsets=q_offsets,
            kv_lengths=kv_lengths,
            block_tables=block_tables,
            write_slots=write_slots,
            q_offsets_cpu=q_offsets_cpu,
            kv_lengths_cpu=kv_lengths_cpu,
        )
        paged_context = PagedForwardContext(kv_store=kv_store, metadata=metadata)

        logits = self.model(
            token_ids=token_ids,
            token_positions=token_positions,
            paged_context=paged_context,
        )

        sampling_positions = []
        policies = []

        for i, req in enumerate(plan.requests):
            if req.do_sample:
                sampling_positions.append(q_offsets_cpu[i + 1] - 1)
                policies.append(req.sampling_params)

        if len(sampling_positions) == 0:
            return ModelOutput(next_tokens_gpu=token_ids.new_empty(size=(0,)))

        sampling_positions = torch.tensor(sampling_positions, dtype=int_dtype, device=device)
        last_token_logits = logits[sampling_positions]
        batch_sampling_args = self.sampler.prepare_sampling_args(policies=policies)

        next_tokens_gpu = self.sampler.sample(
            last_token_logits=last_token_logits,
            sampling_params=batch_sampling_args,
        )
        return ModelOutput(next_tokens_gpu=next_tokens_gpu)
