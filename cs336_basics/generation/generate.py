from typing import Optional

import torch
import torch.nn as nn

from .requests import Batch, Req
from .results import FinishReason, GenerationResult
from .sampling import BatchSamplingArgs, Sampler


def generate(
    *,
    model: nn.Module,
    req: Req,
    sampler: Sampler,
    context_capacity: int,
    device: torch.device | str = "cpu",
) -> GenerationResult:

    if context_capacity <= 0 or req.sampling_params.max_new_tokens <= 0:
        raise ValueError()
    if len(req.token_ids) >= context_capacity:
        raise ValueError()

    return batched_generate(
        model=model,
        sampler=sampler,
        requests=[req],
        context_capacity=context_capacity,
        device=device,
    )[0]


def batched_generate(
    *,
    model: nn.Module,
    requests: list[Req],
    sampler: Sampler,
    context_capacity: int,
    device: torch.device | str = "cpu",
    pad_id: int = 0,
) -> list[GenerationResult]:
    if context_capacity <= 0:
        raise ValueError()
    if any(req.prompt_tokens_count >= context_capacity for req in requests):
        raise ValueError()

    model.eval()

    while True:
        for req in requests:
            if req.last_token() == req.sampling_params.eos_token_id:
                req.finish(FinishReason.EOS)
            elif req.generated_tokens_count >= req.sampling_params.max_new_tokens:
                req.finish(FinishReason.MAX_NEW_TOKENS)
            elif req.total_tokens_count >= context_capacity:
                req.finish(FinishReason.MAX_CONTEXT_CAPACITY)

        requests_to_generate = [
            req
            for req in requests
            if (
                not req.finished
                and req.generated_tokens_count < req.sampling_params.max_new_tokens
                and req.total_tokens_count < context_capacity
                and req.last_token() != req.sampling_params.eos_token_id
            )
        ]

        for req in requests_to_generate:
            req.start()

        if not requests_to_generate:
            break

        curr_batch = Batch(reqs=requests_to_generate)
        sampling_params: BatchSamplingArgs = sampler.prepare_batch_sampling_args(
            batch=curr_batch
        )

        max_seq_len = max(req.total_tokens_count for req in requests_to_generate)
        padded_batch = torch.stack(
            [
                torch.tensor(
                    req.token_ids + [pad_id] * (max_seq_len - req.total_tokens_count)
                )
                for req in requests_to_generate
            ]
        )

        padded_batch = padded_batch.to(device)
        with torch.inference_mode():
            logits = model.forward(padded_batch)  # [batch_size, seq_len, vocab_size]

        batch_indices = torch.arange(
            len(requests_to_generate),
            device=device,
        )

        last_token_positions = torch.tensor(
            [req.total_tokens_count - 1 for req in requests_to_generate],
            dtype=torch.int32,
            device=device,
        )

        last_token_logits = logits[
            batch_indices, last_token_positions, :
        ]  # [batch, vocab]
        generated_tokens = sampler.sample_impl(
            last_token_logits, sampling_params
        )  # [batch]
        generated_token_ids = generated_tokens.tolist()

        for req, token_id in zip(
            requests_to_generate,
            generated_token_ids,
            strict=True,
        ):
            req.append_token(token_id)

    result = []
    for req in requests:
        reason = req.finish_reason

        if reason is None:
            raise RuntimeError("Cannot return an unfinished request")

        res = GenerationResult(
            token_ids=tuple(req.token_ids),
            finish_reason=reason,
            # error=...,
        )
        result.append(res)

    return result
