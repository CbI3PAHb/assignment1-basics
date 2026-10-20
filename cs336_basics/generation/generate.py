from typing import Optional

import torch
import torch.nn as nn

from .requests import Req
from .results import FinishReason, GenerationResult


def generate(
    *,
    model: nn.Module,
    token_ids: list[int],
    eos_id: Optional[int],
    max_new_tokens: int,
    context_capacity: int,
    device: torch.device | str = "cpu",
) -> GenerationResult:
    if not token_ids:
        raise ValueError()
    if context_capacity <= 0 or max_new_tokens <= 0:
        raise ValueError()
    if len(token_ids) >= context_capacity:
        raise ValueError()

    return batched_generate(
        model=model,
        requests=[Req(token_ids=token_ids.copy())],
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        context_capacity=context_capacity,
        device=device,
    )[0]


def batched_generate(
    *,
    model: nn.Module,
    requests: list[Req],
    eos_id: Optional[int],
    max_new_tokens: int,
    context_capacity: int,
    device: torch.device | str = "cpu",
    pad_id: int = 0,
) -> list[GenerationResult]:

    model.eval()

    while True:
        for req in requests:
            if req.last_token() == eos_id:
                req.finish(FinishReason.EOS)
            elif req.generated_tokens_count >= max_new_tokens:
                req.finish(FinishReason.MAX_NEW_TOKENS)
            elif req.total_tokens_count >= context_capacity:
                req.finish(FinishReason.MAX_CONTEXT_CAPACITY)

        requests_to_generate = [
            req
            for req in requests
            if (
                not req.finished
                and req.generated_tokens_count < max_new_tokens
                and req.total_tokens_count < context_capacity
                and req.last_token() != eos_id
            )
        ]

        for req in requests_to_generate:
            req.start()

        if not requests_to_generate:
            break

        max_seq_len = max(req.total_tokens_count for req in requests_to_generate)
        batch = torch.stack(
            [
                torch.tensor(
                    req.token_ids + [pad_id] * (max_seq_len - req.total_tokens_count)
                )
                for req in requests_to_generate
            ]
        )

        batch = batch.to(device)
        with torch.inference_mode():
            logits = model.forward(batch) # [batch_size, seq_len, vocab_size]

        batch_indices = torch.arange(
            len(requests_to_generate),
            device=device,
        )

        last_token_positions = torch.tensor(
            [req.total_tokens_count - 1 for req in requests_to_generate],
            dtype=torch.int32,
            device=device,
        )

        last_token_logits = logits[batch_indices, last_token_positions, :]  # [batch, vocab]
        generated_tokens = torch.argmax(last_token_logits, dim=-1)  # [batch]
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
