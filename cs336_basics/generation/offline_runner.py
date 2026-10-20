import torch
import torch.nn as nn

from cs336_basics.tokenization.tokenizer import Tokenizer

from .generate import generate
from .results import FinishReason, GenerationResult


def runner(
    *,
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: list[str],
    eos_id: int | None,
    max_new_tokens: int,
    context_capacity: int,
    device: torch.device | str = "cpu",
) -> list[GenerationResult]:
    generation_results = []

    for prompt in prompts:
        token_ids: list[int] = []

        try:
            token_ids = tokenizer.encode(prompt)
            generation_result: GenerationResult = generate(
                model=model,
                token_ids=token_ids,
                eos_id=eos_id,
                max_new_tokens=max_new_tokens,
                context_capacity=context_capacity,
                device=device,
            )
        except Exception as e:
            print(f"Generation of prompt {prompt!r} failed: {e}")
            generation_result = GenerationResult(
                token_ids=tuple(token_ids),
                finish_reason=FinishReason.ERROR,
                error=f"{type(e).__name__}: {e}",
            )
        else:
            print(f"generation of the prompt: {prompt} completed successfully")

        generation_results.append(generation_result)

    return generation_results
