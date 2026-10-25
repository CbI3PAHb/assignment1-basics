from .engine import Engine
from .requests import Batch, Req
from .results import FinishReason, GenerationResult, ModelOutput
from .sampling import BatchSamplingArgs


def batched_generate(
    *,
    engine: Engine,
    requests: list[Req],
    context_capacity: int,
    pad_id: int = 0,
) -> list[GenerationResult]:
    if context_capacity <= 0:
        raise ValueError()
    if any(req.prompt_tokens_count >= context_capacity for req in requests):
        raise ValueError()

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
        sampling_params: BatchSamplingArgs = engine.sampler.prepare_batch_sampling_args(
            batch=curr_batch
        )

        model_output: ModelOutput = engine.forward_batch(
            batch=curr_batch,
            batch_sampling_args=sampling_params,
            pad_id=pad_id,
        )
        generated_token_ids = model_output.next_tokens_gpu.tolist()

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
            # error=..., # TODO: handle errors
        )
        result.append(res)

    return result
