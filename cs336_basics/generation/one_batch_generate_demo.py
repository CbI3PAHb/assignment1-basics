import time
from pathlib import Path

import torch

from cs336_basics.checkpoint.checkpoint import load_model_weights
from cs336_basics.config.inference import SamplingParams
from cs336_basics.config.io import load_config
from cs336_basics.config.tokenizer import TokenizerConfig
from cs336_basics.config.training import ExperimentConfig
from cs336_basics.generation import GenerationResult, Req, Sampler, batched_generate
from cs336_basics.tokenization.build import build_tokenizer
from cs336_basics.transformers.build import build_model


def main() -> None:
    prompts: list[str] = [
        "Once upon a time",
        "Explain how a black hole works",
        "Write a short story about a robot exploring Mars",
        "What are the main differences between Python and C++?",
        "The secret to happiness is",
    ]

    sampling_params = [
        SamplingParams(
            max_new_tokens=13,
            temperature=1,
            top_p=0.95,
            top_k=20,
        ),
        SamplingParams(
            max_new_tokens=10,
            temperature=0.3,
            top_p=0.8,
        ),
        SamplingParams(
            max_new_tokens=7,
            temperature=0.7,
        ),
        SamplingParams(
            max_new_tokens=8,
            top_p=0.9,
        ),
        SamplingParams(
            max_new_tokens=11,
        ),
    ]

    context_capacity = 64
    device = "cuda"

    repository_root = Path(__file__).resolve().parents[2]
    tokenizer_path = repository_root / "artifacts/tokenizers/owt_train/tokenizer.pkl"
    model_path = repository_root / "artifacts/training/train_small_adamw"
    checkpoint_path = model_path / "checkpoint_0000000100.pt"
    checkpoint_config_path = model_path / "config.yaml"

    tokenizer_config = TokenizerConfig(path=tokenizer_path)
    tokenizer = build_tokenizer(tokenizer_config)

    print(f"Loading model...")
    experiment_config = load_config(checkpoint_config_path, ExperimentConfig)
    model = build_model(experiment_config.model)
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

    end_time = time.time()
    print(f"Generation ends in {(end_time - start_time) * 1000:.3f}ms")

    for request, generation_result in zip(requests, generation_results, strict=True):
        generated_tokens = generation_result.token_ids[request.prompt_tokens_count :]
        prompt_tokens = generation_result.token_ids[: request.prompt_tokens_count]
        print(f"Prompt IDs: {prompt_tokens}")
        print(f"Generated IDs: {generated_tokens}")
        print(f"Decoded text : {tokenizer.decode(generation_result.token_ids)}")
        print(f"Decoded continuation: {tokenizer.decode(generated_tokens)}")
        print(f"Finish reason: {generation_result.finish_reason}")


if __name__ == "__main__":
    main()
