import time
from pathlib import Path

import torch

from cs336_basics.checkpoint.checkpoint import load_model_weights
from cs336_basics.config.io import load_config
from cs336_basics.config.tokenizer import TokenizerConfig
from cs336_basics.config.training import ExperimentConfig
from cs336_basics.generation import GenerationResult, generate
from cs336_basics.tokenization.build import build_tokenizer
from cs336_basics.transformers.build import build_model


def main() -> None:
    prompt = "Once upon a time"
    eos_id = None
    max_new_tokens = 8
    context_capacity = 64
    device = "cpu"

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

    token_ids = tokenizer.encode(prompt)

    print("generation process starts")
    start_time = time.time()

    with torch.inference_mode():
        generation_result: GenerationResult = generate(
            model=model,
            token_ids=token_ids,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            context_capacity=context_capacity,
            device=device,
        )

    end_time = time.time()
    generated_tokens = generation_result.token_ids[len(token_ids) :]

    print(f"Generation ends in {(end_time - start_time) * 1000:.3f}ms")
    print(f"Prompt IDs: {token_ids}")
    print(f"Generated IDs: {generated_tokens}")
    print(f"Decoded text : {tokenizer.decode(generation_result.token_ids)}")
    print(f"Decoded continuation: {tokenizer.decode(generated_tokens)}")
    print(f"Finish reason: {generation_result.finish_reason}")


if __name__ == "__main__":
    main()
