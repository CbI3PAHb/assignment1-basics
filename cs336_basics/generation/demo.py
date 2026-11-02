import argparse
import json
from pathlib import Path
from typing import Sequence

import cattrs
import torch

from cs336_basics.config import EngineConfig, SamplingParams, TokenizerConfig, dump_config, load_config
from cs336_basics.logging_setup import setup_logging
from cs336_basics.tokenization.build import build_tokenizer
from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.transformers.paged_kv import PagedKVStore

from .engine import Engine
from .generate import generate_offline
from .kv_cache import BlockManager
from .requests import Req, SamplingParams
from .results import GenerationResult, ModelOutput
from .scheduler import Scheduler, StepPlan


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Offline inference")
    parser.add_argument(
        "--prompts-file-path",
        type=Path,
        default="/home/parii-artem/wip/EFDL/assignment1-basics/artifacts/generation/dummy_requests_5_short.txt",
    )
    parser.add_argument(
        "--tokenizer-pickle-path",
        type=Path,
        default="/home/parii-artem/wip/EFDL/assignment1-basics/artifacts/tokenizers/owt_train/tokenizer.pkl",
    )
    parser.add_argument(
        "--engine-config-path",
        type=Path,
        default="/home/parii-artem/wip/EFDL/assignment1-basics/artifacts/generation/engine_config.yaml",
    )

    parser.add_argument("--num-blocks", default=128, type=int, help="physical pages in the KV pool")
    parser.add_argument("--block-size", default=8, type=int, help="token slots per page")

    parser.add_argument("--max-step-input-tokens", type=int, default=32, help="")  # TODO: make help
    parser.add_argument("--max-active-requests", type=int, default=4, help="")  # TODO: make help
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=16, help="")  # TODO: make help
    return parser.parse_args(argv)


def create_requests_from_file(
    path: Path,
    tokenizer: Tokenizer,
) -> list[Req]:
    converter = cattrs.Converter(forbid_extra_keys=True)
    requests: list[Req] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            prompt = record.pop("prompt")
            sampling_params = converter.structure(record, SamplingParams)

            requests.append(
                Req(
                    token_ids=list(tokenizer.encode(prompt)),
                    sampling_params=sampling_params,
                )
            )

    return requests


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    # print(args)

    engine_config = load_config(args.engine_config_path, EngineConfig)
    tokenizer_config = TokenizerConfig(path=args.tokenizer_pickle_path)
    tokenizer: Tokenizer = build_tokenizer(config=tokenizer_config)

    setup_logging(silent=False)

    engine = Engine(engine_config)

    block_manager = BlockManager(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
    )
    scheduler = Scheduler(
        block_manager=block_manager,
        max_step_input_tokens=args.max_step_input_tokens,
        max_active_requests=args.max_active_requests,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        max_context_tokens=engine_config.model.context_length,
    )

    head_dim = engine_config.model.d_model // engine_config.model.num_heads
    dtype = torch.float32

    kv_store = PagedKVStore(
        num_layers=engine_config.model.num_layers,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        num_heads=engine_config.model.num_heads,
        head_dim=head_dim,
        device=engine_config.runtime.device,
        dtype=dtype,
    )
    requests: list[Req] = create_requests_from_file(
        path=args.prompts_file_path,
        tokenizer=tokenizer,
    )

    # for req in requests:
    #     print(f"\t{req}")

    generation_results: list[GenerationResult] = generate_offline(
        engine=engine,
        scheduler=scheduler,
        kv_store=kv_store,
        requests=requests,
    )
    for result, req in zip(generation_results, requests, strict=True):
        res = json.dumps(
            {
                "prompt": tokenizer.decode(result.token_ids[: req.prompt_tokens_count]),
                "text": tokenizer.decode(result.token_ids[req.prompt_tokens_count :]),
                "finish_reason": result.finish_reason,
            }
        )
        print(res)


if __name__ == "__main__":
    main()
