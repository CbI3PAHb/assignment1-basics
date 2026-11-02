"""The packed demo loads local artifacts and prints generated text as JSON lines.

Public entrypoint: generation.demo.main(argv=None) -> None, also callable with
uv run python -m cs336_basics.generation.demo. Pass --engine-config-path
(EngineConfig YAML), --tokenizer-pickle-path and --prompts-file-path (JSONL).
Each nonblank input line contains a prompt plus flat SamplingParams fields;
omitted fields keep the class defaults, and JSON null becomes None. Sampling
settings belong to each request, not global CLI arguments. These tests also pass
--num-blocks, --block-size, --max-step-input-tokens, --max-active-requests and
--max-prefill-chunk-tokens explicitly.

Use generate_offline once for all requests, preserving their policies. Scheduler
context capacity comes from the model config; the store uses that model's layers,
heads and head dimension, plus its actual parameter dtype/device. Print one JSON
object per input prompt, in order: {"prompt": original, "text": generated suffix
decoded WITHOUT the prompt, "finish_reason": reason.value}. Return None; loading
and generation exceptions propagate. No new config schema or validation policy.

The real checkpoint, tokenizer, Engine and model are used. Tiny artifacts are
created in tmp_path; these tests never download weights or training data.
"""

import importlib
import json
import pickle
from types import SimpleNamespace

import pytest
import torch

from cs336_basics.config import EngineConfig, ModelConfig, RuntimeConfig, SamplingParams, dump_config
from cs336_basics.generation.engine import Engine
from cs336_basics.generation.results import FinishReason
from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.transformers.build import build_model


@pytest.fixture
def demo_module():
    try:
        module = importlib.import_module("cs336_basics.generation.demo")
    except ModuleNotFoundError as error:
        if error.name != "cs336_basics.generation.demo":
            raise
        pytest.fail("Implement main(argv=None) in cs336_basics/generation/demo.py", pytrace=False)
    if not callable(getattr(module, "main", None)):
        pytest.fail("Implement main(argv=None) in cs336_basics/generation/demo.py", pytrace=False)
    return module


def _tokenizer_payload():
    return {
        "vocab": {**{i: bytes([i]) for i in range(256)}, 256: b"<eos>"},
        "merges": [],
        "special_tokens": ["<eos>"],
    }


def _artifacts(tmp_path, device):
    model_config = ModelConfig(
        vocab_size=257,
        context_length=16,
        d_model=12,
        num_layers=2,
        num_heads=3,
        d_ff=24,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71126)
        reference = build_model(model_config).eval()
    checkpoint = tmp_path / "tiny-checkpoint.pt"
    torch.save({"model_state_dict": reference.state_dict()}, checkpoint)
    config = EngineConfig(model=model_config, checkpoint=checkpoint, runtime=RuntimeConfig(device=device.type))
    config_path = tmp_path / "engine.yaml"
    config_path.write_text(dump_config(config), encoding="utf-8")

    payload = _tokenizer_payload()
    tokenizer_path = tmp_path / "tiny-tokenizer.pkl"
    tokenizer_path.write_bytes(pickle.dumps(payload))
    tokenizer = Tokenizer(**payload)
    return SimpleNamespace(
        config=config,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        reference=reference.to(device),
    )


def _write_requests(tmp_path, records):
    path = tmp_path / "requests.jsonl"
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    return path


def _argv(artifacts, prompts_path):
    return [
        "--engine-config-path",
        str(artifacts.config_path),
        "--tokenizer-pickle-path",
        str(artifacts.tokenizer_path),
        "--prompts-file-path",
        str(prompts_path),
        "--num-blocks",
        "8",
        "--block-size",
        "2",
        "--max-step-input-tokens",
        "3",
        "--max-active-requests",
        "2",
        "--max-prefill-chunk-tokens",
        "2",
    ]


def _observe_offline_driver(monkeypatch, demo_module):
    module = importlib.import_module("cs336_basics.generation.generate")
    original = module.generate_offline
    calls = []

    def observed(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module, "generate_offline", observed)
    # Support both module-qualified calls and direct imports, including aliases.
    for name, value in list(vars(demo_module).items()):
        if value is original:
            monkeypatch.setattr(demo_module, name, observed)
    return calls


@pytest.mark.usefixtures("isolated_root_logging")
def test_demo_loads_real_artifacts_and_matches_full_prefix_generation(
    demo_module, tmp_path, monkeypatch, capsys, execution_device
):
    artifacts = _artifacts(tmp_path, execution_device)
    prompts = ["hi", "éabc"]  # Two and five byte tokens.
    output_limits = [3, 2]
    prompts_path = _write_requests(
        tmp_path,
        [{"prompt": prompt, "max_new_tokens": limit} for prompt, limit in zip(prompts, output_limits, strict=True)],
    )
    expected_histories, expected_texts = [], []
    with torch.inference_mode():
        for prompt, limit in zip(prompts, output_limits, strict=True):
            history = artifacts.tokenizer.encode(prompt)
            prompt_length = len(history)
            for _ in range(limit):
                ids = torch.tensor([history], dtype=torch.int64, device=execution_device)
                history.append(int(artifacts.reference(ids)[0, -1].argmax().item()))
            expected_histories.append(tuple(history))
            expected_texts.append(artifacts.tokenizer.decode(history[prompt_length:]))

    calls = _observe_offline_driver(monkeypatch, demo_module)
    executions, forward_sizes, hooks = [], [], []
    original_execute = Engine.execute

    def record_forward(_module, args, kwargs, output):
        assert torch.is_inference_mode_enabled()
        assert torch.isfinite(output).all()
        forward_sizes.append(kwargs["token_ids"].numel())

    def execute(self, *, plan, kv_store):
        if not hooks:
            hooks.append(self.model.register_forward_hook(record_forward, with_kwargs=True))
        executions.append((plan, kv_store))
        return original_execute(self, plan=plan, kv_store=kv_store)

    monkeypatch.setattr(Engine, "execute", execute)
    try:
        assert demo_module.main(_argv(artifacts, prompts_path)) is None
    finally:
        for hook in hooks:
            hook.remove()

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records == [
        {"prompt": prompt, "text": text, "finish_reason": FinishReason.MAX_NEW_TOKENS.value}
        for prompt, text in zip(prompts, expected_texts, strict=True)
    ]
    assert len(calls) == 1, "Submit all prompts together through generate_offline"
    call = calls[0]
    engine, scheduler, store, requests = (call[key] for key in ("engine", "scheduler", "kv_store", "requests"))
    assert engine.config == artifacts.config
    assert not engine.model.training
    assert [tuple(req.token_ids) for req in requests] == expected_histories
    assert [req.prompt_tokens_count for req in requests] == [2, 5]
    assert [req.sampling_params.max_new_tokens for req in requests] == output_limits
    assert all(req.sampling_params.temperature is None and req.sampling_params.eos_token_id is None for req in requests)
    assert scheduler.max_step_input_tokens == 3
    assert scheduler.max_active_requests == 2
    assert scheduler.max_prefill_chunk_tokens == 2
    assert scheduler.max_context_tokens == artifacts.config.model.context_length
    assert scheduler.block_manager.num_blocks == store.num_blocks == 8
    assert scheduler.block_manager.block_size == store.block_size == 2
    assert store.k_pages.shape == store.v_pages.shape == (2, 8, 2, 3, 4)
    parameter = next(engine.model.parameters())
    assert store.k_pages.dtype == store.v_pages.dtype == parameter.dtype
    assert store.k_pages.device == store.v_pages.device == parameter.device == execution_device
    for key, expected in artifacts.reference.state_dict().items():
        torch.testing.assert_close(engine.model.state_dict()[key], expected, rtol=0, atol=0)

    a, b = [req.request_id for req in requests]
    assert [
        [(item.request_id, item.start_pos, item.num_input_tokens) for item in plan.requests] for plan, _ in executions
    ] == [
        [(a, 0, 2), (b, 0, 1)],
        [(a, 2, 1), (b, 1, 2)],
        [(a, 3, 1), (b, 3, 2)],
        [(b, 5, 1)],
    ]
    assert forward_sizes == [3, 3, 3, 1]
    assert all(used_store is store for _, used_store in executions)
    assert scheduler.schedule() is None
    assert scheduler.block_manager.free_block_count == scheduler.block_manager.available_block_count == 8


@pytest.mark.usefixtures("isolated_root_logging")
def test_demo_initial_eos_prints_empty_generated_text_without_forward(demo_module, tmp_path, monkeypatch, capsys):
    artifacts = _artifacts(tmp_path, torch.device("cpu"))
    prompts = ["already<eos>", "<eos>"]
    prompts_path = _write_requests(tmp_path, [{"prompt": prompt, "eos_token_id": 256} for prompt in prompts])
    calls = _observe_offline_driver(monkeypatch, demo_module)

    def unexpected_execute(self, *, plan, kv_store):
        pytest.fail("Initial EOS prompts need no model execution")

    monkeypatch.setattr(Engine, "execute", unexpected_execute)
    assert demo_module.main(_argv(artifacts, prompts_path)) is None

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records == [{"prompt": prompt, "text": "", "finish_reason": FinishReason.EOS.value} for prompt in prompts]
    assert len(calls) == 1
    scheduler = calls[0]["scheduler"]
    assert all(req.finished and req.num_computed_tokens == 0 for req in calls[0]["requests"])
    assert scheduler.schedule() is None
    assert scheduler.block_manager.free_block_count == scheduler.block_manager.available_block_count == 8


def test_jsonl_parser_preserves_prompt_whitespace_and_per_request_sampling(demo_module, tmp_path):
    tokenizer = Tokenizer(**_tokenizer_payload())
    records = [
        {"prompt": "  hi \t"},
        {"prompt": "éabc", "max_new_tokens": 7, "temperature": 0.7, "top_p": 0.9, "top_k": 5, "eos_token_id": 256},
        {"prompt": "last", "max_new_tokens": 5, "temperature": None, "top_p": 1.0, "top_k": None, "eos_token_id": None},
    ]
    path = tmp_path / "requests.jsonl"
    path.write_text(
        "\n \t\n" + "\n\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n \n",
        encoding="utf-8",
    )

    requests = demo_module.create_requests_from_file(path=path, tokenizer=tokenizer)

    assert len(requests) == 3
    expected_ids = [tokenizer.encode(record["prompt"]) for record in records]
    assert [req.token_ids for req in requests] == expected_ids
    assert [req.prompt_tokens_count for req in requests] == [len(ids) for ids in expected_ids]
    assert [req.sampling_params for req in requests] == [
        SamplingParams(),
        SamplingParams(max_new_tokens=7, temperature=0.7, top_p=0.9, top_k=5, eos_token_id=256),
        SamplingParams(max_new_tokens=5, temperature=None, top_p=1.0, top_k=None, eos_token_id=None),
    ]
    assert all(req.num_computed_tokens == req.generated_tokens_count == 0 and not req.finished for req in requests)
