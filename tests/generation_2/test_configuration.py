"""Regressions for config forwarding and request-local EOS in offline inference."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import attrs
import pytest
import yaml
from cattrs.errors import ClassValidationError

from cs336_basics import cli
from cs336_basics.config.inference import InferenceMapperConfig, SamplingParams
from cs336_basics.config.io import dump_config, load_config


@pytest.mark.parametrize("eos_token_id", [None, 0, 7])
def test_eos_survives_yaml_round_trip(mapper_raw, write_mapper_config, eos_token_id):
    """Preserve EOS exactly, including zero and an explicitly disabled EOS stop."""
    mapper_raw["generation"]["eos_token_id"] = eos_token_id
    path = write_mapper_config(mapper_raw)

    config = load_config(path, InferenceMapperConfig)
    dumped = yaml.safe_load(dump_config(config))

    assert config.generation.eos_token_id == eos_token_id
    assert dumped["generation"]["eos_token_id"] == eos_token_id
    assert type(dumped["generation"]["eos_token_id"]) is type(eos_token_id)
    path.write_text(dump_config(config), encoding="utf-8")
    assert load_config(path, InferenceMapperConfig) == config
    assert not config.output_dir.exists()


@pytest.mark.parametrize("eos_token_id", [True, False, -1, 0.0, 1.5, "0"])
def test_yaml_rejects_invalid_eos(mapper_raw, write_mapper_config, eos_token_id):
    """Reject invalid token IDs before any runtime component can be constructed."""
    mapper_raw["generation"]["eos_token_id"] = eos_token_id

    with pytest.raises(ClassValidationError):
        load_config(write_mapper_config(mapper_raw), InferenceMapperConfig)


@pytest.mark.parametrize("eos_token_id", [True, -1, 0.0])
def test_python_api_also_validates_eos(eos_token_id):
    """Enforce token-ID constraints for callers that bypass YAML completely."""
    with pytest.raises((TypeError, ValueError)):
        SamplingParams(eos_token_id=eos_token_id)


@pytest.mark.parametrize("source", ["prompt", "file"])
def test_mapper_forwards_original_nested_configs(mapper_config, tmp_path, monkeypatch, source):
    """Read either input source and pass the existing config objects through unchanged."""
    from cs336_basics.generation_2 import mapper
    from cs336_basics.generation_2.core import GenerationResult

    text = "  Preserve spaces\nand newlines\n"
    if source == "file":
        input_path = tmp_path / "prompt.txt"
        input_path.write_text(text, encoding="utf-8")
        config = attrs.evolve(mapper_config, prompt=None, input_path=input_path)
    else:
        config = attrs.evolve(mapper_config, prompt=text)
    expected = GenerationResult(text="answer", token_ids=(0,), finish_reason="eos")
    llm = Mock()
    llm.generate.return_value = expected
    llm_factory = Mock(return_value=llm)
    monkeypatch.setattr(mapper, "LLM", llm_factory)

    actual = mapper.run_offline_inference(config)

    assert actual is expected
    llm_factory.assert_called_once()
    assert llm_factory.call_args.args[0] is config.engine
    llm.generate.assert_called_once()
    assert llm.generate.call_args.args[0] == text
    assert llm.generate.call_args.args[1] is config.generation


def test_cli_dispatches_the_original_mapper_config(mapper_config, monkeypatch):
    """The CLI selects the program and forwards its complete loaded configuration."""
    loader = Mock(return_value=mapper_config)
    run_mapper = Mock()
    runtime = ModuleType("cs336_basics.generation_2")
    runtime.run_offline_inference = run_mapper
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(cli, "load_config", loader)

    cli.main(["inference", "mapper", "run", "--config", "mapper.yaml"])

    loader.assert_called_once_with(Path("mapper.yaml"), InferenceMapperConfig)
    run_mapper.assert_called_once()
    assert run_mapper.call_args.args[0] is mapper_config
    assert run_mapper.call_args.kwargs == {}


def test_cli_config_prints_eos_without_importing_runtime(
    mapper_raw, write_mapper_config, monkeypatch, capsys
):
    """Config inspection works even when importing the inference package is forbidden."""
    path = write_mapper_config(mapper_raw)
    monkeypatch.setitem(sys.modules, "cs336_basics.generation_2", None)

    cli.main(["inference", "mapper", "config", "--config", str(path)])

    captured = capsys.readouterr()
    printed = yaml.safe_load(captured.out)
    assert printed["generation"]["eos_token_id"] == 0
    assert captured.err == ""
    assert sys.modules["cs336_basics.generation_2"] is None
    assert not Path(printed["output_dir"]).exists()


def test_reused_llm_uses_each_requests_eos(mapper_config, monkeypatch):
    """Run the real LLM/Scheduler lifecycle with new EOS settings on every call.

    The fake Engine emits the same sequence for each request, making differences
    in completion depend only on that request's SamplingParams. No model,
    checkpoint, tokenizer file, tensor computation or GPU is needed.
    """
    from cs336_basics.generation_2 import llm as llm_module
    from cs336_basics.generation_2 import scheduler as scheduler_module
    from cs336_basics.generation_2.core import ForwardOutput

    class FakeTokenizer:
        """Provide deterministic prompt tokens and a visible text representation."""

        def encode(self, prompt: str) -> list[int]:
            """Produce a nonempty prefix without loading a tokenizer vocabulary."""
            return [1, 2]

        def decode(self, token_ids: list[int]) -> str:
            """Expose selected token IDs so the result can be checked end to end."""
            return " ".join(str(token_id) for token_id in token_ids)

    class FakeEngine:
        """Replace model execution while retaining the production Scheduler."""

        def __init__(self, config) -> None:
            """Accept the usual Engine config without opening any referenced paths."""
            self.config = config

        def forward_batch(self, batch) -> ForwardOutput:
            """Emit 0, then 7, then 9 based on this request's current progress."""
            req = batch.reqs[0]
            token_id = (0, 7, 9)[len(req.generated_token_ids)]
            return ForwardOutput(next_token_ids=(token_id,))

    tokenizer_factory = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr(llm_module, "build_tokenizer", tokenizer_factory)
    monkeypatch.setattr(scheduler_module, "Engine", FakeEngine)
    llm = llm_module.LLM(mapper_config.engine)

    for eos_token_id, limit, expected_tokens, expected_reason in (
        (0, 3, (0,), "eos"),
        (7, 3, (0, 7), "eos"),
        (None, 3, (0, 7, 9), "length"),
        (7, 2, (0, 7), "eos"),
    ):
        params = SamplingParams(
            max_new_tokens=limit,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=eos_token_id,
        )

        result = llm.generate("prompt", params)

        assert result.token_ids == expected_tokens
        assert result.text == " ".join(str(token_id) for token_id in expected_tokens)
        assert result.finish_reason == expected_reason
        assert not llm.scheduler.has_work()

    tokenizer_factory.assert_called_once_with(mapper_config.engine.tokenizer)
